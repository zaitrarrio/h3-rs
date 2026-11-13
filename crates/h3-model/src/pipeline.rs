//! Prepared conditioning → packed latents → denoising loop → decoded video
//! and audio, for every released sampler: Euler on DMD rungs or a uniform
//! grid (H3 base, FastH3 V1 / V2 / Trim), Pika's renoise knots, and FastH3
//! Omni's Parallel Decoding Distillation blocks. Regen-2K is in
//! [`crate::regen`].

use rayon::prelude::*;
use crate::{
    audio_vae::{AudioLatents, AudioVae},
    dit::{Attention, H3Dit, OutputHeads, StepMods, Workspace},
    taeh3::TaeH3,
    vae::{Posterior, Tiling, VideoVae, Volume},
};
use h3_core::{
    Error, Result,
    backend::{Backend, FlowStep},
    lora::Adapter,
    packing::{self, AUDIO_CHANNELS, Anchor, Geometry, Layout, RefGeom, TEXT_TAG, row_timesteps},
    rng::Rng,
    schedule::{KEYFRAME_NOISE_AUG, PddPlan, Schedule, StepRule},
    weights::WeightSource,
};
use std::time::Instant;

/// The reference's fixed seed for the VAE posterior sample of keyframes and
/// references (`MINIMAX_H3_KEYFRAME_ENCODE_SEED`).
pub const KEYFRAME_ENCODE_SEED: u64 = 42;

/// One prepared reference (Ref2VA): pixels already resized / trimmed by the
/// caller (h3-text's presentation builder does the geometry).
pub enum Reference {
    /// `[3, 1, H, W]` in `[0, 1]`.
    Image(Volume),
    /// `[3, T, H, W]` (T = 17n + 5) in `[0, 1]`, with an optional 32 kHz
    /// stereo soundtrack `[2][samples]`.
    Video { pixels: Volume, audio: Option<[Vec<f32>; 2]> },
    /// 32 kHz stereo.
    Audio([Vec<f32>; 2]),
}

/// A clip the target continues (Ref2VA): 5 / 22 / 39 frames with its soundtrack.
pub struct Continuation {
    pub pixels: Volume,
    pub audio: Option<[Vec<f32>; 2]>,
}

/// What the text side produced: the conditioner's `[rows, text_dim]` hidden
/// states and each row's tag (1 text, 0 vision placeholder).
pub struct Prompt {
    pub embeds: Vec<f32>,
    pub tags: Vec<u8>,
}

impl Prompt {
    pub fn rows(&self, text_dim: usize) -> usize {
        self.embeds.len() / text_dim
    }
    /// A text-only prompt (every row tagged text).
    pub fn text_only(embeds: Vec<f32>, text_dim: usize) -> Self {
        let n = embeds.len() / text_dim;
        Self { embeds, tags: vec![TEXT_TAG; n] }
    }
}

/// How the denoising loop walks its timesteps.
pub enum SamplerPlan {
    /// Video and audio schedules of equal length; `rule` Euler or Renoise.
    Schedules { video: Schedule, audio: Schedule },
    /// Fused PDD heads over node sigmas.
    Pdd(PddPlan),
}

pub struct Request {
    pub prompt: Prompt,
    pub geometry: Geometry,
    pub seed: u64,
    /// FL2VA keyframes, in anchor order; `[3, 1, H, W]` at the canvas size.
    pub keyframes: Vec<(Anchor, Volume)>,
    /// Ref2VA references, in presentation order.
    pub references: Vec<Reference>,
    pub continuation: Option<Continuation>,
    pub sampler: SamplerPlan,
    pub attention: Attention,
    /// Steps run dense before VSA kicks in (`vsa_dense_first_n_steps`).
    pub dense_first_steps: usize,
    pub ffn_chunk: usize,
    pub tiling: Option<Tiling>,
    /// Decode with TAEH3 instead of the VAE decoder.
    pub preview: bool,
}

pub struct Pipeline<B: Backend> {
    pub dit: H3Dit<B>,
    pub vae: Option<VideoVae<B>>,
    pub audio_vae: Option<AudioVae<B>>,
    pub tae: Option<TaeH3<B>>,
}

#[derive(Debug, Clone, Default)]
pub struct Timings {
    pub prepare_s: f64,
    pub adaln_s: f64,
    pub steps_s: Vec<f64>,
    pub decode_s: f64,
    pub audio_s: f64,
}

impl Timings {
    pub fn denoise_s(&self) -> f64 {
        self.steps_s.iter().sum()
    }
}

/// Progress of a generation, in order: `Prepared`, one `Step` per forward,
/// then `Frames` per decoded chunk (8-bit RGB), then `Audio`.
pub enum Event<'a> {
    Prepared { secs: f64, tokens: usize },
    Step { index: usize, total: usize, secs: f64 },
    Frames { first: usize, count: usize, width: usize, height: usize, rgb: &'a [u8] },
    Audio { sample_rate: usize, stereo: &'a [Vec<f32>; 2] },
}

/// A finished clip: frames × height × width × RGB (8-bit) and 32 kHz stereo.
pub struct Generation {
    pub frames: usize,
    pub width: usize,
    pub height: usize,
    pub rgb: Vec<u8>,
    pub audio: Option<[Vec<f32>; 2]>,
    pub sample_rate: usize,
    pub timings: Timings,
}

/// The packed latent state of one request before denoising.
pub struct Prepared {
    pub layout: Layout,
    /// `[Nv, video_patch_dim]` rows: condition rows then the target.
    pub video_rows: Vec<f32>,
    /// `[Na, audio_in]` rows: condition rows then the target.
    pub audio_rows: Vec<f32>,
}

fn stereo_latents<B: Backend>(be: &B, avae: &AudioVae<B>, audio: &[Vec<f32>; 2]) -> Result<Vec<f32>> {
    // Channel-major rows: all of channel 0's frames, then channel 1's.
    let mut rows = Vec::new();
    let mut frames = None;
    for ch in audio {
        let mut z = avae.encode(be, ch)?.mode();
        avae.normalize_latents(&mut z);
        if let Some(f) = frames {
            if f != z.frames {
                return Err(Error::Geometry("stereo channels encode to different lengths".into()));
            }
        }
        frames = Some(z.frames);
        rows.extend(z.rows());
    }
    Ok(rows)
}

impl<B: Backend> Pipeline<B> {
    /// Normalised latents of pixels, sampled from the posterior with the
    /// reference's fixed seed and rounded through f16 as the reference does.
    fn encode_visual(&self, be: &B, pixels: &Volume, keyframe: bool, tiling: Option<Tiling>) -> Result<Volume> {
        let vae = self.vae.as_ref().ok_or_else(|| Error::Msg("conditioning needs the video VAE encoder".into()))?;
        let m = if keyframe { vae.encode_keyframe(be, pixels, tiling)? } else { vae.encode_pixels(be, pixels, tiling)? };
        let mut rng = Rng::new(KEYFRAME_ENCODE_SEED);
        let mut z = Posterior::from_moments(&m).sample(&mut rng);
        for v in &mut z.data {
            *v = half::f16::from_f32(*v).to_f32();
        }
        vae.normalize_latents(&mut z);
        Ok(z)
    }

    fn patch_rows(&self, z: &Volume) -> Vec<f32> {
        patch_rows_of(z, self.dit.cfg.patch)
    }
}

/// Patchified rows of a latent volume (channel-major features).
pub fn patch_rows_of(z: &Volume, (pt, ph, pw): (usize, usize, usize)) -> Vec<f32> {
    {
        let (c, t, h, w) = (z.c, z.t, z.h, z.w);
        let (tt, hh, ww) = (t / pt, h / ph, w / pw);
        let f = c * pt * ph * pw;
        let mut out = vec![0.0f32; tt * hh * ww * f];
        for row in 0..tt * hh * ww {
            let (ti, rem) = (row / (hh * ww), row % (hh * ww));
            let (hi, wi) = (rem / ww, rem % ww);
            for ci in 0..c {
                for kt in 0..pt {
                    for kh in 0..ph {
                        for kw in 0..pw {
                            out[row * f + ((ci * pt + kt) * ph + kh) * pw + kw] = z.at(ci, ti * pt + kt, hi * ph + kh, wi * pw + kw);
                        }
                    }
                }
            }
        }
        out
    }
}

impl<B: Backend> Pipeline<B> {
    /// `0.999·clean + 0.001·noise` with noise from `rng` (`scale_noise` at
    /// the keyframe augmentation level).
    fn augment(rows: &mut [f32], rng: &mut Rng) {
        let noise = rng.normal(rows.len());
        for (r, n) in rows.iter_mut().zip(noise) {
            *r = KEYFRAME_NOISE_AUG * *r + (1.0 - KEYFRAME_NOISE_AUG) * n;
        }
    }

    /// Encode conditions, build the layout, draw the target noise.
    pub fn prepare(&self, be: &B, req: &Request) -> Result<Prepared> {
        let cfg = &self.dit.cfg;
        let g = &req.geometry;
        let ratio = self.vae.as_ref().map_or(16, |v| v.spatial_ratio());
        g.validate(ratio)?;
        let (lt, lh, lw) = g.latent(ratio);
        let na = g.audio_latents();
        let mut rng = Rng::new(req.seed);
        let text_dim = cfg.text_dim;
        if req.prompt.tags.len() != req.prompt.rows(text_dim) {
            return Err(Error::Geometry("prompt tags and rows disagree".into()));
        }
        let mut cond_video: Vec<f32> = Vec::new();
        let mut cond_audio: Vec<f32> = Vec::new();
        let layout;
        if !req.references.is_empty() || req.continuation.is_some() {
            let mut refs = Vec::new();
            // Visual rows first (all references), then the clip; the keyframe
            // noise uses the request generator, the clip its own.
            let mut visual_rows = Vec::new();
            let mut shapes = Vec::new();
            for r in &req.references {
                match r {
                    Reference::Image(px) => {
                        let z = self.encode_visual(be, px, true, req.tiling)?;
                        refs.push(RefGeom::Image { latent_h: z.h, latent_w: z.w });
                        shapes.push((z.t, z.h, z.w));
                        visual_rows.push(self.patch_rows(&z));
                    }
                    Reference::Video { pixels, audio } => {
                        let trimmed = packing::trim_reference_frames(pixels.t);
                        let px = pixels.crop(0, trimmed, 0, pixels.h, 0, pixels.w);
                        let z = self.encode_visual(be, &px, false, req.tiling)?;
                        let mut al = 0;
                        if let Some(a) = audio {
                            let avae = self.audio_vae.as_ref().ok_or_else(|| Error::Msg("references with audio need the audio VAE".into()))?;
                            let rows = stereo_latents(be, avae, a)?;
                            al = rows.len() / cfg.audio_in_channels / AUDIO_CHANNELS;
                            cond_audio.extend(rows);
                        }
                        refs.push(RefGeom::Video { latent_t: z.t, latent_h: z.h, latent_w: z.w, audio_latents: al });
                        shapes.push((z.t, z.h, z.w));
                        visual_rows.push(self.patch_rows(&z));
                    }
                    Reference::Audio(a) => {
                        let avae = self.audio_vae.as_ref().ok_or_else(|| Error::Msg("audio references need the audio VAE".into()))?;
                        let rows = stereo_latents(be, avae, a)?;
                        let al = rows.len() / cfg.audio_in_channels / AUDIO_CHANNELS;
                        cond_audio.extend(rows);
                        refs.push(RefGeom::Audio { audio_latents: al });
                    }
                }
            }
            let mut all: Vec<f32> = visual_rows.concat();
            Self::augment(&mut all, &mut rng);
            cond_video.extend(all);
            let mut cont = None;
            if let Some(c) = &req.continuation {
                let z = self.encode_visual(be, &c.pixels, false, req.tiling)?;
                if (z.h, z.w) != (lh, lw) {
                    return Err(Error::Geometry("the continuation clip must be at the target canvas".into()));
                }
                let mut rows = self.patch_rows(&z);
                let mut clip_rng = Rng::new(KEYFRAME_ENCODE_SEED + 1);
                Self::augment(&mut rows, &mut clip_rng);
                cond_video.extend(rows);
                let mut al = 0;
                if let Some(a) = &c.audio {
                    let avae = self.audio_vae.as_ref().ok_or_else(|| Error::Msg("a continuation with audio needs the audio VAE".into()))?;
                    let rows = stereo_latents(be, avae, a)?;
                    al = rows.len() / cfg.audio_in_channels / AUDIO_CHANNELS;
                    cond_audio.extend(rows);
                }
                cont = Some((z.t, al));
            }
            layout = packing::build_ref2va_layout(&req.prompt.tags, &refs, lt, lh, lw, na, cfg.patch, cont)?;
        } else {
            let anchors: Vec<Anchor> = req.keyframes.iter().map(|(a, _)| *a).collect();
            let mut rows = Vec::new();
            for (_, px) in &req.keyframes {
                let z = self.encode_visual(be, px, true, req.tiling)?;
                if (z.h, z.w) != (lh, lw) {
                    return Err(Error::Geometry("keyframes must be at the canvas size".into()));
                }
                rows.extend(self.patch_rows(&z));
            }
            Self::augment(&mut rows, &mut rng);
            cond_video.extend(rows);
            layout = packing::build_layout(&req.prompt.tags, lt, lh, lw, na, cfg.patch, &anchors)?;
        }
        // Target noise: video latent [C, T, H, W] then patchified; audio rows.
        let noise = rng.normal(cfg.in_channels * lt * lh * lw);
        let mut video_rows = cond_video;
        video_rows.extend(self.patch_rows(&Volume::from_data(cfg.in_channels, lt, lh, lw, noise)));
        let mut audio_rows = cond_audio;
        audio_rows.extend(rng.normal(na * AUDIO_CHANNELS * cfg.audio_in_channels));
        if video_rows.len() != layout.video_indices.len() * cfg.video_patch_dim() {
            return Err(Error::Geometry("packed video row count does not match the layout".into()));
        }
        if audio_rows.len() != layout.audio_indices.len() * cfg.audio_in_channels {
            return Err(Error::Geometry("packed audio row count does not match the layout".into()));
        }
        Ok(Prepared { layout, video_rows, audio_rows })
    }

    /// Denoise a prepared request. Returns the target video latent
    /// `[C, T, H, W]` (normalised) and the target audio rows.
    #[allow(clippy::too_many_arguments)]
    pub fn denoise(
        &self,
        be: &B,
        src: &dyn WeightSource,
        adapters: &[Adapter],
        tables: Option<&crate::dit::AdalnTables>,
        req: &Request,
        prep: &Prepared,
        t: &mut Timings,
        on: &mut dyn FnMut(Event<'_>),
    ) -> Result<(Volume, Vec<f32>)> {
        let cfg = &self.dit.cfg;
        let layout = &prep.layout;
        let ratio = self.vae.as_ref().map_or(16, |v| v.spatial_ratio());
        let (lt, lh, lw) = req.geometry.latent(ratio);
        let t0 = Instant::now();
        // Timestep plan.
        let (video_sched, audio_sched) = match &req.sampler {
            SamplerPlan::Schedules { video, audio } => (video.clone(), audio.clone()),
            SamplerPlan::Pdd(p) => p.schedules(),
        };
        if video_sched.steps() != audio_sched.steps() {
            return Err(Error::Config("video and audio schedules must have the same number of steps".into()));
        }
        let steps = video_sched.steps();
        let mut per_step = Vec::with_capacity(steps);
        let mut all_ts: Vec<f32> = Vec::new();
        for i in 0..steps {
            let (tv, ta) = (video_sched.timestep(i), audio_sched.timestep(i));
            let (unique, inverse) = row_timesteps(layout, tv, ta, tv.max(KEYFRAME_NOISE_AUG), 1.0);
            for &u in &unique {
                if !all_ts.contains(&u) {
                    all_ts.push(u);
                }
            }
            per_step.push((unique, inverse));
        }
        all_ts.sort_by(|a, b| a.total_cmp(b));
        let text = self.dit.refine_text(be, &req.prompt.embeds);
        let ws: Workspace<B> = self.dit.workspace(be, layout, req.ffn_chunk, req.attention)?;
        be.sync();
        t.prepare_s = t0.elapsed().as_secs_f64();
        let t1 = Instant::now();
        let computed;
        let tables = match tables {
            Some(tb) => {
                if !tb.covers(&all_ts) {
                    return Err(Error::Config(format!(
                        "the AdaLN table file lacks some of this run's timesteps ({all_ts:?}); recompute it with `h3 adaln-tables`"
                    )));
                }
                tb
            }
            None => {
                computed = self.dit.adaln_tables(be, src, adapters, &all_ts, None)?;
                &computed
            }
        };
        be.sync();
        t.adaln_s = t1.elapsed().as_secs_f64();
        on(Event::Prepared { secs: t.prepare_s + t.adaln_s, tokens: layout.sequence_length });

        let vp = cfg.video_patch_dim();
        let ap = cfg.audio_in_channels;
        let heads_n = cfg.pdd_steps.unwrap_or(1);
        let x_v = be.upload(&prep.video_rows);
        let x_a = be.upload(&prep.audio_rows);
        let nv_t = layout.num_target_video_rows() * vp;
        let na_t = layout.num_target_audio_rows() * ap;
        let (off_v, off_a) = (layout.num_condition_video_rows * vp, layout.num_condition_audio_rows * ap);
        if let SamplerPlan::Pdd(p) = &req.sampler {
            // PDD targets start at the first node's noise level.
            be.axpby(&x_v, off_v, p.video_sigmas[0], &x_v, off_v, 0.0, nv_t);
            be.axpby(&x_a, off_a, p.audio_sigmas[0], &x_a, off_a, 0.0, na_t);
        }
        let _ = heads_n;
        let v_out = be.alloc(layout.video_indices.len() * vp);
        let a_out = be.alloc(layout.audio_indices.len() * ap);
        // `flow_step` reads the noise at the same offset as `x`, so the buffers
        // span the conditioning rows too (zeros there; only the target rows are
        // written). With references or keyframes `off_v` > 0: an `n`-long
        // buffer made every conditioned Pika run fail on CUDA.
        let noise_v = be.alloc(off_v + nv_t);
        let noise_a = be.alloc(off_a + na_t);
        // Renoise draws from a generator seeded seed + 1 (Pika).
        let mut renoise_rng = Rng::new(req.seed.wrapping_add(1));
        for i in 0..steps {
            let ts = Instant::now();
            let (unique, inverse) = &per_step[i];
            let dense = i < req.dense_first_steps;
            let mods: StepMods<B> = self.dit.step_mods(be, tables, unique, inverse, &layout.token_tags, dense)?;
            let heads: Option<OutputHeads<B>> = match &req.sampler {
                SamplerPlan::Pdd(p) => {
                    let (s, e) = p.block(i);
                    Some(self.dit.fused_heads(be, &p.video_weights, &p.audio_weights, s, e))
                }
                _ => None,
            };
            self.dit.forward(be, &ws, &x_v, &x_a, &text, &mods, heads.as_ref(), &v_out, &a_out);
            let sv = video_sched.step(i);
            let sa = audio_sched.step(i);
            let step_of = |c: h3_core::schedule::StepCoeffs, off: usize, n: usize, noise: &B::Buf, rng: &mut Rng| -> FlowStep {
                match c.rule {
                    StepRule::Euler => FlowStep::Euler { sigma: c.sigma, next: c.next },
                    StepRule::Renoise => {
                        if c.next == 0.0 {
                            FlowStep::Terminal { sigma: c.sigma }
                        } else {
                            let mut z = vec![0.0; off];
                            z.extend(rng.normal(n));
                            be.write(noise, &z);
                            FlowStep::Renoise { sigma: c.sigma, next: c.next }
                        }
                    }
                }
            };
            let fv = step_of(sv, off_v, nv_t, &noise_v, &mut renoise_rng);
            be.flow_step(&x_v, off_v, &v_out, off_v, Some(&noise_v), nv_t, fv);
            let fa = step_of(sa, off_a, na_t, &noise_a, &mut renoise_rng);
            be.flow_step(&x_a, off_a, &a_out, off_a, Some(&noise_a), na_t, fa);
            be.sync();
            t.steps_s.push(ts.elapsed().as_secs_f64());
            on(Event::Step { index: i, total: steps, secs: ts.elapsed().as_secs_f64() });
        }
        let rows = be.download(&x_v);
        let latent = unpatch_rows(&rows[off_v..off_v + nv_t], cfg.in_channels, lt, lh, lw, cfg.patch);
        let audio = be.download(&x_a)[off_a..off_a + na_t].to_vec();
        Ok((latent, audio))
    }

    /// Decode the target latent (normalised `[C, T, H, W]`) to 8-bit frames.
    pub fn decode_video(&self, be: &B, latent: &Volume, tiling: Option<Tiling>, preview: bool, t: &mut Timings, on: &mut dyn FnMut(Event<'_>)) -> Result<(usize, usize, usize, Vec<u8>)> {
        let t0 = Instant::now();
        let mut rgb = Vec::new();
        let (mut frames, mut width, mut height) = (0usize, 0usize, 0usize);
        if preview {
            let tae = self.tae.as_ref().ok_or_else(|| Error::Msg("preview decode needs TAEH3".into()))?;
            let mut st = tae.state();
            let c = latent.c;
            let plane = latent.h * latent.w;
            let mut f0 = 0;
            while f0 < latent.t {
                let n = 5.min(latent.t - f0);
                let mut z = Vec::with_capacity(n * c * plane);
                for f in f0..f0 + n {
                    for ch in 0..c {
                        z.extend_from_slice(&latent.data[(ch * latent.t + f) * plane..][..plane]);
                    }
                }
                let fr = tae.decode_chunk(be, &mut st, &z, n, latent.h, latent.w);
                (width, height) = (fr.width, fr.height);
                let start = rgb.len();
                let px = fr.width * fr.height;
                for f in 0..fr.frames {
                    for i in 0..px {
                        for ch in 0..3 {
                            rgb.push((fr.data[(f * 3 + ch) * px + i].clamp(0.0, 1.0) * 255.0 + 0.5) as u8);
                        }
                    }
                }
                on(Event::Frames { first: frames, count: fr.frames, width, height, rgb: &rgb[start..] });
                frames += fr.frames;
                f0 += n;
            }
        } else {
            let vae = self.vae.as_ref().ok_or_else(|| Error::Msg("decoding needs the video VAE".into()))?;
            let mut z = latent.clone();
            vae.denormalize_latents(&mut z);
            vae.decode(be, &z, tiling, &mut |chunk: &Volume| {
                (width, height) = (chunk.w, chunk.h);
                let start = rgb.len();
                let px = chunk.h * chunk.w;
                // Planar [3, T, H, W] in [0, 1] → interleaved 8-bit frames, one task per frame.
                rgb.resize(start + chunk.t * px * 3, 0);
                rgb[start..].par_chunks_mut(px * 3).enumerate().for_each(|(f, out)| {
                    let planes: [&[f32]; 3] = std::array::from_fn(|ch| &chunk.data[(ch * chunk.t + f) * px..][..px]);
                    for (i, o) in out.chunks_exact_mut(3).enumerate() {
                        for ch in 0..3 {
                            o[ch] = (planes[ch][i] * 255.0).clamp(0.0, 255.0) as u8;
                        }
                    }
                });
                on(Event::Frames { first: frames, count: chunk.t, width, height, rgb: &rgb[start..] });
                frames += chunk.t;
            })?;
        }
        t.decode_s = t0.elapsed().as_secs_f64();
        Ok((frames, width, height, rgb))
    }

    /// Decode the target audio rows (`[2·A, audio_in]`, normalised) to 32 kHz stereo.
    pub fn decode_audio(&self, be: &B, rows: &[f32], num_latents: usize, t: &mut Timings) -> Result<[Vec<f32>; 2]> {
        let t0 = Instant::now();
        let avae = self.audio_vae.as_ref().ok_or_else(|| Error::Msg("audio decoding needs the audio VAE".into()))?;
        let lc = self.dit.cfg.audio_in_channels;
        if lc != avae.cfg.latent_channels {
            return Err(Error::Config(format!("DiT audio_in_channels {lc} != audio VAE latent_channels {}", avae.cfg.latent_channels)));
        }
        let per = num_latents * lc;
        let mut out = [Vec::new(), Vec::new()];
        for ch in 0..AUDIO_CHANNELS {
            let mut z = AudioLatents::from_rows(&rows[ch * per..(ch + 1) * per], lc);
            avae.denormalize_latents(&mut z);
            out[ch] = avae.decode(be, &z)?;
        }
        t.audio_s = t0.elapsed().as_secs_f64();
        Ok(out)
    }

    /// The distinct clean times a request's sampler visits (for `h3 adaln-tables`).
    pub fn timesteps_of(&self, req: &Request) -> Result<Vec<f32>> {
        let (video_sched, audio_sched) = match &req.sampler {
            SamplerPlan::Schedules { video, audio } => (video.clone(), audio.clone()),
            SamplerPlan::Pdd(p) => p.schedules(),
        };
        let mut all = vec![KEYFRAME_NOISE_AUG, 1.0];
        for i in 0..video_sched.steps() {
            for t in [video_sched.timestep(i), audio_sched.timestep(i)] {
                if !all.contains(&t) {
                    all.push(t);
                }
            }
        }
        all.sort_by(|a, b| a.total_cmp(b));
        Ok(all)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn generate(&self, be: &B, src: &dyn WeightSource, adapters: &[Adapter], tables: Option<&crate::dit::AdalnTables>, req: &Request, on: &mut dyn FnMut(Event<'_>)) -> Result<Generation> {
        let mut t = Timings::default();
        let prep = self.prepare(be, req)?;
        let (latent, audio_rows) = self.denoise(be, src, adapters, tables, req, &prep, &mut t, on)?;
        let (frames, width, height, rgb) = self.decode_video(be, &latent, req.tiling, req.preview, &mut t, on)?;
        let audio = if self.audio_vae.is_some() {
            let a = self.decode_audio(be, &audio_rows, prep.layout.num_audio_latents, &mut t)?;
            on(Event::Audio { sample_rate: self.audio_vae.as_ref().unwrap().cfg.sample_rate, stereo: &a });
            Some(a)
        } else {
            None
        };
        Ok(Generation {
            frames,
            width,
            height,
            rgb,
            audio,
            sample_rate: self.audio_vae.as_ref().map_or(32000, |a| a.cfg.sample_rate),
            timings: t,
        })
    }
}

/// Rows (channel-major features) → `[c, t, h, w]`.
pub fn unpatch_rows(rows: &[f32], c: usize, t: usize, h: usize, w: usize, (pt, ph, pw): (usize, usize, usize)) -> Volume {
    let (tt, hh, ww) = (t / pt, h / ph, w / pw);
    let f = c * pt * ph * pw;
    let mut v = Volume::zeros(c, t, h, w);
    for row in 0..tt * hh * ww {
        let (ti, rem) = (row / (hh * ww), row % (hh * ww));
        let (hi, wi) = (rem / ww, rem % ww);
        for ci in 0..c {
            for kt in 0..pt {
                for kh in 0..ph {
                    for kw in 0..pw {
                        let val = rows[row * f + ((ci * pt + kt) * ph + kh) * pw + kw];
                        v.data[((ci * t + ti * pt + kt) * h + hi * ph + kh) * w + wi * pw + kw] = val;
                    }
                }
            }
        }
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audio_vae::AudioVae;
    use crate::dit::{Attention, H3Dit, LoadOptions};
    use crate::taeh3::TaeH3;
    use crate::vae::VideoVae;
    use h3_core::config::{AudioVaeConfig, H3Config, TaeH3Config, VideoVaeConfig};
    use h3_core::packing::{Anchor, Geometry};
    use h3_core::schedule::{Schedule, StepRule, PIKA_KNOTS};
    use h3_core::weights::Synthetic;
    use h3_cpu::Cpu;

    /// Pika's renoise steps draw noise for the target rows only, but
    /// `flow_step` reads it at the rows' offset: with a keyframe (conditioning
    /// rows first) the noise buffer must span the whole latent. This ran only
    /// on the Server Edition before, where the CUDA backend caught it.
    #[test]
    fn renoise_with_a_keyframe_runs() {
        let be = Cpu;
        let cfg = H3Config::tiny();
        let src = Synthetic { seed: 1 };
        let pipe = Pipeline {
            dit: H3Dit::load(&be, cfg.clone(), &src, &LoadOptions { quant: Default::default(), stream_blocks: false }, &[]).unwrap(),
            vae: Some(VideoVae::load(&be, VideoVaeConfig::tiny(), &src, true, true).unwrap()),
            audio_vae: Some(AudioVae::load(&be, AudioVaeConfig::tiny(), &src, false, true).unwrap()),
            tae: Some(TaeH3::load(&be, TaeH3Config::tiny(), &src).unwrap()),
        };
        let g = Geometry { width: 64, height: 32, frames: 124 };
        let mut r = Rng::new(3);
        let keyframe = Volume::from_data(3, 1, 32, 64, r.uniform(3 * 32 * 64, 0.5).iter().map(|v| v + 0.5).collect());
        let req = Request {
            prompt: Prompt::text_only(r.normal(4 * cfg.text_dim), cfg.text_dim),
            geometry: g,
            seed: 7,
            keyframes: vec![(Anchor::First, keyframe)],
            references: vec![],
            continuation: None,
            sampler: SamplerPlan::Schedules {
                video: Schedule::from_rungs(&PIKA_KNOTS, 12.0, StepRule::Renoise).unwrap(),
                audio: Schedule::from_rungs(&PIKA_KNOTS, 12.0, StepRule::Renoise).unwrap(),
            },
            attention: Attention::Dense { qk8: false, pv8: false },
            dense_first_steps: 0,
            ffn_chunk: usize::MAX,
            tiling: None,
            preview: true,
        };
        let out = pipe.generate(&be, &src, &[], None, &req, &mut |_| {}).unwrap();
        assert_eq!((out.width, out.height), (64, 32));
        assert!(out.frames > 0 && out.rgb.len() == out.frames * 64 * 32 * 3);
    }
}
