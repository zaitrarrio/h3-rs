//! Open-H3-Regenerate-2K: a 768p H3 draft → 2560×1440 through the latent
//! upscaler, a rank-576 LoRA merged into the DiT, the Sharp curve AdaLN
//! branches, three fixed-sigma Euler forwards and the 24-block fast decoder.
//! The released pipeline writes a silent video: the audio stream (encoded
//! silence) only conditions the DiT.

use crate::{
    audio_vae::AudioVae,
    dit::{Attention, H3Dit, StepMods, Workspace},
    pipeline::{Event, KEYFRAME_ENCODE_SEED, Prompt, Timings, unpatch_rows},
    upscaler::Upscaler,
    vae::{Posterior, Tiling, VideoVae, Volume},
};
use h3_core::{
    Error, Result,
    backend::{Backend, FlowStep},
    config::H3Config,
    lora::Adapter,
    packing::{self, AUDIO_CHANNELS, FPS, row_timesteps},
    rng::Rng,
    schedule::regen_sigmas,
    weights::{SafetensorsDir, WeightSource},
};
use std::path::Path;
use std::time::Instant;

pub const DRAFT_FRAMES: usize = 121;
pub const DRAFT_PADDED_FRAMES: usize = 124;
pub const OUTPUT_SIZE: (usize, usize) = (1440, 2560);
pub const CURVE_TABLE_ROWS: usize = 1025;
pub const CURVE_DIM: usize = 8;

/// The Sharp curve branches: `adaln_proj.linear` output of block `l` gets
/// `curve_B·curve_A·interp(table, t)`.
pub struct SharpCurve {
    /// `[1025, 8]`.
    pub table: Vec<f32>,
    /// Per block `(A [64, 8], B [18·d, 64])`.
    pub factors: Vec<(Vec<f32>, Vec<f32>)>,
    pub rank: usize,
}

impl SharpCurve {
    /// `branches`: `sharp_curve_branches.safetensors`; `table`: the raw
    /// `adaln_t_table.bin` (1025×8 f32, not redistributed with the weights:
    /// fetch it from the Comfy-Org MiniMax-H3 repository's `adaln_t_table` tensor).
    pub fn load(branches: &Path, table: &Path, cfg: &H3Config) -> Result<Self> {
        let src = SafetensorsDir::open(branches)?;
        let bytes = std::fs::read(table)?;
        if bytes.len() != CURVE_TABLE_ROWS * CURVE_DIM * 4 {
            return Err(Error::Weights(format!("{} must hold a {CURVE_TABLE_ROWS}×{CURVE_DIM} f32 table", table.display())));
        }
        let table: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let mods = H3Config::MOD_SLOTS * H3Config::MODALITIES * cfg.hidden;
        let mut factors = Vec::with_capacity(cfg.layers);
        let mut rank = 0;
        for l in 0..cfg.layers {
            let p = format!("transformer.transformer_blocks.{l}.adaln_proj.linear");
            let (_, ashape, _) = src.raw(&format!("{p}.curve_A.weight"))?;
            rank = ashape[0];
            let a = src.get(&format!("{p}.curve_A.weight"), &[rank, CURVE_DIM])?;
            let b = src.get(&format!("{p}.curve_B.weight"), &[mods, rank])?;
            factors.push((a, b));
        }
        Ok(Self { table, factors, rank })
    }

    /// Linear interpolation of the table at model time `t ∈ [0, 1]`.
    pub fn coords(&self, t: f32) -> Vec<f32> {
        let pos = t.clamp(0.0, 1.0) * (CURVE_TABLE_ROWS - 1) as f32;
        let lo = (pos.floor() as usize).min(CURVE_TABLE_ROWS - 2);
        let f = pos - lo as f32;
        (0..CURVE_DIM).map(|i| self.table[lo * CURVE_DIM + i] * (1.0 - f) + self.table[(lo + 1) * CURVE_DIM + i] * f).collect()
    }

    /// `[U, 18·d]` delta of block `l` at `timesteps`.
    pub fn delta(&self, l: usize, timesteps: &[f32]) -> Vec<f32> {
        let (a, b) = &self.factors[l];
        let mut out = Vec::new();
        for &t in timesteps {
            let c = self.coords(t);
            let mid = h3_core::timestep::matvec(a, None, &c);
            out.extend(h3_core::timestep::matvec(b, None, &mid));
        }
        out
    }
}

pub struct Regen<B: Backend> {
    pub dit: H3Dit<B>,
    /// Encoder from the base VAE.
    pub encoder: VideoVae<B>,
    /// The fast 24-block decoder.
    pub decoder: VideoVae<B>,
    pub audio_vae: AudioVae<B>,
    pub upscaler: Upscaler<B>,
    pub curve: SharpCurve,
}

pub struct RegenRequest {
    pub prompt: Prompt,
    /// `[3, 124, 768, 1344]` in `[0, 1]`: the 121 draft frames with the last
    /// held three more times (`draft_video.py`).
    pub draft: Volume,
    pub seed: u64,
    pub steps: usize,
    pub start_sigma: Option<f64>,
    pub attention: Attention,
    pub ffn_chunk: usize,
    pub tiling: Option<Tiling>,
}

impl<B: Backend> Regen<B> {
    /// Encoded silence for `frames` at 24 fps, as normalised channel-major rows.
    pub fn silent_audio_rows(&self, be: &B, frames: usize) -> Result<Vec<f32>> {
        let samples = ((frames as f64 / FPS as f64) * self.audio_vae.cfg.sample_rate as f64).round() as usize;
        let silence = vec![0.0f32; samples];
        let mut z = self.audio_vae.encode(be, &silence)?.mode();
        self.audio_vae.normalize_latents(&mut z);
        let rows = z.rows();
        Ok([rows.clone(), rows].concat())
    }

    pub fn run(&self, be: &B, src: &dyn WeightSource, adapters: &[Adapter], req: &RegenRequest, on: &mut dyn FnMut(Event<'_>)) -> Result<(usize, usize, usize, Vec<u8>, Timings)> {
        let cfg = &self.dit.cfg;
        let mut t = Timings::default();
        let t0 = Instant::now();
        if req.draft.t != DRAFT_PADDED_FRAMES {
            return Err(Error::Geometry(format!("the draft must hold {DRAFT_PADDED_FRAMES} frames, got {}", req.draft.t)));
        }
        let sigmas = regen_sigmas(req.steps, req.start_sigma)?;
        // Draft latent: posterior sampled with the fixed seed, normalised.
        let m = self.encoder.encode_pixels(be, &req.draft, req.tiling)?;
        let mut rng42 = Rng::new(KEYFRAME_ENCODE_SEED);
        let mut draft = Posterior::from_moments(&m).sample(&mut rng42);
        self.encoder.normalize_latents(&mut draft);
        let up = self.upscaler.upscale(be, &draft, OUTPUT_SIZE)?;
        let (lt, lh, lw) = (up.t, up.h, up.w);
        let na = packing::audio_latents(DRAFT_PADDED_FRAMES);
        let layout = packing::build_layout(&req.prompt.tags, lt, lh, lw, na, cfg.patch, &[])?;
        let up_rows = crate::pipeline::patch_rows_of(&up, cfg.patch);
        let mut rng = Rng::new(req.seed);
        let noise = rng.normal(up_rows.len());
        let s0 = sigmas[0] as f32;
        let x_rows: Vec<f32> = up_rows.iter().zip(&noise).map(|(u, n)| (1.0 - s0) * u + s0 * n).collect();
        let audio_rows = self.silent_audio_rows(be, DRAFT_PADDED_FRAMES)?;
        if audio_rows.len() != layout.audio_indices.len() * cfg.audio_in_channels {
            return Err(Error::Geometry("silent audio rows do not match the layout".into()));
        }
        let text = self.dit.refine_text(be, &req.prompt.embeds);
        let ws: Workspace<B> = self.dit.workspace(be, &layout, req.ffn_chunk, req.attention)?;
        let all_ts: Vec<f32> = {
            let mut v: Vec<f32> = sigmas[..sigmas.len() - 1].iter().map(|&s| 1.0 - s as f32).collect();
            v.push(1.0);
            v.sort_by(|a, b| a.total_cmp(b));
            v.dedup();
            v
        };
        let curve = &self.curve;
        let extra = |l: usize, ts: &[f32]| curve.delta(l, ts);
        let tables = self.dit.adaln_tables(be, src, adapters, &all_ts, Some(&extra))?;
        be.sync();
        t.prepare_s = t0.elapsed().as_secs_f64();
        on(Event::Prepared { secs: t.prepare_s, tokens: layout.sequence_length });
        let x_v = be.upload(&x_rows);
        let x_a = be.upload(&audio_rows);
        let v_out = be.alloc(x_rows.len());
        let a_out = be.alloc(audio_rows.len());
        for i in 0..sigmas.len() - 1 {
            let ts = Instant::now();
            let (sigma, next) = (sigmas[i] as f32, sigmas[i + 1] as f32);
            let (unique, inverse) = row_timesteps(&layout, 1.0 - sigma, 1.0, 1.0, 1.0);
            let mods: StepMods<B> = self.dit.step_mods(be, &tables, &unique, &inverse, &layout.token_tags, false)?;
            self.dit.forward(be, &ws, &x_v, &x_a, &text, &mods, None, &v_out, &a_out);
            be.flow_step(&x_v, 0, &v_out, 0, None, x_rows.len(), FlowStep::Euler { sigma, next });
            be.sync();
            t.steps_s.push(ts.elapsed().as_secs_f64());
            on(Event::Step { index: i, total: sigmas.len() - 1, secs: ts.elapsed().as_secs_f64() });
        }
        let rows = be.download(&x_v);
        let mut latent = unpatch_rows(&rows, cfg.in_channels, lt, lh, lw, cfg.patch);
        self.decoder.denormalize_latents(&mut latent);
        let td = Instant::now();
        let mut rgb = Vec::new();
        let (mut frames, mut width, mut height) = (0usize, 0usize, 0usize);
        self.decoder.decode(be, &latent, req.tiling, &mut |chunk: &Volume| {
            if frames >= DRAFT_FRAMES {
                return;
            }
            let keep = chunk.t.min(DRAFT_FRAMES - frames);
            (width, height) = (chunk.w, chunk.h);
            let start = rgb.len();
            let px = chunk.h * chunk.w;
            for f in 0..keep {
                for i in 0..px {
                    for ch in 0..3 {
                        rgb.push((chunk.data[(ch * chunk.t + f) * px + i] * 255.0).clamp(0.0, 255.0) as u8);
                    }
                }
            }
            on(Event::Frames { first: frames, count: keep, width, height, rgb: &rgb[start..] });
            frames += keep;
        })?;
        t.decode_s = td.elapsed().as_secs_f64();
        let _ = AUDIO_CHANNELS;
        Ok((frames, width, height, rgb, t))
    }
}
