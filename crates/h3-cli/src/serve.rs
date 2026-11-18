//! `h3 serve`: a local engine plus the prompt-to-clip demo page.
//!
//! One engine thread owns the backend, the resident DiT, the VAEs and the
//! conditioner (or the gallery); it takes jobs one at a time from a queue.
//! The HTTP thread only reads the shared job table.
//!
//! ```text
//! GET  /                         the demo page (web/demo.html)
//! GET  /api/info                 backend, model, limits, tasks, ready / error
//! POST /api/generate             {prompt, width, height, frames, seed?, upscale?,
//!                                 task?, first_image?, last_image?, refs?, continue?}
//!                                → {id, seed, task}
//! GET  /api/jobs/<id>            status, per-stage timings, frames decoded so far
//! GET  /api/jobs/<id>/frames?from=N
//!                                binary: u32 LE width, height, first, count; then RGB8
//! GET  /api/jobs/<id>/audio      WAV (16-bit stereo, 32 kHz) once the clip is done
//! GET  /api/jobs/<id>/hd?frame=N one upscaled frame (JPEG), once upscaled
//! ```
//!
//! One request carries a task's inputs, so a hosted worker needs no per-job
//! `h3 generate` (a full model load) for keyframes or references:
//! `task` is `t2va` / `fl2va` / `ref2va` (default: what the inputs imply),
//! `first_image` / `last_image` are `{"b64", "mime": "image/png" |
//! "image/jpeg"}` (decoded like `generate --first-image`), `refs` is an
//! ordered array of `{"kind": "image" | "video" | "audio", "b64", "mime"?,
//! "audio_b64"?}` (a video is Y4M bytes like `--ref video:F.y4m`, its
//! `audio_b64` a WAV; an audio reference is a WAV), and `continue` is
//! `{"b64": y4m, "audio_b64"?: wav}` like `--continue`. A body with any of
//! these fields may be [`MAX_BODY`] (64 MiB); one without stays at
//! [`SMALL_BODY`] (64 KiB); bigger is 413. `/api/info` lists the tasks the
//! resident engine takes per request (`tasks`): FL2VA needs a preset that
//! lists it, Ref2VA an engine started with `--task ref2va` (it loads
//! `transformer_ref` and the task's adapter), and a gallery engine has no
//! conditioner for either. The job's `task` and `ref_count` are in its
//! status. The media are decoded on the HTTP thread (a bad file is a 400),
//! sized and encoded (VAE, vision tower) on the engine thread with the job.
//!
//! A size off the 32-pixel grid (480×272) renders on the next canvas up
//! (480×288) and the frames are centre-cropped as they decode. With
//! `--upscale realesr-animevideov3.safetensors`, a request's `upscale`
//! (`true` for the default factor, or 2 / 3 / 4) runs Real-ESRGAN on the
//! finished clip: the low-resolution frames stream as usual, then the job
//! enters `upscaling` and its frames become available one by one as JPEGs
//! (`up_frames`), which keeps a clip of 1080p frames to tens of MB.
//!
//! It binds 127.0.0.1 by default and sends no CORS headers, so other web
//! pages cannot queue jobs on your GPU. `--allow-origin <url>` opens the API
//! to exactly one origin and to nothing else; `/api/generate` also rejects
//! non-JSON bodies and foreign `Origin`s (a cross-site form POST skips the
//! preflight).

use crate::{
    AttnArgs, Resolved, finish,
    inputs::{self, Inputs, RefInput, Sizing},
    media, open_conditioner, open_transformer, sampler_plan,
};
use anyhow::{Context, Result, anyhow};
use h3_core::{
    Backend,
    config::{AudioVaeConfig, H3Config, TaeH3Config, VideoVaeConfig},
    packing::{self, Anchor, Geometry},
    rng::Rng,
    schedule::Schedule,
    variant::{Task, Variant},
    weights::{SafetensorsDir, Synthetic},
};
use h3_model::{
    audio_vae::AudioVae,
    dit::{AdalnTables, H3Dit, LoadOptions},
    esrgan::{Esrgan, EsrganConfig},
    pipeline::{Event, Pipeline, Prompt, Request, SamplerPlan},
    taeh3::TaeH3,
    vae::{Tiling, VideoVae},
};
use h3_text::{PresentedRef, image::Image};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;

const PAGE: &str = include_str!("../../../web/demo.html");
const KEEP_JOBS: usize = 8;
const MAX_QUEUE: usize = 3;
/// The body cap of a generate request without media fields.
pub const SMALL_BODY: usize = 64 * 1024;
/// The body cap of a generate request carrying keyframes / references / a
/// continuation (base64 of a 124-frame 768p Y4M is ~60 MiB at 4:2:0).
pub const MAX_BODY: usize = 64 << 20;
/// The request fields that carry media.
const MEDIA_FIELDS: [&str; 4] = ["first_image", "last_image", "refs", "continue"];

fn task_name(t: Task) -> &'static str {
    match t {
        Task::T2va => "t2va",
        Task::Fl2va => "fl2va",
        Task::Ref2va => "ref2va",
        Task::Regen2k => "regen2k",
    }
}

pub struct Settings {
    pub addr: String,
    /// `None` with `synthetic`.
    pub model: Option<Resolved>,
    pub gallery: Option<PathBuf>,
    pub synthetic: bool,
    pub max_pixels: usize,
    pub max_frames: usize,
    pub allow_origin: Option<String>,
    pub attn: AttnArgs,
    pub tiling: Option<Tiling>,
    pub preview: bool,
    /// Real-ESRGAN weights (any value with `synthetic`: a tiny random net).
    pub upscale: Option<PathBuf>,
    /// Text encoder placement and the prompt cache.
    pub encoder: crate::encoder::EncoderArgs,
}

impl Settings {
    /// The tasks the resident engine takes per request. The DiT loaded at
    /// start decides: `--task ref2va` loads `transformer_ref` (and Pika's
    /// Ref2VA adapter), so that engine serves Ref2VA only; any other start
    /// serves the preset's T2VA / FL2VA, unless the preset's adapters are
    /// per task (Pika), when only the start task's adapter is resident. A
    /// gallery engine has no conditioner, so no vision tower: T2VA only.
    /// The synthetic engine runs the tiny pipeline on all three.
    pub fn served_tasks(&self) -> Vec<Task> {
        if self.synthetic {
            return vec![Task::T2va, Task::Fl2va, Task::Ref2va];
        }
        let Some(r) = &self.model else {
            return vec![];
        };
        let per_task_adapters = r.variant.adapters.iter().any(|a| a.task.is_some());
        // Keyframes and references need the live encoder's vision tower: a
        // gallery engine has one only when `--encoder` asks for it.
        let no_encoder = match self.encoder.encoder {
            Some(m) => m == crate::encoder::EncoderMode::None,
            None => self.gallery.is_some(),
        };
        r.variant
            .tasks
            .iter()
            .copied()
            .filter(|&t| match t {
                Task::Regen2k => false,
                Task::Ref2va => r.task == Task::Ref2va && !no_encoder,
                Task::Fl2va => {
                    r.task != Task::Ref2va
                        && !no_encoder
                        && !(per_task_adapters && r.task != t)
                }
                Task::T2va => r.task != Task::Ref2va && !(per_task_adapters && r.task != t),
            })
            .collect()
    }

    /// The sample rate soundtracks are resampled to: the resident audio
    /// VAE's contract (32 kHz for H3; the tiny model's is 320 Hz, and at
    /// 32 kHz a 0.3 s clip would be ~1900 reference rows).
    fn audio_rate(&self) -> usize {
        if self.synthetic {
            AudioVaeConfig::tiny().sample_rate
        } else {
            AudioVaeConfig::h3().sample_rate
        }
    }

    /// The task of a request that names none and carries no media: T2VA
    /// when served, else the engine's start task.
    fn default_task(&self) -> Task {
        let served = self.served_tasks();
        if served.contains(&Task::T2va) {
            Task::T2va
        } else {
            self.model
                .as_ref()
                .map_or(Task::T2va, |r| r.task)
        }
    }
}

/// What a job asked of the upscaler.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Upscale {
    Off,
    /// [`finish::serve_factor`].
    Auto,
    Factor(usize),
}

/// A parsed `/api/generate` body.
#[derive(Debug)]
struct Parsed {
    prompt: String,
    /// The canvas the model renders.
    geometry: Geometry,
    /// The size the client gets (the canvas, or a centre crop of it).
    out: (usize, usize),
    seed: u64,
    upscale: Upscale,
    task: Task,
    /// Decoded media, sized on the engine thread.
    inputs: Inputs,
}

#[derive(Clone, Copy, PartialEq)]
enum Stage {
    Queued,
    Encoding,
    Preparing,
    Denoising,
    Decoding,
    Upscaling,
    Done,
    Failed,
}

impl Stage {
    fn name(self) -> &'static str {
        match self {
            Stage::Queued => "queued",
            Stage::Encoding => "encoding",
            Stage::Preparing => "preparing",
            Stage::Denoising => "denoising",
            Stage::Decoding => "decoding",
            Stage::Upscaling => "upscaling",
            Stage::Done => "done",
            Stage::Failed => "failed",
        }
    }
}

struct Job {
    id: u64,
    prompt: String,
    geometry: Geometry,
    out: (usize, usize),
    seed: u64,
    upscale: Upscale,
    task: Task,
    ref_count: usize,
    /// Taken by the engine thread when the job starts.
    inputs: Option<Inputs>,
    stage: Stage,
    tokens: usize,
    encode_s: f64,
    prepare_s: f64,
    steps: Vec<f64>,
    total_steps: usize,
    decode_s: f64,
    started: Option<Instant>,
    total_s: f64,
    width: usize,
    height: usize,
    frames: usize,
    rgb: Vec<u8>,
    audio: Option<Vec<u8>>,
    error: Option<String>,
    up_factor: Option<usize>,
    /// Upscaled frames, JPEG.
    hd: Vec<Vec<u8>>,
    up_s: f64,
    up_error: Option<String>,
    /// Where the prompt's rows came from: gallery, cache, encoder, synthetic.
    prompt_source: &'static str,
}

impl Job {
    fn new(id: u64, p: Parsed, total_steps: usize) -> Self {
        Job {
            id,
            prompt: p.prompt,
            geometry: p.geometry,
            out: p.out,
            seed: p.seed,
            upscale: p.upscale,
            task: p.task,
            ref_count: p.inputs.refs.len(),
            inputs: Some(p.inputs),
            stage: Stage::Queued,
            tokens: 0,
            encode_s: 0.0,
            prepare_s: 0.0,
            steps: Vec::new(),
            total_steps,
            decode_s: 0.0,
            started: None,
            total_s: 0.0,
            width: 0,
            height: 0,
            frames: 0,
            rgb: Vec::new(),
            audio: None,
            error: None,
            up_factor: None,
            hd: Vec::new(),
            up_s: 0.0,
            up_error: None,
            prompt_source: "",
        }
    }

    fn status(&self) -> Value {
        json!({
            "id": self.id, "prompt": self.prompt, "stage": self.stage.name(), "seed": self.seed,
            "task": task_name(self.task), "ref_count": self.ref_count,
            "width": self.out.0, "height": self.out.1, "total_frames": self.geometry.frames,
            "canvas_width": self.geometry.width, "canvas_height": self.geometry.height,
            "tokens": self.tokens, "prompt_source": self.prompt_source, "encode_s": self.encode_s, "prepare_s": self.prepare_s,
            "steps": self.steps, "total_steps": self.total_steps, "decode_s": self.decode_s,
            "elapsed_s": self.started.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0),
            "total_s": self.total_s, "frames": self.frames, "audio": self.audio.is_some(), "error": self.error,
            "upscale": self.up_factor, "up_frames": self.hd.len(), "up_s": self.up_s, "up_error": self.up_error,
            "up_width": self.up_factor.map(|k| self.out.0 * k), "up_height": self.up_factor.map(|k| self.out.1 * k),
        })
    }
}

pub struct Shared {
    jobs: Mutex<Vec<Job>>,
    info: Mutex<Value>,
    next: Mutex<u64>,
}

impl Shared {
    fn with<R>(&self, id: u64, f: impl FnOnce(&mut Job) -> R) -> Option<R> {
        self.jobs
            .lock()
            .unwrap()
            .iter_mut()
            .find(|j| j.id == id)
            .map(f)
    }
}

enum Text<B: Backend> {
    Synthetic {
        dim: usize,
        only: Option<crate::gallery::Gallery>,
    },
    /// Gallery prompts first, then the prompt cache, then the live encoder.
    Real {
        dim: usize,
        gallery: Option<crate::gallery::Gallery>,
        cache: Option<crate::encoder::PromptCache>,
        encoder: Option<Box<h3_text::Conditioner<B>>>,
    },
}

impl<B: Backend> Text<B> {
    /// The prompt's rows, its token count, and where it came from. With
    /// `keyframes` the FL2VA presentation, with `presented` references the
    /// Ref2VA one (both need the live encoder's vision tower: gallery and
    /// cache entries are text-only; the synthetic engine makes random text
    /// rows whatever it gets, which the tiny pipeline accepts).
    fn encode(
        &self,
        be: &B,
        prompt: &str,
        keyframes: &[Image],
        presented: &[PresentedRef],
    ) -> Result<(Prompt, usize, &'static str)> {
        match self {
            Text::Synthetic { dim, only } => {
                if let Some(g) = only {
                    if g.find(prompt).is_none() {
                        return Err(anyhow!(
                            "This engine has no text encoder: pick one of its {} gallery prompts.",
                            g.entries.len()
                        ));
                    }
                }
                let seed = prompt.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| {
                    (h ^ b as u64).wrapping_mul(0x100_0000_01b3)
                });
                let n = prompt.split_whitespace().count().max(1);
                Ok((
                    Prompt::text_only(Rng::new(seed).normal(n * dim), *dim),
                    n,
                    "synthetic",
                ))
            }
            Text::Real {
                dim,
                gallery,
                cache,
                encoder,
            } => {
                let visual = !keyframes.is_empty() || !presented.is_empty();
                if visual {
                    let c = encoder.as_ref().ok_or_else(|| {
                        anyhow!(
                            "This engine has no text encoder, so it cannot present keyframes or references."
                        )
                    })?;
                    let p = if !presented.is_empty() {
                        c.encode_ref2va(be, prompt, presented)?
                    } else {
                        c.encode_fl2va(be, prompt, keyframes)?
                    };
                    let n = p.rows(*dim);
                    return Ok((p, n, "encoder"));
                }
                if let Some(e) = gallery.as_ref().and_then(|g| g.find(prompt)) {
                    let g = gallery.as_ref().unwrap();
                    return Ok((g.prompt(e, *dim)?, e.tokens, "gallery"));
                }
                if let Some(p) = cache.as_ref().and_then(|c| c.get(prompt)) {
                    let n = p.rows(*dim);
                    return Ok((p, n, "cache"));
                }
                let Some(c) = encoder else {
                    return Err(anyhow!(
                        "This engine has no text encoder: pick one of its {} gallery prompts.",
                        gallery.as_ref().map_or(0, |g| g.entries.len())
                    ));
                };
                let p = c.encode_text(be, prompt)?;
                if let Some(cache) = cache
                    && let Err(e) = cache.put(prompt, &p)
                {
                    eprintln!("prompt cache: {e:#}");
                }
                let n = p.rows(*dim);
                Ok((p, n, "encoder"))
            }
        }
    }
}

/// Loads everything, then runs jobs forever. Runs on the engine thread.
pub fn engine<B: Backend>(
    be: B,
    s: &Settings,
    shared: Arc<Shared>,
    rx: mpsc::Receiver<u64>,
) -> Result<()> {
    let t0 = Instant::now();
    let stage = |what: &str| {
        eprintln!("engine: {what} ({:.1}s)", t0.elapsed().as_secs_f64());
        let mut info = shared.info.lock().unwrap();
        info["loading"] = json!(what);
        info["loading_s"] = json!(t0.elapsed().as_secs_f64());
    };
    stage(&format!("starting on {}", be.name()));
    let gallery_list = |g: &crate::gallery::Gallery| {
        json!(g.entries.iter().map(|e| &e.prompt).collect::<Vec<_>>())
    };
    let (pipe, text, model, variant, src, adapters, tables): (
        Pipeline<B>,
        Text<B>,
        String,
        Variant,
        Box<dyn h3_core::weights::WeightSource>,
        &[h3_core::lora::Adapter],
        Option<AdalnTables>,
    ) = if s.synthetic {
        let cfg = H3Config::tiny_vsa();
        let src = Synthetic { seed: 1 };
        let only = match &s.gallery {
            Some(dir) => {
                let g = crate::gallery::Gallery::open(dir)?;
                {
                    let mut info = shared.info.lock().unwrap();
                    info["gallery"] = gallery_list(&g);
                    info["gallery_only"] = json!(true);
                }
                Some(g)
            }
            None => None,
        };
        let text = Text::Synthetic {
            dim: cfg.text_dim,
            only,
        };
        let mut variant = Variant::by_name("fasth3-v2")?;
        variant.config = cfg.clone();
        let pipe = Pipeline {
            dit: H3Dit::load(
                &be,
                cfg,
                &src,
                &LoadOptions {
                    quant: s.attn.weights,
                    stream_blocks: false,
                },
                &[],
            )?,
            // Both encoders: the tiny pipeline takes keyframes, references
            // and continuations.
            vae: Some(VideoVae::load(
                &be,
                VideoVaeConfig::tiny(),
                &src,
                true,
                true,
            )?),
            audio_vae: Some(AudioVae::load(
                &be,
                AudioVaeConfig::tiny(),
                &src,
                true,
                true,
            )?),
            tae: Some(TaeH3::load(&be, TaeH3Config::tiny(), &src)?),
        };
        (
            pipe,
            text,
            "synthetic tiny model (random weights: output is noise)".to_string(),
            variant,
            Box::new(src),
            &[][..],
            None,
        )
    } else {
        let r = s
            .model
            .as_ref()
            .expect("a resolved model when not synthetic");
        let cfg = r.variant.config.clone();
        let model_dir = r
            .model_dir
            .as_ref()
            .ok_or_else(|| anyhow!("--model-dir is needed"))?;
        stage("checking the checkpoint");
        let (src, _) = open_transformer(r)?;
        // Keyframes and references go through the video VAE encoder and the
        // conditioner's vision tower; soundtracks through the audio VAE
        // encoder. Load them only for the tasks this engine serves.
        let served = s.served_tasks();
        let visual_inputs = served.contains(&Task::Fl2va) || served.contains(&Task::Ref2va);
        let audio_inputs = served.contains(&Task::Ref2va);
        let gallery = match &s.gallery {
            Some(dir) => {
                stage("reading the prompt gallery");
                let g = crate::gallery::Gallery::open(dir)?;
                shared.info.lock().unwrap()["gallery"] = gallery_list(&g);
                Some(g)
            }
            None => None,
        };
        let tables = match &r.tables {
            Some(p) => Some(AdalnTables::load(p, &cfg)?),
            None => None,
        };
        stage(&format!(
            "loading the transformer onto the device ({} blocks, {})",
            cfg.layers, s.attn.weights
        ));
        let dit = H3Dit::load(
            &be,
            cfg.clone(),
            src.as_ref(),
            &LoadOptions {
                quant: s.attn.weights,
                stream_blocks: s.attn.stream_blocks,
            },
            &r.adapters,
        )?;
        stage("loading the decoders");
        let vae_dir = model_dir.join("vae");
        let vae = VideoVae::load_with(
            &be,
            VideoVaeConfig::h3(),
            &SafetensorsDir::open(&vae_dir)?,
            visual_inputs,
            !s.preview,
            s.attn.decoder_weights,
        )?;
        let audio_vae = AudioVae::load(
            &be,
            AudioVaeConfig::h3(),
            &SafetensorsDir::open(&model_dir.join("audio_vae"))?,
            audio_inputs,
            true,
        )?;
        let tae = match (&r.tae, s.preview) {
            (Some(p), true) => Some(TaeH3::load(
                &be,
                TaeH3Config::taeh3(),
                &SafetensorsDir::open(p)?,
            )?),
            (None, true) => return Err(anyhow!("--preview needs --tae")),
            _ => None,
        };
        let pipe = Pipeline {
            dit,
            vae: Some(vae),
            audio_vae: Some(audio_vae),
            tae,
        };
        // The conditioner last, so `--encoder auto` sees what the rest left free.
        let enc = &s.encoder;
        let mode = enc.encoder.unwrap_or(if gallery.is_some() {
            crate::encoder::EncoderMode::None
        } else {
            crate::encoder::EncoderMode::Auto
        });
        let (encoder, placement) = if mode == crate::encoder::EncoderMode::None {
            (None, None)
        } else {
            let placement =
                crate::encoder::choose(&be, model_dir, mode, enc.encoder_weights, true)?;
            stage(&format!(
                "opening the text encoder ({placement:?}, {:?})",
                enc.encoder_weights
            ));
            let c = open_conditioner(&be, model_dir, placement, enc, visual_inputs)?;
            (Some(Box::new(c)), Some(placement))
        };
        let cache = if encoder.is_some() || enc.prompt_cache.is_some() {
            Some(crate::encoder::PromptCache::new(
                enc.prompt_cache.clone(),
                crate::encoder::identity(
                    model_dir,
                    &crate::tokenizer_path(model_dir)?,
                    enc.encoder_weights,
                )?,
                cfg.text_dim,
            )?)
        } else {
            None
        };
        {
            let mut info = shared.info.lock().unwrap();
            info["gallery_only"] = json!(encoder.is_none() && cache.is_none());
            info["encoder"] = json!(placement.map(|p| format!("{p:?}").to_lowercase()));
        }
        let has_gallery = gallery.is_some();
        let text = Text::Real {
            dim: cfg.text_dim,
            gallery,
            cache,
            encoder,
        };
        let model = format!(
            "{} · {} forwards · {}{}{}",
            r.variant
                .summary
                .split(':')
                .next()
                .unwrap_or(&r.variant.name),
            r.variant.forwards(),
            if s.preview {
                "TAEH3 preview"
            } else {
                "VAE decoder"
            },
            match (has_gallery, placement) {
                (true, None) => " · prompt gallery".to_string(),
                (true, Some(p)) => format!(" · gallery + {p:?} text encoder").to_lowercase(),
                (false, Some(p)) => format!(" · {p:?} text encoder").to_lowercase(),
                (false, None) => " · prompt cache".to_string(),
            },
            if s.attn.weights != h3_core::quant::WeightQuant::Native {
                format!(" · {}", s.attn.weights)
            } else {
                String::new()
            }
        );
        (
            pipe,
            text,
            model,
            r.variant.clone(),
            src,
            &r.adapters[..],
            tables,
        )
    };
    let upscaler = match &s.upscale {
        None => None,
        Some(p) => {
            stage("loading the upscaler");
            Some(if s.synthetic {
                Esrgan::load(&be, EsrganConfig::tiny(), &Synthetic { seed: 5 })?
            } else {
                Esrgan::load(
                    &be,
                    EsrganConfig::anime_video_v3(),
                    &SafetensorsDir::open(p)?,
                )?
            })
        }
    };
    stage("waiting for the device");
    be.sync();
    {
        let mut info = shared.info.lock().unwrap();
        info["ready"] = json!(true);
        info["upscale"] = json!(upscaler.as_ref().map(|u| u.cfg.scale));
        info["upscale_max_pixels"] = json!(finish::SERVE_MAX_OUT_PIXELS);
        info["model"] = json!(model);
        info["backend"] = json!(be.name());
        info["load_s"] = json!(t0.elapsed().as_secs_f64());
        info["steps"] = json!(variant.forwards());
    }
    eprintln!(
        "engine ready on {} in {:.1}s",
        be.name(),
        t0.elapsed().as_secs_f64()
    );
    let sampler = match sampler_plan(&variant) {
        Ok(p) => p,
        Err(e) => return Err(anyhow!("{e}")),
    };
    let _ = Schedule::uniform;
    let pika = variant.pika_buckets;
    let sizing = if s.synthetic {
        Sizing::Canvas
    } else if pika {
        Sizing::Release {
            ref_short_edge: packing::pika::tier("768p")
                .map_err(|e| anyhow!("{e}"))?
                .reference_short_edge,
        }
    } else {
        Sizing::Release {
            ref_short_edge: packing::REFERENCE_IMAGE_SHORT_EDGE,
        }
    };

    for id in rx {
        let Some((prompt, geometry, out, seed, upscale, inputs)) = shared.with(id, |j| {
            j.started = Some(Instant::now());
            j.stage = Stage::Encoding;
            (
                j.prompt.clone(),
                j.geometry,
                j.out,
                j.seed,
                j.upscale,
                j.inputs.take().unwrap_or_default(),
            )
        }) else {
            continue;
        };
        let t = Instant::now();
        let run = || -> Result<()> {
            // Sizing is host work; the VAE / vision encodes follow on this
            // thread, inside the prompt encode (vision tower) and the
            // pipeline's prepare (VAE).
            let inputs::Prepared {
                keyframes,
                kf_images,
                references,
                presented,
                continuation,
            } = inputs::prepare(inputs, geometry, pika, sizing)?;
            let (p, tokens, source) = text.encode(&be, &prompt, &kf_images, &presented)?;
            drop(kf_images);
            drop(presented);
            shared.with(id, |j| {
                j.encode_s = t.elapsed().as_secs_f64();
                j.tokens = tokens;
                j.prompt_source = source;
                j.stage = Stage::Preparing;
            });
            let req = Request {
                prompt: p,
                geometry,
                seed,
                keyframes,
                references,
                continuation,
                sampler: match &sampler {
                    SamplerPlan::Schedules { video, audio } => SamplerPlan::Schedules {
                        video: video.clone(),
                        audio: audio.clone(),
                    },
                    SamplerPlan::Pdd(p) => SamplerPlan::Pdd(p.clone()),
                },
                attention: s.attn.attention(&variant),
                dense_first_steps: s.attn.dense_first_steps,
                ffn_chunk: s.attn.ffn_chunk.unwrap_or(usize::MAX),
                tiling: s.tiling,
                preview: s.synthetic || s.preview,
            };
            let generation = pipe.generate(
                &be,
                src.as_ref(),
                adapters,
                tables.as_ref(),
                &req,
                &mut |ev| match ev {
                    Event::Prepared { secs, .. } => {
                        shared.with(id, |j| {
                            j.prepare_s = secs;
                            j.total_steps = variant.forwards();
                            j.stage = Stage::Denoising;
                        });
                    }
                    Event::Step { index, total, secs } => {
                        shared.with(id, |j| {
                            j.steps.push(secs);
                            if index + 1 == total {
                                j.stage = Stage::Decoding;
                            }
                        });
                    }
                    Event::Frames {
                        count,
                        width,
                        height,
                        rgb,
                        ..
                    } => {
                        let (ow, oh) = (out.0.min(width), out.1.min(height));
                        let cropped = finish::crop(rgb, count, width, height, ow, oh);
                        shared.with(id, |j| {
                            j.width = ow;
                            j.height = oh;
                            j.rgb.extend_from_slice(&cropped);
                            j.frames += count;
                            j.decode_s = j.started.unwrap().elapsed().as_secs_f64()
                                - j.encode_s
                                - j.prepare_s
                                - j.steps.iter().sum::<f64>();
                        });
                    }
                    Event::Audio {
                        sample_rate,
                        stereo,
                    } => {
                        let mut buf = Vec::new();
                        let tmp = std::env::temp_dir()
                            .join(format!("h3-serve-{}-{id}.wav", std::process::id()));
                        if media::write_wav(&tmp, stereo, sample_rate as u32).is_ok() {
                            buf = std::fs::read(&tmp).unwrap_or_default();
                            let _ = std::fs::remove_file(&tmp);
                        }
                        shared.with(id, |j| j.audio = Some(buf));
                    }
                },
            )?;
            let _ = generation;
            if upscale != Upscale::Off {
                let res = upscale_job(&be, upscaler.as_ref(), &shared, id, upscale);
                shared.with(id, |j| j.up_error = res.err().map(|e| format!("{e:#}")));
            }
            Ok(())
        };
        let res = run();
        shared.with(id, |j| {
            j.total_s = t.elapsed().as_secs_f64();
            match res {
                Ok(()) => j.stage = Stage::Done,
                Err(e) => {
                    j.stage = Stage::Failed;
                    j.error = Some(format!("{e:#}"));
                }
            }
        });
    }
    Ok(())
}

/// Upscale a finished job's frames and keep them as JPEGs, frame by frame.
fn upscale_job<B: Backend>(
    be: &B,
    up: Option<&Esrgan<B>>,
    shared: &Shared,
    id: u64,
    upscale: Upscale,
) -> Result<()> {
    let up = up.ok_or_else(|| anyhow!("this engine has no upscaler (start it with --upscale)"))?;
    let (rgb, frames, w, h) = shared
        .with(id, |j| (j.rgb.clone(), j.frames, j.width, j.height))
        .ok_or_else(|| anyhow!("no such job"))?;
    let factor = match upscale {
        Upscale::Factor(k) => k,
        _ => finish::serve_factor(w, h, up.cfg.scale),
    };
    shared.with(id, |j| {
        j.stage = Stage::Upscaling;
        j.up_factor = Some(factor);
    });
    let t = Instant::now();
    let (tw, th) = (w * factor, h * factor);
    finish::upscale(be, up, &rgb, frames, w, h, factor, &mut |_, n, x| {
        let mut jpegs = vec![Vec::new(); n];
        std::thread::scope(|sc| -> Result<()> {
            let hs: Vec<_> = x
                .chunks(tw * th * 3)
                .map(|fr| sc.spawn(move || jpeg(fr, tw, th)))
                .collect();
            for (o, hnd) in jpegs.iter_mut().zip(hs) {
                *o = hnd.join().map_err(|_| anyhow!("JPEG encoder panicked"))??;
            }
            Ok(())
        })?;
        shared.with(id, |j| {
            j.hd.extend(jpegs);
            j.up_s = t.elapsed().as_secs_f64();
        });
        Ok(())
    })?;
    Ok(())
}

fn jpeg(rgb: &[u8], w: usize, h: usize) -> Result<Vec<u8>> {
    let mut buf = Vec::new();
    image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, 92).encode(
        rgb,
        w as u32,
        h as u32,
        image::ExtendedColorType::Rgb8,
    )?;
    Ok(buf)
}

fn hdr(k: &str, v: &str) -> tiny_http::Header {
    tiny_http::Header::from_bytes(k, v).expect("valid header")
}

fn send<R: std::io::Read>(
    req: tiny_http::Request,
    resp: tiny_http::Response<R>,
    cors: &[tiny_http::Header],
) {
    let mut resp = resp;
    for h in cors {
        resp = resp.with_header(h.clone());
    }
    let _ = req.respond(resp);
}

fn respond_json(req: tiny_http::Request, code: u16, v: &Value, cors: &[tiny_http::Header]) {
    let r = tiny_http::Response::from_string(v.to_string())
        .with_status_code(code)
        .with_header(hdr("Content-Type", "application/json"));
    send(req, r, cors);
}

/// CORS headers when the request's `Origin` is the one allowed origin.
fn cors_for(req: &tiny_http::Request, allow: Option<&str>) -> Vec<tiny_http::Header> {
    let origin = req
        .headers()
        .iter()
        .find(|h| h.field.equiv("Origin"))
        .map(|h| h.value.as_str().to_string());
    match (origin, allow) {
        (Some(o), Some(a)) if o.trim_end_matches('/') == a.trim_end_matches('/') => vec![
            hdr("Access-Control-Allow-Origin", &o),
            hdr("Vary", "Origin"),
            hdr("Access-Control-Allow-Methods", "GET, POST, OPTIONS"),
            hdr("Access-Control-Allow-Headers", "Content-Type"),
            hdr("Access-Control-Allow-Private-Network", "true"),
            hdr("Access-Control-Max-Age", "600"),
        ],
        _ => Vec::new(),
    }
}

/// Decode standard or URL-safe base64 (whitespace ignored, padding optional,
/// a `data:…;base64,` prefix allowed).
fn base64_decode(s: &str) -> Result<Vec<u8>, String> {
    let s = match s.strip_prefix("data:") {
        Some(rest) => rest.split_once(',').map_or(rest, |(_, b)| b),
        None => s,
    };
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let (mut acc, mut bits) = (0u32, 0u32);
    let mut padding = 0;
    for &c in s.as_bytes() {
        let v = match c {
            b'A'..=b'Z' => c - b'A',
            b'a'..=b'z' => c - b'a' + 26,
            b'0'..=b'9' => c - b'0' + 52,
            b'+' | b'-' => 62,
            b'/' | b'_' => 63,
            b'=' => {
                padding += 1;
                continue;
            }
            b' ' | b'\n' | b'\r' | b'\t' => continue,
            _ => return Err(format!("base64: unexpected byte {c:#04x}")),
        };
        if padding > 0 {
            return Err("base64: data after padding".into());
        }
        acc = (acc << 6) | v as u32;
        bits += 6;
        if bits >= 8 {
            bits -= 8;
            out.push((acc >> bits) as u8);
            acc &= (1 << bits) - 1;
        }
    }
    if bits >= 6 || (bits > 0 && acc != 0) {
        return Err("base64: truncated".into());
    }
    Ok(out)
}

/// The JSON fields `.b64` (required) and `.mime` of a media object.
fn media_object(v: &Value, what: &str) -> Result<(Vec<u8>, Option<String>), String> {
    if !v.is_object() {
        return Err(format!("`{what}` must be an object {{\"b64\": …, \"mime\": …}}."));
    }
    let b64 = v["b64"]
        .as_str()
        .ok_or_else(|| format!("`{what}` needs its bytes in `b64`."))?;
    let bytes = base64_decode(b64).map_err(|e| format!("`{what}`: {e}."))?;
    if bytes.is_empty() {
        return Err(format!("`{what}` is empty."));
    }
    let mime = match &v["mime"] {
        Value::Null => None,
        Value::String(m) => Some(m.trim().to_ascii_lowercase()),
        _ => return Err(format!("`{what}.mime` must be a string.")),
    };
    Ok((bytes, mime))
}

/// A PNG / JPEG keyframe or reference image; `mime` picks the decoder.
fn image_field(v: &Value, what: &str) -> Result<Image, String> {
    let (bytes, mime) = media_object(v, what)?;
    let fmt = match mime.as_deref() {
        Some("image/png") => image::ImageFormat::Png,
        Some("image/jpeg" | "image/jpg") => image::ImageFormat::Jpeg,
        Some(m) => {
            return Err(format!(
                "`{what}.mime` is {m}: send image/png or image/jpeg."
            ));
        }
        None => return Err(format!("`{what}` needs `mime` (image/png or image/jpeg).")),
    };
    media::decode_image(&bytes, Some(fmt)).map_err(|e| format!("`{what}`: {e:#}."))
}

/// Y4M frames (`mime` may be absent, video/x-yuv4mpeg or application/octet-stream).
fn video_field(v: &Value, what: &str) -> Result<(Vec<Image>, Option<f64>), String> {
    let (bytes, mime) = media_object(v, what)?;
    match mime.as_deref() {
        None | Some("video/x-yuv4mpeg" | "application/octet-stream") => {}
        Some(m) => {
            return Err(format!(
                "`{what}.mime` is {m}: send Y4M bytes (video/x-yuv4mpeg)."
            ));
        }
    }
    let (frames, fps) = media::decode_y4m(&bytes).map_err(|e| format!("`{what}`: {e:#}."))?;
    if frames.is_empty() {
        return Err(format!("`{what}` holds no frames."));
    }
    Ok((frames, fps))
}

/// A WAV soundtrack from `v[key]` (any PCM layout, resampled to `rate`).
fn wav_field(
    v: &Value,
    key: &str,
    what: &str,
    rate: usize,
) -> Result<Option<[Vec<f32>; 2]>, String> {
    let s = match &v[key] {
        Value::Null => return Ok(None),
        Value::String(s) => s,
        _ => return Err(format!("`{what}.{key}` must be a base64 string.")),
    };
    let bytes = base64_decode(s).map_err(|e| format!("`{what}.{key}`: {e}."))?;
    let a = media::decode_wav(&bytes, rate).map_err(|e| format!("`{what}.{key}`: {e:#}."))?;
    if a[0].is_empty() {
        return Err(format!("`{what}.{key}` holds no samples."));
    }
    Ok(Some(a))
}

/// Whether a generate body names any media field (the 64 MiB cap applies).
fn has_media_fields(v: &Value) -> bool {
    MEDIA_FIELDS.iter().any(|k| !v[*k].is_null())
}

/// The media and task fields of a generate body, against the tasks this
/// engine serves.
fn parse_inputs(v: &Value, s: &Settings) -> Result<(Task, Inputs), String> {
    let rate = s.audio_rate();
    let mut inputs = Inputs::default();
    for (key, anchor) in [("first_image", Anchor::First), ("last_image", Anchor::Last)] {
        if !v[key].is_null() {
            inputs.keyframes.push((anchor, image_field(&v[key], key)?));
        }
    }
    match &v["refs"] {
        Value::Null => {}
        Value::Array(items) => {
            for (i, r) in items.iter().enumerate() {
                let what = format!("refs[{i}]");
                let kind = r["kind"].as_str().unwrap_or("");
                inputs.refs.push(match kind {
                    "image" => RefInput::Image(image_field(r, &what)?),
                    "video" => {
                        let (frames, fps) = video_field(r, &what)?;
                        if frames.len() < inputs::MIN_REFERENCE_FRAMES {
                            return Err(format!(
                                "`{what}` has {} frames: a reference video needs at least {} (17n + 5 at 24 fps).",
                                frames.len(),
                                inputs::MIN_REFERENCE_FRAMES
                            ));
                        }
                        RefInput::Video {
                            frames,
                            fps,
                            audio: wav_field(r, "audio_b64", &what, rate)?,
                        }
                    }
                    "audio" => {
                        let (bytes, mime) = media_object(r, &what)?;
                        match mime.as_deref() {
                            None
                            | Some(
                                "audio/wav" | "audio/x-wav" | "audio/wave" | "audio/vnd.wave",
                            ) => {}
                            Some(m) => {
                                return Err(format!("`{what}.mime` is {m}: send a WAV."));
                            }
                        }
                        let a = media::decode_wav(&bytes, rate)
                            .map_err(|e| format!("`{what}`: {e:#}."))?;
                        if a[0].is_empty() {
                            return Err(format!("`{what}` holds no samples."));
                        }
                        RefInput::Audio(a)
                    }
                    other => {
                        return Err(format!(
                            "`{what}.kind` is `{other}`: image, video or audio."
                        ));
                    }
                });
            }
        }
        _ => return Err("`refs` must be an array.".into()),
    }
    let count = |k: &str| inputs.refs.iter().filter(|r| r.kind() == k).count();
    if count("image") > packing::MAX_REFERENCE_IMAGES
        || count("video") > packing::MAX_REFERENCE_VIDEOS
        || count("audio") > packing::MAX_REFERENCE_AUDIOS
    {
        return Err(format!(
            "At most {} image, {} video and {} audio references.",
            packing::MAX_REFERENCE_IMAGES,
            packing::MAX_REFERENCE_VIDEOS,
            packing::MAX_REFERENCE_AUDIOS
        ));
    }
    if !v["continue"].is_null() {
        let (frames, _) = video_field(&v["continue"], "continue")?;
        if !packing::CONTINUATION_FRAMES.contains(&frames.len()) {
            return Err(format!(
                "`continue` has {} frames: a continuation clip has 5, 22 or 39.",
                frames.len()
            ));
        }
        let audio = wav_field(&v["continue"], "audio_b64", "continue", rate)?;
        inputs.continuation = Some((frames, audio));
    }
    let served = s.served_tasks();
    let task = match &v["task"] {
        Value::Null => {
            if inputs.is_empty() {
                s.default_task()
            } else {
                inputs.implied_task()
            }
        }
        Value::String(t) => t.parse::<Task>()?,
        _ => return Err("`task` must be t2va, fl2va or ref2va.".into()),
    };
    if !served.contains(&task) {
        let hint = match task {
            Task::Ref2va => " (start h3 serve with --task ref2va)",
            Task::Regen2k => " (regen2k is `h3 regen2k`, not a request)",
            _ => "",
        };
        return Err(format!(
            "task {} not supported by this engine: it serves {}{hint}.",
            task_name(task),
            served
                .iter()
                .map(|t| task_name(*t))
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    match task {
        Task::T2va if !inputs.is_empty() => {
            return Err("task t2va takes no images, references or continuation: set `task`.".into());
        }
        Task::Fl2va if inputs.keyframes.is_empty() => {
            return Err("task fl2va needs `first_image` and / or `last_image`.".into());
        }
        Task::Fl2va if !inputs.refs.is_empty() || inputs.continuation.is_some() => {
            return Err("task fl2va takes keyframes only, not `refs` / `continue`.".into());
        }
        Task::Ref2va if inputs.refs.is_empty() => {
            return Err("task ref2va needs at least one entry in `refs`.".into());
        }
        Task::Ref2va if !inputs.keyframes.is_empty() => {
            return Err("task ref2va takes `refs` / `continue`, not keyframes.".into());
        }
        _ => {}
    }
    Ok((task, inputs))
}

#[cfg(test)]
fn parse_request(body: &str, s: &Settings, pika: bool) -> Result<Parsed, String> {
    let v: Value = serde_json::from_str(body).map_err(|e| format!("body is not JSON: {e}"))?;
    parse_value(&v, s, pika)
}

fn parse_value(v: &Value, s: &Settings, pika: bool) -> Result<Parsed, String> {
    if !v.is_object() {
        return Err("body is not a JSON object".into());
    }
    let prompt = v["prompt"].as_str().unwrap_or("").trim().to_string();
    if prompt.is_empty() {
        return Err("Write a prompt first.".into());
    }
    if prompt.chars().count() > 2000 {
        return Err("Prompts are limited to 2000 characters.".into());
    }
    let num = |k: &str, d: u64| v[k].as_u64().unwrap_or(d) as usize;
    let frames = if pika {
        packing::pika::resolve_frames(num("frames", 124))
    } else {
        packing::resolve_num_frames(num("frames", 124))
    }
    .map_err(|e| e.to_string())?;
    let out = (num("width", 1344), num("height", 768));
    // Off the 32-pixel grid: render the next canvas up and crop back (not
    // for Pika's bucketed canvases).
    let crop = !pika && out.0 % 16 == 0 && out.1 % 16 == 0;
    let g = Geometry {
        width: if crop {
            finish::canvas_edge(out.0)
        } else {
            out.0
        },
        height: if crop {
            finish::canvas_edge(out.1)
        } else {
            out.1
        },
        frames,
    };
    g.validate(16).map_err(|e| {
        if pika {
            e.to_string()
        } else {
            format!(
                "{}×{}: width and height must be positive multiples of 16.",
                out.0, out.1
            )
        }
    })?;
    if g.width * g.height > s.max_pixels {
        return Err(format!(
            "{}×{} is above this engine's limit of {} pixels per frame.",
            g.width, g.height, s.max_pixels
        ));
    }
    if g.frames > s.max_frames {
        return Err(format!(
            "{} frames is above this engine's limit of {}.",
            g.frames, s.max_frames
        ));
    }
    let upscale = match &v["upscale"] {
        Value::Null | Value::Bool(false) => Upscale::Off,
        Value::Bool(true) => Upscale::Auto,
        x => match x.as_u64() {
            Some(k @ 2..=4) => Upscale::Factor(k as usize),
            _ => return Err("`upscale` is true, false, 2, 3 or 4.".into()),
        },
    };
    if upscale != Upscale::Off && s.upscale.is_none() {
        return Err(
            "This engine has no upscaler: start it with --upscale realesr-animevideov3.safetensors."
                .into(),
        );
    }
    let seed = v["seed"].as_u64().unwrap_or_else(|| {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64 % 1_000_000_000)
            .unwrap_or(7)
    });
    let (task, inputs) = parse_inputs(v, s)?;
    Ok(Parsed {
        prompt,
        geometry: g,
        out,
        seed,
        upscale,
        task,
        inputs,
    })
}

/// Serve until killed. `start_engine` builds the backend on the engine thread.
pub fn serve(
    s: Settings,
    start_engine: impl FnOnce(&Settings, Arc<Shared>, mpsc::Receiver<u64>) -> Result<()>
    + Send
    + 'static,
) -> Result<()> {
    let pika = s.model.as_ref().is_some_and(|m| m.variant.pika_buckets);
    let tasks: Vec<&str> = s.served_tasks().into_iter().map(task_name).collect();
    let shared = Arc::new(Shared {
        jobs: Mutex::new(Vec::new()),
        info: Mutex::new(json!({
            "ready": false, "error": null, "backend": null, "model": null,
            "synthetic": s.synthetic, "gallery": null,
            "max_pixels": s.max_pixels, "max_frames": s.max_frames, "fps": packing::FPS, "sample_rate": 32000,
            "variant": s.model.as_ref().map(|m| m.variant.name.clone()), "pika": pika,
            "task": s.model.as_ref().map(|m| task_name(m.task)),
            "tasks": tasks,
            "inputs": {"max_body_bytes": MAX_BODY, "max_body_bytes_without_media": SMALL_BODY,
                       "max_image_pixels": media::MAX_DECODED_PIXELS},
        })),
        next: Mutex::new(1),
    });
    let (tx, rx) = mpsc::channel::<u64>();
    let server = tiny_http::Server::http(&s.addr).map_err(|e| anyhow!("bind {}: {e}", s.addr))?;
    eprintln!("h3 demo: http://{}/", s.addr);
    let s = Arc::new(s);
    {
        let (shared, s) = (shared.clone(), s.clone());
        std::thread::Builder::new()
            .name("engine".into())
            .spawn(move || {
                let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    start_engine(&s, shared.clone(), rx)
                }));
                let msg = match run {
                    Ok(Ok(())) => return,
                    Ok(Err(e)) => format!("{e:#}"),
                    Err(p) => format!(
                        "engine crashed: {}",
                        p.downcast_ref::<String>()
                            .map(String::as_str)
                            .or_else(|| p.downcast_ref::<&str>().copied())
                            .unwrap_or("panic")
                    ),
                };
                eprintln!("engine failed: {msg}");
                {
                    let mut info = shared.info.lock().unwrap();
                    info["ready"] = json!(false);
                    info["error"] = json!(msg.clone());
                }
                for j in shared.jobs.lock().unwrap().iter_mut() {
                    if !matches!(j.stage, Stage::Done | Stage::Failed) {
                        j.stage = Stage::Failed;
                        j.error = Some(msg.clone());
                    }
                }
            })
            .context("spawn engine")?;
    }

    for mut req in server.incoming_requests() {
        let url = req.url().to_string();
        let (path, query) = url.split_once('?').unwrap_or((&url, ""));
        let method = req.method().clone();
        let cors = cors_for(&req, s.allow_origin.as_deref());
        match (method, path) {
            (tiny_http::Method::Options, _) => {
                let code = if cors.is_empty() { 403 } else { 204 };
                send(req, tiny_http::Response::empty(code), &cors);
            }
            (tiny_http::Method::Get, "/") | (tiny_http::Method::Get, "/index.html") => {
                let h = hdr("Content-Type", "text/html; charset=utf-8");
                send(
                    req,
                    tiny_http::Response::from_string(PAGE).with_header(h),
                    &cors,
                );
            }
            (tiny_http::Method::Get, "/api/info") => {
                let mut info = shared.info.lock().unwrap().clone();
                let jobs = shared.jobs.lock().unwrap();
                info["queued"] = json!(jobs.iter().filter(|j| j.stage == Stage::Queued).count());
                drop(jobs);
                respond_json(req, 200, &info, &cors);
            }
            (tiny_http::Method::Post, "/api/generate") => {
                let header = |k: &str| {
                    req.headers()
                        .iter()
                        .find(|h| h.field.as_str().as_str().eq_ignore_ascii_case(k))
                        .map(|h| h.value.as_str().to_string())
                };
                let json_body =
                    header("Content-Type").is_some_and(|c| c.starts_with("application/json"));
                let origin_ok = match header("Origin") {
                    None => true,
                    Some(o) => {
                        !cors.is_empty()
                            || header("Host").is_some_and(|h| {
                                o == format!("http://{h}") || o == format!("https://{h}")
                            })
                    }
                };
                if !json_body || !origin_ok {
                    respond_json(
                        req,
                        403,
                        &json!({"error": "Requests must be JSON and come from this engine's page or its --allow-origin site."}),
                        &cors,
                    );
                    continue;
                }
                // Up to MAX_BODY with media fields, SMALL_BODY without; the
                // declared length refuses an oversize body before it is read.
                let too_big = |req: tiny_http::Request, cors: &[tiny_http::Header]| {
                    respond_json(
                        req,
                        413,
                        &json!({"error": format!(
                            "The request body is limited to {} MiB with images / references and {} KiB without.",
                            MAX_BODY >> 20, SMALL_BODY >> 10
                        )}),
                        cors,
                    );
                };
                if req.body_length().is_some_and(|n| n > MAX_BODY) {
                    too_big(req, &cors);
                    continue;
                }
                let mut raw = Vec::new();
                if req
                    .as_reader()
                    .take(MAX_BODY as u64 + 1)
                    .read_to_end(&mut raw)
                    .is_err()
                {
                    respond_json(req, 400, &json!({"error": "unreadable body"}), &cors);
                    continue;
                }
                if raw.len() > MAX_BODY {
                    too_big(req, &cors);
                    continue;
                }
                let value: Result<Value, String> = String::from_utf8(raw)
                    .map_err(|_| "body is not UTF-8".to_string())
                    .and_then(|body| {
                        let v: Value = serde_json::from_str(&body)
                            .map_err(|e| format!("body is not JSON: {e}"))?;
                        if body.len() > SMALL_BODY && !has_media_fields(&v) {
                            return Err(String::new());
                        }
                        Ok(v)
                    });
                let value = match value {
                    Ok(v) => v,
                    Err(e) if e.is_empty() => {
                        too_big(req, &cors);
                        continue;
                    }
                    Err(e) => {
                        respond_json(req, 400, &json!({"error": e}), &cors);
                        continue;
                    }
                };
                let info = shared.info.lock().unwrap().clone();
                if !info["ready"].as_bool().unwrap_or(false) {
                    let msg = info["error"]
                        .as_str()
                        .map(|e| format!("The engine failed to start: {e}"))
                        .unwrap_or_else(|| {
                            "The engine is still loading the model. Try again in a moment.".into()
                        });
                    respond_json(req, 503, &json!({"error": msg}), &cors);
                    continue;
                }
                let parsed = parse_value(&value, &s, pika).and_then(|r| {
                    let info = shared.info.lock().unwrap();
                    match info["gallery"].as_array().filter(|_| info["gallery_only"].as_bool().unwrap_or(false)) {
                        Some(g) if !g.iter().any(|p| p.as_str().is_some_and(|p| p.split_whitespace().eq(r.prompt.split_whitespace()))) => {
                            Err(format!("This engine has no text encoder: pick one of its {} gallery prompts.", g.len()))
                        }
                        _ => Ok(r),
                    }
                });
                match parsed {
                    Err(e) => respond_json(req, 400, &json!({"error": e}), &cors),
                    Ok(parsed) => {
                        let mut jobs = shared.jobs.lock().unwrap();
                        if jobs
                            .iter()
                            .filter(|j| matches!(j.stage, Stage::Queued))
                            .count()
                            >= MAX_QUEUE
                        {
                            drop(jobs);
                            respond_json(
                                req,
                                429,
                                &json!({"error": "Three clips are already waiting. Try again when one finishes."}),
                                &cors,
                            );
                            continue;
                        }
                        let id = {
                            let mut n = shared.next.lock().unwrap();
                            *n += 1;
                            *n - 1
                        };
                        let total_steps = info["steps"].as_u64().unwrap_or(8) as usize;
                        let seed = parsed.seed;
                        let task = task_name(parsed.task);
                        jobs.push(Job::new(id, parsed, total_steps));
                        while jobs.len() > KEEP_JOBS {
                            match jobs
                                .iter()
                                .position(|j| matches!(j.stage, Stage::Done | Stage::Failed))
                            {
                                Some(i) => {
                                    jobs.remove(i);
                                }
                                None => break,
                            }
                        }
                        drop(jobs);
                        let _ = tx.send(id);
                        respond_json(
                            req,
                            200,
                            &json!({"id": id, "seed": seed, "task": task}),
                            &cors,
                        );
                    }
                }
            }
            (tiny_http::Method::Get, p) if p.starts_with("/api/jobs/") => {
                let rest = &p["/api/jobs/".len()..];
                let (id_s, what) = match rest.rsplit_once('/') {
                    Some((i, w)) => (i, w),
                    None => (rest, ""),
                };
                let Ok(id) = id_s.parse::<u64>() else {
                    respond_json(req, 404, &json!({"error": "no such job"}), &cors);
                    continue;
                };
                match what {
                    "frames" => {
                        let from = query
                            .split('&')
                            .find_map(|kv| kv.strip_prefix("from="))
                            .and_then(|v| v.parse::<usize>().ok())
                            .unwrap_or(0);
                        let body = shared.with(id, |j| {
                            let px = j.width * j.height * 3;
                            let first = from.min(j.frames);
                            let count = j.frames - first;
                            let mut out = Vec::with_capacity(16 + count * px);
                            for v in [j.width, j.height, first, count] {
                                out.extend_from_slice(&(v as u32).to_le_bytes());
                            }
                            out.extend_from_slice(&j.rgb[first * px..(first + count) * px]);
                            out
                        });
                        match body {
                            Some(b) => send(
                                req,
                                tiny_http::Response::from_data(b)
                                    .with_header(hdr("Content-Type", "application/octet-stream")),
                                &cors,
                            ),
                            None => respond_json(req, 404, &json!({"error": "no such job"}), &cors),
                        }
                    }
                    "hd" => {
                        let n = query
                            .split('&')
                            .find_map(|kv| kv.strip_prefix("frame="))
                            .and_then(|v| v.parse::<usize>().ok())
                            .unwrap_or(0);
                        match shared.with(id, |j| j.hd.get(n).cloned()) {
                            Some(Some(b)) => send(
                                req,
                                tiny_http::Response::from_data(b)
                                    .with_header(hdr("Content-Type", "image/jpeg")),
                                &cors,
                            ),
                            Some(None) => respond_json(
                                req,
                                404,
                                &json!({"error": "that frame is not upscaled (yet)"}),
                                &cors,
                            ),
                            None => respond_json(req, 404, &json!({"error": "no such job"}), &cors),
                        }
                    }
                    "audio" => match shared.with(id, |j| j.audio.clone()) {
                        Some(Some(b)) => send(
                            req,
                            tiny_http::Response::from_data(b)
                                .with_header(hdr("Content-Type", "audio/wav")),
                            &cors,
                        ),
                        Some(None) => {
                            respond_json(req, 404, &json!({"error": "no audio yet"}), &cors)
                        }
                        None => respond_json(req, 404, &json!({"error": "no such job"}), &cors),
                    },
                    "" => match shared.with(id, |j| j.status()) {
                        Some(v) => respond_json(req, 200, &v, &cors),
                        None => respond_json(req, 404, &json!({"error": "no such job"}), &cors),
                    },
                    _ => respond_json(req, 404, &json!({"error": "not found"}), &cors),
                }
            }
            _ => respond_json(req, 404, &json!({"error": "not found"}), &cors),
        }
    }
    Ok(())
}

use std::io::Read as _;

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;

    #[derive(Parser)]
    struct A {
        #[command(flatten)]
        attn: AttnArgs,
    }

    fn settings(upscale: bool) -> Settings {
        Settings {
            addr: String::new(),
            model: None,
            gallery: None,
            synthetic: true,
            max_pixels: 1344 * 768,
            max_frames: 243,
            allow_origin: None,
            attn: A::parse_from(["t"]).attn,
            tiling: None,
            preview: false,
            upscale: upscale.then(|| PathBuf::from("x")),
            encoder: crate::encoder::EncoderArgs {
                encoder: None,
                encoder_weights: Default::default(),
                prompt_cache: None,
                encoder_int8: None,
            },
        }
    }

    #[test]
    fn sizes_off_the_grid_render_the_next_canvas_up() {
        let p = parse_request(
            r#"{"prompt":"a fox","width":480,"height":272}"#,
            &settings(false),
            false,
        )
        .unwrap();
        assert_eq!(
            (p.geometry.width, p.geometry.height, p.out),
            (480, 288, (480, 272))
        );
        let p = parse_request(
            r#"{"prompt":"a fox","width":272,"height":480}"#,
            &settings(false),
            false,
        )
        .unwrap();
        assert_eq!(
            (p.geometry.width, p.geometry.height, p.out),
            (288, 480, (272, 480))
        );
        let p = parse_request(
            r#"{"prompt":"a fox","width":1344,"height":768}"#,
            &settings(false),
            false,
        )
        .unwrap();
        assert_eq!(
            (p.geometry.width, p.geometry.height, p.out),
            (1344, 768, (1344, 768))
        );
        assert!(
            parse_request(
                r#"{"prompt":"a fox","width":481,"height":272}"#,
                &settings(false),
                false
            )
            .is_err()
        );
        // Pika's canvases are not cropped.
        assert!(
            parse_request(
                r#"{"prompt":"a fox","width":480,"height":272}"#,
                &settings(false),
                true
            )
            .is_err()
        );
    }

    #[test]
    fn upscale_needs_an_upscaler_and_a_known_factor() {
        let body =
            |u: &str| format!(r#"{{"prompt":"a fox","width":480,"height":272,"upscale":{u}}}"#);
        assert_eq!(
            parse_request(&body("false"), &settings(false), false)
                .unwrap()
                .upscale,
            Upscale::Off
        );
        assert!(
            parse_request(&body("true"), &settings(false), false)
                .unwrap_err()
                .contains("--upscale")
        );
        assert_eq!(
            parse_request(&body("true"), &settings(true), false)
                .unwrap()
                .upscale,
            Upscale::Auto
        );
        assert_eq!(
            parse_request(&body("3"), &settings(true), false)
                .unwrap()
                .upscale,
            Upscale::Factor(3)
        );
        assert!(parse_request(&body("5"), &settings(true), false).is_err());
        assert!(parse_request(&body("\"yes\""), &settings(true), false).is_err());
    }

    #[test]
    fn upscaled_frames_are_jpegs_of_the_factor() {
        let be = h3_cpu::Cpu;
        let up = Esrgan::load(&be, EsrganConfig::tiny(), &Synthetic { seed: 5 }).unwrap();
        let shared = Shared {
            jobs: Mutex::new(Vec::new()),
            info: Mutex::new(json!({})),
            next: Mutex::new(1),
        };
        let p = parse_request(
            r#"{"prompt":"a fox","width":480,"height":272,"upscale":true}"#,
            &settings(true),
            false,
        )
        .unwrap();
        let mut j = Job::new(1, p, 4);
        (j.width, j.height, j.frames) = (20, 12, 3);
        j.rgb = (0..3 * 20 * 12 * 3).map(|i| (i * 13 % 256) as u8).collect();
        shared.jobs.lock().unwrap().push(j);
        upscale_job(&be, Some(&up), &shared, 1, Upscale::Auto).unwrap();
        let st = shared.with(1, |j| j.status()).unwrap();
        assert_eq!(
            (
                st["stage"].as_str(),
                st["upscale"].as_u64(),
                st["up_frames"].as_u64()
            ),
            (Some("upscaling"), Some(2), Some(3))
        );
        let hd = shared.with(1, |j| j.hd.clone()).unwrap();
        for f in &hd {
            let im = image::load_from_memory_with_format(f, image::ImageFormat::Jpeg).unwrap();
            assert_eq!((im.width(), im.height()), (40, 24));
        }
        assert!(upscale_job(&be, None, &shared, 1, Upscale::Auto).is_err());
    }

    // ---- request inputs ----------------------------------------------------

    /// Settings for a real preset started with `--task`, without weights.
    fn model_settings(variant: &str, task: Task, gallery: bool) -> Settings {
        let mut s = settings(false);
        s.synthetic = false;
        s.gallery = gallery.then(|| PathBuf::from("gallery"));
        s.model = Some(Resolved {
            variant: Variant::by_name(variant).unwrap(),
            task,
            model_dir: None,
            variant_dir: None,
            transformer: PathBuf::from("transformer"),
            adapters: vec![],
            tae: None,
            tables: None,
        });
        s
    }

    fn b64(bytes: &[u8]) -> String {
        const A: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
        let mut out = String::new();
        for c in bytes.chunks(3) {
            let n = (c[0] as u32) << 16
                | (c.get(1).copied().unwrap_or(0) as u32) << 8
                | c.get(2).copied().unwrap_or(0) as u32;
            out.push(A[(n >> 18) as usize & 63] as char);
            out.push(A[(n >> 12) as usize & 63] as char);
            out.push(if c.len() > 1 { A[(n >> 6) as usize & 63] as char } else { '=' });
            out.push(if c.len() > 2 { A[n as usize & 63] as char } else { '=' });
        }
        out
    }

    fn png(w: u32, h: u32, shade: u8) -> Vec<u8> {
        let mut buf = std::io::Cursor::new(Vec::new());
        image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([shade, (x * 30) as u8, (y * 30) as u8])
        })
        .write_to(&mut buf, image::ImageFormat::Png)
        .unwrap();
        buf.into_inner()
    }

    fn y4m(frames: usize, w: usize, h: usize) -> Vec<u8> {
        let rgb: Vec<u8> = (0..frames * w * h * 3).map(|i| (i * 7 % 256) as u8).collect();
        let mut out = Vec::new();
        media::write_y4m_to(&mut out, frames, w, h, &rgb, 24).unwrap();
        out
    }

    fn wav(seconds: f32) -> Vec<u8> {
        let n = (seconds * 32000.0) as usize;
        let s: Vec<f32> = (0..n).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        let mut out = Vec::new();
        media::write_wav_to(&mut out, &[s.clone(), s], 32000).unwrap();
        out
    }

    fn image_json(bytes: &[u8], mime: &str) -> Value {
        json!({"b64": b64(bytes), "mime": mime})
    }

    #[test]
    fn base64_round_trips_and_rejects_junk() {
        for n in 0..20usize {
            let data: Vec<u8> = (0..n).map(|i| (i * 37 + 11) as u8).collect();
            assert_eq!(base64_decode(&b64(&data)).unwrap(), data);
            // Unpadded and URL-safe alphabets decode too.
            let s = b64(&data).replace('=', "").replace('+', "-").replace('/', "_");
            assert_eq!(base64_decode(&s).unwrap(), data);
        }
        assert_eq!(
            base64_decode("data:image/png;base64,AQID").unwrap(),
            vec![1, 2, 3]
        );
        assert_eq!(base64_decode("AQ ID\n").unwrap(), vec![1, 2, 3]);
        assert!(base64_decode("AQ$D").is_err());
        assert!(base64_decode("A").is_err());
        assert!(base64_decode("AQ==ID").is_err());
    }

    #[test]
    fn served_tasks_follow_the_resident_dit() {
        let names = |s: &Settings| {
            s.served_tasks()
                .into_iter()
                .map(task_name)
                .collect::<Vec<_>>()
        };
        assert_eq!(names(&settings(false)), ["t2va", "fl2va", "ref2va"]);
        assert_eq!(
            names(&model_settings("h3-plug4", Task::T2va, false)),
            ["t2va", "fl2va"]
        );
        assert_eq!(
            names(&model_settings("h3-plug4", Task::Fl2va, false)),
            ["t2va", "fl2va"]
        );
        // `--task ref2va` loads transformer_ref: Ref2VA only.
        assert_eq!(
            names(&model_settings("h3", Task::Ref2va, false)),
            ["ref2va"]
        );
        assert_eq!(
            names(&model_settings("h3-turbo4-ref", Task::Ref2va, false)),
            ["ref2va"]
        );
        assert_eq!(
            names(&model_settings("fasth3-v2", Task::T2va, false)),
            ["t2va"]
        );
        // Pika's adapters are per task: only the loaded one's task.
        assert_eq!(
            names(&model_settings("pika-flex", Task::Fl2va, false)),
            ["fl2va"]
        );
        assert_eq!(
            names(&model_settings("pika-flex", Task::Ref2va, false)),
            ["ref2va"]
        );
        // A gallery engine has no vision tower.
        assert_eq!(
            names(&model_settings("h3-plug4", Task::T2va, true)),
            ["t2va"]
        );
        assert!(names(&model_settings("h3", Task::Ref2va, true)).is_empty());
        let mut none = settings(false);
        none.synthetic = false;
        assert!(names(&none).is_empty());
    }

    #[test]
    fn request_task_defaults_and_gating() {
        let s = settings(false);
        let base = |extra: Value| {
            let mut v = json!({"prompt": "a fox", "width": 64, "height": 32});
            for (k, x) in extra.as_object().unwrap() {
                v[k] = x.clone();
            }
            v.to_string()
        };
        let p = parse_request(&base(json!({})), &s, false).unwrap();
        assert_eq!(p.task, Task::T2va);
        assert!(p.inputs.is_empty());
        // Media imply the task when `task` is absent.
        let p = parse_request(
            &base(json!({"first_image": image_json(&png(8, 8, 1), "image/png")})),
            &s,
            false,
        )
        .unwrap();
        assert_eq!(p.task, Task::Fl2va);
        assert_eq!(p.inputs.keyframes.len(), 1);
        assert_eq!(p.inputs.keyframes[0].0, Anchor::First);
        let p = parse_request(
            &base(json!({
                "task": "ref2va",
                "refs": [
                    {"kind": "image", "b64": b64(&png(8, 8, 2)), "mime": "image/jpeg"},
                ],
            })),
            &s,
            false,
        );
        // A PNG sent as image/jpeg does not decode.
        assert!(p.unwrap_err().contains("refs[0]"));
        let p = parse_request(
            &base(json!({
                "refs": [
                    {"kind": "image", "b64": b64(&png(8, 8, 2)), "mime": "image/png"},
                    {"kind": "video", "b64": b64(&y4m(22, 16, 8)), "audio_b64": b64(&wav(0.1))},
                    {"kind": "audio", "b64": b64(&wav(0.1)), "mime": "audio/wav"},
                ],
                "continue": {"b64": b64(&y4m(22, 16, 8))},
            })),
            &s,
            false,
        )
        .unwrap();
        assert_eq!(p.task, Task::Ref2va);
        assert_eq!(p.inputs.refs.len(), 3);
        assert!(matches!(&p.inputs.refs[1], RefInput::Video { frames, audio: Some(_), .. } if frames.len() == 22));
        assert_eq!(p.inputs.continuation.as_ref().unwrap().0.len(), 22);
        // The explicit task must match the media.
        let err = |extra: Value| parse_request(&base(extra), &s, false).unwrap_err();
        assert!(err(json!({"task": "fl2va"})).contains("first_image"));
        assert!(err(json!({"task": "ref2va"})).contains("refs"));
        assert!(
            err(json!({"task": "t2va", "first_image": image_json(&png(8, 8, 1), "image/png")}))
                .contains("takes no images")
        );
        assert!(err(json!({"task": "regen2k"})).contains("not supported by this engine"));
        assert!(err(json!({"task": "flying"})).contains("unknown task"));
        assert!(err(json!({"task": 3})).contains("`task`"));

        // A resident T2VA / FL2VA DiT cannot take references.
        let plug = model_settings("h3-plug4", Task::T2va, false);
        let body = base(json!({
            "refs": [{"kind": "image", "b64": b64(&png(8, 8, 2)), "mime": "image/png"}],
        }));
        let e = parse_request(&body, &plug, false).unwrap_err();
        assert!(e.contains("task ref2va not supported by this engine"), "{e}");
        assert!(e.contains("--task ref2va"), "{e}");
        let p = parse_request(
            &base(json!({"last_image": image_json(&png(8, 8, 1), "image/jpeg")})),
            &plug,
            false,
        );
        assert!(p.unwrap_err().contains("last_image"));
        let p = parse_request(
            &base(json!({"last_image": image_json(&png(8, 8, 1), "image/png")})),
            &plug,
            false,
        )
        .unwrap();
        assert_eq!((p.task, p.inputs.keyframes[0].0), (Task::Fl2va, Anchor::Last));
        // A Ref2VA engine serves Ref2VA alone; with no media the default is
        // its start task, whose check then names what it needs.
        let r2v = model_settings("h3", Task::Ref2va, false);
        assert!(parse_request(&base(json!({})), &r2v, false).unwrap_err().contains("refs"));
        assert!(
            parse_request(
                &base(json!({"first_image": image_json(&png(8, 8, 1), "image/png")})),
                &r2v,
                false
            )
            .unwrap_err()
            .contains("not supported by this engine")
        );
        // A gallery engine takes text only.
        let g = model_settings("h3-plug4", Task::T2va, true);
        assert!(
            parse_request(
                &base(json!({"first_image": image_json(&png(8, 8, 1), "image/png")})),
                &g,
                false
            )
            .unwrap_err()
            .contains("not supported by this engine")
        );
    }

    #[test]
    fn bad_media_is_refused_with_a_reason() {
        let s = settings(false);
        let body = |extra: Value| {
            let mut v = json!({"prompt": "a fox", "width": 64, "height": 32});
            for (k, x) in extra.as_object().unwrap() {
                v[k] = x.clone();
            }
            v.to_string()
        };
        let err = |extra: Value| parse_request(&body(extra), &s, false).unwrap_err();
        let im = png(8, 8, 1);
        assert!(err(json!({"first_image": image_json(&im, "image/gif")})).contains("image/png or image/jpeg"));
        assert!(err(json!({"first_image": {"b64": b64(&im)}})).contains("needs `mime`"));
        assert!(err(json!({"first_image": {"b64": "@@@", "mime": "image/png"}})).contains("base64"));
        assert!(err(json!({"first_image": {"b64": "", "mime": "image/png"}})).contains("empty"));
        assert!(err(json!({"first_image": "abc"})).contains("must be an object"));
        assert!(err(json!({"first_image": {"b64": b64(b"not a png"), "mime": "image/png"}})).contains("first_image"));
        assert!(err(json!({"refs": "x"})).contains("array"));
        assert!(err(json!({"refs": [{"kind": "gif", "b64": b64(&im), "mime": "image/png"}]})).contains("image, video or audio"));
        assert!(err(json!({"refs": [{"kind": "video", "b64": b64(&im)}]})).contains("YUV4MPEG2"));
        assert!(err(json!({"refs": [{"kind": "video", "b64": b64(&y4m(22, 8, 8)), "mime": "video/mp4"}]})).contains("Y4M"));
        assert!(err(json!({"refs": [{"kind": "video", "b64": b64(&y4m(5, 8, 8))}]})).contains("at least 22"));
        assert!(err(json!({"refs": [{"kind": "audio", "b64": b64(&im)}]})).contains("WAV"));
        assert!(err(json!({"refs": [{"kind": "video", "b64": b64(&y4m(22, 8, 8)), "audio_b64": b64(&im)}]})).contains("audio_b64"));
        assert!(err(json!({"continue": {"b64": b64(&y4m(6, 8, 8))}})).contains("5, 22 or 39"));
        let many: Vec<Value> = (0..10).map(|_| json!({"kind": "image", "b64": b64(&im), "mime": "image/png"})).collect();
        assert!(err(json!({"refs": many})).contains("At most"));

        // An image over the pixel limit is refused from its header alone: a
        // PNG whose IHDR claims 9000×9000 and holds no pixels.
        let mut big = vec![0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];
        let mut ihdr = b"IHDR".to_vec();
        ihdr.extend_from_slice(&9000u32.to_be_bytes());
        ihdr.extend_from_slice(&9000u32.to_be_bytes());
        ihdr.extend_from_slice(&[8, 2, 0, 0, 0]);
        big.extend_from_slice(&13u32.to_be_bytes());
        big.extend_from_slice(&ihdr);
        big.extend_from_slice(&crc32(&ihdr).to_be_bytes());
        for chunk in [b"IDAT", b"IEND"] {
            big.extend_from_slice(&0u32.to_be_bytes());
            big.extend_from_slice(chunk);
            big.extend_from_slice(&crc32(chunk).to_be_bytes());
        }
        let e = err(json!({"first_image": image_json(&big, "image/png")}));
        assert!(e.contains("limit"), "{e}");

        // The small cap applies to bodies without media fields.
        assert!(!has_media_fields(&json!({"prompt": "x"})));
        assert!(has_media_fields(&json!({"prompt": "x", "refs": []})));
        assert!(has_media_fields(&json!({"continue": {}})));
    }

    fn crc32(data: &[u8]) -> u32 {
        let mut c = !0u32;
        for &b in data {
            c ^= b as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xedb8_8320 ^ (c >> 1) } else { c >> 1 };
            }
        }
        !c
    }

    /// One HTTP/1.1 request to the test engine: (status, body).
    fn http(addr: &str, method: &str, path: &str, body: Option<&str>) -> (u16, Vec<u8>) {
        use std::io::Write as _;
        let mut st = std::net::TcpStream::connect(addr).unwrap();
        let body = body.unwrap_or("");
        let mut req = format!("{method} {path} HTTP/1.1\r\nHost: {addr}\r\nConnection: close\r\n");
        if method == "POST" {
            req.push_str(&format!(
                "Content-Type: application/json\r\nContent-Length: {}\r\n",
                body.len()
            ));
        }
        req.push_str("\r\n");
        st.write_all(req.as_bytes()).unwrap();
        st.write_all(body.as_bytes()).unwrap();
        let mut raw = Vec::new();
        st.read_to_end(&mut raw).unwrap();
        let sep = raw.windows(4).position(|w| w == b"\r\n\r\n").expect("headers");
        let head = std::str::from_utf8(&raw[..sep]).unwrap();
        let status: u16 = head.split_whitespace().nth(1).unwrap().parse().unwrap();
        (status, raw[sep + 4..].to_vec())
    }

    fn http_json(addr: &str, method: &str, path: &str, body: Option<&str>) -> (u16, Value) {
        let (code, b) = http(addr, method, path, body);
        (code, serde_json::from_slice(&b).unwrap_or(Value::Null))
    }

    /// Poll a job to its end.
    fn wait_job(addr: &str, id: u64) -> Value {
        let t = Instant::now();
        loop {
            let (_, s) = http_json(addr, "GET", &format!("/api/jobs/{id}"), None);
            if matches!(s["stage"].as_str(), Some("done" | "failed")) {
                return s;
            }
            assert!(t.elapsed().as_secs() < 900, "job {id} did not finish: {s}");
            std::thread::sleep(std::time::Duration::from_millis(200));
        }
    }

    /// The synthetic engine on the CPU backend: keyframes and references
    /// travel with the request and come back as finished jobs.
    #[test]
    fn synthetic_engine_takes_keyframes_and_references() {
        let port = {
            let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
            l.local_addr().unwrap().port()
        };
        let addr = format!("127.0.0.1:{port}");
        let mut s = settings(false);
        s.addr = addr.clone();
        s.max_frames = 124;
        std::thread::spawn(move || {
            serve(s, |s, shared, rx| engine(h3_cpu::Cpu, s, shared, rx)).unwrap();
        });
        let t = Instant::now();
        let info = loop {
            std::thread::sleep(std::time::Duration::from_millis(200));
            if let Ok(st) = std::net::TcpStream::connect(&addr) {
                drop(st.shutdown(std::net::Shutdown::Both));
                let (_, info) = http_json(&addr, "GET", "/api/info", None);
                assert!(info["error"].is_null(), "{info}");
                if info["ready"].as_bool() == Some(true) {
                    break info;
                }
            }
            assert!(t.elapsed().as_secs() < 300, "engine did not come up");
        };
        assert_eq!(info["tasks"], json!(["t2va", "fl2va", "ref2va"]));
        assert_eq!(info["inputs"]["max_body_bytes"], json!(MAX_BODY));
        assert_eq!(info["max_frames"], json!(124));

        // FL2VA: a first and a last frame.
        let body = json!({
            "prompt": "a fox", "width": 32, "height": 32, "frames": 124, "seed": 3,
            "first_image": image_json(&png(40, 24, 10), "image/png"),
            "last_image": image_json(&png(64, 32, 200), "image/png"),
        });
        let (code, v) = http_json(&addr, "POST", "/api/generate", Some(&body.to_string()));
        assert_eq!(code, 200, "{v}");
        assert_eq!(v["task"], json!("fl2va"));
        let st = wait_job(&addr, v["id"].as_u64().unwrap());
        assert_eq!(st["stage"], json!("done"), "{st}");
        assert_eq!((st["task"].as_str(), st["ref_count"].as_u64()), (Some("fl2va"), Some(0)));
        assert!(st["frames"].as_u64().unwrap() > 0);
        assert_eq!((st["width"].as_u64(), st["height"].as_u64()), (Some(32), Some(32)));

        // Ref2VA: an image, a video with its soundtrack, an audio clip and a
        // continuation.
        let body = json!({
            "prompt": "a fox", "width": 32, "height": 32, "frames": 124, "seed": 3,
            "task": "ref2va",
            "refs": [
                {"kind": "image", "b64": b64(&png(48, 24, 2)), "mime": "image/png"},
                {"kind": "video", "b64": b64(&y4m(22, 32, 16)), "mime": "video/x-yuv4mpeg",
                 "audio_b64": b64(&wav(0.3))},
                {"kind": "audio", "b64": b64(&wav(0.2)), "mime": "audio/wav"},
            ],
            "continue": {"b64": b64(&y4m(5, 32, 32)), "audio_b64": b64(&wav(0.2))},
        });
        let (code, v) = http_json(&addr, "POST", "/api/generate", Some(&body.to_string()));
        assert_eq!(code, 200, "{v}");
        let st = wait_job(&addr, v["id"].as_u64().unwrap());
        assert_eq!(st["stage"], json!("done"), "{st}");
        assert_eq!((st["task"].as_str(), st["ref_count"].as_u64()), (Some("ref2va"), Some(3)));
        assert!(st["frames"].as_u64().unwrap() > 0);
        let f = |k: &str| st[k].as_f64().unwrap_or(0.0);
        eprintln!(
            "ref2va job: encode {:.1}s prepare {:.1}s steps {} decode {:.1}s total {:.1}s",
            f("encode_s"),
            f("prepare_s"),
            st["steps"],
            f("decode_s"),
            f("total_s")
        );

        // Refusals.
        let bad = json!({"prompt": "a fox", "width": 32, "height": 32, "frames": 124, "task": "t2va",
                         "first_image": image_json(&png(8, 8, 1), "image/png")});
        let (code, v) = http_json(&addr, "POST", "/api/generate", Some(&bad.to_string()));
        assert_eq!(code, 400);
        assert!(v["error"].as_str().unwrap().contains("takes no images"));
        let bad = json!({"prompt": "a fox", "width": 32, "height": 32, "frames": 124,
                         "first_image": image_json(&png(8, 8, 1), "image/bmp")});
        let (code, _) = http_json(&addr, "POST", "/api/generate", Some(&bad.to_string()));
        assert_eq!(code, 400);
        // Over 64 KiB without media is 413; the same size with media is taken.
        let pad = "x".repeat(SMALL_BODY);
        let big = json!({"prompt": "a fox", "width": 32, "height": 32, "frames": 124, "note": pad});
        let (code, _) = http_json(&addr, "POST", "/api/generate", Some(&big.to_string()));
        assert_eq!(code, 413);
        let big = json!({"prompt": "a fox", "width": 32, "height": 32, "frames": 124, "note": pad,
                         "first_image": image_json(&png(8, 8, 1), "image/png")});
        let (code, v) = http_json(&addr, "POST", "/api/generate", Some(&big.to_string()));
        assert_eq!(code, 200, "{v}");
        assert_eq!(v["task"], json!("fl2va"));
        let (_, st) = http_json(&addr, "GET", &format!("/api/jobs/{}", v["id"]), None);
        assert!(matches!(st["stage"].as_str(), Some("queued" | "encoding" | "preparing" | "denoising" | "decoding" | "done")), "{st}");
    }
}
