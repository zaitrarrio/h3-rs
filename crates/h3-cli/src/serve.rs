//! `h3 serve`: a local engine plus the prompt-to-clip demo page.
//!
//! One engine thread owns the backend, the resident DiT, the VAEs and the
//! conditioner (or the gallery); it takes jobs one at a time from a queue.
//! The HTTP thread only reads the shared job table.
//!
//! ```text
//! GET  /                         the demo page (web/demo.html)
//! GET  /api/info                 backend, model, limits, ready / error
//! POST /api/generate             {prompt, width, height, frames, seed?} → {id}
//! GET  /api/jobs/<id>            status, per-stage timings, frames decoded so far
//! GET  /api/jobs/<id>/frames?from=N
//!                                binary: u32 LE width, height, first, count; then RGB8
//! GET  /api/jobs/<id>/audio      WAV (16-bit stereo, 32 kHz) once the clip is done
//! ```
//!
//! It binds 127.0.0.1 by default and sends no CORS headers, so other web
//! pages cannot queue jobs on your GPU. `--allow-origin <url>` opens the API
//! to exactly one origin and to nothing else; `/api/generate` also rejects
//! non-JSON bodies and foreign `Origin`s (a cross-site form POST skips the
//! preflight).

use crate::{AttnArgs, Resolved, media, open_conditioner, open_transformer, sampler_plan};
use anyhow::{Context, Result, anyhow};
use h3_core::{
    Backend,
    config::{AudioVaeConfig, H3Config, TaeH3Config, VideoVaeConfig},
    packing::{self, Geometry},
    rng::Rng,
    schedule::Schedule,
    variant::Variant,
    weights::{SafetensorsDir, Synthetic},
};
use h3_model::{
    audio_vae::AudioVae,
    dit::{AdalnTables, H3Dit, LoadOptions},
    pipeline::{Event, Pipeline, Prompt, Request, SamplerPlan},
    taeh3::TaeH3,
    vae::{Tiling, VideoVae},
};
use serde_json::{Value, json};
use std::path::PathBuf;
use std::sync::{Arc, Mutex, mpsc};
use std::time::Instant;

const PAGE: &str = include_str!("../../../web/demo.html");
const KEEP_JOBS: usize = 8;
const MAX_QUEUE: usize = 3;

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
}

#[derive(Clone, Copy, PartialEq)]
enum Stage {
    Queued,
    Encoding,
    Preparing,
    Denoising,
    Decoding,
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
            Stage::Done => "done",
            Stage::Failed => "failed",
        }
    }
}

struct Job {
    id: u64,
    prompt: String,
    geometry: Geometry,
    seed: u64,
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
}

impl Job {
    fn status(&self) -> Value {
        json!({
            "id": self.id, "prompt": self.prompt, "stage": self.stage.name(), "seed": self.seed,
            "width": self.geometry.width, "height": self.geometry.height, "total_frames": self.geometry.frames,
            "tokens": self.tokens, "encode_s": self.encode_s, "prepare_s": self.prepare_s,
            "steps": self.steps, "total_steps": self.total_steps, "decode_s": self.decode_s,
            "elapsed_s": self.started.map(|t| t.elapsed().as_secs_f64()).unwrap_or(0.0),
            "total_s": self.total_s, "frames": self.frames, "audio": self.audio.is_some(), "error": self.error,
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
        self.jobs.lock().unwrap().iter_mut().find(|j| j.id == id).map(f)
    }
}

enum Text<B: Backend> {
    Synthetic { dim: usize, only: Option<crate::gallery::Gallery> },
    Qwen(Box<h3_text::Conditioner<B>>),
    Gallery(crate::gallery::Gallery, usize),
}

impl<B: Backend> Text<B> {
    fn encode(&self, be: &B, prompt: &str) -> Result<(Prompt, usize)> {
        match self {
            Text::Synthetic { dim, only } => {
                if let Some(g) = only {
                    if g.find(prompt).is_none() {
                        return Err(anyhow!("This engine has no text encoder: pick one of its {} gallery prompts.", g.entries.len()));
                    }
                }
                let seed = prompt.bytes().fold(0xcbf2_9ce4_8422_2325u64, |h, b| (h ^ b as u64).wrapping_mul(0x100_0000_01b3));
                let n = prompt.split_whitespace().count().max(1);
                Ok((Prompt::text_only(Rng::new(seed).normal(n * dim), *dim), n))
            }
            Text::Qwen(c) => {
                let p = c.encode_text(be, prompt)?;
                let n = p.rows(c.model.cfg.hidden);
                Ok((p, n))
            }
            Text::Gallery(g, dim) => {
                let e = g.find(prompt).ok_or_else(|| anyhow!("This engine has no text encoder: pick one of its {} gallery prompts.", g.entries.len()))?;
                Ok((g.prompt(e, *dim)?, e.tokens))
            }
        }
    }
}

/// Loads everything, then runs jobs forever. Runs on the engine thread.
pub fn engine<B: Backend>(be: B, s: &Settings, shared: Arc<Shared>, rx: mpsc::Receiver<u64>) -> Result<()> {
    let t0 = Instant::now();
    let stage = |what: &str| {
        eprintln!("engine: {what} ({:.1}s)", t0.elapsed().as_secs_f64());
        let mut info = shared.info.lock().unwrap();
        info["loading"] = json!(what);
        info["loading_s"] = json!(t0.elapsed().as_secs_f64());
    };
    stage(&format!("starting on {}", be.name()));
    let gallery_list = |g: &crate::gallery::Gallery| json!(g.entries.iter().map(|e| &e.prompt).collect::<Vec<_>>());
    let (pipe, text, model, variant, src, adapters, tables): (Pipeline<B>, Text<B>, String, Variant, Box<dyn h3_core::weights::WeightSource>, &[h3_core::lora::Adapter], Option<AdalnTables>) = if s.synthetic {
        let cfg = H3Config::tiny_vsa();
        let src = Synthetic { seed: 1 };
        let only = match &s.gallery {
            Some(dir) => {
                let g = crate::gallery::Gallery::open(dir)?;
                shared.info.lock().unwrap()["gallery"] = gallery_list(&g);
                Some(g)
            }
            None => None,
        };
        let text = Text::Synthetic { dim: cfg.text_dim, only };
        let mut variant = Variant::by_name("fasth3-v2")?;
        variant.config = cfg.clone();
        let pipe = Pipeline {
            dit: H3Dit::load(&be, cfg, &src, &LoadOptions { quant: s.attn.weights, stream_blocks: false }, &[])?,
            vae: Some(VideoVae::load(&be, VideoVaeConfig::tiny(), &src, false, true)?),
            audio_vae: Some(AudioVae::load(&be, AudioVaeConfig::tiny(), &src, false, true)?),
            tae: Some(TaeH3::load(&be, TaeH3Config::tiny(), &src)?),
        };
        (pipe, text, "synthetic tiny model (random weights: output is noise)".to_string(), variant, Box::new(src), &[][..], None)
    } else {
        let r = s.model.as_ref().expect("a resolved model when not synthetic");
        let cfg = r.variant.config.clone();
        let model_dir = r.model_dir.as_ref().ok_or_else(|| anyhow!("--model-dir is needed"))?;
        stage("checking the checkpoint");
        let (src, _) = open_transformer(r)?;
        let text = match &s.gallery {
            Some(dir) => {
                stage("reading the prompt gallery");
                let g = crate::gallery::Gallery::open(dir)?;
                shared.info.lock().unwrap()["gallery"] = gallery_list(&g);
                Text::Gallery(g, cfg.text_dim)
            }
            None => {
                stage("opening the conditioner");
                Text::Qwen(Box::new(open_conditioner(&be, model_dir, false, false)?))
            }
        };
        let tables = match &r.tables {
            Some(p) => Some(AdalnTables::load(p, &cfg)?),
            None => None,
        };
        stage(&format!("loading the transformer onto the device ({} blocks, {})", cfg.layers, s.attn.weights));
        let dit = H3Dit::load(&be, cfg.clone(), src.as_ref(), &LoadOptions { quant: s.attn.weights, stream_blocks: s.attn.stream_blocks }, &r.adapters)?;
        stage("loading the decoders");
        let vae_dir = model_dir.join("vae");
        let vae = VideoVae::load_with(&be, VideoVaeConfig::h3(), &SafetensorsDir::open(&vae_dir)?, false, !s.preview, s.attn.decoder_weights)?;
        let audio_vae = AudioVae::load(&be, AudioVaeConfig::h3(), &SafetensorsDir::open(&model_dir.join("audio_vae"))?, false, true)?;
        let tae = match (&r.tae, s.preview) {
            (Some(p), true) => Some(TaeH3::load(&be, TaeH3Config::taeh3(), &SafetensorsDir::open(p)?)?),
            (None, true) => return Err(anyhow!("--preview needs --tae")),
            _ => None,
        };
        let pipe = Pipeline { dit, vae: Some(vae), audio_vae: Some(audio_vae), tae };
        let model = format!(
            "{} · {} forwards · {}{}{}",
            r.variant.summary.split(':').next().unwrap_or(&r.variant.name),
            r.variant.forwards(),
            if s.preview { "TAEH3 preview" } else { "VAE decoder" },
            if matches!(text, Text::Gallery(..)) { " · prompt gallery" } else { "" },
            if s.attn.weights != h3_core::quant::WeightQuant::Native { format!(" · {}", s.attn.weights) } else { String::new() }
        );
        (pipe, text, model, r.variant.clone(), src, &r.adapters[..], tables)
    };
    stage("waiting for the device");
    be.sync();
    {
        let mut info = shared.info.lock().unwrap();
        info["ready"] = json!(true);
        info["model"] = json!(model);
        info["backend"] = json!(be.name());
        info["load_s"] = json!(t0.elapsed().as_secs_f64());
        info["steps"] = json!(variant.forwards());
    }
    eprintln!("engine ready on {} in {:.1}s", be.name(), t0.elapsed().as_secs_f64());
    let sampler = match sampler_plan(&variant) {
        Ok(p) => p,
        Err(e) => return Err(anyhow!("{e}")),
    };
    let _ = Schedule::uniform;

    for id in rx {
        let Some((prompt, geometry, seed)) = shared.with(id, |j| {
            j.started = Some(Instant::now());
            j.stage = Stage::Encoding;
            (j.prompt.clone(), j.geometry, j.seed)
        }) else {
            continue;
        };
        let t = Instant::now();
        let run = || -> Result<()> {
            let (p, tokens) = text.encode(&be, &prompt)?;
            shared.with(id, |j| {
                j.encode_s = t.elapsed().as_secs_f64();
                j.tokens = tokens;
                j.stage = Stage::Preparing;
            });
            let req = Request {
                prompt: p,
                geometry,
                seed,
                keyframes: vec![],
                references: vec![],
                continuation: None,
                sampler: match &sampler {
                    SamplerPlan::Schedules { video, audio } => SamplerPlan::Schedules { video: video.clone(), audio: audio.clone() },
                    SamplerPlan::Pdd(p) => SamplerPlan::Pdd(p.clone()),
                },
                attention: s.attn.attention(&variant),
                dense_first_steps: s.attn.dense_first_steps,
                ffn_chunk: s.attn.ffn_chunk.unwrap_or(usize::MAX),
                tiling: s.tiling,
                preview: s.synthetic || s.preview,
            };
            let generation = pipe.generate(&be, src.as_ref(), adapters, tables.as_ref(), &req, &mut |ev| match ev {
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
                Event::Frames { count, width, height, rgb, .. } => {
                    shared.with(id, |j| {
                        j.width = width;
                        j.height = height;
                        j.rgb.extend_from_slice(rgb);
                        j.frames += count;
                        j.decode_s = j.started.unwrap().elapsed().as_secs_f64() - j.encode_s - j.prepare_s - j.steps.iter().sum::<f64>();
                    });
                }
                Event::Audio { sample_rate, stereo } => {
                    let mut buf = Vec::new();
                    let tmp = std::env::temp_dir().join(format!("h3-serve-{}-{id}.wav", std::process::id()));
                    if media::write_wav(&tmp, stereo, sample_rate as u32).is_ok() {
                        buf = std::fs::read(&tmp).unwrap_or_default();
                        let _ = std::fs::remove_file(&tmp);
                    }
                    shared.with(id, |j| j.audio = Some(buf));
                }
            })?;
            let _ = generation;
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

fn hdr(k: &str, v: &str) -> tiny_http::Header {
    tiny_http::Header::from_bytes(k, v).expect("valid header")
}

fn send<R: std::io::Read>(req: tiny_http::Request, resp: tiny_http::Response<R>, cors: &[tiny_http::Header]) {
    let mut resp = resp;
    for h in cors {
        resp = resp.with_header(h.clone());
    }
    let _ = req.respond(resp);
}

fn respond_json(req: tiny_http::Request, code: u16, v: &Value, cors: &[tiny_http::Header]) {
    let r = tiny_http::Response::from_string(v.to_string()).with_status_code(code).with_header(hdr("Content-Type", "application/json"));
    send(req, r, cors);
}

/// CORS headers when the request's `Origin` is the one allowed origin.
fn cors_for(req: &tiny_http::Request, allow: Option<&str>) -> Vec<tiny_http::Header> {
    let origin = req.headers().iter().find(|h| h.field.equiv("Origin")).map(|h| h.value.as_str().to_string());
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

fn parse_request(body: &str, s: &Settings, pika: bool) -> Result<(String, Geometry, u64), String> {
    let v: Value = serde_json::from_str(body).map_err(|e| format!("body is not JSON: {e}"))?;
    let prompt = v["prompt"].as_str().unwrap_or("").trim().to_string();
    if prompt.is_empty() {
        return Err("Write a prompt first.".into());
    }
    if prompt.chars().count() > 2000 {
        return Err("Prompts are limited to 2000 characters.".into());
    }
    let num = |k: &str, d: u64| v[k].as_u64().unwrap_or(d) as usize;
    let frames = if pika { packing::pika::resolve_frames(num("frames", 124)) } else { packing::resolve_num_frames(num("frames", 124)) }.map_err(|e| e.to_string())?;
    let g = Geometry { width: num("width", 1344), height: num("height", 768), frames };
    g.validate(16).map_err(|e| e.to_string())?;
    if g.width * g.height > s.max_pixels {
        return Err(format!("{}×{} is above this engine's limit of {} pixels per frame.", g.width, g.height, s.max_pixels));
    }
    if g.frames > s.max_frames {
        return Err(format!("{} frames is above this engine's limit of {}.", g.frames, s.max_frames));
    }
    let seed = v["seed"].as_u64().unwrap_or_else(|| {
        std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64 % 1_000_000_000).unwrap_or(7)
    });
    Ok((prompt, g, seed))
}

/// Serve until killed. `start_engine` builds the backend on the engine thread.
pub fn serve(s: Settings, start_engine: impl FnOnce(&Settings, Arc<Shared>, mpsc::Receiver<u64>) -> Result<()> + Send + 'static) -> Result<()> {
    let pika = s.model.as_ref().is_some_and(|m| m.variant.pika_buckets);
    let shared = Arc::new(Shared {
        jobs: Mutex::new(Vec::new()),
        info: Mutex::new(json!({
            "ready": false, "error": null, "backend": null, "model": null,
            "synthetic": s.synthetic, "gallery": null,
            "max_pixels": s.max_pixels, "max_frames": s.max_frames, "fps": packing::FPS, "sample_rate": 32000,
            "variant": s.model.as_ref().map(|m| m.variant.name.clone()), "pika": pika,
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
                let run = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| start_engine(&s, shared.clone(), rx)));
                let msg = match run {
                    Ok(Ok(())) => return,
                    Ok(Err(e)) => format!("{e:#}"),
                    Err(p) => format!(
                        "engine crashed: {}",
                        p.downcast_ref::<String>().map(String::as_str).or_else(|| p.downcast_ref::<&str>().copied()).unwrap_or("panic")
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
                send(req, tiny_http::Response::from_string(PAGE).with_header(h), &cors);
            }
            (tiny_http::Method::Get, "/api/info") => {
                let mut info = shared.info.lock().unwrap().clone();
                let jobs = shared.jobs.lock().unwrap();
                info["queued"] = json!(jobs.iter().filter(|j| j.stage == Stage::Queued).count());
                drop(jobs);
                respond_json(req, 200, &info, &cors);
            }
            (tiny_http::Method::Post, "/api/generate") => {
                let header = |k: &str| req.headers().iter().find(|h| h.field.as_str().as_str().eq_ignore_ascii_case(k)).map(|h| h.value.as_str().to_string());
                let json_body = header("Content-Type").is_some_and(|c| c.starts_with("application/json"));
                let origin_ok = match header("Origin") {
                    None => true,
                    Some(o) => !cors.is_empty() || header("Host").is_some_and(|h| o == format!("http://{h}") || o == format!("https://{h}")),
                };
                if !json_body || !origin_ok {
                    respond_json(req, 403, &json!({"error": "Requests must be JSON and come from this engine's page or its --allow-origin site."}), &cors);
                    continue;
                }
                let mut body = String::new();
                if req.as_reader().take(64 * 1024).read_to_string(&mut body).is_err() {
                    respond_json(req, 400, &json!({"error": "unreadable body"}), &cors);
                    continue;
                }
                let info = shared.info.lock().unwrap().clone();
                if !info["ready"].as_bool().unwrap_or(false) {
                    let msg = info["error"].as_str().map(|e| format!("The engine failed to start: {e}")).unwrap_or_else(|| "The engine is still loading the model. Try again in a moment.".into());
                    respond_json(req, 503, &json!({"error": msg}), &cors);
                    continue;
                }
                let parsed = parse_request(&body, &s, pika).and_then(|r| {
                    let info = shared.info.lock().unwrap();
                    match info["gallery"].as_array() {
                        Some(g) if !g.iter().any(|p| p.as_str().is_some_and(|p| p.split_whitespace().eq(r.0.split_whitespace()))) => {
                            Err(format!("This engine has no text encoder: pick one of its {} gallery prompts.", g.len()))
                        }
                        _ => Ok(r),
                    }
                });
                match parsed {
                    Err(e) => respond_json(req, 400, &json!({"error": e}), &cors),
                    Ok((prompt, geometry, seed)) => {
                        let mut jobs = shared.jobs.lock().unwrap();
                        if jobs.iter().filter(|j| matches!(j.stage, Stage::Queued)).count() >= MAX_QUEUE {
                            drop(jobs);
                            respond_json(req, 429, &json!({"error": "Three clips are already waiting. Try again when one finishes."}), &cors);
                            continue;
                        }
                        let id = {
                            let mut n = shared.next.lock().unwrap();
                            *n += 1;
                            *n - 1
                        };
                        let total_steps = info["steps"].as_u64().unwrap_or(8) as usize;
                        jobs.push(Job {
                            id,
                            prompt,
                            geometry,
                            seed,
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
                        });
                        while jobs.len() > KEEP_JOBS {
                            match jobs.iter().position(|j| matches!(j.stage, Stage::Done | Stage::Failed)) {
                                Some(i) => {
                                    jobs.remove(i);
                                }
                                None => break,
                            }
                        }
                        drop(jobs);
                        let _ = tx.send(id);
                        respond_json(req, 200, &json!({"id": id, "seed": seed}), &cors);
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
                        let from = query.split('&').find_map(|kv| kv.strip_prefix("from=")).and_then(|v| v.parse::<usize>().ok()).unwrap_or(0);
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
                            Some(b) => send(req, tiny_http::Response::from_data(b).with_header(hdr("Content-Type", "application/octet-stream")), &cors),
                            None => respond_json(req, 404, &json!({"error": "no such job"}), &cors),
                        }
                    }
                    "audio" => match shared.with(id, |j| j.audio.clone()) {
                        Some(Some(b)) => send(req, tiny_http::Response::from_data(b).with_header(hdr("Content-Type", "audio/wav")), &cors),
                        Some(None) => respond_json(req, 404, &json!({"error": "no audio yet"}), &cors),
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
