//! `h3`: plan, check, bench, encode and generate MiniMax H3 and its FastH3,
//! Pika and Regenerate-2K variants on CPU, WebGPU or CUDA.

use anyhow::{Context, Result, anyhow, bail};
use clap::{Parser, Subcommand, ValueEnum};
use h3_core::{
    Backend,
    config::{AudioVaeConfig, Qwen3VlConfig, TaeH3Config, UpscalerConfig, VideoVaeConfig},
    lora::Adapter,
    packing::{self, Anchor, Geometry},
    plan::{Device, plan},
    quant::WeightQuant,
    schedule::PddPlan,
    variant::{Sampler, Task, Variant},
    weights::{SafetensorsDir, Synthetic, WeightSource, h3_manifest, missing},
};
use h3_model::{
    Attention,
    audio_vae::AudioVae,
    dit::{AdalnTables, H3Dit, LoadOptions},
    pipeline::{Event, Pipeline, Prompt, Request, SamplerPlan},
    regen::{Regen, RegenRequest, SharpCurve},
    taeh3::TaeH3,
    upscaler::Upscaler,
    vae::{Tiling, VideoVae},
};
use h3_text::image::Image;
use std::path::{Path, PathBuf};

mod encoder;
mod finish;
mod gallery;
mod inputs;
mod media;
mod parity;
mod roundtrip;
mod serve;
mod worker;

#[derive(Parser)]
#[command(
    name = "h3",
    about = "MiniMax H3 video + audio generation on CPU, WebGPU or CUDA"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Debug)]
enum BackendKind {
    /// CUDA when built in and a device is present, else WebGPU.
    Auto,
    Cpu,
    Wgpu,
    Cuda,
}

#[derive(Clone, Copy, ValueEnum, PartialEq)]
enum AttnKind {
    /// The variant's own choice: VSA on FastH3 checkpoints, dense on the base.
    Auto,
    Dense,
    Vsa,
}

#[derive(clap::Args, Clone)]
struct ModelArgs {
    /// Which release: h3 | h3-plug4 | fasth3-v1 | fasth3-v2 | fasth3-trim | fasth3-omni | pika-flex | regen-2k.
    #[arg(long, default_value = "fasth3-v2")]
    variant: String,
    /// t2va | fl2va | ref2va (default: the variant's first task).
    #[arg(long)]
    task: Option<String>,
    /// The MiniMaxAI/MiniMax-H3 download (transformer/, transformer_ref/, vae/, audio_vae/, text_encoder/, tokenizer/).
    #[arg(long)]
    model_dir: Option<PathBuf>,
    /// The variant's own download (its transformer/, adapters, fastvideo_inference.json); defaults to --model-dir.
    #[arg(long)]
    variant_dir: Option<PathBuf>,
    /// Override the DiT weights directory or file (e.g. an `h3 quantize` output).
    #[arg(long)]
    transformer: Option<PathBuf>,
    /// Extra adapters merged at load (repeatable); the variant's own are added automatically.
    #[arg(long = "adapter")]
    adapters: Vec<PathBuf>,
    /// Strength of the variant's own adapters (default: the preset's, 1). The
    /// LongLive-Plug CFG adapter was swept at 0.5–5 by its authors; other
    /// values than 1 are exploratory.
    #[arg(long)]
    adapter_strength: Option<f32>,
    /// `taeh3.safetensors` for `--preview` decoding.
    #[arg(long)]
    tae: Option<PathBuf>,
    /// Precomputed AdaLN tables (`h3 adaln-tables`), so the 13 B modulation weights are never read.
    #[arg(long)]
    tables: Option<PathBuf>,
    /// Euler step count for the base model (its release runs 49; ComfyUI's template 25). Distilled
    /// variants keep their own ladders.
    #[arg(long)]
    steps: Option<usize>,
}

#[derive(clap::Args, Clone)]
struct AttnArgs {
    #[arg(long, value_enum, default_value_t = AttnKind::Auto)]
    attention: AttnKind,
    /// Override the VSA sparsity (the variant's contract sets it otherwise).
    #[arg(long)]
    vsa_sparsity: Option<f64>,
    /// Steps run dense before VSA starts.
    #[arg(long, default_value_t = 0)]
    dense_first_steps: usize,
    /// INT8 Q·Kᵀ (SageAttention's recipe; int8 tensor cores on CUDA) in the VSA fine stage and in dense attention.
    #[arg(long, visible_alias = "qk8", default_value_t = false)]
    vsa_qk8: bool,
    /// FP8 P·V as well (SageAttention2's recipe), in dense attention and in the VSA fine stage (implies --qk8; fp8 tensor cores on sm_89+).
    #[arg(long, default_value_t = false)]
    pv8: bool,
    /// Block linears on the device: native (bf16 / f16) | int8 | int4[-hqq][-gN][-ffn] | nvfp4.
    #[arg(long, default_value = "native")]
    weights: WeightQuant,
    /// Keep the block weights in host memory and upload them per step (small GPUs).
    #[arg(long, default_value_t = false)]
    stream_blocks: bool,
    /// The video decoder's block linears: native | mxfp8 (FP8 tensor cores on sm_100+) | int8.
    #[arg(long, default_value = "native")]
    decoder_weights: WeightQuant,
    /// Decode with the fast (Sol-H3) VAE decoder instead of the released one: its `.safetensors` file or the Open-H3-Regenerate-2K tree (`vae_fast/`). Its block linears follow --decoder-weights.
    #[arg(long)]
    fast_vae: Option<PathBuf>,
    /// Tokens per FFN chunk (default: all).
    #[arg(long)]
    ffn_chunk: Option<usize>,
}

impl AttnArgs {
    /// `--fast-vae`: load the fast decoder into `vae` (not for a TAE preview).
    fn attach_fast_vae<B: Backend>(
        &self,
        be: &B,
        vae: &mut VideoVae<B>,
        preview: bool,
    ) -> Result<()> {
        if let (Some(p), false) = (&self.fast_vae, preview) {
            let t0 = std::time::Instant::now();
            vae.fast = Some(h3_model::fast_vae::FastDecoder::open(
                be,
                p,
                self.decoder_weights,
            )?);
            eprintln!(
                "fast VAE decoder: {} in {:.1}s",
                p.display(),
                t0.elapsed().as_secs_f64()
            );
        }
        Ok(())
    }

    fn attention(&self, v: &Variant) -> Attention {
        match (self.attention, v.vsa) {
            (AttnKind::Dense, _) | (AttnKind::Auto, None) => Attention::Dense {
                qk8: self.vsa_qk8 || self.pv8,
                pv8: self.pv8,
            },
            (AttnKind::Vsa, None) => Attention::Vsa {
                sparsity: self.vsa_sparsity.unwrap_or(0.8),
                tile: 64,
                ref_sparsity: 0.9,
                qk8: self.vsa_qk8 || self.pv8,
                pv8: self.pv8,
            },
            (_, Some(p)) => Attention::Vsa {
                sparsity: self.vsa_sparsity.unwrap_or(p.sparsity),
                tile: p.tile,
                ref_sparsity: p
                    .ref_keep_rate
                    .map_or(self.vsa_sparsity.unwrap_or(p.sparsity), |k| 1.0 - k),
                qk8: self.vsa_qk8 || self.pv8,
                pv8: self.pv8,
            },
        }
    }
}

#[derive(clap::Args, Clone)]
struct Shape {
    #[arg(long)]
    width: Option<usize>,
    #[arg(long)]
    height: Option<usize>,
    /// `W:H`, used when no width/height is given (the canvas follows the variant's rules).
    #[arg(long, default_value = "16:9")]
    aspect: String,
    /// Resolution tier when no width/height is given: 768p (default; the
    /// model's 1344×768 at 16:9), 480p (832×480, the draft tier's fast
    /// setting) or 288p (480×288, H3's smallest canvas on its 32-pixel
    /// grid; below its trained sizes, it can ghost). Pika H3 Flex samples
    /// its own 480p / 768p buckets.
    #[arg(long)]
    tier: Option<String>,
    /// Frames at 24 fps, aligned up to 17n + 5 (124 = 5.17 s).
    #[arg(long, default_value_t = 124)]
    frames: usize,
}

impl Shape {
    fn geometry(&self, v: &Variant) -> Result<Geometry> {
        Ok(self.resolve(v)?.0)
    }

    /// The render canvas (a multiple of 32 on both edges) and the output
    /// size: equal, unless `--width` / `--height` are off the 32-pixel grid
    /// (e.g. 480×272), in which case the
    /// canvas rounds each edge up and the decoded frames are centre-cropped.
    fn resolve(&self, v: &Variant) -> Result<(Geometry, (usize, usize))> {
        let (aw, ah) = packing::pika::parse_aspect(&self.aspect)?;
        let (height, width, frames) = if v.pika_buckets {
            let tier = packing::pika::tier(self.tier.as_deref().unwrap_or("768p"))?;
            let (h, w, _) = packing::pika::resolve_canvas(self.height, self.width, (aw, ah), tier)?;
            (h, w, packing::pika::resolve_frames(self.frames)?)
        } else {
            let (h, w) = match (self.height, self.width) {
                (Some(_), _) | (_, Some(_)) if self.tier.is_some() => {
                    bail!("--tier sets the canvas: give it or --width / --height, not both")
                }
                (Some(h), Some(w)) => (h, w),
                (None, None) => match self.tier.as_deref().unwrap_or("768p") {
                    "768p" | "p768" | "768" => packing::canvas_size(
                        aw as f64,
                        ah as f64,
                        packing::SHORT_EDGE,
                        packing::MAX_PIXELS,
                    )?,
                    "480p" | "p480" | "480" => {
                        packing::canvas_size(aw as f64, ah as f64, 480, 832 * 480)?
                    }
                    "288p" | "p288" | "288" => {
                        packing::canvas_size(aw as f64, ah as f64, 288, 480 * 288)?
                    }
                    t => bail!("unknown tier `{t}` (288p | 480p | 768p)"),
                },
                _ => bail!("give --width and --height together, or neither"),
            };
            (h, w, packing::resolve_num_frames(self.frames)?)
        };
        let grid = finish::canvas_edge;
        let g = Geometry {
            width: grid(width),
            height: grid(height),
            frames,
        };
        g.validate(16)?;
        Ok((g, (width, height)))
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Memory plan for a device, before allocating anything.
    Plan {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        shape: Shape,
        #[command(flatten)]
        attn: AttnArgs,
        #[arg(long, default_value_t = 24.0)]
        vram_gib: f64,
        /// Largest buffer binding (WebGPU adapters cap this), MiB.
        #[arg(long)]
        max_binding_mib: Option<u64>,
        #[arg(long, value_enum, default_value_t = BackendKind::Cuda)]
        backend: BackendKind,
        /// Prompt tokens to assume.
        #[arg(long, default_value_t = 256)]
        text_tokens: usize,
    },
    /// Every op and the tiny models against the CPU reference.
    Check {
        #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
        backend: BackendKind,
        /// Only the op checks.
        #[arg(long, default_value_t = false)]
        ops_only: bool,
        /// Also write the results as JSON (backend, tolerance, every case).
        #[arg(long)]
        report: Option<PathBuf>,
    },
    /// Time denoising steps with synthetic weights at a real geometry.
    Bench {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        shape: Shape,
        #[command(flatten)]
        attn: AttnArgs,
        #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
        backend: BackendKind,
        /// Transformer blocks to build (50 for the real model; fewer for a quick look).
        #[arg(long)]
        layers: Option<usize>,
        /// Forwards to time.
        #[arg(long, default_value_t = 2)]
        steps: usize,
        #[arg(long, default_value_t = 256)]
        text_tokens: usize,
    },
    /// The DiT against a PyTorch reference dump (`scripts/reference/full_parity.py`): real widths and weights, its first N blocks.
    ParityDit {
        #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
        backend: BackendKind,
        /// The checkpoint's `transformer/` directory (with --tiny: the fixture file).
        #[arg(long)]
        transformer: PathBuf,
        /// The dump (`input.*`, `expected.*`).
        #[arg(long)]
        fixture: PathBuf,
        /// The tiny config (checks the script and this command end to end without real weights).
        #[arg(long, default_value_t = false)]
        tiny: bool,
        /// Relative tolerance (default 2e-3 on the CPU, 3e-2 elsewhere).
        #[arg(long)]
        tolerance: Option<f32>,
    },
    /// Encode a video with the H3 VAE and decode it with the released and the fast (Sol-H3) decoders: time and PSNR.
    VaeRoundtrip {
        #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
        backend: BackendKind,
        /// A tree with `vae/` (h3-base).
        #[arg(long)]
        model_dir: PathBuf,
        /// The video (Y4M, PNG frames, …), sides multiples of 16.
        #[arg(long)]
        input: PathBuf,
        /// The fast decoder (file or Open-H3-Regenerate-2K tree).
        #[arg(long)]
        fast_vae: Option<PathBuf>,
        /// Use the first N frames.
        #[arg(long)]
        frames: Option<usize>,
        /// Spatial tile of the released VAE in pixels (0: none; `generate`'s default is 256).
        #[arg(long, default_value_t = 256)]
        tile: usize,
        #[arg(long, default_value = "native")]
        decoder_weights: WeightQuant,
        /// Only the fast decoder.
        #[arg(long, default_value_t = false)]
        skip_base: bool,
        /// Decode each this many times (the first includes one-time setup).
        #[arg(long, default_value_t = 1)]
        repeat: usize,
        /// Write base.y4m / fast.y4m here.
        #[arg(long)]
        out_dir: Option<PathBuf>,
    },
    /// Encode text prompts with Qwen3-VL into a gallery (`serve --gallery`, `generate --embeds`).
    Encode {
        #[arg(long)]
        model_dir: PathBuf,
        #[arg(long = "prompt")]
        prompts: Vec<String>,
        /// One prompt per line (blank lines and `#` lines skipped).
        #[arg(long)]
        prompt_file: Option<PathBuf>,
        #[arg(long)]
        out_dir: PathBuf,
        #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
        backend: BackendKind,
        /// Same as `--encoder resident`.
        #[arg(long, default_value_t = false)]
        resident: bool,
        #[command(flatten)]
        enc: encoder::EncoderArgs,
    },
    /// Write the text encoder's language layers as int8 codes (`--encoder-weights
    /// int8` then reads them instead of quantizing 64 GB of bf16 at every load).
    QuantizeEncoder {
        #[arg(long)]
        model_dir: PathBuf,
        /// Default `<model-dir>/text_encoder_int8`.
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Precompute the AdaLN modulation tables for a variant's timestep ladder.
    AdalnTables {
        #[command(flatten)]
        model: ModelArgs,
        #[arg(long)]
        out: PathBuf,
        #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
        backend: BackendKind,
        /// Extra clean times to include (e.g. the Regen-2K sigmas).
        #[arg(long = "timestep")]
        timesteps: Vec<f32>,
    },
    /// Write a quantized (and adapter-merged) copy of the DiT.
    Quantize {
        #[command(flatten)]
        model: ModelArgs,
        /// int8 | int4[-hqq][-gN][-ffn]; `native` only converts / merges.
        #[arg(long, default_value = "int8")]
        weights: WeightQuant,
        #[arg(long)]
        out: PathBuf,
        /// Leave the 13 B AdaLN weights out (use with `h3 adaln-tables`).
        #[arg(long, default_value_t = false)]
        skip_adaln: bool,
    },
    /// Generate a clip (video + 32 kHz stereo audio).
    Generate {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        shape: Shape,
        #[command(flatten)]
        attn: AttnArgs,
        #[command(flatten)]
        enc: encoder::EncoderArgs,
        #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
        backend: BackendKind,
        #[arg(long, required_unless_present = "embeds")]
        prompt: Option<String>,
        /// A gallery file with precomputed `prompt_embeds` (text-only prompts).
        #[arg(long)]
        embeds: Option<PathBuf>,
        /// FL2VA first keyframe (png / jpg / ppm); stretched to the canvas (Pika: cover-cropped).
        #[arg(long)]
        first_image: Option<PathBuf>,
        /// FL2VA last keyframe; cover-cropped to the canvas.
        #[arg(long)]
        last_image: Option<PathBuf>,
        /// Ref2VA references in order: `image:FILE`, `video:FILE.y4m[,audio=FILE.wav]`, `audio:FILE.wav`.
        #[arg(long = "ref")]
        refs: Vec<String>,
        /// Ref2VA continuation clip: `FILE.y4m[,audio=FILE.wav]` (5 / 22 / 39 frames at the canvas).
        #[arg(long = "continue")]
        continuation: Option<String>,
        #[arg(long, default_value_t = 7)]
        seed: u64,
        /// Decode with TAEH3 (needs --tae) instead of the full VAE decoder.
        #[arg(long, default_value_t = false)]
        preview: bool,
        /// Upscale the finished frames with Real-ESRGAN `realesr-animevideov3`
        /// (its safetensors; fastwan-rs's scripts/upscale/convert_esrgan.py).
        #[arg(long)]
        upscale: Option<PathBuf>,
        /// Output scale of --upscale: 4 (the network's), or 3 / 2 (the ×4
        /// frames Lanczos-resampled). Default 4 up to 480p, 2 above.
        #[arg(long)]
        upscale_factor: Option<usize>,
        /// VAE spatial tile in pixels (0 = whole frame).
        #[arg(long, default_value_t = 256)]
        tile: usize,
        #[arg(long, default_value_t = 1)]
        runs: usize,
        #[arg(long, default_value = "out.y4m")]
        out: PathBuf,
    },
    /// Open-H3-Regenerate-2K: a 768p draft → 2560×1440 (silent output).
    Regen2k {
        /// The Efficient-Large-Model/Open-H3-Regenerate-2K download.
        #[arg(long)]
        regen_dir: PathBuf,
        /// The MiniMaxAI/MiniMax-H3 download (text encoder, VAE encoder, audio VAE).
        #[arg(long)]
        model_dir: PathBuf,
        /// The draft: a .y4m (any length ≥ 121 frames; 24 fps) or a directory of frames.
        #[arg(long)]
        draft: PathBuf,
        #[arg(long, required_unless_present = "embeds")]
        prompt: Option<String>,
        #[arg(long)]
        embeds: Option<PathBuf>,
        /// `adaln_t_table.bin` (1025×8 f32; not redistributed with the weights).
        #[arg(long)]
        adaln_table: PathBuf,
        #[command(flatten)]
        attn: AttnArgs,
        #[command(flatten)]
        enc: encoder::EncoderArgs,
        #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
        backend: BackendKind,
        #[arg(long, default_value_t = 3)]
        steps: usize,
        #[arg(long)]
        start_sigma: Option<f64>,
        #[arg(long, default_value_t = 1766580932)]
        seed: u64,
        #[arg(long, default_value_t = 256)]
        tile: usize,
        #[arg(long, default_value = "out-2k.y4m")]
        /// Decode with the released 36-block VAE instead of the tree's `vae_fast/` decoder (which the release, and this command, use by default).
        #[arg(long, default_value_t = false)]
        released_vae: bool,
        out: PathBuf,
    },
    /// RunPod serverless worker: take jobs (`{"args": [...], "uploads": {...}}`),
    /// run `h3 <args>` for each, PUT the named outputs, post the result.
    Worker {
        /// Run this one job (a JSON file: the job's input) without RunPod.
        #[arg(long)]
        test_input: Option<PathBuf>,
        /// Where each job gets its own working directory.
        #[arg(long, default_value = "/tmp/h3-jobs")]
        work_dir: PathBuf,
    },
    /// Run the local engine and the prompt-to-clip demo page.
    Serve {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        attn: AttnArgs,
        #[arg(long, default_value = "127.0.0.1:7860")]
        addr: String,
        #[arg(long, value_enum, default_value_t = BackendKind::Auto)]
        backend: BackendKind,
        /// Precomputed prompts (`h3 encode`), used before the text encoder. With no
        /// `--encoder` given, the engine runs without one: gallery prompts only.
        #[arg(long)]
        gallery: Option<PathBuf>,
        /// Tiny random model: test the engine and page without weights.
        #[arg(long, default_value_t = false)]
        synthetic: bool,
        #[arg(long, default_value_t = 1344 * 768)]
        max_pixels: usize,
        #[arg(long, default_value_t = 243)]
        max_frames: usize,
        /// Let one web origin call this engine from the browser.
        #[arg(long)]
        allow_origin: Option<String>,
        #[arg(long, default_value_t = 256)]
        tile: usize,
        #[arg(long, default_value_t = false)]
        preview: bool,
        #[command(flatten)]
        enc: encoder::EncoderArgs,
        /// Offer Real-ESRGAN upscaling of finished clips
        /// (`realesr-animevideov3.safetensors`; with `--synthetic`, any value
        /// loads a tiny random ×2 network).
        #[arg(long)]
        upscale: Option<PathBuf>,
    },
}

/// Run `f` with the selected backend.
macro_rules! with_backend {
    ($kind:expr, |$be:ident| $body:expr) => {{
        let kind = resolve($kind);
        match kind {
            BackendKind::Auto => unreachable!(),
            BackendKind::Cpu => {
                let $be = h3_cpu::Cpu;
                $body
            }
            BackendKind::Wgpu => {
                #[cfg(feature = "wgpu")]
                {
                    let $be = h3_wgpu::Wgpu::new().map_err(|e| anyhow!("{e}"))?;
                    if $be.is_software() && !allow_software() {
                        bail!(
                            "WebGPU found only a software rasterizer ({}): the GPU's Vulkan/DX12/Metal driver did not load. \
                             Fix the driver, or set H3_ALLOW_SOFTWARE=1 to run on the CPU rasterizer anyway",
                            $be.adapter_info().name
                        );
                    }
                    $body
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    bail!("built without WebGPU: cargo build --release --features wgpu")
                }
            }
            BackendKind::Cuda => {
                #[cfg(any(feature = "cuda", feature = "cuda-13"))]
                {
                    let $be = h3_cuda::Cuda::new(0).map_err(|e| anyhow!("{e}"))?;
                    $body
                }
                #[cfg(not(any(feature = "cuda", feature = "cuda-13")))]
                {
                    bail!("built without CUDA: cargo build --release --features cuda")
                }
            }
        }
    }};
}

#[cfg_attr(not(feature = "wgpu"), allow(dead_code))]
fn allow_software() -> bool {
    std::env::var("H3_ALLOW_SOFTWARE").is_ok_and(|v| v == "1")
        || ALLOW_SW.load(std::sync::atomic::Ordering::Relaxed)
}
static ALLOW_SW: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

fn resolve(kind: BackendKind) -> BackendKind {
    if kind != BackendKind::Auto {
        return kind;
    }
    #[cfg(any(feature = "cuda", feature = "cuda-13"))]
    if h3_cuda::Cuda::new(0).is_ok() {
        return BackendKind::Cuda;
    }
    #[cfg(feature = "wgpu")]
    {
        return BackendKind::Wgpu;
    }
    #[allow(unreachable_code)]
    BackendKind::Cpu
}

/// Everything the model arguments resolve to.
pub struct Resolved {
    pub variant: Variant,
    pub task: Task,
    pub model_dir: Option<PathBuf>,
    pub variant_dir: Option<PathBuf>,
    pub transformer: PathBuf,
    pub adapters: Vec<Adapter>,
    pub tae: Option<PathBuf>,
    pub tables: Option<PathBuf>,
}

impl ModelArgs {
    /// The named preset with `--steps` applied (the base model's Euler ladder only).
    fn preset(&self) -> Result<Variant> {
        let mut variant = Variant::by_name(&self.variant)?;
        if let Some(n) = self.steps {
            match variant.sampler {
                Sampler::Uniform(_) if n >= 1 => variant.sampler = Sampler::Uniform(n + 1),
                Sampler::Uniform(_) => bail!("--steps must be at least 1"),
                _ => bail!(
                    "--steps applies to the base model's uniform Euler ladder; {} has its own",
                    variant.name
                ),
            }
        }
        Ok(variant)
    }

    fn resolve(&self) -> Result<Resolved> {
        let mut variant = self.preset()?;
        let variant_dir = self.variant_dir.clone().or_else(|| self.model_dir.clone());
        if let Some(vd) = &variant_dir {
            for name in [
                "fastvideo_inference.json",
                "transformer/fastvideo_inference.json",
            ] {
                let p = vd.join(name);
                if p.is_file() {
                    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&p)?)?;
                    variant
                        .apply_contract(&v)
                        .with_context(|| format!("{}", p.display()))?;
                    eprintln!("contract: {}", p.display());
                    break;
                }
            }
        }
        let task = match &self.task {
            Some(t) => t.parse::<Task>().map_err(|e| anyhow!(e))?,
            None => variant.tasks[0],
        };
        if !variant.tasks.contains(&task) {
            bail!(
                "{} does not serve {task:?} (it serves {:?})",
                variant.name,
                variant.tasks
            );
        }
        // Ref2VA on the base and on Pika uses `transformer_ref`.
        let tdir = if task == Task::Ref2va && variant.transformer_dir == "transformer" {
            "transformer_ref".to_string()
        } else {
            variant.transformer_dir.clone()
        };
        let transformer = match &self.transformer {
            Some(t) => t.clone(),
            None => {
                let base = variant_dir
                    .clone()
                    .or_else(|| self.model_dir.clone())
                    .ok_or_else(|| {
                        anyhow!("give --model-dir (or --variant-dir / --transformer)")
                    })?;
                let p = base.join(&tdir);
                if p.exists() {
                    p
                } else {
                    self.model_dir
                        .clone()
                        .map(|m| m.join(&tdir))
                        .filter(|p| p.exists())
                        .unwrap_or(p)
                }
            }
        };
        let mut adapters = Vec::new();
        for spec in &variant.adapters {
            if spec.task.is_some_and(|t| t != task) {
                continue;
            }
            let p = variant_dir
                .as_ref()
                .map(|d| d.join(&spec.path))
                .filter(|p| p.is_file());
            match p {
                Some(p) => {
                    eprintln!("adapter: {}", p.display());
                    let strength = self.adapter_strength.unwrap_or(spec.strength);
                    adapters.push(Adapter::open_with(&p, spec.alpha, spec.rank, strength)?);
                }
                // The variant's sampler is the adapter's (Pika's renoise knots, the
                // LongLive-Plug four steps): without it the base weights would run a
                // schedule they were never trained for and make noise, not a clip.
                None => bail!(
                    "{} needs its adapter {} under --variant-dir{}",
                    variant.name,
                    spec.path,
                    if spec.path == h3_core::variant::PLUG4_ADAPTER {
                        " (convert the release's generator_lora.pt with scripts/reference/convert_longlive_plug.py; for pika-plug4, link it beside Pika's fl2va/ and ref2va/)"
                    } else {
                        ""
                    }
                ),
            }
        }
        for p in &self.adapters {
            adapters.push(Adapter::open(p, None, None)?);
        }
        Ok(Resolved {
            variant,
            task,
            model_dir: self.model_dir.clone(),
            variant_dir,
            transformer,
            adapters,
            tae: self.tae.clone(),
            tables: self.tables.clone(),
        })
    }
}

fn open_transformer(r: &Resolved) -> Result<(Box<dyn WeightSource>, bool)> {
    let dir = SafetensorsDir::open(&r.transformer)?;
    let src: Box<dyn WeightSource> = if h3_core::weights::ComfySource::detect(&dir) {
        eprintln!("{}: ComfyUI single-file layout", r.transformer.display());
        Box::new(h3_core::weights::ComfySource { inner: dir })
    } else {
        Box::new(dir)
    };
    let src_box = src;
    let src = &*src_box;
    let skip_adaln = r.tables.is_some();
    let with_gate = src.has_linear("transformer_blocks.0.attn.to_gate_compress");
    let miss: Vec<String> = missing(src, &h3_manifest(&r.variant.config, with_gate))
        .into_iter()
        .filter(|n| !(skip_adaln && n.contains(".adaln_proj.")))
        .collect();
    if !miss.is_empty() {
        bail!(
            "{} DiT tensors missing from {}, e.g. {:?}",
            miss.len(),
            r.transformer.display(),
            &miss[..miss.len().min(3)]
        );
    }
    Ok((src_box, with_gate))
}

fn sampler_plan(v: &Variant) -> Result<SamplerPlan> {
    Ok(match &v.sampler {
        Sampler::Pdd { indices, .. } => SamplerPlan::Pdd(PddPlan::new(indices, v.shifts)?),
        _ => {
            let (video, audio) = v.schedules()?;
            SamplerPlan::Schedules { video, audio }
        }
    })
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if matches!(cli.cmd, Cmd::Check { .. } | Cmd::Plan { .. }) {
        ALLOW_SW.store(true, std::sync::atomic::Ordering::Relaxed);
    }
    match cli.cmd {
        Cmd::Plan {
            model,
            shape,
            attn,
            vram_gib,
            max_binding_mib,
            backend,
            text_tokens,
        } => {
            let variant = model.preset()?;
            let g = shape.geometry(&variant)?;
            let (lt, lh, lw) = g.latent(16);
            let layout = packing::build_layout(
                &vec![packing::TEXT_TAG; text_tokens],
                lt,
                lh,
                lw,
                g.audio_latents(),
                variant.config.patch,
                &[],
            )?;
            let act = if backend == BackendKind::Cuda { 2 } else { 4 };
            let mut dev = Device::gib(vram_gib, 2, act);
            if let Some(m) = max_binding_mib {
                dev.max_binding_bytes = m << 20;
            }
            let steps = variant.forwards();
            let unique = 2 * steps + 2;
            let p = plan(
                &variant.config,
                &VideoVaeConfig::h3(),
                &layout,
                unique,
                attn.weights,
                variant.vsa.is_some(),
                &dev,
            );
            println!(
                "{} · {}x{}x{} · {} forwards",
                variant.name, g.width, g.height, g.frames, steps
            );
            print!("{p}");
        }
        Cmd::Check {
            backend,
            ops_only,
            report,
        } => {
            with_backend!(backend, |be| check(
                &be,
                resolve(backend),
                ops_only,
                report.as_deref()
            )?)
        }
        Cmd::Bench {
            model,
            shape,
            attn,
            backend,
            layers,
            steps,
            text_tokens,
        } => {
            with_backend!(backend, |be| bench(
                &be,
                &model,
                &shape,
                &attn,
                layers,
                steps,
                text_tokens
            )?)
        }
        Cmd::ParityDit {
            backend,
            transformer,
            fixture,
            tiny,
            tolerance,
        } => {
            let a = parity::Args {
                transformer,
                fixture,
                tiny,
                tolerance,
            };
            with_backend!(backend, |be| parity::run(&be, &a)?)
        }
        Cmd::VaeRoundtrip {
            backend,
            model_dir,
            input,
            fast_vae,
            frames,
            tile,
            decoder_weights,
            skip_base,
            repeat,
            out_dir,
        } => {
            let a = roundtrip::Args {
                model_dir,
                input,
                fast_vae,
                frames,
                tiling: (tile > 0).then_some(Tiling { tile, overlap: 64 }),
                decoder_weights,
                skip_base,
                repeat,
                out_dir,
            };
            with_backend!(backend, |be| roundtrip::run(&be, &a)?)
        }
        Cmd::Encode {
            model_dir,
            mut prompts,
            prompt_file,
            out_dir,
            backend,
            resident,
            enc,
        } => {
            if let Some(f) = prompt_file {
                let text =
                    std::fs::read_to_string(&f).with_context(|| format!("read {}", f.display()))?;
                prompts.extend(
                    text.lines()
                        .map(str::trim)
                        .filter(|l| !l.is_empty() && !l.starts_with('#'))
                        .map(str::to_string),
                );
            }
            if prompts.is_empty() {
                bail!("give --prompt (repeatable) or --prompt-file");
            }
            with_backend!(backend, |be| {
                let t0 = std::time::Instant::now();
                let mode = if resident {
                    encoder::EncoderMode::Resident
                } else {
                    enc.encoder.unwrap_or(encoder::EncoderMode::Auto)
                };
                let placement = encoder::choose(
                    &be,
                    &model_dir,
                    mode,
                    enc.encoder_weights,
                    prompts.len() > 1,
                )?;
                let cond = open_conditioner(&be, &model_dir, placement, &enc, false)?;
                println!(
                    "qwen3-vl opened in {:.1}s on {}",
                    t0.elapsed().as_secs_f64(),
                    be.name()
                );
                let dim = cond.model.cfg.hidden;
                for p in &prompts {
                    let t = std::time::Instant::now();
                    let e = cond.encode_text(&be, p)?;
                    let entry = gallery::Gallery::add(&out_dir, p, &e, dim)?;
                    println!(
                        "{} ({} tokens, {:.1}s): {p}",
                        entry.file,
                        entry.tokens,
                        t.elapsed().as_secs_f64()
                    );
                }
                println!("gallery: {}", out_dir.join("gallery.json").display());
            })
        }
        Cmd::AdalnTables {
            model,
            out,
            backend,
            timesteps,
        } => {
            let r = model.resolve()?;
            let (src, _) = open_transformer(&r)?;
            let mut ts: Vec<f32> = vec![h3_core::schedule::KEYFRAME_NOISE_AUG, 1.0];
            let (vs, as_) = r.variant.schedules()?;
            for i in 0..vs.steps() {
                ts.push(vs.timestep(i));
                ts.push(as_.timestep(i));
            }
            if let Sampler::Pdd { indices, .. } = &r.variant.sampler {
                let p = PddPlan::new(indices, r.variant.shifts)?;
                let (v, a) = p.schedules();
                for i in 0..v.steps() {
                    ts.push(v.timestep(i));
                    ts.push(a.timestep(i));
                }
            }
            ts.extend(timesteps);
            ts.sort_by(|a, b| a.total_cmp(b));
            ts.dedup();
            with_backend!(backend, |be| {
                let t0 = std::time::Instant::now();
                // Only the small parts load; the AdaLN weights stream through once.
                let dit = H3Dit::load(
                    &be,
                    r.variant.config.clone(),
                    src.as_ref(),
                    &LoadOptions {
                        quant: WeightQuant::Int8,
                        stream_blocks: true,
                    },
                    &r.adapters,
                )?;
                let tables = dit.adaln_tables(&be, src.as_ref(), &r.adapters, &ts, None)?;
                tables.save(&out, &r.variant.name, &transformer_name(&r.transformer))?;
                println!(
                    "{} timesteps {:?} → {} ({:.1} MB) in {:.1}s",
                    ts.len(),
                    ts,
                    out.display(),
                    tables.bytes() as f64 / 2e6,
                    t0.elapsed().as_secs_f64()
                );
            })
        }
        Cmd::Quantize {
            model,
            weights,
            out,
            skip_adaln,
        } => {
            let r = model.resolve()?;
            let (src, _) = open_transformer(&r)?;
            if weights == WeightQuant::Native && r.adapters.is_empty() && !skip_adaln {
                bail!(
                    "--weights native with no adapters and without --skip-adaln would only convert to f16"
                );
            }
            if r.adapters
                .iter()
                .any(|a| !a.modules().is_empty() && a.modules().iter().any(|m| a.has_lora(m)))
            {
                eprintln!(
                    "warning: merging LoRA pairs into a {} file keeps only part of a small delta (LongLive-Plug's H3 adapter: a third in bf16, a tenth in MXFP8; \
                     docs/runs/2026-11-15-runpod-6000s-longlive-plug). `generate` keeps them as branches: quantize the base and pass the adapter there instead.",
                    weights
                );
            }
            let t0 = std::time::Instant::now();
            let adapters = &r.adapters;
            let patch = |name: &str, w: &mut [f32], shape: &[usize]| -> h3_core::Result<()> {
                if let Some(key) = name.strip_suffix(".weight") {
                    if shape.len() == 2 {
                        for a in adapters {
                            a.apply_weight(key, w, shape[0], shape[1])?;
                        }
                    } else {
                        for a in adapters {
                            a.apply_vector(key, w)?;
                        }
                    }
                } else if let Some(key) = name.strip_suffix(".bias") {
                    for a in adapters {
                        a.apply_bias(key, w)?;
                    }
                }
                Ok(())
            };
            let tensors = h3_core::export::quantize_dit(
                src.as_ref(),
                &r.variant.config,
                weights,
                Some(&patch),
                skip_adaln,
            )?;
            let bytes: usize = tensors.iter().map(|t| t.3.len()).sum();
            h3_core::export::write(
                &out,
                &tensors,
                h3_core::export::metadata(weights, &r.variant.name),
            )?;
            println!(
                "{weights}: {} tensors, {:.3} GB -> {} in {:.1}s",
                tensors.len(),
                bytes as f64 / 1e9,
                out.display(),
                t0.elapsed().as_secs_f64()
            );
        }
        Cmd::Generate {
            model,
            shape,
            attn,
            enc,
            backend,
            prompt,
            embeds,
            first_image,
            last_image,
            refs,
            continuation,
            seed,
            preview,
            upscale,
            upscale_factor,
            tile,
            runs,
            out,
        } => {
            let r = model.resolve()?;
            let (g, out_size) = shape.resolve(&r.variant)?;
            let finish = Finish {
                out_size,
                upscale: upscale.map(|p| (p, upscale_factor)),
            };
            with_backend!(backend, |be| generate(
                &be,
                &r,
                g,
                &attn,
                &enc,
                prompt,
                embeds,
                first_image,
                last_image,
                &refs,
                continuation,
                seed,
                preview,
                &finish,
                tile,
                runs,
                &out
            )?)
        }
        Cmd::Regen2k {
            regen_dir,
            model_dir,
            draft,
            prompt,
            embeds,
            adaln_table,
            attn,
            enc,
            backend,
            steps,
            start_sigma,
            seed,
            tile,
            released_vae,
            out,
        } => {
            // The release decodes with its fast decoder (`vae_fast/`); so do we
            // unless asked for the released one.
            let mut attn = attn;
            if released_vae {
                if attn.fast_vae.is_some() {
                    bail!("--released-vae and --fast-vae contradict each other");
                }
            } else if attn.fast_vae.is_none() {
                let f = regen_dir.join("vae_fast/fast_h3_vae_noncausal_24l.safetensors");
                if f.is_file() {
                    attn.fast_vae = Some(f);
                } else {
                    eprintln!(
                        "note: {} not found; decoding with the released VAE",
                        f.display()
                    );
                }
            }
            with_backend!(backend, |be| regen2k(
                &be,
                &regen_dir,
                &model_dir,
                &draft,
                prompt,
                embeds,
                &adaln_table,
                &attn,
                &enc,
                steps,
                start_sigma,
                seed,
                tile,
                &out
            )?)
        }
        Cmd::Worker {
            test_input,
            work_dir,
        } => worker::run(test_input, work_dir)?,
        Cmd::QuantizeEncoder { model_dir, out } => {
            let out = out.unwrap_or_else(|| model_dir.join("text_encoder_int8"));
            let src = SafetensorsDir::open(&model_dir.join("text_encoder"))?;
            let t0 = std::time::Instant::now();
            let mut total = 0usize;
            h3_text::write_int8_tree(&src, &Qwen3VlConfig::h3(), &out, &mut |l, b| {
                total += b;
                eprintln!(
                    "layer {l}: {:.2} GB ({:.1}s)",
                    b as f64 / 1e9,
                    t0.elapsed().as_secs_f64()
                );
            })?;
            println!(
                "wrote {} ({:.1} GB) in {:.1}s",
                out.display(),
                total as f64 / 1e9,
                t0.elapsed().as_secs_f64()
            );
        }
        Cmd::Serve {
            model,
            attn,
            addr,
            backend,
            gallery,
            synthetic,
            max_pixels,
            max_frames,
            allow_origin,
            tile,
            preview,
            enc,
            upscale,
        } => {
            let resolved = if synthetic {
                None
            } else {
                Some(model.resolve()?)
            };
            let settings = serve::Settings {
                addr,
                model: resolved,
                gallery,
                synthetic,
                max_pixels,
                max_frames,
                allow_origin,
                attn,
                tiling: (tile > 0).then_some(Tiling { tile, overlap: 64 }),
                preview,
                upscale,
                encoder: enc,
            };
            serve::serve(settings, move |s, shared, rx| {
                with_backend!(backend, |be| serve::engine(be, s, shared, rx))
            })?;
        }
    }
    Ok(())
}

fn check<B: Backend>(
    be: &B,
    kind: BackendKind,
    ops_only: bool,
    report: Option<&Path>,
) -> Result<()> {
    println!("{}", be.name());
    // bf16 activations on CUDA: ~3 significant digits per op; f16 weight
    // storage on WebGPU: ~3 digits in the partial sums.
    let tol = match kind {
        BackendKind::Cuda => 3e-2,
        BackendKind::Wgpu => 1e-2,
        _ => 2e-3,
    };
    let res = if ops_only {
        h3_model::check::run_ops(be, tol)
    } else {
        h3_model::check::run_all(be, tol)
    };
    let mut bad = 0;
    for r in &res {
        println!(
            "{:<52} rel {:.2e}  {}",
            r.name,
            r.rel_err,
            if r.ok { "ok" } else { "FAIL" }
        );
        bad += (!r.ok) as usize;
    }
    if let Some(p) = report {
        let cases: Vec<_> = res
            .iter()
            .map(|r| serde_json::json!({"name": r.name, "rel_err": r.rel_err, "ok": r.ok}))
            .collect();
        let v = serde_json::json!({
            "backend": be.name(), "tolerance": tol, "passed": res.len() - bad,
            "total": res.len(), "cases": cases,
        });
        std::fs::write(p, serde_json::to_vec_pretty(&v)?)
            .with_context(|| format!("write {}", p.display()))?;
    }
    if bad > 0 {
        bail!("{bad} of {} checks failed (tolerance {tol:e})", res.len());
    }
    println!("all {} checks pass (tolerance {tol:e})", res.len());
    Ok(())
}

fn bench<B: Backend>(
    be: &B,
    model: &ModelArgs,
    shape: &Shape,
    attn: &AttnArgs,
    layers: Option<usize>,
    steps: usize,
    text_tokens: usize,
) -> Result<()> {
    let variant = Variant::by_name(&model.variant)?;
    let mut cfg = variant.config.clone();
    if let Some(l) = layers {
        cfg.layers = l;
    }
    let g = shape.geometry(&variant)?;
    let (lt, lh, lw) = g.latent(16);
    let layout = packing::build_layout(
        &vec![packing::TEXT_TAG; text_tokens],
        lt,
        lh,
        lw,
        g.audio_latents(),
        cfg.patch,
        &[],
    )?;
    println!(
        "{}  |  {} {}x{}x{}  {} tokens  {} blocks  weights {}",
        be.name(),
        variant.name,
        g.width,
        g.height,
        g.frames,
        layout.sequence_length,
        cfg.layers,
        attn.weights
    );
    let src = Synthetic { seed: 1 };
    let t0 = std::time::Instant::now();
    let dit = H3Dit::load(
        be,
        cfg.clone(),
        &src,
        &LoadOptions {
            quant: attn.weights,
            stream_blocks: attn.stream_blocks,
        },
        &[],
    )?;
    be.sync();
    println!("load (synthetic) {:.2}s", t0.elapsed().as_secs_f64());
    let attention = if variant.vsa.is_some() || attn.attention == AttnKind::Vsa {
        attn.attention(&variant)
    } else {
        Attention::Dense {
            qk8: attn.vsa_qk8 || attn.pv8,
            pv8: attn.pv8,
        }
    };
    let text = dit.refine_text(be, &vec![0.0; text_tokens * cfg.text_dim]);
    let ws = dit.workspace(be, &layout, attn.ffn_chunk.unwrap_or(usize::MAX), attention)?;
    let t1 = std::time::Instant::now();
    let tables = dit.adaln_tables(be, &src, &[], &[0.3, 0.7, 0.999, 1.0], None)?;
    be.sync();
    println!(
        "adaln tables {:.2}s ({} per block)",
        t1.elapsed().as_secs_f64(),
        tables.blocks[0].len() / 4
    );
    let (unique, inverse) = packing::row_timesteps(&layout, 0.3, 0.7, 0.999, 1.0);
    let mods = dit.step_mods(be, &tables, &unique, &inverse, &layout.token_tags, false)?;
    let vrows = be.alloc(layout.video_indices.len() * cfg.video_patch_dim());
    let arows = be.alloc(layout.audio_indices.len() * cfg.audio_in_channels);
    let vout = be.alloc(layout.video_indices.len() * cfg.video_patch_dim());
    let aout = be.alloc(layout.audio_indices.len() * cfg.audio_in_channels);
    be.profile_reset();
    for i in 0..steps.max(1) {
        let t = std::time::Instant::now();
        dit.forward(be, &ws, &vrows, &arows, &text, &mods, None, &vout, &aout);
        be.sync();
        let s = t.elapsed().as_secs_f64();
        println!(
            "forward {i}: {:.3}s ({:.1} ms/block; ×{}/{} blocks ≈ {:.2}s)",
            s,
            s * 1e3 / cfg.layers as f64,
            variant.config.layers,
            cfg.layers,
            s * variant.config.layers as f64 / cfg.layers as f64
        );
    }
    if let Some(r) = be.profile_report() {
        print!("{r}");
    }
    Ok(())
}

fn tokenizer_path(model_dir: &Path) -> Result<PathBuf> {
    ["tokenizer/tokenizer.json", "processor/tokenizer.json"]
        .iter()
        .map(|p| model_dir.join(p))
        .find(|p| p.is_file())
        .ok_or_else(|| anyhow!("no tokenizer.json under {}", model_dir.display()))
}

fn open_conditioner<B: Backend>(
    be: &B,
    model_dir: &Path,
    placement: h3_text::Placement,
    enc: &encoder::EncoderArgs,
    vision: bool,
) -> Result<h3_text::Conditioner<B>> {
    let tok = tokenizer_path(model_dir)?;
    let t0 = std::time::Instant::now();
    let weights = enc.encoder_weights.weights();
    let tree = enc.int8_tree(model_dir);
    let c = h3_text::Conditioner::open(
        be,
        &tok,
        &model_dir.join("text_encoder"),
        Qwen3VlConfig::h3(),
        placement,
        weights,
        tree.as_deref(),
        vision,
    )?;
    eprintln!(
        "text encoder: {placement:?}, {weights:?}{}, opened in {:.1}s",
        tree.map_or(String::new(), |t| format!(" from {}", t.display())),
        t0.elapsed().as_secs_f64()
    );
    Ok(c)
}

/// A one-prompt text encode through the prompt cache (when given): a hit
/// skips the conditioner entirely.
fn encode_text_cached<B: Backend>(
    be: &B,
    model_dir: &Path,
    enc: &encoder::EncoderArgs,
    prompt: &str,
    text_dim: usize,
) -> Result<Prompt> {
    let cache = match &enc.prompt_cache {
        Some(d) => Some(encoder::PromptCache::new(
            Some(d.clone()),
            encoder::identity(model_dir, &tokenizer_path(model_dir)?, enc.encoder_weights)?,
            text_dim,
        )?),
        None => None,
    };
    if let Some(p) = cache.as_ref().and_then(|c| c.get(prompt)) {
        eprintln!("prompt: from the cache");
        return Ok(p);
    }
    let mode = enc.encoder.unwrap_or(encoder::EncoderMode::Auto);
    if mode == encoder::EncoderMode::None {
        bail!("--encoder none and the prompt is not in the cache");
    }
    let placement = encoder::choose(be, model_dir, mode, enc.encoder_weights, false)?;
    let p = open_conditioner(be, model_dir, placement, enc, false)?.encode_text(be, prompt)?;
    if let Some(c) = &cache {
        c.put(prompt, &p)?;
    }
    Ok(p)
}

/// Parse `kind:path[,audio=path]`.
fn parse_media_spec(spec: &str) -> Result<(String, PathBuf, Option<PathBuf>)> {
    let (kind, rest) = spec
        .split_once(':')
        .ok_or_else(|| anyhow!("media spec must be kind:path, got {spec}"))?;
    let mut parts = rest.split(",audio=");
    let path = PathBuf::from(parts.next().unwrap());
    let audio = parts.next().map(PathBuf::from);
    Ok((kind.to_string(), path, audio))
}

#[allow(clippy::too_many_arguments)]
/// The DiT a run loads, by its directory or file name (`transformer`,
/// `transformer_ref`, a quantized file's name).
fn transformer_name(p: &Path) -> String {
    p.file_name().map_or_else(
        || p.display().to_string(),
        |n| n.to_string_lossy().into_owned(),
    )
}

/// What happens to a clip's decoded frames before they are written: the
/// centre crop to the requested size (canvases off the 32-pixel grid, e.g.
/// `--width 480 --height 272`) and the optional Real-ESRGAN upscale.
struct Finish {
    out_size: (usize, usize),
    /// Weights and the requested factor (2 / 3 / 4; `None` picks).
    upscale: Option<(PathBuf, Option<usize>)>,
}

impl Finish {
    fn apply<B: Backend>(&self, be: &B, g: &mut h3_model::pipeline::Generation) -> Result<()> {
        let (ow, oh) = self.out_size;
        if (ow, oh) != (g.width, g.height) {
            g.rgb = finish::crop(&g.rgb, g.frames, g.width, g.height, ow, oh);
            println!("crop {}x{} -> {ow}x{oh}", g.width, g.height);
            (g.width, g.height) = (ow, oh);
        }
        let Some((weights, factor)) = &self.upscale else {
            return Ok(());
        };
        let factor = factor.unwrap_or(finish::cli_factor(g.width, g.height));
        let t0 = std::time::Instant::now();
        let src = SafetensorsDir::open(weights)?;
        let up = h3_model::esrgan::Esrgan::load(
            be,
            h3_model::esrgan::EsrganConfig::anime_video_v3(),
            &src,
        )?;
        let (w, h) = (g.width, g.height);
        let (tw, th) = (w * factor, h * factor);
        let mut rgb = vec![0u8; g.frames * tw * th * 3];
        let batch = finish::upscale(be, &up, &g.rgb, g.frames, w, h, factor, &mut |f0, n, x| {
            rgb[f0 * tw * th * 3..(f0 + n) * tw * th * 3].copy_from_slice(&x);
            Ok(())
        })?;
        println!(
            "upscale x{factor} ({batch} frames per batch): {w}x{h} -> {tw}x{th} in {:.2}s",
            t0.elapsed().as_secs_f64()
        );
        (g.rgb, g.width, g.height) = (rgb, tw, th);
        Ok(())
    }
}

fn generate<B: Backend>(
    be: &B,
    r: &Resolved,
    g: Geometry,
    attn: &AttnArgs,
    enc: &encoder::EncoderArgs,
    prompt: Option<String>,
    embeds: Option<PathBuf>,
    first_image: Option<PathBuf>,
    last_image: Option<PathBuf>,
    refs: &[String],
    continuation: Option<String>,
    seed: u64,
    preview: bool,
    finish: &Finish,
    tile: usize,
    runs: usize,
    out: &Path,
) -> Result<()> {
    let cfg = &r.variant.config;
    let model_dir = r
        .model_dir
        .as_ref()
        .ok_or_else(|| anyhow!("--model-dir is needed for the VAEs and the text encoder"))?;
    let vae_dir = r
        .variant_dir
        .as_ref()
        .map(|d| d.join("vae"))
        .filter(|p| p.is_dir() && r.variant.nvfp4)
        .unwrap_or_else(|| model_dir.join("vae"));
    let vae_cfg = match std::fs::read(vae_dir.join("config.json")) {
        Ok(b) => VideoVaeConfig::from_json(&serde_json::from_slice(&b)?)?,
        Err(_) => VideoVaeConfig::h3(),
    };
    println!("{}", be.name());
    let t0 = std::time::Instant::now();
    let (src, _) = open_transformer(r)?;
    let tables = match &r.tables {
        Some(p) => {
            // A LoRA on the modulation (H3 Turbo 8-step) changes the tables: a
            // file computed without it would silently drop that part.
            let touches = r.adapters.iter().any(|a| {
                (0..cfg.layers)
                    .any(|l| a.touches(&format!("transformer_blocks.{l}.adaln_proj.linear")))
            });
            if touches {
                let of = AdalnTables::variant_of(p)?;
                if of.as_deref() != Some(r.variant.name.as_str()) {
                    bail!(
                        "{} was computed for {}, but {}'s adapter changes the AdaLN modulation: run `h3 adaln-tables --variant {}` or leave out --tables",
                        p.display(),
                        of.unwrap_or_else(|| "an unknown variant".into()),
                        r.variant.name,
                        r.variant.name
                    );
                }
            }
            // T2VA / FL2VA run `transformer/`, Ref2VA `transformer_ref/`: their
            // modulation weights differ, so a table from the other one is wrong.
            if let Some(t) = AdalnTables::transformer_of(p)?
                && t != transformer_name(&r.transformer)
            {
                bail!(
                    "{} was computed from {t}, this run uses {}: run `h3 adaln-tables` with the same --task",
                    p.display(),
                    transformer_name(&r.transformer)
                );
            }
            Some(AdalnTables::load(p, cfg)?)
        }
        None => None,
    };
    let need_encoder =
        first_image.is_some() || last_image.is_some() || !refs.is_empty() || continuation.is_some();
    let dit = H3Dit::load(
        be,
        cfg.clone(),
        src.as_ref(),
        &LoadOptions {
            quant: attn.weights,
            stream_blocks: attn.stream_blocks,
        },
        &r.adapters,
    )?;
    let mut vae = VideoVae::load_with(
        be,
        vae_cfg,
        &SafetensorsDir::open(&vae_dir)?,
        need_encoder,
        !preview && attn.fast_vae.is_none(),
        attn.decoder_weights,
    )?;
    attn.attach_fast_vae(be, &mut vae, preview)?;
    let audio_vae = AudioVae::load(
        be,
        AudioVaeConfig::h3(),
        &SafetensorsDir::open(&model_dir.join("audio_vae"))?,
        refs.iter().any(|s| s.contains("audio"))
            || continuation.as_deref().is_some_and(|c| c.contains("audio")),
        true,
    )?;
    let tae = match (&r.tae, preview) {
        (Some(p), true) => Some(TaeH3::load(
            be,
            TaeH3Config::taeh3(),
            &SafetensorsDir::open(p)?,
        )?),
        (None, true) => bail!("--preview needs --tae taeh3.safetensors"),
        _ => None,
    };
    let pipe = Pipeline {
        dit,
        vae: Some(vae),
        audio_vae: Some(audio_vae),
        tae,
    };
    be.sync();
    println!("load {:.1}s", t0.elapsed().as_secs_f64());
    let tiling = (tile > 0).then_some(Tiling { tile, overlap: 64 });

    // Conditioning: read the files, then size them like `h3 serve` does.
    let pika = r.variant.pika_buckets;
    let mut inputs = inputs::Inputs::default();
    for (anchor, path) in [(Anchor::First, first_image), (Anchor::Last, last_image)] {
        if let Some(p) = path {
            inputs.keyframes.push((anchor, media::read_image(&p)?));
        }
    }
    for spec in refs {
        let (kind, path, audio) = parse_media_spec(spec)?;
        inputs.refs.push(match kind.as_str() {
            "image" => inputs::RefInput::Image(media::read_image(&path)?),
            "video" => {
                let (frames, fps) = media::read_frames(&path)?;
                inputs::RefInput::Video {
                    frames,
                    fps,
                    audio: audio
                        .as_deref()
                        .map(|p| media::read_wav(p, 32000))
                        .transpose()?,
                }
            }
            "audio" => inputs::RefInput::Audio(media::read_wav(&path, 32000)?),
            other => bail!("unknown reference kind `{other}` (image | video | audio)"),
        });
    }
    if let Some(spec) = continuation {
        let (path, audio) = match spec.split_once(",audio=") {
            Some((p, a)) => (PathBuf::from(p), Some(PathBuf::from(a))),
            None => (PathBuf::from(&spec), None),
        };
        let (frames, _) = media::read_frames(&path)?;
        inputs.continuation = Some((
            frames,
            audio
                .as_deref()
                .map(|p| media::read_wav(p, 32000))
                .transpose()?,
        ));
    }
    let ref_short_edge = if pika {
        packing::pika::tier("768p")?.reference_short_edge
    } else {
        packing::REFERENCE_IMAGE_SHORT_EDGE
    };
    let inputs::Prepared {
        keyframes,
        kf_images,
        references,
        presented,
        continuation: cont,
    } = inputs::prepare(inputs, g, pika, inputs::Sizing::Release { ref_short_edge })?;
    be.profile_reset();
    let tp = std::time::Instant::now();
    let prompt_rows: Prompt = match (&prompt, &embeds) {
        (_, Some(path)) if !need_encoder => gallery::load_prompt(path, cfg.text_dim)?,
        (Some(p), _) if presented.is_empty() && kf_images.is_empty() => {
            encode_text_cached(be, model_dir, enc, p, cfg.text_dim)?
        }
        (Some(p), _) => {
            let mode = enc.encoder.unwrap_or(encoder::EncoderMode::Auto);
            let placement = encoder::choose(be, model_dir, mode, enc.encoder_weights, false)?;
            let cond = open_conditioner(be, model_dir, placement, enc, need_encoder)?;
            if !presented.is_empty() {
                cond.encode_ref2va(be, p, &presented)?
            } else {
                cond.encode_fl2va(be, p, &kf_images)?
            }
        }
        _ => bail!("give --prompt, or --embeds for a text-only clip"),
    };
    println!(
        "prompt: {} rows in {:.1}s",
        prompt_rows.rows(cfg.text_dim),
        tp.elapsed().as_secs_f64()
    );
    let req = Request {
        prompt: prompt_rows,
        geometry: g,
        seed,
        keyframes,
        references,
        continuation: cont,
        sampler: sampler_plan(&r.variant)?,
        attention: attn.attention(&r.variant),
        dense_first_steps: attn.dense_first_steps,
        ffn_chunk: attn.ffn_chunk.unwrap_or(usize::MAX),
        tiling,
        preview,
    };
    // With H3_PROFILE=1 every stage boundary prints its own kernel table.
    let stage_report = |name: &str| {
        if let Some(rep) = be.profile_report() {
            println!("== profile: {name}\n{rep}");
            be.profile_reset();
        }
    };
    stage_report("prompt encoding (Qwen3-VL)");
    let mut last = None;
    for run in 0..runs.max(1) {
        let t0 = std::time::Instant::now();
        let generation = pipe.generate(
            be,
            src.as_ref(),
            &r.adapters,
            tables.as_ref(),
            &req,
            &mut |ev| match ev {
                Event::Prepared { .. } => stage_report("prepare (latent encode, AdaLN tables)"),
                Event::Step { index, total, secs } => {
                    eprintln!("step {}/{total} {secs:.2}s", index + 1);
                    if index + 1 == total {
                        stage_report("denoise");
                    }
                }
                Event::Frames { .. } => stage_report("video decode"),
                Event::Audio { .. } => stage_report("audio decode"),
            },
        )?;
        let t = &generation.timings;
        println!(
            "run {run}: total {:.2}s  prepare {:.2}s  adaln {:.2}s  denoise {:.2}s ({:?})  decode {:.2}s  audio {:.2}s",
            t0.elapsed().as_secs_f64(),
            t.prepare_s,
            t.adaln_s,
            t.denoise_s(),
            t.steps_s
                .iter()
                .map(|s| format!("{s:.2}"))
                .collect::<Vec<_>>(),
            t.decode_s,
            t.audio_s
        );
        last = Some(generation);
    }
    stage_report("remainder");
    let mut generation = last.expect("at least one run");
    finish.apply(be, &mut generation)?;
    media::write_y4m(
        out,
        generation.frames,
        generation.width,
        generation.height,
        &generation.rgb,
        packing::FPS as u32,
    )?;
    let wav = out.with_extension("wav");
    if let Some(a) = &generation.audio {
        media::write_wav(&wav, a, generation.sample_rate as u32)?;
    }
    println!(
        "wrote {} ({} frames {}x{}) and {}",
        out.display(),
        generation.frames,
        generation.width,
        generation.height,
        wav.display()
    );
    println!(
        "mux: ffmpeg -i {} -i {} -c:v libx264 -pix_fmt yuv420p -c:a aac {}",
        out.display(),
        wav.display(),
        out.with_extension("mp4").display()
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn regen2k<B: Backend>(
    be: &B,
    regen_dir: &Path,
    model_dir: &Path,
    draft: &Path,
    prompt: Option<String>,
    embeds: Option<PathBuf>,
    adaln_table: &Path,
    attn: &AttnArgs,
    enc: &encoder::EncoderArgs,
    steps: usize,
    start_sigma: Option<f64>,
    seed: u64,
    tile: usize,
    out: &Path,
) -> Result<()> {
    let variant = Variant::by_name("regen-2k")?;
    let cfg = variant.config.clone();
    println!("{}", be.name());
    let t0 = std::time::Instant::now();
    let tdir = [
        regen_dir.join("base_model/transformer"),
        model_dir.join("transformer"),
    ]
    .into_iter()
    .find(|p| p.is_dir())
    .ok_or_else(|| {
        anyhow!(
            "no transformer/ under {} or {}",
            regen_dir.display(),
            model_dir.display()
        )
    })?;
    let src = SafetensorsDir::open(&tdir)?;
    let lora = regen_dir.join("lora/open_h3_regenerate_2k_lora.safetensors");
    let adapters =
        vec![Adapter::open(&lora, None, None).with_context(|| format!("{}", lora.display()))?];
    let dit = H3Dit::load(
        be,
        cfg.clone(),
        &src,
        &LoadOptions {
            quant: attn.weights,
            stream_blocks: attn.stream_blocks,
        },
        &adapters,
    )?;
    let curve = SharpCurve::load(
        &regen_dir.join("lora/sharp_curve_branches.safetensors"),
        adaln_table,
        &cfg,
    )?;
    // FastVideo's pipeline decodes with the base model's VAE (36-block
    // full-attention decoder); `--fast-vae` takes the release's `vae_fast/`
    // Sol decoder instead (`fast_vae.rs`).
    let mut vae = VideoVae::load_with(
        be,
        VideoVaeConfig::h3(),
        &SafetensorsDir::open(&model_dir.join("vae"))?,
        true,
        attn.fast_vae.is_none(),
        attn.decoder_weights,
    )?;
    attn.attach_fast_vae(be, &mut vae, false)?;
    let audio_vae = AudioVae::load(
        be,
        AudioVaeConfig::h3(),
        &SafetensorsDir::open(&model_dir.join("audio_vae"))?,
        true,
        false,
    )?;
    let upscaler = Upscaler::load(
        be,
        UpscalerConfig::regen_2k(),
        &SafetensorsDir::open(&regen_dir.join("latent_upsampler"))?,
    )?;
    let regen = Regen {
        dit,
        vae,
        audio_vae,
        upscaler,
        curve,
    };
    be.sync();
    println!("load {:.1}s", t0.elapsed().as_secs_f64());
    let (frames, _) = media::read_frames(draft)?;
    if frames.len() < h3_model::regen::DRAFT_FRAMES {
        bail!(
            "the draft has {} frames; Regenerate-2K needs {}",
            frames.len(),
            h3_model::regen::DRAFT_FRAMES
        );
    }
    // The draft's own aspect decides both canvases (2560×1440 at 16:9,
    // 1440×2560 at 9:16, …); a 480p or 288p draft is fitted to the 768p one.
    let (dw, dh) = (frames[0].width, frames[0].height);
    let ((h, w), output) = h3_model::regen::regen_canvases(dw, dh)?;
    println!(
        "draft {dw}x{dh} -> canvas {w}x{h} -> {}x{}",
        output.1, output.0
    );
    let mut fr: Vec<Image> = frames[..h3_model::regen::DRAFT_FRAMES]
        .iter()
        .map(|f| f.fit_canvas(w, h, false))
        .collect();
    let last = fr[fr.len() - 1].clone();
    fr.extend(std::iter::repeat_n(
        last,
        h3_model::regen::DRAFT_PADDED_FRAMES - h3_model::regen::DRAFT_FRAMES,
    ));
    let draft_vol = Image::frames_to_volume(&fr);
    let prompt_rows = match (&prompt, &embeds) {
        (_, Some(p)) => gallery::load_prompt(p, cfg.text_dim)?,
        (Some(p), _) => encode_text_cached(be, model_dir, enc, p, cfg.text_dim)?,
        _ => bail!("give --prompt or --embeds"),
    };
    let req = RegenRequest {
        prompt: prompt_rows,
        draft: draft_vol,
        output,
        seed,
        steps,
        start_sigma,
        attention: attn.attention(&variant),
        ffn_chunk: attn.ffn_chunk.unwrap_or(usize::MAX),
        tiling: (tile > 0).then_some(Tiling { tile, overlap: 64 }),
    };
    let t1 = std::time::Instant::now();
    let (frames, width, height, rgb, t) = regen.run(be, &src, &adapters, &req, &mut |ev| {
        if let Event::Step { index, total, secs } = ev {
            eprintln!("step {}/{total} {secs:.2}s", index + 1);
        }
    })?;
    println!(
        "regen: total {:.1}s  prepare {:.1}s  steps {:?}  decode {:.1}s",
        t1.elapsed().as_secs_f64(),
        t.prepare_s,
        t.steps_s
            .iter()
            .map(|s| format!("{s:.1}"))
            .collect::<Vec<_>>(),
        t.decode_s
    );
    media::write_y4m(out, frames, width, height, &rgb, packing::FPS as u32)?;
    println!(
        "wrote {} ({frames} frames {width}x{height}, silent)",
        out.display()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_288p_is_on_the_grid() {
        let v = Variant::by_name("draft").unwrap();
        for (aspect, want) in [
            ("16:9", (480, 288)),
            ("9:16", (288, 480)),
            ("1:1", (288, 288)),
        ] {
            let cli = Cli::parse_from(["h3", "plan", "--tier", "288p", "--aspect", aspect]);
            let Cmd::Plan { shape, .. } = cli.cmd else {
                unreachable!()
            };
            let (g, out) = shape.resolve(&v).unwrap();
            assert_eq!(((g.width, g.height), out), (want, want), "{aspect}");
        }
        let cli = Cli::parse_from(["h3", "plan", "--tier", "272p"]);
        let Cmd::Plan { shape, .. } = cli.cmd else {
            unreachable!()
        };
        assert!(shape.resolve(&v).is_err());
    }
}
