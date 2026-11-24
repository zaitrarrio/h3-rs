//! `h3 parity-dit`: the DiT against a PyTorch reference dump at full width.
//!
//! `scripts/reference/full_parity.py` runs FastVideo's
//! `MiniMaxH3Transformer3DModel` on the real checkpoint (its first `layers`
//! blocks, the embedders, the text refiner and the head) for a small packed
//! sequence and writes the inputs and outputs; this command loads the same
//! blocks here and compares. The committed tiny fixtures (`hf_parity`) hold
//! the structure at toy sizes; this holds the real widths, head counts, RoPE
//! split and weights.

use anyhow::{Result, bail};
use h3_core::{
    Backend, H3Config, packing,
    packing::Anchor,
    variant::Variant,
    weights::{SafetensorsDir, WeightSource},
};
use h3_model::dit::{Attention, H3Dit, LoadOptions};
use std::path::PathBuf;

pub struct Args {
    /// The checkpoint's `transformer/` directory (or the fixture itself with `tiny`).
    pub transformer: PathBuf,
    pub fixture: PathBuf,
    pub tiny: bool,
    pub tolerance: Option<f32>,
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())) + 1e-6;
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
}

pub fn run<B: Backend>(be: &B, a: &Args) -> Result<()> {
    let fx = SafetensorsDir::open(&a.fixture)?;
    let g = fx.get("input.geometry", &[5])?;
    let [lt, lh, lw, na, layers] = [g[0], g[1], g[2], g[3], g[4]].map(|v| v as usize);
    let ts = fx.get("input.timesteps", &[4])?;
    let mut cfg = if a.tiny {
        H3Config::tiny()
    } else {
        Variant::by_name("base")?.config.clone()
    };
    cfg.layers = layers;
    let nt = fx.shape("input.text_tags")?[0];
    let tags: Vec<u8> = fx
        .get("input.text_tags", &[nt])?
        .iter()
        .map(|v| *v as u8)
        .collect();
    let layout = packing::build_layout(&tags, lt, lh, lw, na, cfg.patch, &[Anchor::First])?;
    let src = SafetensorsDir::open(&a.transformer)?;
    let t0 = std::time::Instant::now();
    let dit = H3Dit::load(be, cfg.clone(), &src, &LoadOptions::default(), &[])?;
    println!(
        "{}: {} blocks of {} loaded in {:.1}s; {} video, {} audio, {} text rows",
        be.name(),
        layers,
        a.transformer.display(),
        t0.elapsed().as_secs_f64(),
        layout.video_indices.len(),
        layout.audio_indices.len(),
        nt
    );
    let text = dit.refine_text(be, &fx.get("input.text", &[nt, cfg.text_dim])?);
    let ws = dit.workspace(
        be,
        &layout,
        usize::MAX,
        Attention::Dense {
            qk8: false,
            pv8: false,
        },
    )?;
    let tables = dit.adaln_tables(be, &src, &[], &ts, None)?;
    let (unique, inverse) = packing::row_timesteps(&layout, ts[0], ts[1], ts[2], ts[3]);
    let mods = dit.step_mods(be, &tables, &unique, &inverse, &layout.token_tags, false)?;
    let (nv, nau) = (layout.video_indices.len(), layout.audio_indices.len());
    let (vd, ad) = (cfg.video_patch_dim(), cfg.audio_in_channels);
    let video = be.upload(&fx.get("input.video_rows", &[nv, vd])?);
    let audio = be.upload(&fx.get("input.audio_rows", &[nau, ad])?);
    let (vo, ao) = (be.alloc(nv * vd), be.alloc(nau * ad));
    dit.forward(be, &ws, &video, &audio, &text, &mods, None, &vo, &ao);
    let rv = rel(&be.download(&vo), &fx.get("expected.video", &[nv, vd])?);
    let ra = rel(&be.download(&ao), &fx.get("expected.audio", &[nau, ad])?);
    let tol = a.tolerance.unwrap_or(if be.name().starts_with("cpu") {
        2e-3
    } else {
        3e-2
    });
    println!("video rel {rv:.2e}  audio rel {ra:.2e}  (tolerance {tol:e})");
    if rv > tol || ra > tol {
        bail!("parity failed: video {rv:.2e}, audio {ra:.2e} over {tol:e}");
    }
    println!("parity ok");
    Ok(())
}
