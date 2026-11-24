//! `h3 vae-roundtrip`: encode a video with the H3 VAE, then decode the same
//! latents with the released decoder and the fast (Sol-H3) one, timing each
//! and reporting PSNR against the input and between the two decoders.

use crate::media;
use anyhow::{Result, bail};
use h3_core::{Backend, config::VideoVaeConfig, quant::WeightQuant, weights::SafetensorsDir};
use h3_model::{
    fast_vae::FastDecoder,
    vae::{Posterior, Tiling, VideoVae, Volume},
};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub struct Args {
    pub model_dir: PathBuf,
    pub input: PathBuf,
    pub fast_vae: Option<PathBuf>,
    pub frames: Option<usize>,
    pub tiling: Option<Tiling>,
    pub decoder_weights: WeightQuant,
    pub skip_base: bool,
    pub repeat: usize,
    pub out_dir: Option<PathBuf>,
}

/// PSNR (dB) over `[0, 1]` values of the frames both clips have.
fn psnr(a: &Volume, b: &Volume) -> f64 {
    let t = a.t.min(b.t);
    let (mut se, mut n) = (0.0f64, 0usize);
    for c in 0..a.c.min(b.c) {
        for f in 0..t {
            let (oa, ob) = ((c * a.t + f) * a.h * a.w, (c * b.t + f) * b.h * b.w);
            for i in 0..a.h * a.w {
                let d = (a.data[oa + i] - b.data[ob + i]) as f64;
                se += d * d;
            }
            n += a.h * a.w;
        }
    }
    10.0 * (1.0 / (se / n.max(1) as f64).max(1e-20)).log10()
}

fn write(dir: &Path, name: &str, v: &Volume) -> Result<()> {
    let px = v.h * v.w;
    let mut rgb = vec![0u8; v.t * px * 3];
    for f in 0..v.t {
        for i in 0..px {
            for c in 0..3 {
                let x = v.data[(c * v.t + f) * px + i];
                rgb[(f * px + i) * 3 + c] = (x.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }
    let p = dir.join(format!("{name}.y4m"));
    media::write_y4m(&p, v.t, v.w, v.h, &rgb, h3_core::packing::FPS as u32)?;
    eprintln!("wrote {}", p.display());
    Ok(())
}

fn decode_all<B: Backend>(
    vae: &VideoVae<B>,
    be: &B,
    z: &Volume,
    tiling: Option<Tiling>,
) -> Result<(Volume, f64)> {
    let t0 = Instant::now();
    let mut parts = Vec::new();
    vae.decode(be, z, tiling, &mut |v| parts.push(v.clone()))?;
    Ok((Volume::cat_t(&parts), t0.elapsed().as_secs_f64()))
}

pub fn run<B: Backend>(be: &B, a: &Args) -> Result<()> {
    let (frames, _) = media::read_frames(&a.input)?;
    let n = a.frames.map_or(frames.len(), |f| f.min(frames.len()));
    if n == 0 {
        bail!("{}: no frames", a.input.display());
    }
    let (w, h) = (frames[0].width, frames[0].height);
    if w % 16 != 0 || h % 16 != 0 {
        bail!("{w}×{h}: the VAE needs multiples of 16");
    }
    let mut px = Volume::zeros(3, n, h, w);
    for (f, im) in frames[..n].iter().enumerate() {
        for i in 0..w * h {
            for c in 0..3 {
                px.data[(c * n + f) * w * h + i] = im.rgb[i * 3 + c] as f32 / 255.0;
            }
        }
    }
    let t0 = Instant::now();
    let mut vae = VideoVae::load_with(
        be,
        VideoVaeConfig::h3(),
        &SafetensorsDir::open(&a.model_dir.join("vae"))?,
        true,
        !a.skip_base,
        a.decoder_weights,
    )?;
    eprintln!("vae load {:.1}s", t0.elapsed().as_secs_f64());
    let t0 = Instant::now();
    let m = vae.encode_pixels(be, &px, a.tiling)?;
    let z = Posterior::from_moments(&m).mode();
    eprintln!(
        "encode {n} frames {w}x{h} -> latents {}x{}x{} in {:.2}s",
        z.t,
        z.h,
        z.w,
        t0.elapsed().as_secs_f64()
    );
    let base = if a.skip_base {
        None
    } else {
        let mut out = None;
        for r in 0..a.repeat.max(1) {
            let (v, s) = decode_all(&vae, be, &z, a.tiling)?;
            eprintln!("base decode {r}: {} frames in {s:.2}s", v.t);
            out = Some(v);
        }
        let v = out.unwrap();
        eprintln!("base vs input: {:.2} dB", psnr(&v, &px));
        Some(v)
    };
    if let Some(p) = &a.fast_vae {
        vae.decoder = None;
        let t0 = Instant::now();
        vae.fast = Some(FastDecoder::open(be, p, a.decoder_weights)?);
        eprintln!("fast vae load {:.1}s", t0.elapsed().as_secs_f64());
        let mut out = None;
        for r in 0..a.repeat.max(1) {
            let (v, s) = decode_all(&vae, be, &z, None)?;
            eprintln!("fast decode {r}: {} frames in {s:.2}s", v.t);
            out = Some(v);
        }
        let v = out.unwrap();
        eprintln!("fast vs input: {:.2} dB", psnr(&v, &px));
        if let Some(b) = &base {
            eprintln!("fast vs base: {:.2} dB", psnr(&v, b));
        }
        if let Some(d) = &a.out_dir {
            write(d, "fast", &v)?;
        }
    }
    if let (Some(d), Some(b)) = (&a.out_dir, &base) {
        write(d, "base", b)?;
    }
    Ok(())
}
