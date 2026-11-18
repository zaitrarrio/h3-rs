//! What happens to a clip's decoded frames before they leave the engine,
//! shared by `h3 generate` and `h3 serve`: the centre crop to the requested
//! size (sizes off the 32-pixel grid, e.g. 480×272 rendered at 480×288)
//! and the Real-ESRGAN upscale (×4, or ×2 / ×3 by resampling the ×4 frames
//! like Real-ESRGAN's `--outscale`).

use anyhow::{Result, bail};
use h3_core::Backend;
use h3_model::esrgan::Esrgan;
use h3_text::image::{Filter, Image};

/// The canvas for a requested size: each edge rounded up to the 32-pixel
/// grid H3 needs. The decoded frames are then cropped back.
pub fn canvas_edge(v: usize) -> usize {
    v.div_ceil(h3_core::packing::CANVAS_MULTIPLE) * h3_core::packing::CANVAS_MULTIPLE
}

/// Centre crop of `frames` RGB frames `w×h` to `ow×oh`.
pub fn crop(rgb: &[u8], frames: usize, w: usize, h: usize, ow: usize, oh: usize) -> Vec<u8> {
    assert!(ow <= w && oh <= h && rgb.len() == frames * w * h * 3);
    if (ow, oh) == (w, h) {
        return rgb.to_vec();
    }
    let (l, t) = ((w - ow) / 2, (h - oh) / 2);
    let mut out = Vec::with_capacity(frames * ow * oh * 3);
    for f in 0..frames {
        for y in t..t + oh {
            let row = (f * h + y) * w;
            out.extend_from_slice(&rgb[(row + l) * 3..(row + l + ow) * 3]);
        }
    }
    out
}

/// `h3 generate`'s default: the network's ×4 up to 480p, ×2 above.
pub fn cli_factor(w: usize, h: usize) -> usize {
    if w * h <= 832 * 480 { 4 } else { 2 }
}

/// Largest output `h3 serve` picks by default: frames go to a browser, which
/// holds a clip's worth of them.
pub const SERVE_MAX_OUT_PIXELS: usize = 2560 * 1440;

/// `h3 serve`'s default: the largest factor ≤ the network's whose output
/// fits [`SERVE_MAX_OUT_PIXELS`], at least 2 (480×288 → ×4 1920×1152,
/// 832×480 → ×3 2496×1440, 1344×768 → ×2 2688×1536).
pub fn serve_factor(w: usize, h: usize, scale: usize) -> usize {
    (2..=scale)
        .rev()
        .find(|k| w * k * h * k <= SERVE_MAX_OUT_PIXELS)
        .unwrap_or(2)
}

/// Upscale `frames` 8-bit RGB frames `w×h` by `factor` (2 ..= the network's
/// scale). `on_batch(first, n, rgb)` receives each batch's `n` frames at
/// `w·factor × h·factor`, in order.
#[allow(clippy::too_many_arguments)]
pub fn upscale<B: Backend>(
    be: &B,
    up: &Esrgan<B>,
    rgb: &[u8],
    frames: usize,
    w: usize,
    h: usize,
    factor: usize,
    on_batch: &mut dyn FnMut(usize, usize, Vec<u8>) -> Result<()>,
) -> Result<usize> {
    let s = up.cfg.scale;
    if !(2..=s).contains(&factor) {
        bail!("the upscale factor is 2 to {s}, got {factor}");
    }
    let (tw, th) = (w * factor, h * factor);
    let batch = up.batch(be, h, w, 1 << 28);
    let err = |e: anyhow::Error| h3_core::Error::Msg(format!("{e:#}"));
    up.upscale(be, rgb, frames, h, w, batch, &mut |f0, n, xs| {
        if factor == s {
            return on_batch(f0, n, xs).map_err(err);
        }
        let mut out = vec![0u8; n * tw * th * 3];
        let (sw, sh) = (w * s, h * s);
        std::thread::scope(|sc| {
            for (f, d) in out.chunks_mut(tw * th * 3).enumerate() {
                let fr = &xs[f * sw * sh * 3..(f + 1) * sw * sh * 3];
                sc.spawn(move || {
                    let im = Image::new(sw, sh, fr.to_vec());
                    d.copy_from_slice(&im.resize(tw, th, Filter::Lanczos).rgb);
                });
            }
        });
        on_batch(f0, n, out).map_err(err)
    })?;
    Ok(batch)
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3_core::weights::Synthetic;
    use h3_model::esrgan::EsrganConfig;

    #[test]
    fn crop_takes_the_centre() {
        // 2 frames of 4×3, pixel value = frame·100 + y·10 + x.
        let (w, h) = (4, 3);
        let rgb: Vec<u8> = (0..2)
            .flat_map(|f| {
                (0..h)
                    .flat_map(move |y| (0..w).flat_map(move |x| [(f * 100 + y * 10 + x) as u8; 3]))
            })
            .collect();
        let c = crop(&rgb, 2, w, h, 2, 1);
        let px: Vec<u8> = c.chunks(3).map(|p| p[0]).collect();
        assert_eq!(px, vec![11, 12, 111, 112]);
        assert_eq!(crop(&rgb, 2, w, h, w, h), rgb);
    }

    #[test]
    fn canvases_and_factors() {
        assert_eq!((canvas_edge(480), canvas_edge(272)), (480, 288));
        assert_eq!(canvas_edge(768), 768);
        assert_eq!(serve_factor(480, 288, 4), 4);
        assert_eq!(serve_factor(288, 480, 4), 4);
        assert_eq!(serve_factor(832, 480, 4), 3);
        assert_eq!(serve_factor(1344, 768, 4), 2);
        assert_eq!(serve_factor(480, 288, 2), 2);
        assert_eq!(
            (
                cli_factor(480, 288),
                cli_factor(832, 480),
                cli_factor(1344, 768)
            ),
            (4, 4, 2)
        );
    }

    #[test]
    fn upscale_resamples_to_the_factor() {
        let be = h3_cpu::Cpu;
        let mut cfg = EsrganConfig::tiny();
        cfg.scale = 4;
        let up = Esrgan::load(&be, cfg, &Synthetic { seed: 3 }).unwrap();
        let (w, h, frames) = (6, 4, 3);
        let rgb: Vec<u8> = (0..frames * w * h * 3)
            .map(|i| (i * 7 % 251) as u8)
            .collect();
        for factor in [4, 3, 2] {
            let mut got = Vec::new();
            upscale(&be, &up, &rgb, frames, w, h, factor, &mut |f0, n, x| {
                assert_eq!(f0, got.len() / (w * factor * h * factor * 3));
                assert_eq!(x.len(), n * w * factor * h * factor * 3);
                got.extend(x);
                Ok(())
            })
            .unwrap();
            assert_eq!(got.len(), frames * w * factor * h * factor * 3);
        }
        assert!(upscale(&be, &up, &rgb, frames, w, h, 5, &mut |_, _, _| Ok(())).is_err());
    }
}
