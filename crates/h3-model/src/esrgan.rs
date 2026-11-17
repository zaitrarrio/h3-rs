//! ×4 pixel upscaler for finished clips: Real-ESRGAN's compact network
//! (SRVGGNetCompact, checkpoint `realesr-animevideov3`, BSD-3), per frame,
//! ported from fastwan-rs's `upscale.rs` (whose real-weight output matches a
//! numpy SRVGGNetCompact exactly).
//!
//! conv3×3 3→F, PReLU → `convs` × (conv3×3 F→F, PReLU) → conv3×3 F→3·s²
//! → pixel-shuffle s, plus the input upsampled ×s (nearest).
//!
//! The convs (the PReLU is a [`ConvEpilogue`]) and the pixel shuffle run on
//! the backend; the residual add, clamp and rounding run on the host, row
//! parallel, against the exact 8-bit input. This is the fast path at every
//! tier and aspect (Regen-2K is the slow, diffusion-quality one). Weights:
//! fastwan-rs's `scripts/upscale/convert_esrgan.py`, 2.5 MB.

use h3_core::{
    Result,
    backend::{Backend, ConvEpilogue, ConvShape},
    weights::WeightSource,
};
use rayon::prelude::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EsrganConfig {
    /// Feature channels.
    pub feat: usize,
    /// Body convs after the first one.
    pub convs: usize,
    pub scale: usize,
}

impl EsrganConfig {
    /// `realesr-animevideov3`.
    pub fn anime_video_v3() -> Self {
        Self {
            feat: 64,
            convs: 16,
            scale: 4,
        }
    }

    pub fn tiny() -> Self {
        Self {
            feat: 8,
            convs: 2,
            scale: 2,
        }
    }

    fn layers(&self) -> Vec<(usize, usize, bool)> {
        let mut v = vec![(3, self.feat, true)];
        v.extend((0..self.convs).map(|_| (self.feat, self.feat, true)));
        v.push((self.feat, 3 * self.scale * self.scale, false));
        v
    }
}

/// Checkpoint keys and shapes (the `nn.Sequential` positions of `body`).
pub fn esrgan_manifest(cfg: &EsrganConfig) -> Vec<(String, Vec<usize>)> {
    let mut v = Vec::new();
    for (i, (cin, cout, prelu)) in cfg.layers().into_iter().enumerate() {
        v.push((format!("body.{}.weight", 2 * i), vec![cout, cin, 3, 3]));
        v.push((format!("body.{}.bias", 2 * i), vec![cout]));
        if prelu {
            v.push((format!("body.{}.weight", 2 * i + 1), vec![cout]));
        }
    }
    v
}

struct Layer<B: Backend> {
    cin: usize,
    cout: usize,
    w: B::Weight,
    b: B::Buf,
    slope: Option<B::Buf>,
}

pub struct Esrgan<B: Backend> {
    pub cfg: EsrganConfig,
    layers: Vec<Layer<B>>,
}

impl<B: Backend> Esrgan<B> {
    pub fn load(be: &B, cfg: EsrganConfig, src: &dyn WeightSource) -> Result<Self> {
        let mut layers = Vec::new();
        for (i, (cin, cout, prelu)) in cfg.layers().into_iter().enumerate() {
            let w = src.get(&format!("body.{}.weight", 2 * i), &[cout, cin, 3, 3])?;
            layers.push(Layer {
                cin,
                cout,
                w: be.weight(&w, cout, cin * 9),
                b: be.upload(&src.get(&format!("body.{}.bias", 2 * i), &[cout])?),
                slope: if prelu {
                    Some(be.upload(&src.get(&format!("body.{}.weight", 2 * i + 1), &[cout])?))
                } else {
                    None
                },
            });
        }
        Ok(Self { cfg, layers })
    }

    /// Frames per batch: the widest activation stays within the backend's
    /// buffer cap and about `budget_elems` elements (device memory), ≥ 1.
    pub fn batch(&self, be: &B, h: usize, w: usize, budget_elems: usize) -> usize {
        let per = self.cfg.feat.max(3 * self.cfg.scale * self.cfg.scale) * h * w;
        (be.max_buffer_elems().min(budget_elems) / per).max(1)
    }

    /// The network on `frames` planar RGB images in `[0, 1]` (`[frames, 3, h,
    /// w]`): every conv, then the pixel shuffle; `[frames, 3, h·s, w·s]` on
    /// the device, before the residual (added on the host by [`Esrgan::upscale`]).
    pub fn network(&self, be: &B, x: &[f32], n: usize, h: usize, w: usize) -> B::Buf {
        let (s, plane) = (self.cfg.scale, h * w);
        let input = be.upload(x);
        let feat = [
            be.alloc(n * self.cfg.feat * plane),
            be.alloc(n * self.cfg.feat * plane),
        ];
        let last = be.alloc(n * 3 * s * s * plane);
        let k = self.layers.len();
        for (i, l) in self.layers.iter().enumerate() {
            let dst = if i + 1 == k { &last } else { &feat[i % 2] };
            let shape = ConvShape {
                frames: n,
                cin_a: l.cin,
                cin_b: 0,
                cout: l.cout,
                h,
                w,
                upsample: false,
                k: 3,
            };
            let ep = match &l.slope {
                Some(sl) => ConvEpilogue::Prelu(sl),
                None => ConvEpilogue::None,
            };
            let inp = if i == 0 { &input } else { &feat[(i - 1) % 2] };
            be.conv2d(inp, None, &shape, &l.w, Some(&l.b), ep, dst);
        }
        let shuffled = be.alloc(n * 3 * s * s * plane);
        be.pixel_shuffle(&last, n, 3, h, w, s, &shuffled);
        shuffled
    }

    /// Upscale `frames` 8-bit RGB frames (`[frames, h, w, 3]`) to
    /// `[frames, h·s, w·s, 3]`, `batch` frames at a time; `on_batch(first,
    /// n, rgb)` receives each batch's output in order (so a caller can
    /// resample or write it without holding the whole ×4 clip).
    pub fn upscale(
        &self,
        be: &B,
        rgb: &[u8],
        frames: usize,
        h: usize,
        w: usize,
        batch: usize,
        on_batch: &mut dyn FnMut(usize, usize, Vec<u8>) -> Result<()>,
    ) -> Result<()> {
        assert_eq!(rgb.len(), frames * h * w * 3);
        let (s, plane) = (self.cfg.scale, h * w);
        let mut f0 = 0;
        while f0 < frames {
            let n = batch.min(frames - f0);
            let src = &rgb[f0 * plane * 3..(f0 + n) * plane * 3];
            let mut x = vec![0f32; n * 3 * plane];
            for f in 0..n {
                for p in 0..plane {
                    for c in 0..3 {
                        x[(f * 3 + c) * plane + p] = src[(f * plane + p) * 3 + c] as f32 / 255.0;
                    }
                }
            }
            let shuffled = self.network(be, &x, n, h, w);
            let y = be.download(&shuffled);
            let (oh, ow) = (h * s, w * s);
            let mut out = vec![0u8; n * oh * ow * 3];
            // + the nearest-upsampled 8-bit input, clamp, round; row parallel.
            out.par_chunks_mut(ow * 3).enumerate().for_each(|(row, o)| {
                let (f, yy) = (row / oh, row % oh);
                for xx in 0..ow {
                    for c in 0..3 {
                        let base = src[(f * plane + (yy / s) * w + xx / s) * 3 + c] as f32;
                        let v = y[((f * 3 + c) * oh + yy) * ow + xx] * 255.0 + base;
                        o[xx * 3 + c] = (v + 0.5).clamp(0.0, 255.0) as u8;
                    }
                }
            });
            on_batch(f0, n, out)?;
            f0 += n;
        }
        Ok(())
    }
}
