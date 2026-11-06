//! Open-H3-Regenerate-2K's latent upscaler: a 3-D convolutional ResNet with a
//! scale embedding around a trilinear resize of the latent grid
//! (`minimax_h3_latent_upscaler.py`, 345 M parameters).

use crate::vae::Volume;
use h3_core::{
    Error, Result,
    backend::{Act, Backend, Conv3dShape, ConvEpilogue},
    config::{H3_LATENTS_MEAN, H3_LATENTS_STD, UpscalerConfig},
    timestep::{matvec, silu},
    weights::WeightSource,
};

pub const UPSCALER_NORM_EPS: f32 = 1e-5;
pub const VAE_DOWNSAMPLE: usize = 16;
pub const PIXEL_ALIGNMENT: usize = 32;

struct Conv3<B: Backend> {
    w: B::Weight,
    b: B::Buf,
}

fn conv<B: Backend>(be: &B, src: &dyn WeightSource, key: &str, cout: usize, cin_g: usize, k: (usize, usize, usize)) -> Result<Conv3<B>> {
    let w = src.get(&format!("{key}.weight"), &[cout, cin_g, k.0, k.1, k.2])?;
    Ok(Conv3 { w: be.weight(&w, cout, cin_g * k.0 * k.1 * k.2), b: be.upload(&src.get(&format!("{key}.bias"), &[cout])?) })
}

struct Res<B: Backend> {
    in_norm: (B::Buf, B::Buf),
    in_conv: Conv3<B>,
    emb_w: Vec<f32>,
    emb_b: Vec<f32>,
    out_norm: (Vec<f32>, Vec<f32>),
    out_conv: Conv3<B>,
}

struct Temporal<B: Backend> {
    norm: (B::Buf, B::Buf),
    dw: Conv3<B>,
    pw: Conv3<B>,
}

enum Block<B: Backend> {
    Res(Res<B>),
    Temporal(Temporal<B>),
}

pub struct Upscaler<B: Backend> {
    pub cfg: UpscalerConfig,
    conv_in: Conv3<B>,
    embed: [(Vec<f32>, Vec<f32>); 2],
    in_blocks: Vec<Block<B>>,
    out_blocks: Vec<Block<B>>,
    norm_out: (B::Buf, B::Buf),
    conv_out: Conv3<B>,
    mean: Vec<f32>,
    std: Vec<f32>,
}

impl<B: Backend> Upscaler<B> {
    pub fn load(be: &B, cfg: UpscalerConfig, src: &dyn WeightSource) -> Result<Self> {
        let c = cfg.channels;
        let vecs = |k: &str, n: usize| -> Result<(Vec<f32>, Vec<f32>)> {
            Ok((src.get(&format!("{k}.weight"), &[n])?, src.get(&format!("{k}.bias"), &[n])?))
        };
        let norm = |k: &str| -> Result<(B::Buf, B::Buf)> {
            let (w, b) = vecs(k, c)?;
            Ok((be.upload(&w), be.upload(&b)))
        };
        let side = |name: &str, count: usize| -> Result<Vec<Block<B>>> {
            let mut v = Vec::new();
            for (i, temporal) in cfg.block_layout(count).into_iter().enumerate() {
                let p = format!("{name}.{i}");
                v.push(if temporal {
                    Block::Temporal(Temporal {
                        norm: norm(&format!("{p}.norm"))?,
                        dw: conv(be, src, &format!("{p}.dwconv"), c, 1, (cfg.temporal_kernel, 1, 1))?,
                        pw: conv(be, src, &format!("{p}.pwconv"), c, c, (1, 1, 1))?,
                    })
                } else {
                    let (ew, eb) = (src.get(&format!("{p}.emb_layers.1.weight"), &[2 * c, cfg.emb_dim])?, src.get(&format!("{p}.emb_layers.1.bias"), &[2 * c])?);
                    Block::Res(Res {
                        in_norm: norm(&format!("{p}.in_layers.0"))?,
                        in_conv: conv(be, src, &format!("{p}.in_layers.2"), c, c, (3, 3, 3))?,
                        emb_w: ew,
                        emb_b: eb,
                        out_norm: vecs(&format!("{p}.out_norm"), c)?,
                        out_conv: conv(be, src, &format!("{p}.out_layers.2"), c, c, (3, 3, 3))?,
                    })
                });
            }
            Ok(v)
        };
        let n = cfg.in_channels;
        Ok(Self {
            conv_in: conv(be, src, "conv_in", c, n, (3, 3, 3))?,
            embed: [
                (src.get("embed.0.weight", &[cfg.emb_dim, 1])?, src.get("embed.0.bias", &[cfg.emb_dim])?),
                (src.get("embed.2.weight", &[cfg.emb_dim, cfg.emb_dim])?, src.get("embed.2.bias", &[cfg.emb_dim])?),
            ],
            in_blocks: side("in_blocks", cfg.in_blocks)?,
            out_blocks: side("out_blocks", cfg.out_blocks)?,
            norm_out: norm("norm_out")?,
            conv_out: conv(be, src, "conv_out", n, c, (3, 3, 3))?,
            mean: if n == 24 { H3_LATENTS_MEAN.to_vec() } else { vec![0.0; n] },
            std: if n == 24 { H3_LATENTS_STD.to_vec() } else { vec![1.0; n] },
            cfg,
        })
    }

    fn run_blocks(&self, be: &B, blocks: &[Block<B>], mut cur: B::Buf, c: usize, t: usize, h: usize, w: usize, emb: &[f32]) -> B::Buf {
        let cfg = &self.cfg;
        let hw = h * w;
        let semb = silu(emb);
        for bl in blocks {
            match bl {
                Block::Res(r) => {
                    let n = be.alloc(c * t * hw);
                    be.group_norm(&cur, c, t, hw, cfg.norm_groups, false, &r.in_norm.0, &r.in_norm.1, UPSCALER_NORM_EPS, Act::Silu, &n);
                    let y = be.alloc(c * t * hw);
                    be.conv3d(&n, &Conv3dShape::same(c, c, t, h, w, 3), &r.in_conv.w, Some(&r.in_conv.b), ConvEpilogue::None, &y);
                    // out_norm · (1 + scale) + shift, then SiLU: folded into the norm's affine.
                    let ss = matvec(&r.emb_w, Some(&r.emb_b), &semb);
                    let (scale, shift) = ss.split_at(c);
                    let w2: Vec<f32> = r.out_norm.0.iter().zip(scale).map(|(w, s)| w * (1.0 + s)).collect();
                    let b2: Vec<f32> = r.out_norm.1.iter().zip(scale).zip(shift).map(|((b, s), sh)| b * (1.0 + s) + sh).collect();
                    let (gw, gb) = (be.upload(&w2), be.upload(&b2));
                    let n2 = be.alloc(c * t * hw);
                    be.group_norm(&y, c, t, hw, cfg.norm_groups, false, &gw, &gb, UPSCALER_NORM_EPS, Act::Silu, &n2);
                    let out = be.alloc(c * t * hw);
                    be.conv3d(&n2, &Conv3dShape::same(c, c, t, h, w, 3), &r.out_conv.w, Some(&r.out_conv.b), ConvEpilogue::Add(&cur), &out);
                    cur = out;
                }
                Block::Temporal(tc) => {
                    let n = be.alloc(c * t * hw);
                    be.group_norm(&cur, c, t, hw, cfg.norm_groups, false, &tc.norm.0, &tc.norm.1, UPSCALER_NORM_EPS, Act::Silu, &n);
                    let k = cfg.temporal_kernel;
                    let dws = Conv3dShape {
                        groups: c,
                        k: (k, 1, 1),
                        pad_t: (k / 2, k / 2),
                        pad_h: (0, 0),
                        pad_w: (0, 0),
                        ..Conv3dShape::same(c, c, t, h, w, 1)
                    };
                    let d = be.alloc(c * t * hw);
                    be.conv3d(&n, &dws, &tc.dw.w, Some(&tc.dw.b), ConvEpilogue::None, &d);
                    let out = be.alloc(c * t * hw);
                    be.conv3d(&d, &Conv3dShape::same(c, c, t, h, w, 1), &tc.pw.w, Some(&tc.pw.b), ConvEpilogue::Add(&cur), &out);
                    cur = out;
                }
            }
        }
        cur
    }

    /// `upscale_latent`: a VAE-normalised latent `[C, T, h, w]` to the latent
    /// grid of `target_pixels` `(height, width)`; the input is returned when
    /// the grid already matches.
    pub fn upscale(&self, be: &B, latent: &Volume, target_pixels: (usize, usize)) -> Result<Volume> {
        let ((h2, w2), scale) = resolve_geometry((latent.h, latent.w), target_pixels);
        if (h2, w2) == (latent.h, latent.w) {
            return Ok(latent.clone());
        }
        if scale < 1.0 {
            return Err(Error::Geometry("the latent upscaler only upscales (scale ≥ 1)".into()));
        }
        let cfg = &self.cfg;
        let (n, c, t) = (latent.c, cfg.channels, latent.t);
        let mut x = latent.clone();
        let inv: Vec<f32> = self.std.iter().map(|s| 1.0 / s).collect();
        let shift: Vec<f32> = self.mean.iter().zip(&self.std).map(|(m, s)| -m / s).collect();
        x.channel_affine(&inv, &shift);
        let e0 = silu(&matvec(&self.embed[0].0, Some(&self.embed[0].1), &[(scale - 1.0) as f32]));
        let emb = matvec(&self.embed[1].0, Some(&self.embed[1].1), &e0);
        let gx = be.upload(&x.data);
        let (h, w) = (latent.h, latent.w);
        let cur = be.alloc(c * t * h * w);
        be.conv3d(&gx, &Conv3dShape::same(n, c, t, h, w, 3), &self.conv_in.w, Some(&self.conv_in.b), ConvEpilogue::None, &cur);
        let cur = self.run_blocks(be, &self.in_blocks, cur, c, t, h, w, &emb);
        let up = be.alloc(c * t * h2 * w2);
        be.trilinear_resize(&cur, c, t, h, w, t, h2, w2, &up);
        let cur = self.run_blocks(be, &self.out_blocks, up, c, t, h2, w2, &emb);
        let nrm = be.alloc(c * t * h2 * w2);
        be.group_norm(&cur, c, t, h2 * w2, cfg.norm_groups, false, &self.norm_out.0, &self.norm_out.1, UPSCALER_NORM_EPS, Act::Silu, &nrm);
        let out = be.alloc(n * t * h2 * w2);
        be.conv3d(&nrm, &Conv3dShape::same(c, n, t, h2, w2, 3), &self.conv_out.w, Some(&self.conv_out.b), ConvEpilogue::None, &out);
        let mut v = Volume::from_data(n, t, h2, w2, be.download(&out));
        v.channel_affine(&self.std, &self.mean);
        Ok(v)
    }
}

/// `resolve_upscale_geometry`: the target latent `(h, w)` after 32-pixel
/// alignment, and the scale the network is conditioned on.
pub fn resolve_geometry(input: (usize, usize), target_pixels: (usize, usize)) -> ((usize, usize), f64) {
    let (h_in, w_in) = input;
    let (ht, wt) = target_pixels;
    let scale = (wt as f64 / (w_in * VAE_DOWNSAMPLE) as f64 + ht as f64 / (h_in * VAE_DOWNSAMPLE) as f64) / 2.0;
    let align = |p: usize| -> usize {
        let r = (p as f64 / PIXEL_ALIGNMENT as f64).round() * PIXEL_ALIGNMENT as f64;
        ((r / VAE_DOWNSAMPLE as f64).round() as usize).max(1)
    };
    ((align(ht), align(wt)), scale)
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3_core::weights::Synthetic;
    use h3_cpu::Cpu;

    #[test]
    fn geometry_and_tiny_forward() {
        assert_eq!(resolve_geometry((48, 84), (1440, 2560)), ((90, 160), (2560.0 / 1344.0 + 1440.0 / 768.0) / 2.0));
        let be = Cpu;
        let up = Upscaler::load(&be, UpscalerConfig::tiny(), &Synthetic { seed: 7 }).unwrap();
        let x = Volume::from_data(4, 2, 4, 4, (0..128).map(|i| (i % 7) as f32 * 0.1).collect());
        let y = up.upscale(&be, &x, (96, 128)).unwrap();
        assert_eq!((y.c, y.t, y.h, y.w), (4, 2, 6, 8));
        assert!(y.data.iter().all(|v| v.is_finite()));
        assert_eq!(up.upscale(&be, &x, (64, 64)).unwrap(), x);
    }
}
