//! `AutoencoderKLMiniMaxH3`: causal 3-D CNN encoder (17 pixel frames → 5
//! latent frames, 16× spatial) and the ViT decoder (7 latent frames → 28
//! pixel frames per chunk, blended over 5). Both run on the backend, one
//! clip or tile at a time; stitching and frame blending happen on the host in
//! f32, exactly as the reference does.

use crate::linear::{Linear, load_plain};
use rayon::prelude::*;
use h3_core::{
    Error, Result,
    backend::{Act, Backend, Conv3dShape, ConvEpilogue, Gate, HeadNorm, Modulation, Pad, Rope},
    config::{PIXEL_MEAN, PIXEL_STD, VideoVaeConfig},
    rng::Rng,
    rope::vae_decoder_rope,
    weights::WeightSource,
};

/// A host `[c, t, h, w]` f32 volume.
#[derive(Debug, Clone, PartialEq)]
pub struct Volume {
    pub c: usize,
    pub t: usize,
    pub h: usize,
    pub w: usize,
    pub data: Vec<f32>,
}

impl Volume {
    pub fn zeros(c: usize, t: usize, h: usize, w: usize) -> Self {
        Self { c, t, h, w, data: vec![0.0; c * t * h * w] }
    }
    pub fn from_data(c: usize, t: usize, h: usize, w: usize, data: Vec<f32>) -> Self {
        assert_eq!(data.len(), c * t * h * w);
        Self { c, t, h, w, data }
    }
    #[inline]
    pub fn at(&self, c: usize, t: usize, y: usize, x: usize) -> f32 {
        self.data[((c * self.t + t) * self.h + y) * self.w + x]
    }
    /// Sub-volume `[t0..t0+nt, y0..y0+nh, x0..x0+nw]`.
    pub fn crop(&self, t0: usize, nt: usize, y0: usize, nh: usize, x0: usize, nw: usize) -> Volume {
        assert!(t0 + nt <= self.t && y0 + nh <= self.h && x0 + nw <= self.w, "crop out of range");
        if (t0, nt, y0, nh, x0, nw) == (0, self.t, 0, self.h, 0, self.w) {
            return self.clone();
        }
        let mut o = Volume::zeros(self.c, nt, nh, nw);
        let (st, sh, sw) = (self.t, self.h, self.w);
        o.data.par_chunks_mut(nw).enumerate().for_each(|(r, row)| {
            let (y, r) = (r % nh, r / nh);
            let (t, c) = (r % nt, r / nt);
            let src = ((c * st + t0 + t) * sh + y0 + y) * sw + x0;
            row.copy_from_slice(&self.data[src..src + nw]);
        });
        o
    }
    /// Concatenate along the time axis.
    pub fn cat_t(parts: &[Volume]) -> Volume {
        let (c, h, w) = (parts[0].c, parts[0].h, parts[0].w);
        let t: usize = parts.iter().map(|p| p.t).sum();
        let mut o = Volume::zeros(c, t, h, w);
        let mut t0 = 0;
        for p in parts {
            assert_eq!((p.c, p.h, p.w), (c, h, w));
            for cc in 0..c {
                let src = cc * p.t * h * w;
                let dst = (cc * t + t0) * h * w;
                o.data[dst..dst + p.t * h * w].copy_from_slice(&p.data[src..src + p.t * h * w]);
            }
            t0 += p.t;
        }
        o
    }
    /// Repeat the last frame `n` times.
    pub fn pad_last_frame(&self, n: usize) -> Volume {
        if n == 0 {
            return self.clone();
        }
        let last = self.crop(self.t - 1, 1, 0, self.h, 0, self.w);
        let mut parts = vec![self.clone()];
        parts.extend(std::iter::repeat_n(last, n));
        Volume::cat_t(&parts)
    }
    pub fn channel_affine(&mut self, scale: &[f32], shift: &[f32]) {
        let inner = self.t * self.h * self.w;
        for c in 0..self.c {
            for v in &mut self.data[c * inner..(c + 1) * inner] {
                *v = *v * scale[c] + shift[c];
            }
        }
    }
}

/// Axis of a blend / stitch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Axis {
    T,
    H,
    W,
}

/// Linear crossfade of `a`'s last `extent` with `b`'s first `extent` along
/// `axis`, followed by the rest of `b` (`AutoencoderKLMiniMaxH3._blend`).
pub fn blend(a: &Volume, b: &Volume, extent: usize, axis: Axis) -> Volume {
    let len = |v: &Volume| match axis {
        Axis::T => v.t,
        Axis::H => v.h,
        Axis::W => v.w,
    };
    assert_eq!(a.c, b.c, "blend: channel counts differ");
    let extent = extent.min(len(a)).min(len(b));
    let mut out = b.clone();
    if extent == 0 {
        return out;
    }
    let la = len(a);
    let (bt, bh, bw) = (b.t, b.h, b.w);
    // One task per output row [c, t, y]; a row is untouched unless its position
    // along `axis` lies in the crossfade.
    out.data.par_chunks_mut(bw).enumerate().for_each(|(r, row)| {
        let (y, r) = (r % bh, r / bh);
        let (t, c) = (r % bt, r / bt);
        match axis {
            Axis::T if t < extent => {
                let wb = t as f32 / extent as f32;
                let src = ((c * a.t + la - extent + t) * a.h + y) * a.w;
                for (o, &av) in row.iter_mut().zip(&a.data[src..src + bw]) {
                    *o = av * (1.0 - wb) + *o * wb;
                }
            }
            Axis::H if y < extent => {
                let wb = y as f32 / extent as f32;
                let src = ((c * a.t + t) * a.h + la - extent + y) * a.w;
                for (o, &av) in row.iter_mut().zip(&a.data[src..src + bw]) {
                    *o = av * (1.0 - wb) + *o * wb;
                }
            }
            Axis::W => {
                let src = ((c * a.t + t) * a.h + y) * a.w + la - extent;
                for (x, o) in row.iter_mut().enumerate().take(extent) {
                    let wb = x as f32 / extent as f32;
                    *o = a.data[src + x] * (1.0 - wb) + *o * wb;
                }
            }
            _ => {}
        }
    });
    out
}

/// `_split_tiles`: tile start indices, lengths and overlaps along one axis.
pub fn split_tiles(length: usize, tile: usize, min_overlap: usize, ratio: usize) -> (Vec<usize>, Vec<usize>, Vec<usize>) {
    if tile >= length {
        return (vec![0], vec![length], vec![]);
    }
    let mut n = length.div_ceil(tile);
    while ((tile * n) as isize - (min_overlap * (n - 1)) as isize - length as isize) < 0 {
        n += 1;
    }
    let mut overlaps = vec![min_overlap; n - 1];
    let remaining = tile * n - overlaps.iter().sum::<usize>() - length;
    for i in 0..remaining / ratio {
        overlaps[i % (n - 1)] += ratio;
    }
    let mut starts = vec![0];
    for i in 0..n - 1 {
        starts.push(starts[i] + tile - overlaps[i]);
    }
    (starts, vec![tile; n], overlaps)
}

/// `_stitch_tiles`: blend overlaps and concatenate a grid of tiles.
///
/// Tile `(ri, ci)` is crossfaded along H with the raw tile above over
/// `h_overlaps[ri − 1]` rows, then along W with the raw tile to its left over
/// `w_overlaps[ci − 1]` columns, then loses its own trailing overlap (except
/// on the last row / column). Computed in one row-parallel pass straight
/// into the output; `stitch_tiles_reference` in the tests is the sequential
/// form.
pub fn stitch_tiles(tiles: &[Vec<Volume>], h_overlaps: &[usize], w_overlaps: &[usize]) -> Volume {
    let (nr, nc) = (tiles.len(), tiles[0].len());
    let (c, t) = (tiles[0][0].c, tiles[0][0].t);
    assert!(tiles.iter().all(|r| r.len() == nc), "ragged tile grid");
    assert!(h_overlaps.len() + 1 >= nr && w_overlaps.len() + 1 >= nc, "missing overlaps");
    // Kept extent of each tile row / column and its start in the output.
    let kept_h: Vec<usize> = (0..nr).map(|ri| tiles[ri][0].h - if ri + 1 < nr { h_overlaps[ri] } else { 0 }).collect();
    let kept_w: Vec<usize> = (0..nc).map(|ci| tiles[0][ci].w - if ci + 1 < nc { w_overlaps[ci] } else { 0 }).collect();
    let mut y_start = vec![0; nr + 1];
    for ri in 0..nr {
        y_start[ri + 1] = y_start[ri] + kept_h[ri];
    }
    let mut x_start = vec![0; nc + 1];
    for ci in 0..nc {
        x_start[ci + 1] = x_start[ci] + kept_w[ci];
    }
    let (h, w) = (y_start[nr], x_start[nc]);
    for row in tiles {
        for (ci, tile) in row.iter().enumerate() {
            assert_eq!((tile.c, tile.t, tile.h, tile.w), (c, t, row[0].h, tiles[0][ci].w), "tile shapes disagree");
        }
    }
    let mut o = Volume::zeros(c, t, h, w);
    o.data.par_chunks_mut(w).enumerate().for_each(|(r, out_row)| {
        let (y, r) = (r % h, r / h);
        let (tt, cc) = (r % t, r / t);
        let ri = y_start[1..].iter().position(|&s| y < s).unwrap();
        let ly = y - y_start[ri];
        let hov = if ri > 0 { h_overlaps[ri - 1].min(tiles[ri - 1][0].h).min(tiles[ri][0].h) } else { 0 };
        let wy = if ly < hov { Some(ly as f32 / hov as f32) } else { None };
        for ci in 0..nc {
            let tile = &tiles[ri][ci];
            let dst = &mut out_row[x_start[ci]..x_start[ci + 1]];
            let n = dst.len();
            let src = ((cc * t + tt) * tile.h + ly) * tile.w;
            dst.copy_from_slice(&tile.data[src..src + n]);
            if let Some(wy) = wy {
                let above = &tiles[ri - 1][ci];
                let asrc = ((cc * t + tt) * above.h + above.h - hov + ly) * above.w;
                for (o, &av) in dst.iter_mut().zip(&above.data[asrc..asrc + n]) {
                    *o = av * (1.0 - wy) + *o * wy;
                }
            }
            if ci > 0 {
                let left = &tiles[ri][ci - 1];
                let wov = w_overlaps[ci - 1].min(left.w).min(tile.w).min(n);
                let lsrc = ((cc * t + tt) * left.h + ly) * left.w + left.w - wov;
                for (x, o) in dst.iter_mut().enumerate().take(wov) {
                    let wx = x as f32 / wov as f32;
                    *o = left.data[lsrc + x] * (1.0 - wx) + *o * wx;
                }
            }
        }
    });
    o
}

/// Spatial tiling in pixels (the release default: 256-pixel tiles, 64
/// overlap).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Tiling {
    pub tile: usize,
    pub overlap: usize,
}

impl Tiling {
    pub const DEFAULT: Tiling = Tiling { tile: 256, overlap: 64 };
}

/// The encoder's diagonal Gaussian: `[latent, …]` mean and log-variance
/// halves of the `[2·latent, …]` moments.
pub struct Posterior {
    pub mean: Volume,
    pub logvar: Volume,
}

impl Posterior {
    pub fn from_moments(m: &Volume) -> Self {
        let l = m.c / 2;
        let inner = m.t * m.h * m.w;
        let mean = Volume::from_data(l, m.t, m.h, m.w, m.data[..l * inner].to_vec());
        let logvar = Volume::from_data(l, m.t, m.h, m.w, m.data[l * inner..].iter().map(|v| v.clamp(-30.0, 20.0)).collect());
        Self { mean, logvar }
    }
    pub fn mode(&self) -> Volume {
        self.mean.clone()
    }
    /// `mean + exp(logvar / 2) · ε` with `ε` from `rng`.
    pub fn sample(&self, rng: &mut Rng) -> Volume {
        let eps = rng.normal(self.mean.data.len());
        let mut v = self.mean.clone();
        for ((x, &lv), e) in v.data.iter_mut().zip(&self.logvar.data).zip(eps) {
            *x += (0.5 * lv).exp() * e;
        }
        v
    }
}

struct Conv<B: Backend> {
    w: B::Weight,
    b: B::Buf,
    cin: usize,
    cout: usize,
    k: usize,
}

fn load_conv<B: Backend>(be: &B, src: &dyn WeightSource, key: &str, cout: usize, cin: usize, k: usize) -> Result<Conv<B>> {
    let w = src.get(&format!("{key}.weight"), &[cout, cin, k, k, k])?;
    Ok(Conv { w: be.weight(&w, cout, cin * k * k * k), b: be.upload(&src.get(&format!("{key}.bias"), &[cout])?), cin, cout, k })
}

struct Norm<B: Backend> {
    w: B::Buf,
    b: B::Buf,
}

fn load_norm<B: Backend>(be: &B, src: &dyn WeightSource, key: &str, c: usize) -> Result<Norm<B>> {
    Ok(Norm { w: be.upload(&src.get(&format!("{key}.weight"), &[c])?), b: be.upload(&src.get(&format!("{key}.bias"), &[c])?) })
}

struct Resnet<B: Backend> {
    norm1: Norm<B>,
    conv1: Conv<B>,
    norm2: Norm<B>,
    conv2: Conv<B>,
    shortcut: Option<Conv<B>>,
}

struct DownBlock<B: Backend> {
    resnets: Vec<Resnet<B>>,
    down: Option<(Conv<B>, usize, usize)>,
}

pub struct Encoder<B: Backend> {
    conv_in: Conv<B>,
    blocks: Vec<DownBlock<B>>,
    norm_out: Norm<B>,
    conv_out: Conv<B>,
    quant_conv: Conv<B>,
}

struct DecBlock<B: Backend> {
    norm1: B::Buf,
    qkv: Linear<B>,
    out: Linear<B>,
    scale1: B::Buf,
    norm2: B::Buf,
    ff_in: Linear<B>,
    ff_out: Linear<B>,
    scale2: B::Buf,
}

pub struct Decoder<B: Backend> {
    post_quant_conv: Conv<B>,
    proj_in: Linear<B>,
    register_tokens: Vec<f32>,
    blocks: Vec<DecBlock<B>>,
    norm_out: Norm<B>,
    proj_out: Linear<B>,
}

pub struct VideoVae<B: Backend> {
    pub cfg: VideoVaeConfig,
    pub encoder: Option<Encoder<B>>,
    pub decoder: Option<Decoder<B>>,
}

/// Device-side scratch: allocate as needed (the VAE runs one tile at a time).
struct Dev<'a, B: Backend> {
    be: &'a B,
}

impl<'a, B: Backend> Dev<'a, B> {
    fn conv(&self, x: &B::Buf, shape: &Conv3dShape, c: &Conv<B>, ep: ConvEpilogue<'_, B::Buf>) -> B::Buf {
        assert_eq!((shape.cin, shape.cout, shape.k.0), (c.cin, c.cout, c.k));
        let out = self.be.alloc(shape.out_len());
        self.be.conv3d(x, shape, &c.w, Some(&c.b), ep, &out);
        out
    }
}

impl<B: Backend> VideoVae<B> {
    pub fn load(be: &B, cfg: VideoVaeConfig, src: &dyn WeightSource, encoder: bool, decoder: bool) -> Result<Self> {
        let enc = if encoder { Some(Self::load_encoder(be, &cfg, src)?) } else { None };
        let dec = if decoder { Some(Self::load_decoder(be, &cfg, src)?) } else { None };
        Ok(Self { cfg, encoder: enc, decoder: dec })
    }

    fn load_encoder(be: &B, cfg: &VideoVaeConfig, src: &dyn WeightSource) -> Result<Encoder<B>> {
        let c0 = cfg.block_out_channels[0];
        let mut blocks = Vec::new();
        let mut cin = c0;
        for (i, &cout) in cfg.block_out_channels.iter().enumerate() {
            let mut resnets = Vec::new();
            for j in 0..cfg.layers_per_block {
                let p = format!("encoder.down_blocks.{i}.resnets.{j}");
                let ci = if j == 0 { cin } else { cout };
                resnets.push(Resnet {
                    norm1: load_norm(be, src, &format!("{p}.norm1"), ci)?,
                    conv1: load_conv(be, src, &format!("{p}.conv1"), cout, ci, 3)?,
                    norm2: load_norm(be, src, &format!("{p}.norm2"), cout)?,
                    conv2: load_conv(be, src, &format!("{p}.conv2"), cout, cout, 3)?,
                    shortcut: if ci != cout { Some(load_conv(be, src, &format!("{p}.conv_shortcut"), cout, ci, 1)?) } else { None },
                });
            }
            let down = if cfg.spatial_down[i] > 1 || cfg.temporal_down[i] > 1 {
                Some((
                    load_conv(be, src, &format!("encoder.down_blocks.{i}.downsamplers.0.conv"), cout, cout, 3)?,
                    cfg.temporal_down[i],
                    cfg.spatial_down[i],
                ))
            } else {
                None
            };
            blocks.push(DownBlock { resnets, down });
            cin = cout;
        }
        let lz = cfg.latent_channels;
        Ok(Encoder {
            conv_in: load_conv(be, src, "encoder.conv_in", c0, cfg.in_channels, 3)?,
            blocks,
            norm_out: load_norm(be, src, "encoder.norm_out", cin)?,
            conv_out: load_conv(be, src, "encoder.conv_out", 2 * lz, cin, 3)?,
            quant_conv: load_conv(be, src, "quant_conv", 2 * lz, 2 * lz, 1)?,
        })
    }

    fn load_decoder(be: &B, cfg: &VideoVaeConfig, src: &dyn WeightSource) -> Result<Decoder<B>> {
        let dd = cfg.decoder_dim();
        let ff = cfg.decoder_ffn_mult * dd;
        let lz = cfg.latent_channels;
        let mut blocks = Vec::new();
        for l in 0..cfg.decoder_layers {
            let p = format!("decoder.transformer_blocks.{l}");
            let mut qkv_w = Vec::with_capacity(3 * dd * dd);
            let mut qkv_b = Vec::with_capacity(3 * dd);
            for m in ["to_q", "to_k", "to_v"] {
                qkv_w.extend(src.get(&format!("{p}.attn.{m}.weight"), &[dd, dd])?);
                qkv_b.extend(src.get(&format!("{p}.attn.{m}.bias"), &[dd])?);
            }
            blocks.push(DecBlock {
                norm1: be.upload(&src.get(&format!("{p}.norm1.weight"), &[dd])?),
                qkv: Linear { w: be.weight(&qkv_w, 3 * dd, dd), b: Some(be.upload(&qkv_b)), rows: 3 * dd, cols: dd },
                out: load_plain(be, src, &format!("{p}.attn.to_out.0"), dd, dd, true)?,
                scale1: be.upload(&src.get(&format!("{p}.scale1"), &[dd])?),
                norm2: be.upload(&src.get(&format!("{p}.norm2.weight"), &[dd])?),
                ff_in: load_plain(be, src, &format!("{p}.ff.net.0.proj"), 2 * ff, dd, true)?,
                ff_out: load_plain(be, src, &format!("{p}.ff.net.2"), dd, ff, true)?,
                scale2: be.upload(&src.get(&format!("{p}.scale2"), &[dd])?),
            });
        }
        let out_dim = cfg.in_channels * cfg.temporal_ratio() * cfg.spatial_ratio() * cfg.spatial_ratio();
        Ok(Decoder {
            post_quant_conv: load_conv(be, src, "post_quant_conv", lz, lz, 1)?,
            proj_in: load_plain(be, src, "decoder.proj_in", dd, lz, true)?,
            register_tokens: src.get("decoder.register_tokens", &[1, cfg.decoder_register_tokens, dd])?,
            blocks,
            norm_out: load_norm(be, src, "decoder.norm_out", dd)?,
            proj_out: load_plain(be, src, "decoder.proj_out", out_dim, dd, true)?,
        })
    }

    pub fn spatial_ratio(&self) -> usize {
        self.cfg.spatial_ratio()
    }

    /// Latents `(z − mean) / std`.
    pub fn normalize_latents(&self, z: &mut Volume) {
        let scale: Vec<f32> = self.cfg.latents_std.iter().map(|s| 1.0 / s).collect();
        let shift: Vec<f32> = self.cfg.latents_mean.iter().zip(&self.cfg.latents_std).map(|(m, s)| -m / s).collect();
        z.channel_affine(&scale, &shift);
    }
    pub fn denormalize_latents(&self, z: &mut Volume) {
        z.channel_affine(&self.cfg.latents_std, &self.cfg.latents_mean);
    }

    /// Encoder moments `[2·latent, t, h, w]` of one ImageNet-normalised clip
    /// `[3, T, H, W]` (T ≤ 17, or 1 for a keyframe), one spatial tile.
    pub fn encode_clip_tile(&self, be: &B, x: &Volume) -> Result<Volume> {
        let enc = self.encoder.as_ref().ok_or_else(|| Error::Msg("the video VAE encoder is not loaded".into()))?;
        let cfg = &self.cfg;
        let dev = Dev { be };
        let (mut t, mut h, mut w) = (x.t, x.h, x.w);
        let gx = be.upload(&x.data);
        let mut cur = dev.conv(&gx, &Conv3dShape::causal(enc.conv_in.cin, enc.conv_in.cout, t, h, w, 3), &enc.conv_in, ConvEpilogue::None);
        let mut c = enc.conv_in.cout;
        for blk in &enc.blocks {
            for r in &blk.resnets {
                let hw = h * w;
                let n = be.alloc(c * t * hw);
                be.group_norm(&cur, c, t, hw, cfg.norm_groups, true, &r.norm1.w, &r.norm1.b, cfg.norm_eps, Act::Silu, &n);
                let y = dev.conv(&n, &Conv3dShape::causal(c, r.conv1.cout, t, h, w, 3), &r.conv1, ConvEpilogue::None);
                let co = r.conv1.cout;
                let n2 = be.alloc(co * t * hw);
                be.group_norm(&y, co, t, hw, cfg.norm_groups, true, &r.norm2.w, &r.norm2.b, cfg.norm_eps, Act::Silu, &n2);
                let res = match &r.shortcut {
                    Some(sc) => {
                        let shape = Conv3dShape { pad_t: (0, 0), pad_h: (0, 0), pad_w: (0, 0), ..Conv3dShape::causal(c, co, t, h, w, 1) };
                        dev.conv(&cur, &shape, sc, ConvEpilogue::None)
                    }
                    None => cur,
                };
                cur = dev.conv(&n2, &Conv3dShape::causal(co, co, t, h, w, 3), &r.conv2, ConvEpilogue::Add(&res));
                c = co;
            }
            if let Some((conv, ts, ss)) = &blk.down {
                let shape = Conv3dShape {
                    cin: c,
                    cout: c,
                    groups: 1,
                    t,
                    h,
                    w,
                    k: (3, 3, 3),
                    stride: (*ts, *ss, *ss),
                    pad_t: (2, 0),
                    pad_h: (0, if *ss == 2 { 1 } else { 0 }),
                    pad_w: (0, if *ss == 2 { 1 } else { 0 }),
                    spatial_pad: Pad::Reflect,
                };
                if *ss == 2 && (h < 2 || w < 2) {
                    return Err(Error::Geometry(format!("a {h}x{w} map cannot be downsampled")));
                }
                let (t2, h2, w2) = shape.out_dims();
                cur = dev.conv(&cur, &shape, conv, ConvEpilogue::None);
                (t, h, w) = (t2, h2, w2);
            }
        }
        let hw = h * w;
        let n = be.alloc(c * t * hw);
        be.group_norm(&cur, c, t, hw, cfg.norm_groups, true, &enc.norm_out.w, &enc.norm_out.b, cfg.norm_eps, Act::Silu, &n);
        let m = dev.conv(&n, &Conv3dShape::causal(c, enc.conv_out.cout, t, h, w, 3), &enc.conv_out, ConvEpilogue::None);
        let q = {
            let lz2 = enc.conv_out.cout;
            let shape = Conv3dShape { pad_t: (0, 0), pad_h: (0, 0), pad_w: (0, 0), ..Conv3dShape::causal(lz2, lz2, t, h, w, 1) };
            dev.conv(&m, &shape, &enc.quant_conv, ConvEpilogue::None)
        };
        let mut data = be.download(&q);
        data.truncate(enc.conv_out.cout * t * h * w);
        Ok(Volume::from_data(enc.conv_out.cout, t, h, w, data))
    }

    /// Moments of one clip with optional spatial tiling (`_encode_clip`).
    pub fn encode_clip(&self, be: &B, x: &Volume, tiling: Option<Tiling>) -> Result<Volume> {
        let Some(tl) = tiling else { return self.encode_clip_tile(be, x) };
        let r = self.spatial_ratio();
        let (ys, yl, yo) = split_tiles(x.h, tl.tile, tl.overlap, r);
        let (xs, xl, xo) = split_tiles(x.w, tl.tile, tl.overlap, r);
        let mut rows = Vec::new();
        for (y0, ny) in ys.iter().zip(&yl) {
            let mut row = Vec::new();
            for (x0, nx) in xs.iter().zip(&xl) {
                row.push(self.encode_clip_tile(be, &x.crop(0, x.t, *y0, *ny, *x0, *nx))?);
            }
            rows.push(row);
        }
        let yo: Vec<usize> = yo.iter().map(|o| o / r).collect();
        let xo: Vec<usize> = xo.iter().map(|o| o / r).collect();
        Ok(stitch_tiles(&rows, &yo, &xo))
    }

    /// Moments of a video: pixels `[3, T, H, W]` in `[0, 1]`, normalised with
    /// the ImageNet statistics, encoded per 17-frame clip (the last clip padded
    /// with its last frame), the final 3 latent frames dropped
    /// (`encode_pixels`). Returns `[2·latent, 5·clips − 3, h, w]`.
    pub fn encode_pixels(&self, be: &B, pixels: &Volume, tiling: Option<Tiling>) -> Result<Volume> {
        let cl = self.cfg.clip_length;
        let mut moments = Vec::new();
        let mut f0 = 0;
        while f0 < pixels.t {
            let n = cl.min(pixels.t - f0);
            let mut clip = pixels.crop(f0, n, 0, pixels.h, 0, pixels.w).pad_last_frame(cl - n);
            normalize_pixels(&mut clip);
            moments.push(self.encode_clip(be, &clip, tiling)?);
            f0 += n;
        }
        let m = Volume::cat_t(&moments);
        let keep = m.t - self.cfg.token_drop;
        Ok(m.crop(0, keep, 0, m.h, 0, m.w))
    }

    /// Moments of one keyframe `[3, 1, H, W]` in `[0, 1]` (`encode_keyframe`:
    /// no clip padding, one latent frame).
    pub fn encode_keyframe(&self, be: &B, pixels: &Volume, tiling: Option<Tiling>) -> Result<Volume> {
        if pixels.t != 1 {
            return Err(Error::Geometry("a keyframe has exactly one frame".into()));
        }
        let mut x = pixels.clone();
        normalize_pixels(&mut x);
        self.encode_clip(be, &x, tiling)
    }

    /// Decode one latent clip tile `[latent, Tc, h, w]` (denormalised) to
    /// ImageNet-normalised pixels `[3, 4·Tc, 16·h, 16·w]`.
    pub fn decode_clip_tile(&self, be: &B, z: &Volume) -> Result<Volume> {
        let (pix, (c, t, h, w)) = self.decode_clip_tile_dev(be, z)?;
        Ok(Volume::from_data(c, t, h, w, be.download(&pix)))
    }

    /// `decode_clip_tile` up to the device buffer: the caller downloads, so a
    /// tiled decode can queue the next tile's kernels before reading this one.
    fn decode_clip_tile_dev(&self, be: &B, z: &Volume) -> Result<(B::Buf, (usize, usize, usize, usize))> {
        let dec = self.decoder.as_ref().ok_or_else(|| Error::Msg("the video VAE decoder is not loaded".into()))?;
        let cfg = &self.cfg;
        let dev = Dev { be };
        let (lz, t, h, w) = (z.c, z.t, z.h, z.w);
        let gz = be.upload(&z.data);
        let shape = Conv3dShape { pad_t: (0, 0), pad_h: (0, 0), pad_w: (0, 0), ..Conv3dShape::causal(lz, lz, t, h, w, 1) };
        let pq = dev.conv(&gz, &shape, &dec.post_quant_conv, ConvEpilogue::None);
        let n = t * h * w;
        let dd = cfg.decoder_dim();
        let reg = cfg.decoder_register_tokens;
        let rows = n + reg + 1;
        let tokens = be.alloc(n * lz);
        be.patchify(&pq, lz, t, h, w, (1, 1, 1), &tokens, 0);
        let hbuf = be.alloc(rows * dd);
        be.linear(&tokens, 0, n, &dec.proj_in.w, dec.proj_in.b.as_ref(), Act::None, &hbuf, 0);
        let regs = be.upload(&dec.register_tokens);
        be.copy(&regs, 0, &hbuf, n * dd, reg * dd);
        // The cls row stays zero (alloc zero-fills).
        let rope = vae_decoder_rope((t, h, w), cfg.decoder_rotary_dim(), cfg.decoder_rope_theta, reg + 1);
        let (rc, rs) = (be.upload_table(&rope.cos), be.upload_table(&rope.sin));
        let rp = Some(Rope { cos: &rc, sin: &rs, rot: cfg.decoder_rotary_dim() });
        let (heads, hd) = (cfg.decoder_heads, cfg.decoder_head_dim);
        let normed = be.alloc(rows * dd);
        let qkv = be.alloc(rows * 3 * dd);
        let (q, k, v, attn, tmp) = (be.alloc(rows * dd), be.alloc(rows * dd), be.alloc(rows * dd), be.alloc(rows * dd), be.alloc(rows * dd));
        let ff = cfg.decoder_ffn_mult * dd;
        let (hid, hid2) = (be.alloc(rows * 2 * ff), be.alloc(rows * ff));
        let eps = cfg.decoder_norm_eps;
        for bl in &dec.blocks {
            be.rms_norm(&hbuf, rows, dd, Some(&bl.norm1), Modulation::None, eps, &normed);
            be.linear(&normed, 0, rows, &bl.qkv.w, bl.qkv.b.as_ref(), Act::None, &qkv, 0);
            let hn = Some(HeadNorm { w: None, eps });
            be.head_split(&qkv, rows, 3 * dd, 0, heads, hd, hn, rp, &q);
            be.head_split(&qkv, rows, 3 * dd, dd, heads, hd, hn, rp, &k);
            be.head_split(&qkv, rows, 3 * dd, 2 * dd, heads, hd, None, None, &v);
            be.attention(&q, &k, &v, heads, heads, rows, rows, hd, false, &attn);
            be.linear(&attn, 0, rows, &bl.out.w, bl.out.b.as_ref(), Act::None, &tmp, 0);
            be.residual_rms_norm(&hbuf, &tmp, rows, dd, Gate::Vector(&bl.scale1), Some(&bl.norm2), Modulation::None, eps, &normed);
            be.linear(&normed, 0, rows, &bl.ff_in.w, bl.ff_in.b.as_ref(), Act::None, &hid, 0);
            be.swiglu(&hid, rows, ff, &hid2);
            be.linear(&hid2, 0, rows, &bl.ff_out.w, bl.ff_out.b.as_ref(), Act::None, &tmp, 0);
            be.gate_residual(&hbuf, &tmp, rows, dd, Gate::Vector(&bl.scale2));
        }
        be.layer_norm(&hbuf, n, dd, Some(&dec.norm_out.w), Some(&dec.norm_out.b), eps, &normed);
        let out_dim = dec.proj_out.rows;
        let proj = be.alloc(n * out_dim);
        be.linear(&normed, 0, n, &dec.proj_out.w, dec.proj_out.b.as_ref(), Act::None, &proj, 0);
        let (pt, ps) = (cfg.temporal_ratio(), cfg.spatial_ratio());
        let (ot, oh, ow) = (t * pt, h * ps, w * ps);
        let pix = be.alloc(cfg.in_channels * ot * oh * ow);
        be.unpatchify(&proj, 0, out_dim, 0, cfg.in_channels, ot, oh, ow, (pt, ps, ps), &pix);
        Ok((pix, (cfg.in_channels, ot, oh, ow)))
    }

    /// One latent clip, optionally spatially tiled (`_decode_clip`).
    ///
    /// Tiles are decoded one ahead of their download: the backends launch
    /// asynchronously, so the GPU runs tile `i + 1` while the host converts
    /// tile `i`; the stitch is one parallel pass at the end.
    pub fn decode_clip(&self, be: &B, z: &Volume, tiling: Option<Tiling>) -> Result<Volume> {
        let Some(tl) = tiling else { return self.decode_clip_tile(be, z) };
        let r = self.spatial_ratio();
        let (ys, yl, yo) = split_tiles(z.h * r, tl.tile, tl.overlap, r);
        let (xs, xl, xo) = split_tiles(z.w * r, tl.tile, tl.overlap, r);
        let mut tiles: Vec<Volume> = Vec::with_capacity(ys.len() * xs.len());
        let mut pending: Option<(B::Buf, (usize, usize, usize, usize))> = None;
        let download = |(pix, (c, t, h, w)): (B::Buf, (usize, usize, usize, usize))| Volume::from_data(c, t, h, w, be.download(&pix));
        for (y0, ny) in ys.iter().zip(&yl) {
            for (x0, nx) in xs.iter().zip(&xl) {
                let dev = self.decode_clip_tile_dev(be, &z.crop(0, z.t, y0 / r, ny / r, x0 / r, nx / r))?;
                tiles.extend(pending.take().map(download));
                pending = Some(dev);
            }
        }
        tiles.extend(pending.map(download));
        let mut it = tiles.into_iter();
        let rows: Vec<Vec<Volume>> = (0..ys.len()).map(|_| it.by_ref().take(xs.len()).collect()).collect();
        Ok(stitch_tiles(&rows, &yo, &xo))
    }

    /// `_temporal_decode_plan`: `(pad tokens, chunks, decoded frames)`.
    pub fn temporal_plan(&self, latent_frames: usize) -> Result<(usize, usize, usize)> {
        let cfg = &self.cfg;
        if latent_frames == 0 {
            return Err(Error::Geometry("no latent frames to decode".into()));
        }
        let (drop, chunk, tr) = (cfg.token_drop, cfg.tokens_chunk(), cfg.temporal_ratio());
        let n = latent_frames + drop;
        let mut pad = (chunk - n % chunk) % chunk;
        let mut chunks = (n + pad) / chunk - usize::from(drop > 0);
        if chunks < 1 {
            pad += chunk;
            chunks = 1;
        }
        let mut frames = chunks * (chunk * tr - cfg.frame_pre_padding());
        if drop > 0 {
            frames += cfg.frame_overlap();
        }
        if pad > 0 {
            let intra_tail = cfg.clip_length % tr;
            let mut pad_frames = 0;
            for off in 0..pad {
                pad_frames += if intra_tail != 0 && (latent_frames + off) % chunk == 0 { intra_tail } else { tr };
            }
            frames = frames.checked_sub(pad_frames).ok_or_else(|| Error::Geometry("decode plan underflow".into()))?;
        }
        if frames == 0 {
            return Err(Error::Geometry(format!("the decode plan yields no frames for {latent_frames} latent frames")));
        }
        Ok((pad, chunks, frames))
    }

    /// Decode denormalised latents `[latent, T, h, w]` into pixels `[3, F, H, W]`
    /// in `[0, 1]` (`_decode_chunks` + ImageNet denormalisation + clamp),
    /// calling `on_chunk` with every finished chunk in order.
    pub fn decode(&self, be: &B, z: &Volume, tiling: Option<Tiling>, on_chunk: &mut dyn FnMut(&Volume)) -> Result<usize> {
        let cfg = &self.cfg;
        let (pad, chunks, total) = self.temporal_plan(z.t)?;
        let z = z.pad_last_frame(pad);
        let chunk_tok = cfg.tokens_chunk();
        let chunk_frames = chunk_tok * cfg.temporal_ratio();
        let pre = cfg.frame_pre_padding();
        let mut written = 0;
        let mut overlap: Option<Volume> = None;
        for ci in 0..chunks {
            let start = ci * chunk_tok;
            let n = (chunk_tok + cfg.token_overlap()).min(z.t - start);
            let clip = self.decode_clip(be, &z.crop(start, n, 0, z.h, 0, z.w), tiling)?;
            let chunk_t = chunk_frames.min(clip.t) - pre;
            let keep = chunk_t.min(total - written);
            let next_overlap = if cfg.token_drop > 0 && clip.t > chunk_frames + pre {
                Some(clip.crop(chunk_frames + pre, clip.t - chunk_frames - pre, 0, clip.h, 0, clip.w))
            } else {
                None
            };
            if keep > 0 {
                written += keep;
                let px = finish_chunk(&clip, pre, chunk_t, keep, overlap.as_ref(), cfg.frame_overlap());
                on_chunk(&px);
            }
            overlap = next_overlap;
        }
        if let Some(ov) = overlap {
            if written < total {
                let keep = (total - written).min(ov.t);
                let px = finish_chunk(&ov, 0, ov.t, keep, None, 0);
                written += keep;
                on_chunk(&px);
            }
        }
        if written != total {
            return Err(Error::Geometry(format!("decoded {written} frames, the plan expected {total}")));
        }
        Ok(total)
    }
}

/// `(p − mean) / std` per channel, in place.
pub fn normalize_pixels(x: &mut Volume) {
    let scale: Vec<f32> = PIXEL_STD.iter().map(|s| 1.0 / s).collect();
    let shift: Vec<f32> = PIXEL_MEAN.iter().zip(PIXEL_STD).map(|(m, s)| -m / s).collect();
    x.channel_affine(&scale[..x.c], &shift[..x.c]);
}

/// One decoded chunk's host post-processing in a single parallel pass:
/// frames `pre .. pre + keep` of `clip`, the first `extent` of them
/// crossfaded with the last `extent` frames of `overlap` (as `blend` along
/// T over the `chunk_t`-frame chunk), then ImageNet-denormalised and clamped
/// (`denormalize_pixels`). Equal to `crop` → `blend` → `crop` → `denormalize_pixels`.
fn finish_chunk(clip: &Volume, pre: usize, chunk_t: usize, keep: usize, overlap: Option<&Volume>, extent: usize) -> Volume {
    assert!(pre + chunk_t <= clip.t && keep <= chunk_t);
    let (c, h, w) = (clip.c, clip.h, clip.w);
    let extent = overlap.map_or(0, |ov| extent.min(ov.t).min(chunk_t));
    let mut out = Volume::zeros(c, keep, h, w);
    out.data.par_chunks_mut(w).enumerate().for_each(|(r, row)| {
        let (y, r) = (r % h, r / h);
        let (f, ch) = (r % keep, r / keep);
        let src = ((ch * clip.t + pre + f) * h + y) * w;
        row.copy_from_slice(&clip.data[src..src + w]);
        if f < extent {
            let ov = overlap.unwrap();
            let wb = f as f32 / extent as f32;
            let osrc = ((ch * ov.t + ov.t - extent + f) * ov.h + y) * ov.w;
            for (o, &av) in row.iter_mut().zip(&ov.data[osrc..osrc + w]) {
                *o = av * (1.0 - wb) + *o * wb;
            }
        }
        let (m, s) = (PIXEL_MEAN[ch.min(2)], PIXEL_STD[ch.min(2)]);
        for v in row {
            *v = (*v * s + m).clamp(0.0, 1.0);
        }
    });
    out
}

/// `p · std + mean`, clamped to `[0, 1]`.
pub fn denormalize_pixels(x: &mut Volume) {
    let inner = x.t * x.h * x.w;
    let plane = (x.h * x.w).max(1);
    x.data.par_chunks_mut(plane).enumerate().for_each(|(i, chunk)| {
        let c = (i * plane / inner.max(1)).min(2);
        let (m, s) = (PIXEL_MEAN[c], PIXEL_STD[c]);
        for v in chunk {
            *v = (*v * s + m).clamp(0.0, 1.0);
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3_core::weights::Synthetic;
    use h3_cpu::Cpu;

    #[test]
    fn tiles_and_blend() {
        let (s, l, o) = split_tiles(768, 256, 64, 16);
        assert_eq!(l, vec![256; 4]);
        assert_eq!(s[0], 0);
        assert_eq!(*s.last().unwrap() + 256, 768);
        assert!(o.iter().all(|&x| x >= 64 && x % 16 == 0));
        assert_eq!(split_tiles(200, 256, 64, 16), (vec![0], vec![200], vec![]));
        let a = Volume::from_data(1, 1, 1, 4, vec![1.0; 4]);
        let b = Volume::from_data(1, 1, 1, 4, vec![3.0; 4]);
        let bl = blend(&a, &b, 2, Axis::W);
        assert_eq!(bl.data, vec![1.0, 2.0, 3.0, 3.0]);
    }

    /// The sequential `_stitch_tiles` (blend H with the tile above, blend W
    /// with the tile to the left, crop the trailing overlaps, concatenate).
    fn stitch_tiles_reference(tiles: &[Vec<Volume>], h_overlaps: &[usize], w_overlaps: &[usize]) -> Volume {
        let mut rows_out = Vec::new();
        for (ri, row) in tiles.iter().enumerate() {
            let mut row_out: Vec<Volume> = Vec::new();
            for (ci, tile) in row.iter().enumerate() {
                let mut t = tile.clone();
                if ri > 0 {
                    t = blend(&tiles[ri - 1][ci], &t, h_overlaps[ri - 1], Axis::H);
                }
                if ci > 0 {
                    t = blend(&row[ci - 1], &t, w_overlaps[ci - 1], Axis::W);
                }
                if ri + 1 < tiles.len() {
                    t = t.crop(0, t.t, 0, t.h - h_overlaps[ri], 0, t.w);
                }
                if ci + 1 < row.len() {
                    t = t.crop(0, t.t, 0, t.h, 0, t.w - w_overlaps[ci]);
                }
                row_out.push(t);
            }
            let (c, t, h) = (row_out[0].c, row_out[0].t, row_out[0].h);
            let w: usize = row_out.iter().map(|p| p.w).sum();
            let mut o = Volume::zeros(c, t, h, w);
            let mut x0 = 0;
            for p in &row_out {
                for cc in 0..c {
                    for tt in 0..t {
                        for y in 0..h {
                            o.data[((cc * t + tt) * h + y) * w + x0..][..p.w].copy_from_slice(&p.data[((cc * t + tt) * h + y) * p.w..][..p.w]);
                        }
                    }
                }
                x0 += p.w;
            }
            rows_out.push(o);
        }
        let (c, t, w) = (rows_out[0].c, rows_out[0].t, rows_out[0].w);
        let h: usize = rows_out.iter().map(|p| p.h).sum();
        let mut o = Volume::zeros(c, t, h, w);
        let mut y0 = 0;
        for p in &rows_out {
            for cc in 0..c {
                for tt in 0..t {
                    o.data[((cc * t + tt) * h + y0) * w..][..p.h * w].copy_from_slice(&p.data[(cc * t + tt) * p.h * w..][..p.h * w]);
                }
            }
            y0 += p.h;
        }
        o
    }

    #[test]
    fn stitch_matches_the_sequential_reference() {
        let mut r = Rng::new(11);
        // Ragged grid: row heights 7 / 9 / 6, column widths 5 / 8 / 4 / 6; overlaps 3 / 2 and 2 / 3 / 1.
        let (hs, ws) = ([7usize, 9, 6], [5usize, 8, 4, 6]);
        let tiles: Vec<Vec<Volume>> =
            hs.iter().map(|&h| ws.iter().map(|&w| Volume::from_data(3, 2, h, w, r.uniform(3 * 2 * h * w, 1.0))).collect()).collect();
        let (ho, wo) = ([3usize, 2], [2usize, 3, 1]);
        let got = stitch_tiles(&tiles, &ho, &wo);
        let want = stitch_tiles_reference(&tiles, &ho, &wo);
        assert_eq!((got.c, got.t, got.h, got.w), (3, 2, 7 + 9 + 6 - 5, 5 + 8 + 4 + 6 - 6));
        assert_eq!(got, want);
        // One tile is the identity; one row and one column exercise each axis alone.
        let one = vec![vec![tiles[0][0].clone()]];
        assert_eq!(stitch_tiles(&one, &[], &[]), tiles[0][0]);
        let row = vec![tiles[1].clone()];
        assert_eq!(stitch_tiles(&row, &[], &wo), stitch_tiles_reference(&row, &[], &wo));
        let col: Vec<Vec<Volume>> = tiles.iter().map(|r| vec![r[2].clone()]).collect();
        assert_eq!(stitch_tiles(&col, &ho, &[]), stitch_tiles_reference(&col, &ho, &[]));
        // Blend along T against the scalar definition, and a crop / denormalise round trip.
        let a = Volume::from_data(2, 4, 3, 5, r.uniform(2 * 4 * 3 * 5, 1.0));
        let b = Volume::from_data(2, 3, 3, 5, r.uniform(2 * 3 * 3 * 5, 1.0));
        let bl = blend(&a, &b, 2, Axis::T);
        for c in 0..2 {
            for t in 0..3 {
                for y in 0..3 {
                    for x in 0..5 {
                        let want = if t < 2 { a.at(c, 2 + t, y, x) * (1.0 - t as f32 / 2.0) + b.at(c, t, y, x) * (t as f32 / 2.0) } else { b.at(c, t, y, x) };
                        assert_eq!(bl.at(c, t, y, x), want);
                    }
                }
            }
        }
        // finish_chunk is crop → blend → crop → denormalise.
        let clip = Volume::from_data(3, 9, 3, 5, r.uniform(3 * 9 * 3 * 5, 1.0));
        let ov = Volume::from_data(3, 4, 3, 5, r.uniform(3 * 4 * 3 * 5, 1.0));
        let mut want = blend(&ov, &clip.crop(2, 6, 0, 3, 0, 5), 3, Axis::T).crop(0, 4, 0, 3, 0, 5);
        denormalize_pixels(&mut want);
        assert_eq!(finish_chunk(&clip, 2, 6, 4, Some(&ov), 3), want);
        let mut want = clip.crop(0, 5, 0, 3, 0, 5);
        denormalize_pixels(&mut want);
        assert_eq!(finish_chunk(&clip, 0, 9, 5, None, 3), want);
        let cr = a.crop(1, 2, 1, 2, 2, 3);
        assert_eq!((cr.t, cr.h, cr.w), (2, 2, 3));
        assert_eq!(cr.at(1, 1, 1, 2), a.at(1, 2, 2, 4));
        let mut px = Volume::from_data(3, 2, 3, 5, r.uniform(3 * 2 * 3 * 5, 1.0));
        let src = px.clone();
        denormalize_pixels(&mut px);
        for c in 0..3 {
            for i in 0..2 * 3 * 5 {
                assert_eq!(px.data[c * 30 + i], (src.data[c * 30 + i] * PIXEL_STD[c] + PIXEL_MEAN[c]).clamp(0.0, 1.0));
            }
        }
    }

    /// Host cost of one 1344×768 decode chunk (28 tiles of [3, 20, 256, 256]):
    /// `cargo test -p h3-model --release -- --ignored --nocapture host_decode_chunk_timing`.
    #[test]
    #[ignore]
    fn host_decode_chunk_timing() {
        let (ys, yl, yo) = split_tiles(768, 256, 64, 16);
        let (xs, xl, xo) = split_tiles(1344, 256, 64, 16);
        let mut r = Rng::new(5);
        let tiles: Vec<Vec<Volume>> = yl.iter().map(|&h| xl.iter().map(|&w| Volume::from_data(3, 20, h, w, r.uniform(3 * 20 * h * w, 1.0))).collect()).collect();
        let _ = (ys, xs);
        let t0 = std::time::Instant::now();
        let clip = stitch_tiles(&tiles, &yo, &xo);
        let t_stitch = t0.elapsed();
        let t0 = std::time::Instant::now();
        let ov = clip.crop(17, 3, 0, clip.h, 0, clip.w);
        let px = finish_chunk(&clip, 1, 16, 16, Some(&ov), 3);
        let t_post = t0.elapsed();
        let t0 = std::time::Instant::now();
        let n = px.h * px.w;
        let mut rgb = vec![0u8; px.t * n * 3];
        rgb.par_chunks_mut(n * 3).enumerate().for_each(|(f, out)| {
            let planes: [&[f32]; 3] = std::array::from_fn(|ch| &px.data[(ch * px.t + f) * n..][..n]);
            for (i, o) in out.chunks_exact_mut(3).enumerate() {
                for ch in 0..3 {
                    o[ch] = (planes[ch][i] * 255.0).clamp(0.0, 255.0) as u8;
                }
            }
        });
        let t_u8 = t0.elapsed();
        println!("host decode chunk on {} threads: stitch {:?}, crop/blend/denormalise {:?}, u8 {:?} (clip {}x{}x{}, {} bytes)", rayon::current_num_threads(), t_stitch, t_post, t_u8, clip.t, clip.h, clip.w, rgb.len());
    }

    #[test]
    fn temporal_plan_matches_reference_counts() {
        let be = Cpu;
        let vae = VideoVae::<Cpu>::load(&be, VideoVaeConfig::tiny(), &Synthetic { seed: 1 }, false, false).unwrap();
        assert_eq!(vae.temporal_plan(37).unwrap(), (0, 7, 124));
        assert_eq!(vae.temporal_plan(2).unwrap(), (5, 1, 5));
        assert_eq!(vae.temporal_plan(7).unwrap(), (0, 1, 22));
        assert_eq!(vae.temporal_plan(107).unwrap(), (0, 21, 362));
    }

    #[test]
    fn tiny_vae_round_trips_shapes() {
        let be = Cpu;
        let cfg = VideoVaeConfig::tiny();
        let vae = VideoVae::load(&be, cfg.clone(), &Synthetic { seed: 2 }, true, true).unwrap();
        // 17 frames, 32×48 pixels → moments [8, 5, 2, 3]; keyframe → [8, 1, 2, 3].
        let mut r = Rng::new(3);
        let px = Volume::from_data(3, 17, 32, 48, r.uniform(3 * 17 * 32 * 48, 0.5).iter().map(|v| v + 0.5).collect());
        let m = vae.encode_pixels(&be, &px, None).unwrap();
        assert_eq!((m.c, m.t, m.h, m.w), (8, 2, 2, 3));
        let kf = vae.encode_keyframe(&be, &px.crop(0, 1, 0, 32, 0, 48), None).unwrap();
        assert_eq!((kf.c, kf.t, kf.h, kf.w), (8, 1, 2, 3));
        // Tiled encode equals untiled when a single tile covers the frame; with
        // two tiles the stitched shape is the same.
        let tiled = vae.encode_keyframe(&be, &px.crop(0, 1, 0, 32, 0, 48), Some(Tiling { tile: 32, overlap: 16 })).unwrap();
        assert_eq!((tiled.c, tiled.t, tiled.h, tiled.w), (8, 1, 2, 3));
        let post = Posterior::from_moments(&m);
        let mut z = post.sample(&mut r);
        vae.normalize_latents(&mut z);
        vae.denormalize_latents(&mut z);
        let mut frames = Vec::new();
        let total = vae.decode(&be, &z, None, &mut |v| frames.push(v.clone())).unwrap();
        assert_eq!(total, 5);
        let out = Volume::cat_t(&frames);
        assert_eq!((out.c, out.t, out.h, out.w), (3, 5, 32, 48));
        assert!(out.data.iter().all(|v| (0.0..=1.0).contains(v)));
    }
}
