//! The fast (Sol-H3) VAE decoder: Open-H3-Regenerate-2K's
//! `vae_fast/fast_h3_vae_noncausal_24l.safetensors`, the released 36-layer
//! H3 ViT decoder pruned to 24 layers and distilled to decode a whole clip in
//! one pass (NVlabs Sana `open_h3/fast_vae/decoder.py`,
//! `NonCausalViTDecoder`; ~30 dB against the released decoder per its
//! README).
//!
//! Tokens are the latents (one per latent pixel, no tiling, no temporal
//! overlap). Self-attention runs in windows of `block` (5) latent frames ×
//! `window` (16) × 16 tokens; odd layers shift them by `t_shift` (2) frames
//! and half a window in space (zero padding in front, never cyclic, so a
//! border window is a smaller box). Each native 5-latent block owns `S` = 5
//! global rows (4 registers + 1 zero row, no RoPE): they attend to every
//! token of their block plus their own 5 rows, and a window attends to its
//! tokens plus the global rows of the 1–2 blocks it overlaps. RoPE is
//! window-local ([`vae_fast_rope`]).
//!
//! On the backend a window is a box of rows: [`Backend::box_rows`] gathers
//! every window of one size in a time slab into `[window·heads, rows, d]`,
//! one [`Backend::attention`] runs them as extra heads, and the output is
//! scattered back token-major. The checkpoint's `to_qkv` rows are per head
//! `[q | k | v]` and `ff.w1` is `[gate | up]`; the loader permutes them into
//! H3's `q | k | v` thirds and `[value | gate]` SwiGLU halves, and folds the
//! 1×1×1 `post_quant_conv` into `x_embedder` (as Sana's deployment engine
//! does).

use crate::linear::{Linear, load_plain};
use crate::vae::{Volume, denormalize_pixels};
use h3_core::backend::{Act, Backend, BoxRows, Gate, HeadNorm, Modulation, Rope};
use h3_core::quant::{Mxfp8, Q8, WeightQuant};
use h3_core::rope::vae_fast_rope;
use h3_core::weights::WeightSource;
use h3_core::{Error, Result};

/// The decoder's geometry. [`FastVaeConfig::h3`] is the released file's.
#[derive(Debug, Clone, PartialEq)]
pub struct FastVaeConfig {
    pub latent: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    pub ffn_mult: usize,
    pub registers: usize,
    /// Spatial window (tokens); odd layers shift by half of it.
    pub window: usize,
    /// Native block / temporal window (latent frames).
    pub block: usize,
    pub t_shift: usize,
    /// Frames dropped at the start of every block (`4·5 − 3 = 17`).
    pub trim: usize,
    pub patch: usize,
    pub patch_t: usize,
    pub out_channels: usize,
    pub rot: usize,
    pub theta: f64,
    pub eps: f32,
}

impl FastVaeConfig {
    /// `fast_h3_vae_noncausal_24l`: 24 layers of 32 × 64, SwiGLU ×4.
    pub fn h3() -> Self {
        Self {
            latent: 24,
            heads: 32,
            head_dim: 64,
            layers: 24,
            ffn_mult: 4,
            registers: 4,
            window: 16,
            block: 5,
            t_shift: 2,
            trim: 3,
            patch: 16,
            patch_t: 4,
            out_channels: 3,
            rot: 48,
            theta: 100.0,
            eps: 1e-5,
        }
    }

    /// The checker's model: both layer parities, windows of 4 tokens.
    pub fn tiny(latent: usize) -> Self {
        Self {
            latent,
            heads: 2,
            head_dim: 16,
            layers: 2,
            ffn_mult: 2,
            registers: 2,
            window: 4,
            block: 5,
            t_shift: 2,
            trim: 3,
            patch: 2,
            patch_t: 4,
            out_channels: 3,
            rot: 12,
            theta: 100.0,
            eps: 1e-5,
        }
    }

    pub fn dim(&self) -> usize {
        self.heads * self.head_dim
    }
    /// Global rows per native block.
    pub fn globals(&self) -> usize {
        self.registers + 1
    }
    /// Video frames for `t` latent frames.
    pub fn frames(&self, t: usize) -> usize {
        t * self.patch_t - self.trim * t.div_ceil(self.block)
    }
    /// `(spatial shift, temporal shift)` of layer `l`.
    pub fn shifts(&self, l: usize) -> (usize, usize) {
        if l % 2 == 1 {
            (self.window / 2, self.t_shift)
        } else {
            (0, 0)
        }
    }
}

/// Windows along one axis of length `len` (window `ws`, front shift `s`),
/// run-length grouped by size: `(first lo, size, count, step)`.
pub fn window_runs(len: usize, ws: usize, s: usize) -> Vec<(usize, usize, usize, usize)> {
    let n = (len + s).div_ceil(ws);
    let mut runs: Vec<(usize, usize, usize, usize)> = Vec::new();
    for i in 0..n {
        let lo = (i * ws).saturating_sub(s);
        let hi = ((i + 1) * ws - s).min(len);
        let size = hi - lo;
        if let Some(r) = runs.last_mut() {
            let step = lo - (r.0 + (r.2 - 1) * r.3);
            if r.1 == size && (r.2 == 1 || r.3 == step) {
                r.3 = step;
                r.2 += 1;
                continue;
            }
        }
        runs.push((lo, size, 1, 0));
    }
    runs
}

struct Layer<B: Backend> {
    norm1: B::Buf,
    qkv: Linear<B>,
    out: Linear<B>,
    scale1: B::Buf,
    norm2: B::Buf,
    ff_in: Linear<B>,
    ff_out: Linear<B>,
    scale2: B::Buf,
}

pub struct FastDecoder<B: Backend> {
    pub cfg: FastVaeConfig,
    /// `x_embedder · post_quant_conv` as one `[dim, latent]` affine map.
    embed: Linear<B>,
    register_tokens: Vec<f32>,
    layers: Vec<Layer<B>>,
    norm_out: (B::Buf, B::Buf),
    proj_out: Linear<B>,
}

/// Rows of the FFN per GEMM (its hidden activation is `2·ffn` wide).
const FFN_ROWS: usize = 16384;

impl<B: Backend> FastDecoder<B> {
    /// Load from the released file's names (`blocks.N.attn.to_qkv`, …), the
    /// block linears in `quant` (`Mxfp8` / `Int8`; the embedding, head and
    /// norms stay native).
    pub fn load(
        be: &B,
        cfg: FastVaeConfig,
        src: &dyn WeightSource,
        quant: WeightQuant,
    ) -> Result<Self> {
        let (dd, hd, heads, lz) = (cfg.dim(), cfg.head_dim, cfg.heads, cfg.latent);
        let ff = cfg.ffn_mult * dd;
        let block_linear = |w: Vec<f32>, b: Vec<f32>, rows: usize, cols: usize| -> Linear<B> {
            let w = match quant {
                WeightQuant::Mxfp8 if cols.is_multiple_of(32) => {
                    be.weight_mxfp8(&Mxfp8::quantize(&w, rows, cols))
                }
                WeightQuant::Int8 => be.weight_q8(&Q8::quantize(&w, rows, cols)),
                _ => be.weight(&w, rows, cols),
            };
            Linear {
                w,
                b: Some(be.upload(&b)),
                rows,
                cols,
                lora: None,
            }
        };
        let get = |k: &str, shape: &[usize]| src.get(k, shape);
        let mut layers = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let p = format!("blocks.{l}");
            // Per head [q | k | v] → q | k | v thirds.
            let w = get(&format!("{p}.attn.to_qkv.weight"), &[3 * dd, dd])?;
            let b = get(&format!("{p}.attn.to_qkv.bias"), &[3 * dd])?;
            let (mut qw, mut qb) = (vec![0.0; 3 * dd * dd], vec![0.0; 3 * dd]);
            for h in 0..heads {
                for part in 0..3 {
                    for c in 0..hd {
                        let (src_r, dst_r) = (h * 3 * hd + part * hd + c, part * dd + h * hd + c);
                        qw[dst_r * dd..(dst_r + 1) * dd]
                            .copy_from_slice(&w[src_r * dd..(src_r + 1) * dd]);
                        qb[dst_r] = b[src_r];
                    }
                }
            }
            // [gate | up] → [up | gate].
            let w1 = get(&format!("{p}.ff.w1.weight"), &[2 * ff, dd])?;
            let b1 = get(&format!("{p}.ff.w1.bias"), &[2 * ff])?;
            let w1 = [&w1[ff * dd..], &w1[..ff * dd]].concat();
            let b1 = [&b1[ff..], &b1[..ff]].concat();
            layers.push(Layer {
                norm1: be.upload(&get(&format!("{p}.norm1.weight"), &[dd])?),
                qkv: block_linear(qw, qb, 3 * dd, dd),
                out: block_linear(
                    get(&format!("{p}.attn.to_out.weight"), &[dd, dd])?,
                    get(&format!("{p}.attn.to_out.bias"), &[dd])?,
                    dd,
                    dd,
                ),
                scale1: be.upload(&get(&format!("{p}.scale1"), &[dd])?),
                norm2: be.upload(&get(&format!("{p}.norm2.weight"), &[dd])?),
                ff_in: block_linear(w1, b1, 2 * ff, dd),
                ff_out: block_linear(
                    get(&format!("{p}.ff.w2.weight"), &[dd, ff])?,
                    get(&format!("{p}.ff.w2.bias"), &[dd])?,
                    dd,
                    ff,
                ),
                scale2: be.upload(&get(&format!("{p}.scale2"), &[dd])?),
            });
        }
        // e = We (Wc z + bc) + be, folded in f64.
        let wc = get("post_quant_conv.weight", &[lz, lz, 1, 1, 1])?;
        let bc = get("post_quant_conv.bias", &[lz])?;
        let we = get("x_embedder.weight", &[dd, lz])?;
        let bev = get("x_embedder.bias", &[dd])?;
        let mut ew = vec![0.0f32; dd * lz];
        let mut eb = vec![0.0f32; dd];
        for o in 0..dd {
            let mut acc_b = bev[o] as f64;
            for m in 0..lz {
                let wom = we[o * lz + m] as f64;
                acc_b += wom * bc[m] as f64;
                for i in 0..lz {
                    let e = &mut ew[o * lz + i];
                    *e = (*e as f64 + wom * wc[m * lz + i] as f64) as f32;
                }
            }
            eb[o] = acc_b as f32;
        }
        let out_dim = cfg.out_channels * cfg.patch_t * cfg.patch * cfg.patch;
        Ok(Self {
            embed: Linear {
                w: be.weight(&ew, dd, lz),
                b: Some(be.upload(&eb)),
                rows: dd,
                cols: lz,
                lora: None,
            },
            register_tokens: get("register_tokens", &[1, cfg.registers, dd])?,
            layers,
            norm_out: (
                be.upload(&get("norm_out.weight", &[dd])?),
                be.upload(&get("norm_out.bias", &[dd])?),
            ),
            proj_out: load_plain(be, src, "proj_out", out_dim, dd, true)?,
            cfg,
        })
    }

    /// The released decoder from `path`: the `.safetensors` file, or a
    /// directory holding `fast_h3_vae_noncausal_24l.safetensors` (directly
    /// or under `vae_fast/`, as in the Open-H3-Regenerate-2K tree).
    pub fn open(be: &B, path: &std::path::Path, quant: WeightQuant) -> Result<Self> {
        const FILE: &str = "fast_h3_vae_noncausal_24l.safetensors";
        let file = if path.is_dir() {
            [path.join(FILE), path.join("vae_fast").join(FILE)]
                .into_iter()
                .find(|p| p.is_file())
                .ok_or_else(|| {
                    Error::Weights(format!(
                        "{}: no {FILE} (or vae_fast/{FILE})",
                        path.display()
                    ))
                })?
        } else {
            path.to_path_buf()
        };
        let src = h3_core::weights::SafetensorsDir::open(&file)?;
        if !Self::detect(&src) {
            return Err(Error::Weights(format!(
                "{}: not the fast VAE decoder (no x_embedder / blocks.0.attn.to_qkv)",
                file.display()
            )));
        }
        Self::load(be, FastVaeConfig::h3(), &src, quant)
    }

    /// Whether `src` is this decoder's file.
    pub fn detect(src: &dyn WeightSource) -> bool {
        src.contains("x_embedder.weight") && src.contains("blocks.0.attn.to_qkv.weight")
    }

    /// The transformer on denormalised latents `[latent, T, h, w]`: the
    /// device buffer of token rows `[T·h·w, dim]` after the final LayerNorm.
    pub fn forward(&self, be: &B, z: &Volume) -> Result<B::Buf> {
        let cfg = &self.cfg;
        if z.c != cfg.latent || z.t == 0 {
            return Err(Error::Geometry(format!(
                "fast VAE decoder: latents {}×{}, need {} channels",
                z.c, z.t, cfg.latent
            )));
        }
        let (t, h, w) = (z.t, z.h, z.w);
        let hw = h * w;
        let n = t * hw;
        let (dd, hd, heads) = (cfg.dim(), cfg.head_dim, cfg.heads);
        let (bk, s_glob) = (cfg.block, cfg.globals());
        let nb = t.div_ceil(bk);
        let g = nb * s_glob;
        let rows_all = n + g;
        let ff = cfg.ffn_mult * dd;
        let eps = cfg.eps;

        // Embedding + the per-block global rows.
        let gz = be.upload(&z.data);
        let tokens = be.alloc(n * cfg.latent);
        be.patchify(&gz, cfg.latent, t, h, w, (1, 1, 1), &tokens, 0);
        let x = be.alloc(rows_all * dd);
        be.linear(
            &tokens,
            0,
            n,
            &self.embed.w,
            self.embed.b.as_ref(),
            Act::None,
            &x,
            0,
        );
        let mut glob = vec![0.0f32; g * dd];
        for b in 0..nb {
            let o = b * s_glob * dd;
            glob[o..o + cfg.registers * dd].copy_from_slice(&self.register_tokens);
        }
        let glob = be.upload(&glob);
        be.copy(&glob, 0, &x, n * dd, g * dd);

        // RoPE: one full-slab table per parity, and the odd layers' first
        // (shifted, shorter) slab.
        let table = |tr: (usize, usize), s: usize, st: usize| {
            let r = vae_fast_rope(tr, (h, w), cfg.rot, cfg.theta, bk, cfg.window, s, st);
            (be.upload_table(&r.cos), be.upload_table(&r.sin))
        };
        let full = [table((0, bk), 0, 0), table((0, bk), cfg.window / 2, 0)];
        let first_odd = table((0, bk - cfg.t_shift), cfg.window / 2, cfg.t_shift);

        let slab = bk * hw;
        let normed = be.alloc(rows_all * dd);
        let o = be.alloc(rows_all * dd);
        let tmp = be.alloc(rows_all * dd);
        let qkv = be.alloc(slab * 3 * dd);
        let qkv_g = be.alloc(g * 3 * dd);
        let (qg, kg, vg) = (be.alloc(g * dd), be.alloc(g * dd), be.alloc(g * dd));
        let q = be.alloc(slab * dd);
        let (mut k, mut v) = (be.alloc(slab * dd), be.alloc(slab * dd));
        let (mut kp, mut vp) = (be.alloc(slab * dd), be.alloc(slab * dd));
        // Packed windows: at most a slab of queries; keys add ≤ 2 blocks of
        // global rows per window.
        let max_win = window_runs(h, cfg.window, cfg.window / 2)
            .iter()
            .map(|r| r.2)
            .sum::<usize>()
            .max(window_runs(h, cfg.window, 0).iter().map(|r| r.2).sum())
            * window_runs(w, cfg.window, cfg.window / 2)
                .iter()
                .map(|r| r.2)
                .sum::<usize>()
                .max(window_runs(w, cfg.window, 0).iter().map(|r| r.2).sum());
        let pk_rows = slab + max_win * 2 * s_glob;
        let (q_pk, k_pk, v_pk) = (
            be.alloc(slab * dd),
            be.alloc(pk_rows * dd),
            be.alloc(pk_rows * dd),
        );
        let o_pk = be.alloc(slab * dd);
        let fc = FFN_ROWS.min(rows_all);
        let (hid, hid2) = (be.alloc(fc * 2 * ff), be.alloc(fc * ff));
        let hn = Some(HeadNorm { w: None, eps });
        // Global rows as a `(G, 1, 1)` grid of `[heads, G, d]`.
        let glob_box = |blk: usize, count: (usize, usize)| BoxRows {
            grid: (g, 1, 1),
            t: (blk * s_glob, (blk + 1) * s_glob),
            y: (0, 1),
            x: (0, 1),
            count,
            step: (0, 0),
            heads,
            width: hd,
            grid_off: 0,
            grid_head: g,
            grid_row: 1,
            packed_off: 0,
            packed_win: 0,
            packed_head: 0,
            packed_row: 1,
        };

        for (li, bl) in self.layers.iter().enumerate() {
            let last = li + 1 == self.layers.len();
            let (s, st) = cfg.shifts(li);
            let par = li % 2;
            be.rms_norm(
                &x,
                rows_all,
                dd,
                Some(&bl.norm1),
                Modulation::None,
                eps,
                &normed,
            );
            be.linear(
                &normed,
                n,
                g,
                &bl.qkv.w,
                bl.qkv.b.as_ref(),
                Act::None,
                &qkv_g,
                0,
            );
            be.head_split(&qkv_g, g, 3 * dd, 0, heads, hd, hn, None, &qg);
            be.head_split(&qkv_g, g, 3 * dd, dd, heads, hd, hn, None, &kg);
            be.head_split(&qkv_g, g, 3 * dd, 2 * dd, heads, hd, None, None, &vg);
            let (yr, xr) = (window_runs(h, cfg.window, s), window_runs(w, cfg.window, s));
            let nt = (t + st).div_ceil(bk);
            let mut prev: Option<(usize, usize)> = None;
            let mut next_block = 0;
            for j in 0..nt {
                let a = (j * bk).saturating_sub(st);
                let b = ((j + 1) * bk - st).min(t);
                let tw = b - a;
                let rows_s = tw * hw;
                be.linear(
                    &normed,
                    a * hw,
                    rows_s,
                    &bl.qkv.w,
                    bl.qkv.b.as_ref(),
                    Act::None,
                    &qkv,
                    0,
                );
                let tab = if j == 0 && st > 0 {
                    &first_odd
                } else {
                    &full[par]
                };
                let rp = Some(Rope {
                    cos: &tab.0,
                    sin: &tab.1,
                    rot: cfg.rot,
                });
                be.head_split(&qkv, rows_s, 3 * dd, 0, heads, hd, hn, rp, &q);
                be.head_split(&qkv, rows_s, 3 * dd, dd, heads, hd, hn, rp, &k);
                be.head_split(&qkv, rows_s, 3 * dd, 2 * dd, heads, hd, None, None, &v);
                let blocks: Vec<usize> = (a / bk..=(b - 1) / bk).collect();
                let ng = blocks.len() * s_glob;
                for &(y0, ny, cy, sy) in &yr {
                    for &(x0, nx, cx, sx) in &xr {
                        let nwin = cy * cx;
                        let nq = tw * ny * nx;
                        let nk = nq + ng;
                        let boxes = BoxRows {
                            grid: (tw, h, w),
                            t: (0, tw),
                            y: (y0, y0 + ny),
                            x: (x0, x0 + nx),
                            count: (cy, cx),
                            step: (sy, sx),
                            heads,
                            width: hd,
                            grid_off: 0,
                            grid_head: rows_s,
                            grid_row: 1,
                            packed_off: 0,
                            packed_win: heads * nq,
                            packed_head: nq,
                            packed_row: 1,
                        };
                        be.box_rows(&q, &q_pk, &boxes, false);
                        let kb = BoxRows {
                            packed_win: heads * nk,
                            packed_head: nk,
                            ..boxes
                        };
                        be.box_rows(&k, &k_pk, &kb, false);
                        be.box_rows(&v, &v_pk, &kb, false);
                        for (m, &blk) in blocks.iter().enumerate() {
                            let gb = BoxRows {
                                packed_off: nq + m * s_glob,
                                packed_win: heads * nk,
                                packed_head: nk,
                                ..glob_box(blk, (cy, cx))
                            };
                            be.box_rows(&kg, &k_pk, &gb, false);
                            be.box_rows(&vg, &v_pk, &gb, false);
                        }
                        let hh = nwin * heads;
                        be.attention(&q_pk, &k_pk, &v_pk, hh, hh, nq, nk, hd, false, &o_pk);
                        let sc = BoxRows {
                            grid_off: a * hw * heads,
                            grid_head: 1,
                            grid_row: heads,
                            packed_off: 0,
                            packed_win: heads,
                            packed_head: 1,
                            packed_row: hh,
                            ..boxes
                        };
                        be.box_rows(&o, &o_pk, &sc, true);
                    }
                }
                // Global attention of every block that now lies in the
                // previous and this slab (the last layer's global rows feed
                // nothing).
                while !last && next_block < nb && ((next_block + 1) * bk).min(t) <= b {
                    let i = next_block;
                    next_block += 1;
                    let (lo, hi) = (i * bk, ((i + 1) * bk).min(t));
                    let nkg = (hi - lo) * hw + s_glob;
                    let qb = BoxRows {
                        packed_head: s_glob,
                        ..glob_box(i, (1, 1))
                    };
                    be.box_rows(&qg, &q_pk, &qb, false);
                    let slabs = [
                        prev.map(|(pa, pb)| (pa, pb, &kp, &vp)),
                        Some((a, b, &k, &v)),
                    ];
                    for (sa, sb, kb_, vb_) in slabs.into_iter().flatten() {
                        let (ol, oh) = (lo.max(sa), hi.min(sb));
                        if ol >= oh {
                            continue;
                        }
                        let bx = BoxRows {
                            grid: (sb - sa, h, w),
                            t: (ol - sa, oh - sa),
                            y: (0, h),
                            x: (0, w),
                            count: (1, 1),
                            step: (0, 0),
                            heads,
                            width: hd,
                            grid_off: 0,
                            grid_head: (sb - sa) * hw,
                            grid_row: 1,
                            packed_off: (ol - lo) * hw,
                            packed_win: 0,
                            packed_head: nkg,
                            packed_row: 1,
                        };
                        be.box_rows(kb_, &k_pk, &bx, false);
                        be.box_rows(vb_, &v_pk, &bx, false);
                    }
                    let gb = BoxRows {
                        packed_off: (hi - lo) * hw,
                        packed_head: nkg,
                        ..glob_box(i, (1, 1))
                    };
                    be.box_rows(&kg, &k_pk, &gb, false);
                    be.box_rows(&vg, &v_pk, &gb, false);
                    be.attention(
                        &q_pk, &k_pk, &v_pk, heads, heads, s_glob, nkg, hd, false, &o_pk,
                    );
                    be.copy(&o_pk, 0, &o, (n + i * s_glob) * dd, s_glob * dd);
                }
                std::mem::swap(&mut k, &mut kp);
                std::mem::swap(&mut v, &mut vp);
                prev = Some((a, b));
            }
            let rows_l = if last { n } else { rows_all };
            be.linear(
                &o,
                0,
                rows_l,
                &bl.out.w,
                bl.out.b.as_ref(),
                Act::None,
                &tmp,
                0,
            );
            be.residual_rms_norm(
                &x,
                &tmp,
                rows_l,
                dd,
                Gate::Vector(&bl.scale1),
                Some(&bl.norm2),
                Modulation::None,
                eps,
                &normed,
            );
            for c0 in (0..rows_l).step_by(fc) {
                let cn = fc.min(rows_l - c0);
                be.linear(
                    &normed,
                    c0,
                    cn,
                    &bl.ff_in.w,
                    bl.ff_in.b.as_ref(),
                    Act::None,
                    &hid,
                    0,
                );
                be.swiglu(&hid, cn, ff, &hid2);
                be.linear(
                    &hid2,
                    0,
                    cn,
                    &bl.ff_out.w,
                    bl.ff_out.b.as_ref(),
                    Act::None,
                    &tmp,
                    c0,
                );
            }
            be.gate_residual(&x, &tmp, rows_l, dd, Gate::Vector(&bl.scale2));
        }
        be.layer_norm(
            &x,
            n,
            dd,
            Some(&self.norm_out.0),
            Some(&self.norm_out.1),
            eps,
            &normed,
        );
        Ok(normed)
    }

    /// Native block `i`'s pixels `[3, 4·frames, H, W]` (ImageNet space,
    /// untrimmed) from [`Self::forward`]'s rows.
    fn head_block(
        &self,
        be: &B,
        rows: &B::Buf,
        (t, h, w): (usize, usize, usize),
        i: usize,
    ) -> (B::Buf, (usize, usize, usize, usize)) {
        let cfg = &self.cfg;
        let (lo, hi) = (i * cfg.block, ((i + 1) * cfg.block).min(t));
        let nr = (hi - lo) * h * w;
        let out_dim = self.proj_out.rows;
        let proj = be.alloc(nr * out_dim);
        be.linear(
            rows,
            lo * h * w,
            nr,
            &self.proj_out.w,
            self.proj_out.b.as_ref(),
            Act::None,
            &proj,
            0,
        );
        let (pt, ps) = (cfg.patch_t, cfg.patch);
        let shape = (cfg.out_channels, (hi - lo) * pt, h * ps, w * ps);
        let pix = be.alloc(shape.0 * shape.1 * shape.2 * shape.3);
        be.unpatchify(
            &proj,
            0,
            out_dim,
            0,
            shape.0,
            shape.1,
            shape.2,
            shape.3,
            (pt, ps, ps),
            &pix,
        );
        (pix, shape)
    }

    /// The whole clip in ImageNet space `[3, F, 16h, 16w]` (tests, checker).
    pub fn decode_normalized(&self, be: &B, z: &Volume) -> Result<Volume> {
        let rows = self.forward(be, z)?;
        let mut parts = Vec::new();
        for i in 0..z.t.div_ceil(self.cfg.block) {
            let (pix, (c, ft, ph, pw)) = self.head_block(be, &rows, (z.t, z.h, z.w), i);
            let v = Volume::from_data(c, ft, ph, pw, be.download(&pix));
            parts.push(v.crop(self.cfg.trim, ft - self.cfg.trim, 0, ph, 0, pw));
        }
        Ok(Volume::cat_t(&parts))
    }

    /// [`crate::vae::VideoVae::decode`]'s contract: denormalised latents in,
    /// frames in `[0, 1]` to `on_chunk` (one native block at a time).
    pub fn decode(&self, be: &B, z: &Volume, on_chunk: &mut dyn FnMut(&Volume)) -> Result<usize> {
        let rows = self.forward(be, z)?;
        let trim = self.cfg.trim;
        let mut written = 0;
        std::thread::scope(|sc| {
            let mut pending: Option<std::thread::ScopedJoinHandle<'_, Volume>> = None;
            for i in 0..z.t.div_ceil(self.cfg.block) {
                let (pix, (c, ft, ph, pw)) = self.head_block(be, &rows, (z.t, z.h, z.w), i);
                let data = be.download(&pix);
                if let Some(p) = pending.take() {
                    let px = p.join().expect("fast decode chunk thread");
                    written += px.t;
                    on_chunk(&px);
                }
                pending = Some(sc.spawn(move || {
                    let mut v =
                        Volume::from_data(c, ft, ph, pw, data).crop(trim, ft - trim, 0, ph, 0, pw);
                    denormalize_pixels(&mut v);
                    v
                }));
            }
            if let Some(p) = pending.take() {
                let px = p.join().expect("fast decode chunk thread");
                written += px.t;
                on_chunk(&px);
            }
        });
        let want = self.cfg.frames(z.t);
        if written != want {
            return Err(Error::Geometry(format!(
                "fast VAE decoder wrote {written} frames, expected {want}"
            )));
        }
        Ok(written)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3_core::rng::Rng;
    use h3_core::weights::Synthetic;
    use h3_cpu::Cpu;

    #[test]
    fn window_runs_group_equal_windows() {
        assert_eq!(window_runs(48, 16, 0), vec![(0, 16, 3, 16)]);
        assert_eq!(
            window_runs(48, 16, 8),
            vec![(0, 8, 1, 0), (8, 16, 2, 16), (40, 8, 1, 0)]
        );
        assert_eq!(window_runs(90, 16, 0), vec![(0, 16, 5, 16), (80, 10, 1, 0)]);
        assert_eq!(window_runs(16, 16, 8), vec![(0, 8, 2, 8)]);
        assert_eq!(window_runs(5, 4, 2), vec![(0, 2, 1, 0), (2, 3, 1, 0)]);
        assert_eq!(window_runs(3, 16, 0), vec![(0, 3, 1, 0)]);
    }

    fn rms(x: &[f32], w: Option<&[f32]>, eps: f32) -> Vec<f32> {
        let m = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let r = 1.0 / (m + eps).sqrt();
        x.iter()
            .enumerate()
            .map(|(i, v)| v * r * w.map_or(1.0, |w| w[i]))
            .collect()
    }

    fn mat(x: &[f32], w: &[f32], b: &[f32], rows_out: usize) -> Vec<f32> {
        let cols = x.len();
        (0..rows_out)
            .map(|o| b[o] + (0..cols).map(|i| w[o * cols + i] * x[i]).sum::<f32>())
            .collect()
    }

    fn attend(q: &[f32], keys: &[&[f32]], vals: &[&[f32]]) -> Vec<f32> {
        let d = q.len();
        let sc = 1.0 / (d as f32).sqrt();
        let s: Vec<f32> = keys
            .iter()
            .map(|k| q.iter().zip(*k).map(|(a, b)| a * b).sum::<f32>() * sc)
            .collect();
        let m = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = s.iter().map(|v| (v - m).exp()).collect();
        let z: f32 = e.iter().sum();
        let mut out = vec![0.0; d];
        for (p, v) in e.iter().zip(vals) {
            for c in 0..d {
                out[c] += p / z * v[c];
            }
        }
        out
    }

    /// Sana's `NonCausalViTDecoder.forward`, transcribed literally on the
    /// host: original weight layouts, padded windows with key masks.
    fn oracle(cfg: &FastVaeConfig, src: &dyn WeightSource, z: &Volume) -> Volume {
        let (lz, t, h, w) = (z.c, z.t, z.h, z.w);
        let (dd, hd, heads) = (cfg.dim(), cfg.head_dim, cfg.heads);
        let ff = cfg.ffn_mult * dd;
        let (tb, ws, sg) = (cfg.block, cfg.window, cfg.globals());
        let nb = t.div_ceil(tb);
        let get = |k: &str, sh: &[usize]| src.get(k, sh).unwrap();
        let wc = get("post_quant_conv.weight", &[lz, lz]);
        let bc = get("post_quant_conv.bias", &[lz]);
        let we = get("x_embedder.weight", &[dd, lz]);
        let bev = get("x_embedder.bias", &[dd]);
        let n = t * h * w;
        let tok = |i: usize| -> (usize, usize, usize) { (i / (h * w), (i / w) % h, i % w) };
        let mut x: Vec<Vec<f32>> = (0..n)
            .map(|i| {
                let (ti, yi, xi) = tok(i);
                let zv: Vec<f32> = (0..lz).map(|c| z.at(c, ti, yi, xi)).collect();
                mat(&mat(&zv, &wc, &bc, lz), &we, &bev, dd)
            })
            .collect();
        let reg = get("register_tokens", &[1, cfg.registers, dd]);
        let mut g: Vec<Vec<f32>> = (0..nb * sg)
            .map(|r| {
                let k = r % sg;
                if k < cfg.registers {
                    reg[k * dd..(k + 1) * dd].to_vec()
                } else {
                    vec![0.0; dd]
                }
            })
            .collect();
        let per_axis = cfg.rot / 6;
        let inv: Vec<f64> = (0..per_axis)
            .map(|k| 1.0 / cfg.theta.powf(k as f64 * 6.0 / cfg.rot as f64))
            .collect();
        for l in 0..cfg.layers {
            let p = format!("blocks.{l}");
            let (s, st) = cfg.shifts(l);
            let n1 = get(&format!("{p}.norm1.weight"), &[dd]);
            let n2 = get(&format!("{p}.norm2.weight"), &[dd]);
            let wq = get(&format!("{p}.attn.to_qkv.weight"), &[3 * dd, dd]);
            let bq = get(&format!("{p}.attn.to_qkv.bias"), &[3 * dd]);
            let wo = get(&format!("{p}.attn.to_out.weight"), &[dd, dd]);
            let bo = get(&format!("{p}.attn.to_out.bias"), &[dd]);
            let w1 = get(&format!("{p}.ff.w1.weight"), &[2 * ff, dd]);
            let b1 = get(&format!("{p}.ff.w1.bias"), &[2 * ff]);
            let w2 = get(&format!("{p}.ff.w2.weight"), &[dd, ff]);
            let b2 = get(&format!("{p}.ff.w2.bias"), &[dd]);
            let s1 = get(&format!("{p}.scale1"), &[dd]);
            let s2 = get(&format!("{p}.scale2"), &[dd]);
            // per row → per head (q, k, v), q / k normed.
            let split = |row: &[f32]| -> Vec<[Vec<f32>; 3]> {
                let qkv = mat(&rms(row, Some(&n1), cfg.eps), &wq, &bq, 3 * dd);
                (0..heads)
                    .map(|hh| {
                        let part = |i: usize| {
                            qkv[hh * 3 * hd + i * hd..hh * 3 * hd + (i + 1) * hd].to_vec()
                        };
                        [
                            rms(&part(0), None, cfg.eps),
                            rms(&part(1), None, cfg.eps),
                            part(2),
                        ]
                    })
                    .collect()
            };
            let rope = |v: &mut Vec<f32>, i: usize| {
                let (ti, yi, xi) = tok(i);
                let id = |a: usize, sh: usize, size: usize| {
                    ((((a + sh) % size) as f64) + 0.5) / size as f64 * 2.0 - 1.0
                };
                let ids = [id(ti, st, tb), id(yi, s, ws), id(xi, s, ws)];
                let mut ang = Vec::new();
                for a in ids {
                    for f in &inv {
                        ang.push((std::f64::consts::TAU * a * f) as f32);
                    }
                }
                let half = cfg.rot / 2;
                let old = v.clone();
                for i in 0..cfg.rot {
                    let a = ang[i % half];
                    let rh = if i < half {
                        -old[i + half]
                    } else {
                        old[i - half]
                    };
                    v[i] = old[i] * a.cos() + rh * a.sin();
                }
            };
            let mut xs: Vec<Vec<[Vec<f32>; 3]>> = x.iter().map(|r| split(r)).collect();
            for (i, hs) in xs.iter_mut().enumerate() {
                for hv in hs.iter_mut() {
                    rope(&mut hv[0], i);
                    rope(&mut hv[1], i);
                }
            }
            let gs: Vec<Vec<[Vec<f32>; 3]>> = g.iter().map(|r| split(r)).collect();
            // window attention over the padded partition
            let tp = (t + st).div_ceil(tb) * tb;
            let hp = (h + s).div_ceil(ws) * ws;
            let wp = (w + s).div_ceil(ws) * ws;
            let mut o = vec![vec![0.0f32; dd]; n];
            for jt in 0..tp / tb {
                let lo = (jt * tb).saturating_sub(st);
                let hi = ((jt + 1) * tb - st - 1).min(t - 1);
                let (b0, b1) = (lo / tb, hi / tb);
                let gblocks: Vec<usize> = if b1 != b0 { vec![b0, b1] } else { vec![b0] };
                for jy in 0..hp / ws {
                    for jx in 0..wp / ws {
                        // valid tokens of this padded window
                        let mut members = Vec::new();
                        for a in 0..tb {
                            for b in 0..ws {
                                for c in 0..ws {
                                    let (pt, py, px) = (jt * tb + a, jy * ws + b, jx * ws + c);
                                    if pt >= st
                                        && pt < st + t
                                        && py >= s
                                        && py < s + h
                                        && px >= s
                                        && px < s + w
                                    {
                                        members.push(((pt - st) * h + py - s) * w + px - s);
                                    }
                                }
                            }
                        }
                        for &qi in &members {
                            for hh in 0..heads {
                                let mut keys: Vec<&[f32]> =
                                    members.iter().map(|&m| &xs[m][hh][1][..]).collect();
                                let mut vals: Vec<&[f32]> =
                                    members.iter().map(|&m| &xs[m][hh][2][..]).collect();
                                for &gb in &gblocks {
                                    for r in gb * sg..(gb + 1) * sg {
                                        keys.push(&gs[r][hh][1]);
                                        vals.push(&gs[r][hh][2]);
                                    }
                                }
                                let a = attend(&xs[qi][hh][0], &keys, &vals);
                                o[qi][hh * hd..(hh + 1) * hd].copy_from_slice(&a);
                            }
                        }
                    }
                }
            }
            // global attention
            let mut og = vec![vec![0.0f32; dd]; nb * sg];
            for bi in 0..nb {
                let toks: Vec<usize> = (bi * tb * h * w..((bi + 1) * tb).min(t) * h * w).collect();
                for r in bi * sg..(bi + 1) * sg {
                    for hh in 0..heads {
                        let mut keys: Vec<&[f32]> =
                            toks.iter().map(|&m| &xs[m][hh][1][..]).collect();
                        let mut vals: Vec<&[f32]> =
                            toks.iter().map(|&m| &xs[m][hh][2][..]).collect();
                        for r2 in bi * sg..(bi + 1) * sg {
                            keys.push(&gs[r2][hh][1]);
                            vals.push(&gs[r2][hh][2]);
                        }
                        let a = attend(&gs[r][hh][0], &keys, &vals);
                        og[r][hh * hd..(hh + 1) * hd].copy_from_slice(&a);
                    }
                }
            }
            let update = |rows: &mut Vec<Vec<f32>>, os: &[Vec<f32>]| {
                for (row, orow) in rows.iter_mut().zip(os) {
                    let a = mat(orow, &wo, &bo, dd);
                    for c in 0..dd {
                        row[c] += a[c] * s1[c];
                    }
                    let hid = mat(&rms(row, Some(&n2), cfg.eps), &w1, &b1, 2 * ff);
                    let act: Vec<f32> = (0..ff)
                        .map(|i| {
                            let gt = hid[i];
                            gt / (1.0 + (-gt).exp()) * hid[ff + i]
                        })
                        .collect();
                    let f = mat(&act, &w2, &b2, dd);
                    for c in 0..dd {
                        row[c] += f[c] * s2[c];
                    }
                }
            };
            update(&mut x, &o);
            update(&mut g, &og);
        }
        let lnw = get("norm_out.weight", &[dd]);
        let lnb = get("norm_out.bias", &[dd]);
        let od = cfg.out_channels * cfg.patch_t * cfg.patch * cfg.patch;
        let pw = get("proj_out.weight", &[od, dd]);
        let pb = get("proj_out.bias", &[od]);
        let (pt, ps) = (cfg.patch_t, cfg.patch);
        let mut full = Volume::zeros(cfg.out_channels, t * pt, h * ps, w * ps);
        for (i, row) in x.iter().enumerate() {
            let m = row.iter().sum::<f32>() / dd as f32;
            let var = row.iter().map(|v| (v - m) * (v - m)).sum::<f32>() / dd as f32;
            let ln: Vec<f32> = (0..dd)
                .map(|c| (row[c] - m) / (var + cfg.eps).sqrt() * lnw[c] + lnb[c])
                .collect();
            let f = mat(&ln, &pw, &pb, od);
            let (ti, yi, xi) = tok(i);
            for c in 0..cfg.out_channels {
                for kt in 0..pt {
                    for kh in 0..ps {
                        for kw in 0..ps {
                            let v = f[((c * pt + kt) * ps + kh) * ps + kw];
                            let (ft, fy, fx) = (ti * pt + kt, yi * ps + kh, xi * ps + kw);
                            let fh = h * ps;
                            let fw = w * ps;
                            full.data[((c * t * pt + ft) * fh + fy) * fw + fx] = v;
                        }
                    }
                }
            }
        }
        let parts: Vec<Volume> = (0..nb)
            .map(|bi| {
                let (lo, hi) = (bi * tb * pt + cfg.trim, ((bi + 1) * tb).min(t) * pt);
                full.crop(lo, hi - lo, 0, full.h, 0, full.w)
            })
            .collect();
        Volume::cat_t(&parts)
    }

    #[test]
    fn matches_the_sana_transcription() {
        let cfg = FastVaeConfig::tiny(4);
        let src = Synthetic { seed: 5 };
        let be = Cpu;
        let dec = FastDecoder::load(&be, cfg.clone(), &src, WeightQuant::Native).unwrap();
        for (t, h, w) in [(7, 5, 7), (3, 4, 4), (11, 3, 9)] {
            let z = Volume::from_data(4, t, h, w, Rng::new(t as u64).normal(4 * t * h * w));
            let got = dec.decode_normalized(&be, &z).unwrap();
            let want = oracle(&cfg, &src, &z);
            assert_eq!(
                (got.c, got.t, got.h, got.w),
                (want.c, want.t, want.h, want.w)
            );
            assert_eq!(got.t, cfg.frames(t));
            let num: f64 = got
                .data
                .iter()
                .zip(&want.data)
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum();
            let den: f64 = want.data.iter().map(|b| (*b as f64).powi(2)).sum();
            let rel = (num / den).sqrt();
            assert!(rel < 1e-4, "{t}x{h}x{w}: rel {rel:.2e}");
        }
    }
}
