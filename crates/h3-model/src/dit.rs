//! The H3 Omni-Transformer (`MiniMaxH3Transformer3DModel`) over one packed
//! sequence. Per block, in order:
//!
//! ```text
//! n   = rms(h)·w1·(1 + scale_msa[idx]) + shift_msa[idx]       rms_norm (modulated)
//! qkv = n·Wqkvᵀ                (+ g = n·Wgateᵀ for VSA)        linear (fused q|k|v)
//! q,k = rope(rms_head(q,k))                                    head_split ×3 (+1)
//! a   = attention(q, k, v) or vsa(q, k, v, g)
//! h  += a·Woutᵀ · gate_msa[idx]; n = rms(h)·w2·(1 + scale_mlp) + shift_mlp   residual_rms_norm
//! h  += (swiglu(n·Wffᵀ)·Wff2ᵀ) · gate_mlp[idx]                 linear, swiglu, linear, gate_residual
//! ```
//!
//! `idx[row] = timestep_index[row]·3 + token_tag[row]` selects the row's
//! modulation entry; the tables are a pure function of the timesteps, so they
//! are computed once per generation ([`H3Dit::adaln_tables`]) and the 13 B
//! AdaLN parameters never live on the device.

use crate::linear::{self, HostLinear, Linear, Shapes};
use h3_core::{
    Error, Result,
    backend::{Act, Backend, Gate, HeadNorm, ModRef, ModTable, Modulation, Rope},
    config::H3Config,
    lora::Adapter,
    packing::Layout,
    quant::WeightQuant,
    rope::h3_rope,
    timestep::TimeEmbedder,
    vsa::{VsaSpec, segments_for_layout},
    weights::{ADALN_TABLE_KEY, WeightSource, h3_manifest},
};

pub const SHIFT_MSA: usize = 0;
pub const SCALE_MSA: usize = 1;
pub const GATE_MSA: usize = 2;
pub const SHIFT_MLP: usize = 3;
pub const SCALE_MLP: usize = 4;
pub const GATE_MLP: usize = 5;

/// One attention + FFN unit (a token-refiner block, or a transformer block
/// without its AdaLN).
pub struct AttnFf<B: Backend> {
    pub qkv: Linear<B>,
    pub out: Linear<B>,
    pub gate: Option<Linear<B>>,
    pub norm_q: B::Buf,
    pub norm_k: B::Buf,
    pub norm1: B::Buf,
    pub norm2: B::Buf,
    pub ff_in: Linear<B>,
    pub ff_out: Linear<B>,
}

/// The same unit kept in host memory (pre-quantized), uploaded per step.
pub struct HostAttnFf {
    qkv: HostLinear,
    out: HostLinear,
    gate: Option<HostLinear>,
    norm_q: Vec<f32>,
    norm_k: Vec<f32>,
    norm1: Vec<f32>,
    norm2: Vec<f32>,
    ff_in: HostLinear,
    ff_out: HostLinear,
}

impl HostAttnFf {
    fn pin<B: Backend>(&self, be: &B, out: &mut Vec<crate::linear::PinGuard>) {
        for l in [&self.qkv, &self.out, &self.ff_in, &self.ff_out].into_iter().chain(self.gate.as_ref()) {
            l.pin(be, out);
        }
    }

    fn upload<B: Backend>(&self, be: &B) -> AttnFf<B> {
        AttnFf {
            qkv: self.qkv.upload(be),
            out: self.out.upload(be),
            gate: self.gate.as_ref().map(|g| g.upload(be)),
            norm_q: be.upload(&self.norm_q),
            norm_k: be.upload(&self.norm_k),
            norm1: be.upload(&self.norm1),
            norm2: be.upload(&self.norm2),
            ff_in: self.ff_in.upload(be),
            ff_out: self.ff_out.upload(be),
        }
    }
    pub fn bytes(&self) -> usize {
        self.qkv.w.bytes() + self.out.w.bytes() + self.ff_in.w.bytes() + self.ff_out.w.bytes() + self.gate.as_ref().map_or(0, |g| g.w.bytes())
    }
}

pub enum Residency<B: Backend> {
    Device(AttnFf<B>),
    /// Streamed: uploaded for one block's forward and dropped.
    Host(HostAttnFf),
}

/// The output projections, possibly a fused PDD block.
pub struct OutputHeads<B: Backend> {
    pub video: Linear<B>,
    pub audio: Linear<B>,
}

/// Widened PDD heads on the host, `[pdd_steps · out, d]`.
pub struct PddHeads {
    pub steps: usize,
    pub video_w: Vec<f32>,
    pub video_b: Vec<f32>,
    pub audio_w: Vec<f32>,
    pub audio_b: Vec<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Attention {
    /// Full attention; `qk8` runs INT8 Q·Kᵀ ([`Backend::attention_qk8`]),
    /// `pv8` FP8 P·V as well.
    Dense { qk8: bool, pv8: bool },
    /// Video sparse attention: target-region sparsity, tile size (64 / 128 /
    /// 256 tokens) and reference-region sparsity.
    Vsa { sparsity: f64, tile: usize, ref_sparsity: f64, qk8: bool, pv8: bool },
}

#[derive(Debug, Clone)]
pub struct LoadOptions {
    pub quant: WeightQuant,
    /// Keep the transformer blocks' linears in host memory and upload them
    /// per step (for devices the plan says they do not fit).
    pub stream_blocks: bool,
}

impl Default for LoadOptions {
    fn default() -> Self {
        Self { quant: WeightQuant::Native, stream_blocks: false }
    }
}

pub struct H3Dit<B: Backend> {
    pub cfg: H3Config,
    pub proj_in: Linear<B>,
    pub audio_proj_in: Linear<B>,
    pub context_embedder: Linear<B>,
    pub refiner: Vec<AttnFf<B>>,
    pub final_norm: B::Buf,
    /// Page-lock guards of the streamed blocks' host buffers: declared before
    /// `blocks` so they drop (unlock) before the buffers are freed.
    #[allow(dead_code)]
    pins: Vec<crate::linear::PinGuard>,
    pub blocks: Vec<Residency<B>>,
    pub norm_out_w: B::Buf,
    pub heads: OutputHeads<B>,
    pub pdd: Option<PddHeads>,
    pub time: TimeEmbedder,
    /// `norm_out.linear` on the host: `[2d, adaln_in]` + bias.
    norm_out_lin: (Vec<f32>, Vec<f32>),
    pub has_gate: bool,
    pub quant: WeightQuant,
}

/// AdaLN tables for a set of timesteps: per block `[U, 18·d]` (row `u` holds
/// the three modalities' six slots), and the final norm's `[U, 2·d]`.
pub struct AdalnTables {
    pub timesteps: Vec<f32>,
    pub blocks: Vec<Vec<f32>>,
    pub head: Vec<f32>,
    pub d: usize,
}

impl AdalnTables {
    pub fn index_of(&self, t: f32) -> Option<usize> {
        self.timesteps.iter().position(|&x| x == t)
    }
    pub fn bytes(&self) -> usize {
        4 * (self.blocks.iter().map(Vec::len).sum::<usize>() + self.head.len())
    }
    /// Every timestep of `needed` has a row.
    pub fn covers(&self, needed: &[f32]) -> bool {
        needed.iter().all(|t| self.index_of(*t).is_some())
    }

    /// Store as one safetensors file (`h3.format = "adaln/1"`): `timesteps`
    /// F32 `[U]`, `block.N` F16 `[U, 18·d]`, `head` F16 `[U, 2·d]`. Values are
    /// bf16-sized in the checkpoint, so f16 storage keeps them exact enough
    /// (|x| < 65504 is checked; F32 otherwise).
    pub fn save(&self, path: &std::path::Path, variant: &str) -> Result<()> {
        use h3_core::export::{Tensor, f16_bytes, f32_bytes};
        use safetensors::Dtype;
        let u = self.timesteps.len();
        let mut t: Vec<Tensor> = vec![("timesteps".into(), Dtype::F32, vec![u], f32_bytes(&self.timesteps))];
        let put = |name: String, v: &[f32], cols: usize| -> Tensor {
            if v.iter().all(|x| x.is_finite() && x.abs() <= 65504.0) {
                (name, Dtype::F16, vec![u, cols], f16_bytes(v))
            } else {
                (name, Dtype::F32, vec![u, cols], f32_bytes(v))
            }
        };
        for (i, b) in self.blocks.iter().enumerate() {
            t.push(put(format!("block.{i}"), b, b.len() / u));
        }
        t.push(put("head".into(), &self.head, 2 * self.d));
        let meta = std::collections::HashMap::from([
            ("h3.format".to_string(), "adaln/1".to_string()),
            ("h3.variant".to_string(), variant.to_string()),
            ("h3.hidden".to_string(), self.d.to_string()),
        ]);
        h3_core::export::write(path, &t, meta)
    }

    pub fn load(path: &std::path::Path, cfg: &H3Config) -> Result<Self> {
        let src = h3_core::weights::SafetensorsDir::open(path)?;
        let (_, ts_shape, _) = src.raw("timesteps")?;
        let u = ts_shape[0];
        let timesteps = src.get("timesteps", &[u])?;
        let d = cfg.hidden;
        let mods = H3Config::MOD_SLOTS * H3Config::MODALITIES * d;
        let mut blocks = Vec::with_capacity(cfg.layers);
        for i in 0..cfg.layers {
            blocks.push(src.get(&format!("block.{i}"), &[u, mods])?);
        }
        let head = src.get("head", &[u, 2 * d])?;
        Ok(Self { timesteps, blocks, head, d })
    }
}

/// Per-step device conditioning: row indices and one modulation table per block.
pub struct StepMods<B: Backend> {
    /// `[rows]` u32: `timestep_index·3 + tag`.
    pub adaln_idx: B::Buf,
    /// `[rows]` u32: timestep index (the final norm).
    pub t_idx: B::Buf,
    /// `[U·3, 6, d]` per block.
    pub blocks: Vec<B::Buf>,
    /// `[U, 2, d]`.
    pub head: B::Buf,
    pub entries: usize,
    /// Run every attention dense this step (VSA warm-up steps).
    pub dense: bool,
}

/// A contiguous run of packed rows: `(packed start, rows, offset in the
/// modality's own row buffer)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Run {
    pub start: usize,
    pub rows: usize,
    pub offset: usize,
}

pub fn runs(indices: &[u32]) -> Vec<Run> {
    let mut v: Vec<Run> = Vec::new();
    let mut off = 0;
    for &i in indices {
        match v.last_mut() {
            Some(r) if r.start + r.rows == i as usize => r.rows += 1,
            _ => v.push(Run { start: i as usize, rows: 1, offset: off }),
        }
        off += 1;
    }
    v
}

/// Activation buffers for one layout, reused by every block and step.
pub struct Workspace<B: Backend> {
    pub attention: Attention,
    pub tokens: usize,
    pub ffn_chunk: usize,
    pub video_runs: Vec<Run>,
    pub audio_runs: Vec<Run>,
    pub text_rows: usize,
    vsa: Option<B::Vsa>,
    vsa_dense: Option<B::Vsa>,
    h: B::Buf,
    normed: B::Buf,
    qkv: B::Buf,
    g: B::Buf,
    q: B::Buf,
    k: B::Buf,
    v: B::Buf,
    attn: B::Buf,
    tmp: B::Buf,
    hid: B::Buf,
    hid2: B::Buf,
    rope_cos: B::Buf,
    rope_sin: B::Buf,
}

impl<B: Backend> H3Dit<B> {
    pub fn load(be: &B, cfg: H3Config, src: &dyn WeightSource, opts: &LoadOptions, adapters: &[Adapter]) -> Result<Self> {
        let d = cfg.hidden;
        let qkv = cfg.dim();
        let has_gate = (0..cfg.layers).all(|l| src.has_linear(&format!("transformer_blocks.{l}.attn.to_gate_compress")))
            || adapters.iter().any(|a| a.touches("transformer_blocks.0.attn.to_gate_compress"));
        let shapes: Shapes = h3_manifest(&cfg, true).into_iter().collect();
        let vec = |k: &str, n: usize| -> Result<Vec<f32>> {
            let mut v = src.get(k, &[n])?;
            for a in adapters {
                a.apply_vector(k.strip_suffix(".weight").unwrap_or(k), &mut v)?;
            }
            Ok(v)
        };
        let unit_host = |p: &str, gate: bool, quant: WeightQuant| -> Result<HostAttnFf> {
            let lin = |keys: &[&str], bias: bool| linear::load_host(src, &shapes, keys, bias, quant, adapters);
            Ok(HostAttnFf {
                qkv: lin(&[&format!("{p}.attn.to_q"), &format!("{p}.attn.to_k"), &format!("{p}.attn.to_v")], false)?,
                out: lin(&[&format!("{p}.attn.to_out.0")], false)?,
                gate: if gate { Some(lin(&[&format!("{p}.attn.to_gate_compress")], false)?) } else { None },
                norm_q: vec(&format!("{p}.attn.norm_q.weight"), cfg.head_dim)?,
                norm_k: vec(&format!("{p}.attn.norm_k.weight"), cfg.head_dim)?,
                norm1: vec(&format!("{p}.norm1.weight"), d)?,
                norm2: vec(&format!("{p}.norm2.weight"), d)?,
                ff_in: lin(&[&format!("{p}.ff.net.0.proj")], false)?,
                ff_out: lin(&[&format!("{p}.ff.net.2")], false)?,
            })
        };
        let mut refiner = Vec::with_capacity(cfg.refiner_layers);
        for l in 0..cfg.refiner_layers {
            refiner.push(unit_host(&format!("token_refiner.refiner_blocks.{l}"), false, WeightQuant::Native)?.upload(be));
        }
        let mut blocks = Vec::with_capacity(cfg.layers);
        let mut pins = Vec::new();
        // MXFP8 leaves the first two and last three blocks in bf16 (`for_block`);
        // `H3_MXFP8_ALL_BLOCKS=1` quantizes them too (a speed / quality experiment).
        let all_blocks = std::env::var("H3_MXFP8_ALL_BLOCKS").is_ok_and(|v| v == "1");
        for l in 0..cfg.layers {
            let q = if all_blocks { opts.quant } else { opts.quant.for_block(l, cfg.layers) };
            let h = unit_host(&format!("transformer_blocks.{l}"), has_gate, q)?;
            blocks.push(if opts.stream_blocks {
                h.pin(be, &mut pins);
                Residency::Host(h)
            } else {
                Residency::Device(h.upload(be))
            });
        }
        let plain = |k: &str| -> Result<Linear<B>> {
            Ok(linear::load_host(src, &shapes, &[k], true, WeightQuant::Native, adapters)?.upload(be))
        };
        let te_in = cfg.time_embed_hidden;
        let adaln_in = cfg.adaln_rank.unwrap_or(cfg.time_embed_dim);
        let time = if src.contains(ADALN_TABLE_KEY) {
            let shape = src.shape(ADALN_TABLE_KEY)?;
            if shape.len() != 2 || shape[1] != adaln_in || shape[0] < 2 {
                return Err(Error::Weights(format!("`{ADALN_TABLE_KEY}` is {shape:?}, expected [grid, {adaln_in}]")));
            }
            let data = src.get(ADALN_TABLE_KEY, &shape)?;
            TimeEmbedder::from_table(h3_core::timestep::AdalnTable { grid: shape[0], width: shape[1], data })
        } else {
        TimeEmbedder {
            table: None,
            freq_dim: cfg.freq_dim,
            l1w: {
                let mut w = src.get("time_embedder.linear_1.weight", &[te_in, cfg.freq_dim])?;
                for a in adapters {
                    a.apply_weight("time_embedder.linear_1", &mut w, te_in, cfg.freq_dim)?;
                }
                w
            },
            l1b: {
                let mut b = src.get("time_embedder.linear_1.bias", &[te_in])?;
                for a in adapters {
                    a.apply_bias("time_embedder.linear_1", &mut b)?;
                }
                b
            },
            l2w: {
                let mut w = src.get("time_embedder.linear_2.weight", &[cfg.time_embed_dim, te_in])?;
                for a in adapters {
                    a.apply_weight("time_embedder.linear_2", &mut w, cfg.time_embed_dim, te_in)?;
                }
                w
            },
            l2b: {
                let mut b = src.get("time_embedder.linear_2.bias", &[cfg.time_embed_dim])?;
                for a in adapters {
                    a.apply_bias("time_embedder.linear_2", &mut b)?;
                }
                b
            },
            basis: match cfg.adaln_rank {
                Some(r) => Some(src.get("adaln_basis.weight", &[r, cfg.time_embed_dim])?),
                None => None,
            },
        }
        };
        let norm_out_lin = {
            let mut w = src.get("norm_out.linear.weight", &[2 * d, adaln_in])?;
            let mut b = src.get("norm_out.linear.bias", &[2 * d])?;
            for a in adapters {
                a.apply_weight("norm_out.linear", &mut w, 2 * d, adaln_in)?;
                a.apply_bias("norm_out.linear", &mut b)?;
            }
            (w, b)
        };
        let heads_n = cfg.pdd_steps.unwrap_or(1);
        let vp = cfg.video_patch_dim();
        let ap = cfg.audio_in_channels;
        let (heads, pdd) = if heads_n > 1 {
            let video_w = src.get("proj_out.weight", &[heads_n * vp, d])?;
            let video_b = src.get("proj_out.bias", &[heads_n * vp])?;
            let audio_w = src.get("audio_proj_out.weight", &[heads_n * ap, d])?;
            let audio_b = src.get("audio_proj_out.bias", &[heads_n * ap])?;
            let pdd = PddHeads { steps: heads_n, video_w, video_b, audio_w, audio_b };
            // Default heads: the first fine-grid interval.
            let w = vec![1.0f64 / heads_n as f64; heads_n];
            let heads = fused_heads(be, &pdd, &w, &w, 0, heads_n, d);
            (heads, Some(pdd))
        } else {
            (OutputHeads { video: plain("proj_out")?, audio: plain("audio_proj_out")? }, None)
        };
        let _ = qkv;
        Ok(Self {
            proj_in: plain("proj_in")?,
            audio_proj_in: plain("audio_proj_in")?,
            context_embedder: plain("context_embedder")?,
            refiner,
            final_norm: be.upload(&vec("token_refiner.final_norm.weight", d)?),
            pins,
            blocks,
            norm_out_w: be.upload(&vec("norm_out.norm.weight", d)?),
            heads,
            pdd,
            time,
            norm_out_lin,
            has_gate,
            quant: opts.quant,
            cfg,
        })
    }

    /// Fuse the PDD heads of fine-grid block `[start, end)` with each
    /// modality's integration weights over the whole grid (normalised within
    /// the block).
    pub fn fused_heads(&self, be: &B, video_weights: &[f64], audio_weights: &[f64], start: usize, end: usize) -> OutputHeads<B> {
        let pdd = self.pdd.as_ref().expect("fused_heads needs a PDD checkpoint");
        fused_heads(be, pdd, video_weights, audio_weights, start, end, self.cfg.hidden)
    }

    /// The modulation tables for `timesteps` (clean times in `[0, 1]`),
    /// streaming each block's AdaLN weight through the backend once. `extra`
    /// adds a per-block, per-timestep delta to the projection (the Regen-2K
    /// Sharp curve branches): `extra(block, timesteps) → [U, 18·d]`.
    pub fn adaln_tables(
        &self,
        be: &B,
        src: &dyn WeightSource,
        adapters: &[Adapter],
        timesteps: &[f32],
        extra: Option<&dyn Fn(usize, &[f32]) -> Vec<f32>>,
    ) -> Result<AdalnTables> {
        let cfg = &self.cfg;
        let d = cfg.hidden;
        let u = timesteps.len();
        let adaln_in = cfg.adaln_rank.unwrap_or(cfg.time_embed_dim);
        let mods = H3Config::MOD_SLOTS * H3Config::MODALITIES * d;
        let mut x = Vec::with_capacity(u * adaln_in);
        for &t in timesteps {
            x.extend(self.time.adaln_input(t));
        }
        let gx = be.upload(&x);
        let gy = be.alloc(u * mods);
        let shapes: Shapes = h3_manifest(cfg, false).into_iter().collect();
        let mut blocks = Vec::with_capacity(cfg.layers);
        for l in 0..cfg.layers {
            let key = format!("transformer_blocks.{l}.adaln_proj.linear");
            let lin = linear::load_host(src, &shapes, &[&key], true, WeightQuant::Native, adapters)?.upload(be);
            be.linear(&gx, 0, u, &lin.w, lin.b.as_ref(), Act::None, &gy, 0);
            let mut t = be.download(&gy);
            t.truncate(u * mods);
            if let Some(f) = extra {
                let delta = f(l, timesteps);
                if delta.len() != t.len() {
                    return Err(Error::Msg(format!("AdaLN delta for block {l} has {} values, expected {}", delta.len(), t.len())));
                }
                t.iter_mut().zip(delta).for_each(|(a, b)| *a += b);
            }
            blocks.push(t);
        }
        let (hw, hb) = &self.norm_out_lin;
        let mut head = Vec::with_capacity(u * 2 * d);
        for i in 0..u {
            head.extend(h3_core::timestep::matvec(hw, Some(hb), &x[i * adaln_in..(i + 1) * adaln_in]));
        }
        Ok(AdalnTables { timesteps: timesteps.to_vec(), blocks, head, d })
    }

    /// Device tables for one step: `unique` are the step's distinct timesteps
    /// (sorted, from [`h3_core::packing::row_timesteps`]), `t_idx[row]` the
    /// row's index into them, `tags[row]` its modality.
    pub fn step_mods(&self, be: &B, tables: &AdalnTables, unique: &[f32], t_idx: &[u32], tags: &[u8], dense: bool) -> Result<StepMods<B>> {
        let d = self.cfg.hidden;
        let slots = H3Config::MOD_SLOTS;
        let mut us = Vec::with_capacity(unique.len());
        for &t in unique {
            us.push(tables.index_of(t).ok_or_else(|| Error::Msg(format!("no AdaLN table for timestep {t}")))?);
        }
        let blocks = tables
            .blocks
            .iter()
            .map(|tb| {
                let mut m = Vec::with_capacity(unique.len() * 3 * slots * d);
                for &u in &us {
                    // Row u of the table: [tag 0: 6 slots][tag 1][tag 2], each slot d wide.
                    m.extend_from_slice(&tb[u * 3 * slots * d..(u + 1) * 3 * slots * d]);
                }
                be.upload_table(&m)
            })
            .collect();
        let mut head = Vec::with_capacity(unique.len() * 2 * d);
        for &u in &us {
            head.extend_from_slice(&tables.head[u * 2 * d..(u + 1) * 2 * d]);
        }
        let idx: Vec<u32> = t_idx.iter().zip(tags).map(|(&t, &g)| t * 3 + g as u32).collect();
        Ok(StepMods {
            adaln_idx: be.upload_u32(&idx),
            t_idx: be.upload_u32(t_idx),
            blocks,
            head: be.upload_table(&head),
            entries: unique.len(),
            dense,
        })
    }

    /// Context embedding + token refiner + final norm of `[rows, text_dim]`
    /// prompt embeddings (once per prompt).
    pub fn refine_text(&self, be: &B, embeds: &[f32]) -> B::Buf {
        let c = &self.cfg;
        let (d, text_dim) = (c.hidden, c.text_dim);
        assert_eq!(embeds.len() % text_dim, 0, "prompt embeds must be [rows, text_dim]");
        let rows = embeds.len() / text_dim;
        let x = be.upload(embeds);
        let h = be.alloc(rows * d);
        be.linear(&x, 0, rows, &self.context_embedder.w, self.context_embedder.b.as_ref(), Act::None, &h, 0);
        let qkv = c.dim();
        let bufs = (
            be.alloc(rows * d),
            be.alloc(rows * 3 * qkv),
            be.alloc(rows * qkv),
            be.alloc(rows * qkv),
            be.alloc(rows * qkv),
            be.alloc(rows * qkv),
            be.alloc(rows * d),
            be.alloc(rows * 2 * c.ffn_dim),
            be.alloc(rows * c.ffn_dim),
        );
        let (normed, qkvb, q, k, v, attn, tmp, hid, hid2) = (&bufs.0, &bufs.1, &bufs.2, &bufs.3, &bufs.4, &bufs.5, &bufs.6, &bufs.7, &bufs.8);
        for bl in &self.refiner {
            be.rms_norm(&h, rows, d, Some(&bl.norm1), Modulation::None, c.norm_eps, normed);
            be.linear(normed, 0, rows, &bl.qkv.w, None, Act::None, qkvb, 0);
            self.split_qkv(be, bl, qkvb, rows, None, q, k, v);
            be.attention(q, k, v, c.heads, c.heads, rows, rows, c.head_dim, false, attn);
            be.linear(attn, 0, rows, &bl.out.w, None, Act::None, tmp, 0);
            be.residual_rms_norm(&h, tmp, rows, d, Gate::None, Some(&bl.norm2), Modulation::None, c.norm_eps, normed);
            be.linear(normed, 0, rows, &bl.ff_in.w, None, Act::None, hid, 0);
            be.swiglu(hid, rows, c.ffn_dim, hid2);
            be.linear(hid2, 0, rows, &bl.ff_out.w, None, Act::None, tmp, 0);
            be.gate_residual(&h, tmp, rows, d, Gate::None);
        }
        let out = be.alloc(rows * d);
        be.rms_norm(&h, rows, d, Some(&self.final_norm), Modulation::None, c.final_norm_eps, &out);
        out
    }

    #[allow(clippy::too_many_arguments)]
    fn split_qkv(&self, be: &B, bl: &AttnFf<B>, qkv: &B::Buf, rows: usize, rope: Option<Rope<'_, B::Buf>>, q: &B::Buf, k: &B::Buf, v: &B::Buf) {
        let c = &self.cfg;
        let w = c.dim();
        let nq = HeadNorm { w: Some(&bl.norm_q), eps: c.qk_norm_eps };
        let nk = HeadNorm { w: Some(&bl.norm_k), eps: c.qk_norm_eps };
        be.head_split(qkv, rows, 3 * w, 0, c.heads, c.head_dim, Some(nq), rope, q);
        be.head_split(qkv, rows, 3 * w, w, c.heads, c.head_dim, Some(nk), rope, k);
        be.head_split(qkv, rows, 3 * w, 2 * w, c.heads, c.head_dim, None, None, v);
    }

    pub fn workspace(&self, be: &B, layout: &Layout, ffn_chunk: usize, attention: Attention) -> Result<Workspace<B>> {
        let c = &self.cfg;
        let s = layout.sequence_length;
        let (d, qkv) = (c.hidden, c.dim());
        // The FFN hidden `[chunk, 2·ffn]` is the largest activation; keep it
        // under the backend's buffer limit (4 GiB on WebGPU).
        let chunk_cap = (be.max_buffer_elems() / (2 * c.ffn_dim)).max(1);
        let chunk = ffn_chunk.min(chunk_cap).clamp(1, s);
        let rope = h3_rope(&layout.position_ids, c.rope_freq_dim, c.rope_theta);
        let (vsa, vsa_dense) = match attention {
            Attention::Dense { .. } => (None, None),
            Attention::Vsa { sparsity, tile, ref_sparsity, qk8, pv8 } => {
                if !be.vsa_supported(c.head_dim) {
                    return Err(Error::Msg(format!("backend {} cannot run VSA at head_dim {}", be.name(), c.head_dim)));
                }
                let segs = segments_for_layout(layout, c.patch, sparsity, ref_sparsity);
                let spec = VsaSpec::new(&segs, tile, c.heads, c.head_dim).with_qk8(qk8).with_pv8(pv8);
                (Some(be.vsa_prepare(&spec)), Some(be.vsa_prepare(&spec.dense())))
            }
        };
        Ok(Workspace {
            attention,
            tokens: s,
            ffn_chunk: chunk,
            video_runs: runs(&layout.video_indices),
            audio_runs: runs(&layout.audio_indices),
            text_rows: layout.num_text_tokens(),
            vsa,
            vsa_dense,
            h: be.alloc(s * d),
            normed: be.alloc(s * d),
            qkv: be.alloc(s * 3 * qkv),
            g: if self.has_gate && matches!(attention, Attention::Vsa { .. }) { be.alloc(s * qkv) } else { be.alloc(1) },
            q: be.alloc(s * qkv),
            k: be.alloc(s * qkv),
            v: be.alloc(s * qkv),
            attn: be.alloc(s * qkv),
            tmp: be.alloc(s * d),
            hid: be.alloc(chunk * 2 * c.ffn_dim),
            hid2: be.alloc(chunk * c.ffn_dim),
            rope_cos: be.upload_table(&rope.cos),
            rope_sin: be.upload_table(&rope.sin),
        })
    }

    /// One denoising forward. `video_rows` is `[Nv, video_patch_dim]` in the
    /// layout's video-row order, `audio_rows` `[Na, audio_in]`, `text`
    /// the refined `[Nt, d]` prompt. Velocities are written to `video_out`
    /// (`[Nv, out_dim]`) and `audio_out`; `heads` overrides the output
    /// projections (PDD).
    #[allow(clippy::too_many_arguments)]
    pub fn forward(
        &self,
        be: &B,
        ws: &Workspace<B>,
        video_rows: &B::Buf,
        audio_rows: &B::Buf,
        text: &B::Buf,
        mods: &StepMods<B>,
        heads: Option<&OutputHeads<B>>,
        video_out: &B::Buf,
        audio_out: &B::Buf,
    ) {
        let c = &self.cfg;
        let (s, d) = (ws.tokens, c.hidden);
        // Assemble the packed sequence.
        be.copy(text, 0, &ws.h, 0, ws.text_rows * d);
        for r in &ws.video_runs {
            be.linear(video_rows, r.offset, r.rows, &self.proj_in.w, self.proj_in.b.as_ref(), Act::None, &ws.h, r.start);
        }
        for r in &ws.audio_runs {
            be.linear(audio_rows, r.offset, r.rows, &self.audio_proj_in.w, self.audio_proj_in.b.as_ref(), Act::None, &ws.h, r.start);
        }
        let rope = Some(Rope { cos: &ws.rope_cos, sin: &ws.rope_sin, rot: c.rotary_dim() });
        // Streamed blocks: block l + 1's weights are uploaded on the backend's
        // side queue while block l computes (`prefetch_*`); at most two blocks
        // are resident, and a block is freed only after the compute queue has
        // passed it (`prefetch_release`). `H3_PREFETCH_DEPTH=2` keeps two
        // blocks in flight (three resident), for links where one block's
        // upload does not fit inside one block's compute.
        let depth: usize = std::env::var("H3_PREFETCH_DEPTH").ok().and_then(|v| v.parse().ok()).unwrap_or(1).clamp(1, 2);
        let prefetch = |l: usize| -> Option<AttnFf<B>> {
            match self.blocks.get(l) {
                Some(Residency::Host(h)) => {
                    be.prefetch_begin();
                    let w = h.upload(be);
                    be.prefetch_end();
                    Some(w)
                }
                _ => None,
            }
        };
        // H3_TRACE=N: magnitude and non-finite count of the hidden state after
        // every block of the first N forwards (then only the output), to localize
        // a blow-up on a backend without a host oracle at this size.
        let trace_n: usize = std::env::var("H3_TRACE").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        let trace = trace_n > 0;
        static FORWARDS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let first = FORWARDS.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < trace_n;
        let stat = |tag: &str, buf: &B::Buf| {
            if trace {
                be.sync();
                let v = be.download(buf);
                let (mx, nf) = v.iter().fold((0f32, 0usize), |(m, n), &x| if x.is_finite() { (m.max(x.abs()), n) } else { (m, n + 1) });
                eprintln!("trace {tag}: max {mx:.4e}, non-finite {nf} of {}", v.len());
            }
        };
        if first {
            stat("input h", &ws.h);
        }
        let mut staged: std::collections::VecDeque<AttnFf<B>> = (0..depth).filter_map(prefetch).collect();
        for (l, (bl, m)) in self.blocks.iter().zip(&mods.blocks).enumerate() {
            let streamed;
            let w = match bl {
                Residency::Device(w) => w,
                Residency::Host(_) => {
                    be.prefetch_wait();
                    streamed = staged.pop_front().expect("prefetched block");
                    &streamed
                }
            };
            let t = ModTable { table: m, idx: &mods.adaln_idx, slots: H3Config::MOD_SLOTS };
            be.rms_norm(&ws.h, s, d, Some(&w.norm1), Modulation::Table { m: t, scale: SCALE_MSA, shift: SHIFT_MSA }, c.norm_eps, &ws.normed);
            be.linear(&ws.normed, 0, s, &w.qkv.w, None, Act::None, &ws.qkv, 0);
            self.split_qkv(be, w, &ws.qkv, s, rope, &ws.q, &ws.k, &ws.v);
            match (&ws.vsa, &w.gate) {
                (Some(vsa), gate) => {
                    let g = gate.as_ref().map(|gl| {
                        be.linear(&ws.normed, 0, s, &gl.w, None, Act::None, &ws.attn, 0);
                        be.head_split(&ws.attn, s, c.dim(), 0, c.heads, c.head_dim, None, None, &ws.g);
                        &ws.g
                    });
                    let st = if mods.dense { ws.vsa_dense.as_ref().unwrap() } else { vsa };
                    be.vsa_attention(&ws.q, &ws.k, &ws.v, g, st, &ws.attn);
                }
                (None, _) if matches!(ws.attention, Attention::Dense { qk8: true, .. }) => {
                    let pv8 = matches!(ws.attention, Attention::Dense { pv8: true, .. });
                    be.attention_qk8(&ws.q, &ws.k, &ws.v, c.heads, c.heads, s, s, c.head_dim, false, pv8, &ws.attn)
                }
                (None, _) => be.attention(&ws.q, &ws.k, &ws.v, c.heads, c.heads, s, s, c.head_dim, false, &ws.attn),
            }
            if first {
                stat(&format!("block {l} attn"), &ws.attn);
            }
            be.linear(&ws.attn, 0, s, &w.out.w, None, Act::None, &ws.tmp, 0);
            be.residual_rms_norm(
                &ws.h,
                &ws.tmp,
                s,
                d,
                Gate::Table(ModRef { table: t, slot: GATE_MSA }),
                Some(&w.norm2),
                Modulation::Table { m: t, scale: SCALE_MLP, shift: SHIFT_MLP },
                c.norm_eps,
                &ws.normed,
            );
            let mut r0 = 0;
            while r0 < s {
                let n = ws.ffn_chunk.min(s - r0);
                be.linear(&ws.normed, r0, n, &w.ff_in.w, None, Act::None, &ws.hid, 0);
                be.swiglu(&ws.hid, n, c.ffn_dim, &ws.hid2);
                be.linear(&ws.hid2, 0, n, &w.ff_out.w, None, Act::None, &ws.tmp, r0);
                r0 += n;
            }
            be.gate_residual(&ws.h, &ws.tmp, s, d, Gate::Table(ModRef { table: t, slot: GATE_MLP }));
            if first {
                stat(&format!("block {l} h"), &ws.h);
            }
            if matches!(bl, Residency::Host(_)) {
                // This block's kernels are queued: fetch the next one behind them,
                // then let the side queue free this block once the compute queue is past it.
                staged.extend(prefetch(l + depth));
                be.prefetch_release();
            }
        }
        let ht = ModTable { table: &mods.head, idx: &mods.t_idx, slots: 2 };
        be.rms_norm(&ws.h, s, d, Some(&self.norm_out_w), Modulation::Table { m: ht, scale: 1, shift: 0 }, c.final_norm_eps, &ws.normed);
        let heads = heads.unwrap_or(&self.heads);
        for r in &ws.video_runs {
            be.linear(&ws.normed, r.start, r.rows, &heads.video.w, heads.video.b.as_ref(), Act::None, video_out, r.offset);
        }
        for r in &ws.audio_runs {
            be.linear(&ws.normed, r.start, r.rows, &heads.audio.w, heads.audio.b.as_ref(), Act::None, audio_out, r.offset);
        }
    }

    /// Resident block weight bytes, as the backend stores them.
    pub fn block_weight_bytes(&self, be: &B) -> usize {
        self.blocks
            .iter()
            .map(|b| match b {
                Residency::Device(w) => {
                    be.weight_bytes(&w.qkv.w)
                        + be.weight_bytes(&w.out.w)
                        + be.weight_bytes(&w.ff_in.w)
                        + be.weight_bytes(&w.ff_out.w)
                        + w.gate.as_ref().map_or(0, |g| be.weight_bytes(&g.w))
                }
                Residency::Host(_) => 0,
            })
            .sum()
    }
}

fn fused_heads<B: Backend>(be: &B, pdd: &PddHeads, vw: &[f64], aw: &[f64], start: usize, end: usize, d: usize) -> OutputHeads<B> {
    assert!(start < end && end <= pdd.steps && vw.len() == pdd.steps && aw.len() == pdd.steps);
    let fuse = |w: &[f32], b: &[f32], weights: &[f64]| -> (Vec<f32>, Vec<f32>) {
        let total: f64 = weights[start..end].iter().sum();
        assert!(total != 0.0 && total.is_finite(), "PDD block weight must be finite and non-zero");
        let out = w.len() / (pdd.steps * d);
        let mut fw = vec![0.0f64; out * d];
        let mut fb = vec![0.0f64; out];
        for j in start..end {
            let n = weights[j] / total;
            for (a, &x) in fw.iter_mut().zip(&w[j * out * d..(j + 1) * out * d]) {
                *a += n * x as f64;
            }
            for (a, &x) in fb.iter_mut().zip(&b[j * out..(j + 1) * out]) {
                *a += n * x as f64;
            }
        }
        (fw.into_iter().map(|v| v as f32).collect(), fb.into_iter().map(|v| v as f32).collect())
    };
    let (vw, vb) = fuse(&pdd.video_w, &pdd.video_b, vw);
    let (aw, ab) = fuse(&pdd.audio_w, &pdd.audio_b, aw);
    let (vo, ao) = (vb.len(), ab.len());
    OutputHeads {
        video: Linear { w: be.weight(&vw, vo, d), b: Some(be.upload(&vb)), rows: vo, cols: d },
        audio: Linear { w: be.weight(&aw, ao, d), b: Some(be.upload(&ab)), rows: ao, cols: d },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runs_split_consecutive_indices() {
        let r = runs(&[3, 4, 5, 9, 10, 20]);
        assert_eq!(
            r,
            vec![Run { start: 3, rows: 3, offset: 0 }, Run { start: 9, rows: 2, offset: 3 }, Run { start: 20, rows: 1, offset: 5 }]
        );
    }
}
