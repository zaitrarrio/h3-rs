//! Qwen3-VL-32B as H3's conditioner, on the backend: the vision tower with
//! DeepStack, the first 50 language layers (hidden state 50, no final norm).
//!
//! The language layers ([`Placement`]) are either read from the
//! memory-mapped checkpoint for every prompt (`Stream`: the 64 GB never
//! stays anywhere), kept in page-locked host memory and uploaded per prompt
//! (`Host`), or kept on the device (`Resident`). Streamed and host layers
//! upload on the backend's side queue while the previous layer computes
//! (`Backend::prefetch_*`, as the DiT's streamed blocks), bf16 checkpoint
//! tensors stay bf16 bits all the way (no f32 round trip), and a streamed
//! layer is read by a background thread with parallel reads while the
//! device works. [`EncoderWeights::Int8`] keeps the linears as int8
//! weight-only (`quant::Q8`, one scale per 128 input columns), half the
//! bytes of bf16.

use crate::present::{Presentation, VisionBlock};
use crate::vision::VisionInput;
use h3_core::{
    Error, Result,
    backend::{Act, Backend, Gate, HeadNorm, Modulation, Rope},
    config::Qwen3VlConfig,
    quant::{Nvfp4, Q8, e4m3_to_f32},
    rope::{qwen3_mrope, qwen3_vision_rope},
    weights::{SafetensorsDir, WeightSource, bytes_to_f32},
};
use h3_model::linear::{HostLinear, HostWeight, Linear, PinGuard};
use rayon::prelude::*;

/// Where the 50 language layers live between prompts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Placement {
    /// Read from the checkpoint for every prompt.
    Stream,
    /// A page-locked host copy, uploaded layer by layer per prompt.
    Host,
    /// On the device.
    Resident,
}

/// The language linears' format on the host and the device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EncoderWeights {
    /// The checkpoint's own precision (bf16 bits; f32 sources stay f32).
    #[default]
    Bf16,
    /// Int8 weight-only (`Q8`): about half of bf16.
    Int8,
}

const LM: &str = "model.language_model";
const VIS: &str = "model.visual";

/// An NVFP4 linear as compressed-tensors stores it (`weight_packed`,
/// row-major E4M3 `weight_scale`, `weight_global_scale`): `w = fp4 ·
/// e4m3(scale) / global_scale`.
fn nvfp4_of(src: &SafetensorsDir, key: &str, rows: usize, cols: usize) -> Result<Nvfp4> {
    let packed_name = format!("{key}.weight_packed");
    let (_, pshape, packed) = src.raw(&packed_name)?;
    if pshape != [rows, cols / 2] {
        return Err(Error::Weights(format!(
            "`{packed_name}` is {pshape:?}, expected [{rows}, {}]",
            cols / 2
        )));
    }
    let (_, sshape, scales) = src.raw(&format!("{key}.weight_scale"))?;
    if sshape != [rows, cols / 16] {
        return Err(Error::Weights(format!(
            "`{key}.weight_scale` is {sshape:?}, expected [{rows}, {}]",
            cols / 16
        )));
    }
    let global = src.scalar(&format!("{key}.weight_global_scale"))?;
    // compressed-tensors packs the first code of each pair in the low nibble.
    Ok(Nvfp4 {
        packed: packed.to_vec(),
        scales: scales.to_vec(),
        alpha: 1.0 / global,
        rows,
        cols,
        first_high: false,
    })
}

/// A linear from a plain or NVFP4 checkpoint.
fn linear<B: Backend>(
    be: &B,
    src: &SafetensorsDir,
    key: &str,
    rows: usize,
    cols: usize,
    bias: bool,
) -> Result<Linear<B>> {
    let b = if bias {
        Some(be.upload(&src.get(&format!("{key}.bias"), &[rows])?))
    } else {
        None
    };
    if src.contains_raw(&format!("{key}.weight")) {
        return Ok(Linear {
            w: be.weight(
                &src.get(&format!("{key}.weight"), &[rows, cols])?,
                rows,
                cols,
            ),
            b,
            rows,
            cols,
            lora: None,
        });
    }
    if src.contains_raw(&format!("{key}.weight_packed")) {
        let q = nvfp4_of(src, key, rows, cols)?;
        let _ = e4m3_to_f32;
        return Ok(Linear {
            w: be.weight_nvfp4(&q),
            b,
            rows,
            cols,
            lora: None,
        });
    }
    Err(Error::Weights(format!("missing linear `{key}`")))
}

struct TextLayer<B: Backend> {
    ln1: B::Buf,
    qkv: Linear<B>,
    o: Linear<B>,
    q_norm: B::Buf,
    k_norm: B::Buf,
    ln2: B::Buf,
    gate_up: Linear<B>,
    down: Linear<B>,
}

/// bf16 bits from little-endian bytes, in parallel (parallel page faults on
/// an mmapped checkpoint are parallel reads).
fn bf16_bits(b: &[u8]) -> Vec<u16> {
    let mut out = vec![0u16; b.len() / 2];
    out.par_chunks_mut(1 << 19)
        .zip(b.par_chunks(1 << 20))
        .for_each(|(o, c)| {
            for (x, p) in o.iter_mut().zip(c.chunks_exact(2)) {
                *x = u16::from_le_bytes([p[0], p[1]]);
            }
        });
    out
}

/// A plain weight's values as stored: bf16 bits kept, anything else f32.
enum Raw {
    Bf16(Vec<u16>),
    F32(Vec<f32>),
}

impl Raw {
    fn into_f32(self) -> Vec<f32> {
        match self {
            Raw::F32(v) => v,
            Raw::Bf16(v) => v
                .par_iter()
                .map(|&b| f32::from_bits((b as u32) << 16))
                .collect(),
        }
    }
}

fn raw_weight(src: &SafetensorsDir, key: &str, rows: usize, cols: usize) -> Result<Raw> {
    let name = format!("{key}.weight");
    if src.contains_raw(&name) {
        let (dt, shape, data) = src.raw(&name)?;
        if shape != [rows, cols] {
            return Err(Error::Weights(format!(
                "`{name}` is {shape:?}, expected [{rows}, {cols}]"
            )));
        }
        return Ok(if dt == safetensors::Dtype::BF16 {
            Raw::Bf16(bf16_bits(data))
        } else {
            Raw::F32(bytes_to_f32(dt, data, &name)?)
        });
    }
    if src.contains_raw(&format!("{key}.weight_packed")) {
        return Ok(Raw::F32(nvfp4_of(src, key, rows, cols)?.dequantize()));
    }
    Err(Error::Weights(format!("missing linear `{key}`")))
}

fn finish(raw: Raw, rows: usize, cols: usize, w: EncoderWeights) -> HostWeight {
    match (raw, w) {
        (Raw::Bf16(v), EncoderWeights::Bf16) => HostWeight::Bf16(v),
        (Raw::F32(v), EncoderWeights::Bf16) => HostWeight::F32(v),
        (r, EncoderWeights::Int8) => HostWeight::Q8(Q8::quantize(&r.into_f32(), rows, cols)),
    }
}

/// A linear the int8 tree stores (`h3 quantize-encoder`): `<key>.weight.q8`
/// + `.scale`, checked against the expected shape.
fn tree_q8(
    tree: Option<&SafetensorsDir>,
    key: &str,
    rows: usize,
    cols: usize,
) -> Result<Option<Q8>> {
    let Some(t) = tree else { return Ok(None) };
    match t.stored(key)? {
        Some(h3_core::quant::Stored::Q8(q)) if q.rows == rows && q.cols == cols => Ok(Some(q)),
        Some(_) => Err(Error::Weights(format!(
            "int8 encoder tree: `{key}` is not a [{rows}, {cols}] int8 linear"
        ))),
        None => Ok(None),
    }
}

/// One language linear on the host; an NVFP4 checkpoint linear stays NVFP4.
fn host_linear(
    src: &SafetensorsDir,
    tree: Option<&SafetensorsDir>,
    key: &str,
    rows: usize,
    cols: usize,
    w: EncoderWeights,
) -> Result<HostLinear> {
    let stored = if w == EncoderWeights::Int8 {
        tree_q8(tree, key, rows, cols)?
    } else {
        None
    };
    let hw = if let Some(q) = stored {
        HostWeight::Q8(q)
    } else if !src.contains_raw(&format!("{key}.weight"))
        && src.contains_raw(&format!("{key}.weight_packed"))
    {
        HostWeight::Nvfp4(nvfp4_of(src, key, rows, cols)?)
    } else {
        finish(raw_weight(src, key, rows, cols)?, rows, cols, w)
    };
    Ok(HostLinear {
        w: hw,
        b: None,
        rows,
        cols,
        lora: None,
    })
}

/// Several linears stacked along the output axis (`fused`: the stacked
/// linear's name in the int8 tree).
fn host_fused(
    src: &SafetensorsDir,
    tree: Option<&SafetensorsDir>,
    fused: &str,
    keys: &[(&str, usize)],
    cols: usize,
    w: EncoderWeights,
) -> Result<HostLinear> {
    let rows: usize = keys.iter().map(|k| k.1).sum();
    if w == EncoderWeights::Int8
        && let Some(q) = tree_q8(tree, fused, rows, cols)?
    {
        return Ok(HostLinear {
            w: HostWeight::Q8(q),
            b: None,
            rows,
            cols,
            lora: None,
        });
    }
    let parts: Vec<Raw> = keys
        .iter()
        .map(|(k, r)| raw_weight(src, k, *r, cols))
        .collect::<Result<_>>()?;
    let raw = if parts.iter().all(|p| matches!(p, Raw::Bf16(_))) {
        Raw::Bf16(
            parts
                .into_iter()
                .flat_map(|p| match p {
                    Raw::Bf16(v) => v,
                    Raw::F32(_) => unreachable!(),
                })
                .collect(),
        )
    } else {
        Raw::F32(parts.into_iter().flat_map(Raw::into_f32).collect())
    };
    Ok(HostLinear {
        w: finish(raw, rows, cols, w),
        b: None,
        rows,
        cols,
        lora: None,
    })
}

/// A language layer on the host: what [`Placement::Host`] keeps and what a
/// streamed layer is read into.
struct HostTextLayer {
    /// Page-lock guards: declared first, so they drop before the buffers.
    pins: Vec<PinGuard>,
    ln1: Vec<f32>,
    qkv: HostLinear,
    o: HostLinear,
    q_norm: Vec<f32>,
    k_norm: Vec<f32>,
    ln2: Vec<f32>,
    gate_up: HostLinear,
    down: HostLinear,
}

impl HostTextLayer {
    fn read(
        src: &SafetensorsDir,
        tree: Option<&SafetensorsDir>,
        cfg: &Qwen3VlConfig,
        l: usize,
        w: EncoderWeights,
    ) -> Result<Self> {
        let p = format!("{LM}.layers.{l}");
        let (h, hd) = (cfg.hidden, cfg.head_dim);
        let v = |k: &str, n: usize| src.get(k, &[n]);
        Ok(HostTextLayer {
            pins: Vec::new(),
            ln1: v(&format!("{p}.input_layernorm.weight"), h)?,
            qkv: host_fused(
                src,
                tree,
                &format!("{p}.self_attn.qkv_proj"),
                &[
                    (&format!("{p}.self_attn.q_proj"), cfg.heads * hd),
                    (&format!("{p}.self_attn.k_proj"), cfg.kv_heads * hd),
                    (&format!("{p}.self_attn.v_proj"), cfg.kv_heads * hd),
                ],
                h,
                w,
            )?,
            o: host_linear(
                src,
                tree,
                &format!("{p}.self_attn.o_proj"),
                h,
                cfg.heads * hd,
                w,
            )?,
            q_norm: v(&format!("{p}.self_attn.q_norm.weight"), hd)?,
            k_norm: v(&format!("{p}.self_attn.k_norm.weight"), hd)?,
            ln2: v(&format!("{p}.post_attention_layernorm.weight"), h)?,
            // SwiGLU takes `value | gate`: up first, then gate.
            gate_up: host_fused(
                src,
                tree,
                &format!("{p}.mlp.up_gate_proj"),
                &[
                    (&format!("{p}.mlp.up_proj"), cfg.intermediate),
                    (&format!("{p}.mlp.gate_proj"), cfg.intermediate),
                ],
                h,
                w,
            )?,
            down: host_linear(
                src,
                tree,
                &format!("{p}.mlp.down_proj"),
                h,
                cfg.intermediate,
                w,
            )?,
        })
    }

    fn pin<B: Backend>(&mut self, be: &B) {
        let mut pins = Vec::new();
        for l in [&self.qkv, &self.o, &self.gate_up, &self.down] {
            l.pin(be, &mut pins);
        }
        self.pins = pins;
    }

    fn upload<B: Backend>(&self, be: &B) -> TextLayer<B> {
        TextLayer {
            ln1: be.upload(&self.ln1),
            qkv: self.qkv.upload(be),
            o: self.o.upload(be),
            q_norm: be.upload(&self.q_norm),
            k_norm: be.upload(&self.k_norm),
            ln2: be.upload(&self.ln2),
            gate_up: self.gate_up.upload(be),
            down: self.down.upload(be),
        }
    }
}

enum Slot<B: Backend> {
    Device(TextLayer<B>),
    Host(HostTextLayer),
    Stream,
}

struct VisionBlockW<B: Backend> {
    norm1: (B::Buf, B::Buf),
    qkv: Linear<B>,
    proj: Linear<B>,
    norm2: (B::Buf, B::Buf),
    fc1: Linear<B>,
    fc2: Linear<B>,
}

struct Merger<B: Backend> {
    norm: (B::Buf, B::Buf),
    /// Whether the norm is over the merged (4×) width.
    post: bool,
    fc1: Linear<B>,
    fc2: Linear<B>,
}

pub struct VisionTower<B: Backend> {
    patch_embed: Linear<B>,
    pos_embed: Vec<f32>,
    blocks: Vec<VisionBlockW<B>>,
    merger: Merger<B>,
    deepstack: Vec<Merger<B>>,
}

pub struct Qwen3Vl<B: Backend> {
    pub cfg: Qwen3VlConfig,
    src: SafetensorsDir,
    /// The pre-quantized int8 language linears (`h3 quantize-encoder`).
    int8_tree: Option<SafetensorsDir>,
    layers: Vec<Slot<B>>,
    pub placement: Placement,
    pub weights: EncoderWeights,
    vision: Option<VisionTower<B>>,
}

/// A conditioning result: `[rows, hidden]` f32.
pub struct Encoded {
    pub embeds: Vec<f32>,
    pub rows: usize,
}

/// Bytes the first `cfg.output_layer` language layers take in `weights`'
/// format (what a host copy or a resident encoder holds; int8 counts its
/// codes and one f32 scale per 128 columns).
pub fn language_bytes(src: &SafetensorsDir, cfg: &Qwen3VlConfig, weights: EncoderWeights) -> u64 {
    let mut total = 0u64;
    for name in src.names() {
        let Some(rest) = name.strip_prefix(&format!("{LM}.layers.")) else {
            continue;
        };
        let Some(l) = rest.split('.').next().and_then(|l| l.parse::<usize>().ok()) else {
            continue;
        };
        if l >= cfg.output_layer {
            continue;
        }
        let Ok((dt, shape, data)) = src.raw(name) else {
            continue;
        };
        let n = data.len() as u64;
        total += match (weights, shape.len(), dt) {
            (
                EncoderWeights::Int8,
                2,
                safetensors::Dtype::BF16 | safetensors::Dtype::F32 | safetensors::Dtype::F16,
            ) => {
                let elems = (shape[0] * shape[1]) as u64;
                elems + elems / 128 * 4
            }
            _ => n,
        };
    }
    total
}

/// Write the int8 tree: one safetensors file per language layer with its four
/// (fused) linears as int8 codes + scales (`crate::export`'s `quant/1`
/// names), loaded by `Qwen3Vl::open` with [`EncoderWeights::Int8`] in place
/// of quantizing the bf16 checkpoint at every load. `on_layer(l, bytes)`
/// reports progress.
pub fn write_int8_tree(
    src: &SafetensorsDir,
    cfg: &Qwen3VlConfig,
    out: &std::path::Path,
    on_layer: &mut dyn FnMut(usize, usize),
) -> Result<()> {
    use h3_core::export::{Tensor, f32_bytes, write};
    std::fs::create_dir_all(out)?;
    for l in 0..cfg.output_layer {
        let h = HostTextLayer::read(src, None, cfg, l, EncoderWeights::Int8)?;
        let p = format!("{LM}.layers.{l}");
        let mut t: Vec<Tensor> = Vec::new();
        for (name, lin) in [
            ("self_attn.qkv_proj", &h.qkv),
            ("self_attn.o_proj", &h.o),
            ("mlp.up_gate_proj", &h.gate_up),
            ("mlp.down_proj", &h.down),
        ] {
            let HostWeight::Q8(q) = &lin.w else {
                // An NVFP4 checkpoint linear stays NVFP4: read from the source.
                continue;
            };
            let key = format!("{p}.{name}.weight");
            t.push((
                format!("{key}.q8"),
                safetensors::Dtype::I8,
                vec![q.rows, q.cols],
                q.q.iter().map(|&v| v as u8).collect(),
            ));
            t.push((
                format!("{key}.scale"),
                safetensors::Dtype::F32,
                vec![q.rows, q.cols / q.group],
                f32_bytes(&q.scales),
            ));
        }
        let bytes: usize = t.iter().map(|x| x.3.len()).sum();
        let path = out.join(format!("layers-{l:02}.safetensors"));
        let tmp = path.with_extension("tmp");
        write(
            &tmp,
            &t,
            [
                ("h3.format".to_string(), "quant/1".to_string()),
                ("h3.weights".to_string(), "int8".to_string()),
            ]
            .into(),
        )?;
        std::fs::rename(&tmp, &path)?;
        on_layer(l, bytes);
    }
    Ok(())
}

impl<B: Backend> Qwen3Vl<B> {
    /// The language layers' [`Placement`] and format; `vision`: load the
    /// vision tower (FL2VA / Ref2VA).
    pub fn open(
        be: &B,
        cfg: Qwen3VlConfig,
        src: SafetensorsDir,
        placement: Placement,
        weights: EncoderWeights,
        int8_tree: Option<SafetensorsDir>,
        vision: bool,
    ) -> Result<Self> {
        let tree = int8_tree
            .as_ref()
            .filter(|_| weights == EncoderWeights::Int8);
        let mut layers = Vec::with_capacity(cfg.output_layer);
        for l in 0..cfg.output_layer {
            layers.push(match placement {
                Placement::Stream => Slot::Stream,
                Placement::Host => {
                    let mut h = HostTextLayer::read(&src, tree, &cfg, l, weights)?;
                    h.pin(be);
                    Slot::Host(h)
                }
                Placement::Resident => {
                    Slot::Device(HostTextLayer::read(&src, tree, &cfg, l, weights)?.upload(be))
                }
            });
        }
        let vision = if vision {
            Some(Self::load_vision(be, &src, &cfg)?)
        } else {
            None
        };
        Ok(Self {
            cfg,
            src,
            int8_tree,
            layers,
            placement,
            weights,
            vision,
        })
    }

    fn load_vision(be: &B, src: &SafetensorsDir, cfg: &Qwen3VlConfig) -> Result<VisionTower<B>> {
        let vh = cfg.vision_hidden;
        let vm = vh * cfg.vision_merge * cfg.vision_merge;
        let pf = 3 * cfg.vision_temporal_patch * cfg.vision_patch * cfg.vision_patch;
        let ln = |k: &str, n: usize| -> Result<(B::Buf, B::Buf)> {
            Ok((
                be.upload(&src.get(&format!("{k}.weight"), &[n])?),
                be.upload(&src.get(&format!("{k}.bias"), &[n])?),
            ))
        };
        let merger = |p: &str, post: bool| -> Result<Merger<B>> {
            Ok(Merger {
                norm: ln(&format!("{p}.norm"), if post { vm } else { vh })?,
                post,
                fc1: linear(be, src, &format!("{p}.linear_fc1"), vm, vm, true)?,
                fc2: linear(
                    be,
                    src,
                    &format!("{p}.linear_fc2"),
                    cfg.vision_out_hidden,
                    vm,
                    true,
                )?,
            })
        };
        let mut blocks = Vec::new();
        for b in 0..cfg.vision_depth {
            let p = format!("{VIS}.blocks.{b}");
            blocks.push(VisionBlockW {
                norm1: ln(&format!("{p}.norm1"), vh)?,
                qkv: linear(be, src, &format!("{p}.attn.qkv"), 3 * vh, vh, true)?,
                proj: linear(be, src, &format!("{p}.attn.proj"), vh, vh, true)?,
                norm2: ln(&format!("{p}.norm2"), vh)?,
                fc1: linear(
                    be,
                    src,
                    &format!("{p}.mlp.linear_fc1"),
                    cfg.vision_intermediate,
                    vh,
                    true,
                )?,
                fc2: linear(
                    be,
                    src,
                    &format!("{p}.mlp.linear_fc2"),
                    vh,
                    cfg.vision_intermediate,
                    true,
                )?,
            });
        }
        let pe = {
            let w = src.get(
                &format!("{VIS}.patch_embed.proj.weight"),
                &[
                    vh,
                    3,
                    cfg.vision_temporal_patch,
                    cfg.vision_patch,
                    cfg.vision_patch,
                ],
            )?;
            Linear {
                w: be.weight(&w, vh, pf),
                b: Some(be.upload(&src.get(&format!("{VIS}.patch_embed.proj.bias"), &[vh])?)),
                rows: vh,
                cols: pf,
                lora: None,
            }
        };
        Ok(VisionTower {
            patch_embed: pe,
            pos_embed: src.get(
                &format!("{VIS}.pos_embed.weight"),
                &[cfg.vision_num_position_embeddings, vh],
            )?,
            blocks,
            merger: merger(&format!("{VIS}.merger"), false)?,
            deepstack: (0..cfg.vision_deepstack.len())
                .map(|i| merger(&format!("{VIS}.deepstack_merger_list.{i}"), true))
                .collect::<Result<_>>()?,
        })
    }

    /// Token embeddings `[rows, hidden]` read straight from the checkpoint.
    pub fn embed(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let (dt, shape, data) = self.src.raw(&format!("{LM}.embed_tokens.weight"))?;
        let h = shape[1];
        let bytes = match dt {
            safetensors::Dtype::BF16 | safetensors::Dtype::F16 => 2,
            safetensors::Dtype::F32 => 4,
            other => {
                return Err(Error::Weights(format!(
                    "embed_tokens: unsupported dtype {other:?}"
                )));
            }
        };
        let mut out = Vec::with_capacity(ids.len() * h);
        for &id in ids {
            if id as usize >= shape[0] {
                return Err(Error::Msg(format!(
                    "token id {id} is out of the vocabulary"
                )));
            }
            let off = id as usize * h * bytes;
            out.extend(bytes_to_f32(
                dt,
                &data[off..off + h * bytes],
                "embed_tokens",
            )?);
        }
        Ok(out)
    }

    /// Bilinear position-embedding gather in merge-block order
    /// (`_interpolate_vision_position_embeddings`).
    fn pos_embed_for(&self, tower: &VisionTower<B>, grid: (usize, usize, usize)) -> Vec<f32> {
        let cfg = &self.cfg;
        let side = (cfg.vision_num_position_embeddings as f64).sqrt() as usize;
        let m = cfg.vision_merge;
        let (t, h, w) = grid;
        let vh = cfg.vision_hidden;
        let mut out = Vec::with_capacity(t * h * w * vh);
        let taps = |i: usize, n: usize| -> [(usize, f32); 2] {
            let pos = i as f32 * (side - 1) as f32 / (n.max(2) - 1) as f32;
            let f = pos.floor();
            let i0 = (f as usize).min(side - 1);
            let i1 = (i0 + 1).min(side - 1);
            let d = pos - f;
            [(i0, (1.0 - d).max(0.0)), (i1, (1.0 - (1.0 - d)).max(0.0))]
        };
        for _ in 0..t {
            for br in 0..h / m {
                for bc in 0..w / m {
                    for mr in 0..m {
                        for mc in 0..m {
                            let (r, c) = (br * m + mr, bc * m + mc);
                            let rt = taps(r, h);
                            let ct = taps(c, w);
                            let mut acc = vec![0.0f32; vh];
                            for (ri, rw) in rt {
                                for (ci, cw) in ct {
                                    let wgt = rw * cw;
                                    if wgt == 0.0 {
                                        continue;
                                    }
                                    let row = &tower.pos_embed
                                        [(ri * side + ci) * vh..(ri * side + ci + 1) * vh];
                                    for (a, &x) in acc.iter_mut().zip(row) {
                                        *a += wgt * x;
                                    }
                                }
                            }
                            out.extend(acc);
                        }
                    }
                }
            }
        }
        out
    }

    /// Vision tower over one input: merged features `[tokens/4, out]` and the
    /// DeepStack features (same shape) per deepstack block.
    pub fn vision_features(
        &self,
        be: &B,
        input: &VisionInput,
    ) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let tower = self
            .vision
            .as_ref()
            .ok_or_else(|| Error::Msg("the vision tower is not loaded".into()))?;
        let cfg = &self.cfg;
        let (vh, heads) = (cfg.vision_hidden, cfg.vision_heads);
        let hd = cfg.vision_head_dim();
        let n = input.tokens();
        let (t, gh, gw) = input.grid;
        let per_frame = gh * gw;
        let m = cfg.vision_merge;
        let x = be.upload(&input.patches);
        let h = be.alloc(n * vh);
        be.linear(
            &x,
            0,
            n,
            &tower.patch_embed.w,
            tower.patch_embed.b.as_ref(),
            Act::None,
            &h,
            0,
        );
        let pe = be.upload(&self.pos_embed_for(tower, input.grid));
        be.axpby(&h, 0, 1.0, &pe, 0, 1.0, n * vh);
        // Rotary over (row, col) in block order; every frame has the same
        // coordinates, so one frame's table serves them all (f32 tables are
        // uploaded, never copied into bf16 scratch: the CUDA backend keeps
        // the two dtypes apart and refuses a mixed copy).
        let mut coords = Vec::with_capacity(per_frame);
        for br in 0..gh / m {
            for bc in 0..gw / m {
                for mr in 0..m {
                    for mc in 0..m {
                        coords.push([(br * m + mr) as u32, (bc * m + mc) as u32]);
                    }
                }
            }
        }
        let rope = qwen3_vision_rope(&coords, hd);
        let (fc, fs) = (be.upload_table(&rope.cos), be.upload_table(&rope.sin));
        let frp = Some(Rope {
            cos: &fc,
            sin: &fs,
            rot: hd,
        });
        let normed = be.alloc(n * vh);
        let qkv = be.alloc(n * 3 * vh);
        let (q, k, v) = (
            be.alloc(heads * per_frame * hd),
            be.alloc(heads * per_frame * hd),
            be.alloc(heads * per_frame * hd),
        );
        let attn = be.alloc(n * vh);
        let tmp = be.alloc(n * vh);
        let hid = be.alloc(n * cfg.vision_intermediate);
        let mut deep = Vec::new();
        let merged = n / (m * m);
        let vm = vh * m * m;
        let run_merger = |mg: &Merger<B>, hin: &B::Buf| -> Vec<f32> {
            let pre = be.alloc(n * vh);
            let (rows, dim) = if mg.post { (merged, vm) } else { (n, vh) };
            be.layer_norm(
                hin,
                rows,
                dim,
                Some(&mg.norm.0),
                Some(&mg.norm.1),
                1e-6,
                &pre,
            );
            let f1 = be.alloc(merged * vm);
            be.linear(
                &pre,
                0,
                merged,
                &mg.fc1.w,
                mg.fc1.b.as_ref(),
                Act::GeluErf,
                &f1,
                0,
            );
            let f2 = be.alloc(merged * cfg.vision_out_hidden);
            be.linear(
                &f1,
                0,
                merged,
                &mg.fc2.w,
                mg.fc2.b.as_ref(),
                Act::None,
                &f2,
                0,
            );
            be.download(&f2)
        };
        for (bi, bl) in tower.blocks.iter().enumerate() {
            be.layer_norm(
                &h,
                n,
                vh,
                Some(&bl.norm1.0),
                Some(&bl.norm1.1),
                1e-6,
                &normed,
            );
            be.linear(
                &normed,
                0,
                n,
                &bl.qkv.w,
                bl.qkv.b.as_ref(),
                Act::None,
                &qkv,
                0,
            );
            // Attention within each frame's tokens (rope rows follow the same order).
            for f in 0..t {
                let r0 = f * per_frame;
                let rows = per_frame;
                let frame_qkv = be.alloc(rows * 3 * vh);
                be.copy(&qkv, r0 * 3 * vh, &frame_qkv, 0, rows * 3 * vh);
                be.head_split(&frame_qkv, rows, 3 * vh, 0, heads, hd, None, frp, &q);
                be.head_split(&frame_qkv, rows, 3 * vh, vh, heads, hd, None, frp, &k);
                be.head_split(&frame_qkv, rows, 3 * vh, 2 * vh, heads, hd, None, None, &v);
                let fa = be.alloc(rows * vh);
                be.attention(&q, &k, &v, heads, heads, rows, rows, hd, false, &fa);
                be.copy(&fa, 0, &attn, r0 * vh, rows * vh);
            }
            be.linear(
                &attn,
                0,
                n,
                &bl.proj.w,
                bl.proj.b.as_ref(),
                Act::None,
                &tmp,
                0,
            );
            be.gate_residual(&h, &tmp, n, vh, Gate::None);
            be.layer_norm(
                &h,
                n,
                vh,
                Some(&bl.norm2.0),
                Some(&bl.norm2.1),
                1e-6,
                &normed,
            );
            be.linear(
                &normed,
                0,
                n,
                &bl.fc1.w,
                bl.fc1.b.as_ref(),
                Act::GeluTanh,
                &hid,
                0,
            );
            be.linear(&hid, 0, n, &bl.fc2.w, bl.fc2.b.as_ref(), Act::None, &tmp, 0);
            be.gate_residual(&h, &tmp, n, vh, Gate::None);
            if let Some(di) = cfg.vision_deepstack.iter().position(|&d| d == bi) {
                deep.push(run_merger(&tower.deepstack[di], &h));
            }
        }
        Ok((run_merger(&tower.merger, &h), deep))
    }

    /// `_get_rope_index`: `(t, h, w)` positions of every token.
    pub fn rope_positions(&self, p: &Presentation) -> Vec<[u32; 3]> {
        let m = self.cfg.vision_merge as u32;
        let mut pos: Vec<[u32; 3]> = Vec::with_capacity(p.ids.len());
        let mut next = 0u32;
        let mut cursor = 0usize;
        let push_text = |pos: &mut Vec<[u32; 3]>, n: usize, next: &mut u32| {
            for i in 0..n {
                pos.push([*next + i as u32; 3]);
            }
            *next += n as u32;
        };
        for b in &p.blocks {
            push_text(&mut pos, b.start - cursor, &mut next);
            let (gt, gh, gw) = (b.grid.0 as u32, b.grid.1 as u32 / m, b.grid.2 as u32 / m);
            let base = next;
            let mut mx = base;
            for tt in 0..gt {
                for hh in 0..gh {
                    for ww in 0..gw {
                        let v = [base + tt, base + hh, base + ww];
                        mx = mx.max(v[0]).max(v[1]).max(v[2]);
                        pos.push(v);
                    }
                }
            }
            next = mx + 1;
            cursor = b.start + b.tokens;
        }
        push_text(&mut pos, p.ids.len() - cursor, &mut next);
        pos
    }

    /// Encode a presentation (token ids + vision inputs in block order).
    pub fn encode(&self, be: &B, p: &Presentation, inputs: &[&VisionInput]) -> Result<Encoded> {
        let cfg = &self.cfg;
        let (h, hd) = (cfg.hidden, cfg.head_dim);
        let n = p.ids.len();
        if n == 0 {
            return Err(Error::Msg("empty prompt".into()));
        }
        let mut emb = self.embed(&p.ids)?;
        // Vision features into the placeholder rows; DeepStack per visual row.
        let mut deep_rows: Vec<Vec<(usize, Vec<f32>)>> =
            vec![Vec::new(); cfg.vision_deepstack.len()];
        let mut blocks_by_input: Vec<Vec<&VisionBlock>> = Vec::new();
        {
            // Image inputs own one block; video inputs own `grid.0` blocks.
            let mut bi = 0;
            for inp in inputs {
                let count = if p.blocks.get(bi).is_some_and(|b| b.video) {
                    inp.grid.0
                } else {
                    1
                };
                blocks_by_input.push(p.blocks[bi..bi + count].iter().collect());
                bi += count;
            }
            if bi != p.blocks.len() {
                return Err(Error::Geometry(format!(
                    "{} vision inputs for {} placeholder blocks",
                    inputs.len(),
                    p.blocks.len()
                )));
            }
        }
        for (inp, blocks) in inputs.iter().zip(&blocks_by_input) {
            let (feat, deep) = self.vision_features(be, inp)?;
            let total: usize = blocks.iter().map(|b| b.tokens).sum();
            if total * h != feat.len() {
                return Err(Error::Geometry(format!(
                    "{} vision features for {total} placeholder tokens",
                    feat.len() / h
                )));
            }
            let mut off = 0;
            for b in blocks {
                emb[b.start * h..(b.start + b.tokens) * h]
                    .copy_from_slice(&feat[off * h..(off + b.tokens) * h]);
                for (k, d) in deep.iter().enumerate() {
                    deep_rows[k].push((b.start, d[off * h..(off + b.tokens) * h].to_vec()));
                }
                off += b.tokens;
            }
        }
        let positions = self.rope_positions(p);
        let rope = qwen3_mrope(&positions, hd, cfg.rope_theta, cfg.mrope_section);
        let (rc, rs) = (be.upload_table(&rope.cos), be.upload_table(&rope.sin));
        let rp = Some(Rope {
            cos: &rc,
            sin: &rs,
            rot: hd,
        });
        let hb = be.upload(&emb);
        let normed = be.alloc(n * h);
        let qkv_w = (cfg.heads + 2 * cfg.kv_heads) * hd;
        let qkv = be.alloc(n * qkv_w);
        let q = be.alloc(cfg.heads * n * hd);
        let k = be.alloc(cfg.kv_heads * n * hd);
        let v = be.alloc(cfg.kv_heads * n * hd);
        let attn = be.alloc(n * cfg.heads * hd);
        let tmp = be.alloc(n * h);
        let gu = be.alloc(n * 2 * cfg.intermediate);
        let act = be.alloc(n * cfg.intermediate);
        // A non-resident layer uploads on the side queue (`prefetch_*`) while
        // the previous one computes, and is freed only after the compute queue
        // is past it (`prefetch_release`); a streamed layer is read on a
        // background thread while the device runs the layer before it.
        let (src, weights) = (&self.src, self.weights);
        let tree = self
            .int8_tree
            .as_ref()
            .filter(|_| weights == EncoderWeights::Int8);
        let read = move |l: usize| HostTextLayer::read(src, tree, cfg, l, weights);
        let stage = |h: &HostTextLayer| -> TextLayer<B> {
            be.prefetch_begin();
            let w = h.upload(be);
            be.prefetch_end();
            w
        };
        let last = cfg.output_layer;
        std::thread::scope(|s| -> Result<()> {
            let mut cur: Option<TextLayer<B>> = match self.layers.first() {
                None | Some(Slot::Device(_)) => None,
                Some(Slot::Host(h)) => Some(stage(h)),
                Some(Slot::Stream) => Some(stage(&read(0)?)),
            };
            for l in 0..last {
                let reader = match self.layers.get(l + 1) {
                    Some(Slot::Stream) => Some(s.spawn(move || read(l + 1))),
                    _ => None,
                };
                let layer = match &self.layers[l] {
                    Slot::Device(x) => x,
                    _ => {
                        be.prefetch_wait();
                        cur.as_ref().expect("staged layer")
                    }
                };
                be.rms_norm(
                    &hb,
                    n,
                    h,
                    Some(&layer.ln1),
                    Modulation::None,
                    cfg.rms_eps,
                    &normed,
                );
                be.linear(&normed, 0, n, &layer.qkv.w, None, Act::None, &qkv, 0);
                let qn = Some(HeadNorm {
                    w: Some(&layer.q_norm),
                    eps: cfg.rms_eps,
                });
                let kn = Some(HeadNorm {
                    w: Some(&layer.k_norm),
                    eps: cfg.rms_eps,
                });
                be.head_split(&qkv, n, qkv_w, 0, cfg.heads, hd, qn, rp, &q);
                be.head_split(&qkv, n, qkv_w, cfg.heads * hd, cfg.kv_heads, hd, kn, rp, &k);
                be.head_split(
                    &qkv,
                    n,
                    qkv_w,
                    (cfg.heads + cfg.kv_heads) * hd,
                    cfg.kv_heads,
                    hd,
                    None,
                    None,
                    &v,
                );
                be.attention(&q, &k, &v, cfg.heads, cfg.kv_heads, n, n, hd, true, &attn);
                be.linear(&attn, 0, n, &layer.o.w, None, Act::None, &tmp, 0);
                be.residual_rms_norm(
                    &hb,
                    &tmp,
                    n,
                    h,
                    Gate::None,
                    Some(&layer.ln2),
                    Modulation::None,
                    cfg.rms_eps,
                    &normed,
                );
                be.linear(&normed, 0, n, &layer.gate_up.w, None, Act::None, &gu, 0);
                be.swiglu(&gu, n, cfg.intermediate, &act);
                be.linear(&act, 0, n, &layer.down.w, None, Act::None, &tmp, 0);
                be.gate_residual(&hb, &tmp, n, h, Gate::None);
                if l < deep_rows.len() {
                    for (start, d) in &deep_rows[l] {
                        let gd = be.upload(d);
                        be.axpby(&hb, start * h, 1.0, &gd, 0, 1.0, d.len());
                    }
                }
                // This layer's kernels are queued: stage the next one behind
                // them, then let the side queue free this one once the compute
                // queue is past it.
                let next = match self.layers.get(l + 1) {
                    Some(Slot::Host(h)) => Some(stage(h)),
                    Some(Slot::Stream) => {
                        let h = reader
                            .expect("reader")
                            .join()
                            .map_err(|_| Error::Msg("layer reader panicked".into()))??;
                        Some(stage(&h))
                    }
                    _ => None,
                };
                if !matches!(self.layers[l], Slot::Device(_)) {
                    be.prefetch_release();
                }
                cur = next;
            }
            Ok(())
        })?;
        Ok(Encoded {
            embeds: be.download(&hb),
            rows: n,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::present::{VisionBlock, present_fl2va, test_tokenizer};
    use h3_core::export::{Tensor, bf16_bytes, f32_bytes, write};
    use h3_core::weights::{Synthetic, qwen3vl_manifest};
    use h3_cpu::Cpu;

    fn tiny_checkpoint(dir: &std::path::Path) -> SafetensorsDir {
        std::fs::create_dir_all(dir).unwrap();
        let cfg = Qwen3VlConfig::tiny();
        let syn = Synthetic { seed: 3 };
        let t: Vec<Tensor> = qwen3vl_manifest(&cfg)
            .into_iter()
            .map(|(n, s)| {
                let v = syn.get(&n, &s).unwrap();
                if n.ends_with("embed_tokens.weight") {
                    (n, safetensors::Dtype::BF16, s, bf16_bytes(&v))
                } else {
                    (n, safetensors::Dtype::F32, s, f32_bytes(&v))
                }
            })
            .collect();
        write(&dir.join("model.safetensors"), &t, Default::default()).unwrap();
        SafetensorsDir::open(dir).unwrap()
    }

    #[test]
    fn rope_positions_follow_get_rope_index() {
        let cfg = Qwen3VlConfig::tiny();
        let dir = std::env::temp_dir().join(format!("h3-qwen-{}", std::process::id()));
        let src = tiny_checkpoint(&dir);
        let be = Cpu;
        let m = Qwen3Vl::open(
            &be,
            cfg,
            src,
            Placement::Stream,
            EncoderWeights::Bf16,
            None,
            false,
        )
        .unwrap();
        // 2 text, a 1×4×4 image block (4 merged tokens) wrapped in start/end, 3 text.
        let p = Presentation {
            ids: vec![5, 6, 1000, 7, 7, 7, 7, 1001, 8, 9, 10],
            tags: vec![1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            blocks: vec![VisionBlock {
                start: 3,
                tokens: 4,
                video: false,
                grid: (1, 4, 4),
            }],
        };
        let pos = m.rope_positions(&p);
        assert_eq!(pos[..3], [[0, 0, 0], [1, 1, 1], [2, 2, 2]]);
        assert_eq!(pos[3..7], [[3, 3, 3], [3, 3, 4], [3, 4, 3], [3, 4, 4]]);
        assert_eq!(pos[7], [5, 5, 5]);
        assert_eq!(pos[10], [8, 8, 8]);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn tiny_conditioner_runs_text_and_vision() {
        let cfg = Qwen3VlConfig::tiny();
        let dir = std::env::temp_dir().join(format!("h3-qwen-v-{}", std::process::id()));
        let src = tiny_checkpoint(&dir);
        let be = Cpu;
        let streamed = Qwen3Vl::open(
            &be,
            cfg.clone(),
            tiny_checkpoint(&dir),
            Placement::Stream,
            EncoderWeights::Bf16,
            None,
            true,
        )
        .unwrap();
        let resident = Qwen3Vl::open(
            &be,
            cfg.clone(),
            src,
            Placement::Resident,
            EncoderWeights::Bf16,
            None,
            true,
        )
        .unwrap();
        let tok = test_tokenizer();
        let mut rgb = vec![0u8; 64 * 64 * 3];
        for (i, v) in rgb.iter_mut().enumerate() {
            *v = (i % 251) as u8;
        }
        let im = crate::image::Image::new(64, 64, rgb);
        let vi = crate::vision::preprocess_image(&im).unwrap();
        let p = present_fl2va(&tok, "a cat on a mat", &[vi]).unwrap();
        // Keep ids inside the tiny vocabulary.
        let mut p = p;
        for id in &mut p.ids {
            *id %= cfg.vocab as u32;
        }
        let a = streamed
            .encode(&be, &p, &[&crate::vision::preprocess_image(&im).unwrap()])
            .unwrap();
        let b = resident
            .encode(&be, &p, &[&crate::vision::preprocess_image(&im).unwrap()])
            .unwrap();
        assert_eq!(a.rows, p.ids.len());
        assert_eq!(a.embeds.len(), p.ids.len() * cfg.hidden);
        assert!(a.embeds.iter().all(|v| v.is_finite()));
        assert_eq!(a.embeds, b.embeds);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// A tiny checkpoint whose linears are bf16 (the real one's format).
    fn tiny_checkpoint_bf16(dir: &std::path::Path) -> SafetensorsDir {
        std::fs::create_dir_all(dir).unwrap();
        let cfg = Qwen3VlConfig::tiny();
        let syn = Synthetic { seed: 3 };
        let t: Vec<Tensor> = qwen3vl_manifest(&cfg)
            .into_iter()
            .map(|(n, s)| {
                let v = syn.get(&n, &s).unwrap();
                if s.len() == 2 {
                    (n, safetensors::Dtype::BF16, s, bf16_bytes(&v))
                } else {
                    (n, safetensors::Dtype::F32, s, f32_bytes(&v))
                }
            })
            .collect();
        write(&dir.join("model.safetensors"), &t, Default::default()).unwrap();
        SafetensorsDir::open(dir).unwrap()
    }

    #[test]
    fn placements_agree_and_int8_stays_close() {
        let cfg = Qwen3VlConfig::tiny();
        let dir = std::env::temp_dir().join(format!("h3-qwen-p-{}", std::process::id()));
        let be = Cpu;
        let tok = test_tokenizer();
        let mut p = present_fl2va(&tok, "a red fox trots through fresh snow", &[]).unwrap();
        for id in &mut p.ids {
            *id %= cfg.vocab as u32;
        }
        let run = |pl: Placement, w: EncoderWeights| {
            Qwen3Vl::open(
                &be,
                cfg.clone(),
                tiny_checkpoint_bf16(&dir),
                pl,
                w,
                None,
                false,
            )
            .unwrap()
            .encode(&be, &p, &[])
            .unwrap()
            .embeds
        };
        let stream = run(Placement::Stream, EncoderWeights::Bf16);
        assert!(stream.iter().all(|v| v.is_finite()));
        // Same bits whatever the placement: the layers are the same weights.
        assert_eq!(stream, run(Placement::Host, EncoderWeights::Bf16));
        assert_eq!(stream, run(Placement::Resident, EncoderWeights::Bf16));
        // Int8 weight-only moves the output a little, the same in every placement.
        let q = run(Placement::Resident, EncoderWeights::Int8);
        assert_eq!(q, run(Placement::Stream, EncoderWeights::Int8));
        // The stored int8 tree gives the bits quantizing at load gives.
        let tree = dir.join("int8");
        let mut layers = 0;
        write_int8_tree(&tiny_checkpoint_bf16(&dir), &cfg, &tree, &mut |_, _| {
            layers += 1
        })
        .unwrap();
        assert_eq!(layers, cfg.output_layer);
        let from_tree = Qwen3Vl::open(
            &be,
            cfg.clone(),
            tiny_checkpoint_bf16(&dir),
            Placement::Resident,
            EncoderWeights::Int8,
            Some(SafetensorsDir::open(&tree).unwrap()),
            false,
        )
        .unwrap()
        .encode(&be, &p, &[])
        .unwrap()
        .embeds;
        assert_eq!(q, from_tree);
        let num: f64 = stream
            .iter()
            .zip(&q)
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum();
        let den: f64 = stream.iter().map(|a| (*a as f64).powi(2)).sum();
        let rel = (num / den).sqrt();
        assert!(rel > 0.0 && rel < 2e-2, "int8 relative error {rel}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
