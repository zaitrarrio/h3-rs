//! Qwen3-VL-32B as H3's conditioner, on the backend: the vision tower with
//! DeepStack, the first 50 language layers (hidden state 50, no final norm).
//! Language layers are streamed from the memory-mapped checkpoint one at a
//! time (the 64 GB of bf16 never has to be resident), or kept resident when
//! the device has room.

use crate::present::{Presentation, VisionBlock};
use crate::vision::VisionInput;
use h3_core::{
    Error, Result,
    backend::{Act, Backend, Gate, HeadNorm, Modulation, Rope},
    config::Qwen3VlConfig,
    quant::{Nvfp4, e4m3_to_f32},
    rope::{qwen3_mrope, qwen3_vision_rope},
    weights::{SafetensorsDir, WeightSource, bytes_to_f32},
};
use h3_model::linear::Linear;

const LM: &str = "model.language_model";
const VIS: &str = "model.visual";

/// An NVFP4 linear as compressed-tensors stores it (`weight_packed`,
/// row-major E4M3 `weight_scale`, `weight_global_scale`): `w = fp4 ·
/// e4m3(scale) / global_scale`.
fn nvfp4_of(src: &SafetensorsDir, key: &str, rows: usize, cols: usize) -> Result<Nvfp4> {
    let packed_name = format!("{key}.weight_packed");
    let (_, pshape, packed) = src.raw(&packed_name)?;
    if pshape != [rows, cols / 2] {
        return Err(Error::Weights(format!("`{packed_name}` is {pshape:?}, expected [{rows}, {}]", cols / 2)));
    }
    let (_, sshape, scales) = src.raw(&format!("{key}.weight_scale"))?;
    if sshape != [rows, cols / 16] {
        return Err(Error::Weights(format!("`{key}.weight_scale` is {sshape:?}, expected [{rows}, {}]", cols / 16)));
    }
    let global = src.scalar(&format!("{key}.weight_global_scale"))?;
    // compressed-tensors packs the first code of each pair in the low nibble.
    Ok(Nvfp4 { packed: packed.to_vec(), scales: scales.to_vec(), alpha: 1.0 / global, rows, cols, first_high: false })
}

/// A linear from a plain or NVFP4 checkpoint.
fn linear<B: Backend>(be: &B, src: &SafetensorsDir, key: &str, rows: usize, cols: usize, bias: bool) -> Result<Linear<B>> {
    let b = if bias { Some(be.upload(&src.get(&format!("{key}.bias"), &[rows])?)) } else { None };
    if src.contains_raw(&format!("{key}.weight")) {
        return Ok(Linear { w: be.weight(&src.get(&format!("{key}.weight"), &[rows, cols])?, rows, cols), b, rows, cols });
    }
    if src.contains_raw(&format!("{key}.weight_packed")) {
        let q = nvfp4_of(src, key, rows, cols)?;
        let _ = e4m3_to_f32;
        return Ok(Linear { w: be.weight_nvfp4(&q), b, rows, cols });
    }
    Err(Error::Weights(format!("missing linear `{key}`")))
}

/// Several linears stacked along the output axis (NVFP4 parts dequantized).
fn fused<B: Backend>(be: &B, src: &SafetensorsDir, keys: &[(&str, usize)], cols: usize) -> Result<Linear<B>> {
    let rows: usize = keys.iter().map(|k| k.1).sum();
    let mut w = Vec::with_capacity(rows * cols);
    for (k, r) in keys {
        if src.contains_raw(&format!("{k}.weight")) {
            w.extend(src.get(&format!("{k}.weight"), &[*r, cols])?);
        } else {
            w.extend(nvfp4_of(src, k, *r, cols)?.dequantize());
        }
    }
    Ok(Linear { w: be.weight(&w, rows, cols), b: None, rows, cols })
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

fn load_text_layer<B: Backend>(be: &B, src: &SafetensorsDir, cfg: &Qwen3VlConfig, l: usize) -> Result<TextLayer<B>> {
    let p = format!("{LM}.layers.{l}");
    let (h, hd) = (cfg.hidden, cfg.head_dim);
    let v = |k: &str, n: usize| -> Result<B::Buf> { Ok(be.upload(&src.get(k, &[n])?)) };
    Ok(TextLayer {
        ln1: v(&format!("{p}.input_layernorm.weight"), h)?,
        qkv: fused(
            be,
            src,
            &[
                (&format!("{p}.self_attn.q_proj"), cfg.heads * hd),
                (&format!("{p}.self_attn.k_proj"), cfg.kv_heads * hd),
                (&format!("{p}.self_attn.v_proj"), cfg.kv_heads * hd),
            ],
            h,
        )?,
        o: linear(be, src, &format!("{p}.self_attn.o_proj"), h, cfg.heads * hd, false)?,
        q_norm: v(&format!("{p}.self_attn.q_norm.weight"), hd)?,
        k_norm: v(&format!("{p}.self_attn.k_norm.weight"), hd)?,
        ln2: v(&format!("{p}.post_attention_layernorm.weight"), h)?,
        // SwiGLU takes `value | gate`: up first, then gate.
        gate_up: fused(be, src, &[(&format!("{p}.mlp.up_proj"), cfg.intermediate), (&format!("{p}.mlp.gate_proj"), cfg.intermediate)], h)?,
        down: linear(be, src, &format!("{p}.mlp.down_proj"), h, cfg.intermediate, false)?,
    })
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
    /// Resident language layers (`None` while streaming).
    layers: Vec<Option<TextLayer<B>>>,
    pub resident: bool,
    vision: Option<VisionTower<B>>,
}

/// A conditioning result: `[rows, hidden]` f32.
pub struct Encoded {
    pub embeds: Vec<f32>,
    pub rows: usize,
}

impl<B: Backend> Qwen3Vl<B> {
    /// `resident`: keep all 50 language layers on the device; `vision`: load
    /// the vision tower (FL2VA / Ref2VA).
    pub fn open(be: &B, cfg: Qwen3VlConfig, src: SafetensorsDir, resident: bool, vision: bool) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.output_layer);
        for l in 0..cfg.output_layer {
            layers.push(if resident { Some(load_text_layer(be, &src, &cfg, l)?) } else { None });
        }
        let vision = if vision { Some(Self::load_vision(be, &src, &cfg)?) } else { None };
        Ok(Self { cfg, src, layers, resident, vision })
    }

    fn load_vision(be: &B, src: &SafetensorsDir, cfg: &Qwen3VlConfig) -> Result<VisionTower<B>> {
        let vh = cfg.vision_hidden;
        let vm = vh * cfg.vision_merge * cfg.vision_merge;
        let pf = 3 * cfg.vision_temporal_patch * cfg.vision_patch * cfg.vision_patch;
        let ln = |k: &str, n: usize| -> Result<(B::Buf, B::Buf)> {
            Ok((be.upload(&src.get(&format!("{k}.weight"), &[n])?), be.upload(&src.get(&format!("{k}.bias"), &[n])?)))
        };
        let merger = |p: &str, post: bool| -> Result<Merger<B>> {
            Ok(Merger {
                norm: ln(&format!("{p}.norm"), if post { vm } else { vh })?,
                post,
                fc1: linear(be, src, &format!("{p}.linear_fc1"), vm, vm, true)?,
                fc2: linear(be, src, &format!("{p}.linear_fc2"), cfg.vision_out_hidden, vm, true)?,
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
                fc1: linear(be, src, &format!("{p}.mlp.linear_fc1"), cfg.vision_intermediate, vh, true)?,
                fc2: linear(be, src, &format!("{p}.mlp.linear_fc2"), vh, cfg.vision_intermediate, true)?,
            });
        }
        let pe = {
            let w = src.get(&format!("{VIS}.patch_embed.proj.weight"), &[vh, 3, cfg.vision_temporal_patch, cfg.vision_patch, cfg.vision_patch])?;
            Linear { w: be.weight(&w, vh, pf), b: Some(be.upload(&src.get(&format!("{VIS}.patch_embed.proj.bias"), &[vh])?)), rows: vh, cols: pf }
        };
        Ok(VisionTower {
            patch_embed: pe,
            pos_embed: src.get(&format!("{VIS}.pos_embed.weight"), &[cfg.vision_num_position_embeddings, vh])?,
            blocks,
            merger: merger(&format!("{VIS}.merger"), false)?,
            deepstack: (0..cfg.vision_deepstack.len()).map(|i| merger(&format!("{VIS}.deepstack_merger_list.{i}"), true)).collect::<Result<_>>()?,
        })
    }

    /// Token embeddings `[rows, hidden]` read straight from the checkpoint.
    pub fn embed(&self, ids: &[u32]) -> Result<Vec<f32>> {
        let (dt, shape, data) = self.src.raw(&format!("{LM}.embed_tokens.weight"))?;
        let h = shape[1];
        let bytes = match dt {
            safetensors::Dtype::BF16 | safetensors::Dtype::F16 => 2,
            safetensors::Dtype::F32 => 4,
            other => return Err(Error::Weights(format!("embed_tokens: unsupported dtype {other:?}"))),
        };
        let mut out = Vec::with_capacity(ids.len() * h);
        for &id in ids {
            if id as usize >= shape[0] {
                return Err(Error::Msg(format!("token id {id} is out of the vocabulary")));
            }
            let off = id as usize * h * bytes;
            out.extend(bytes_to_f32(dt, &data[off..off + h * bytes], "embed_tokens")?);
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
                                    let row = &tower.pos_embed[(ri * side + ci) * vh..(ri * side + ci + 1) * vh];
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
    pub fn vision_features(&self, be: &B, input: &VisionInput) -> Result<(Vec<f32>, Vec<Vec<f32>>)> {
        let tower = self.vision.as_ref().ok_or_else(|| Error::Msg("the vision tower is not loaded".into()))?;
        let cfg = &self.cfg;
        let (vh, heads) = (cfg.vision_hidden, cfg.vision_heads);
        let hd = cfg.vision_head_dim();
        let n = input.tokens();
        let (t, gh, gw) = input.grid;
        let per_frame = gh * gw;
        let m = cfg.vision_merge;
        let x = be.upload(&input.patches);
        let h = be.alloc(n * vh);
        be.linear(&x, 0, n, &tower.patch_embed.w, tower.patch_embed.b.as_ref(), Act::None, &h, 0);
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
        let frp = Some(Rope { cos: &fc, sin: &fs, rot: hd });
        let normed = be.alloc(n * vh);
        let qkv = be.alloc(n * 3 * vh);
        let (q, k, v) = (be.alloc(heads * per_frame * hd), be.alloc(heads * per_frame * hd), be.alloc(heads * per_frame * hd));
        let attn = be.alloc(n * vh);
        let tmp = be.alloc(n * vh);
        let hid = be.alloc(n * cfg.vision_intermediate);
        let mut deep = Vec::new();
        let merged = n / (m * m);
        let vm = vh * m * m;
        let run_merger = |mg: &Merger<B>, hin: &B::Buf| -> Vec<f32> {
            let pre = be.alloc(n * vh);
            let (rows, dim) = if mg.post { (merged, vm) } else { (n, vh) };
            be.layer_norm(hin, rows, dim, Some(&mg.norm.0), Some(&mg.norm.1), 1e-6, &pre);
            let f1 = be.alloc(merged * vm);
            be.linear(&pre, 0, merged, &mg.fc1.w, mg.fc1.b.as_ref(), Act::GeluErf, &f1, 0);
            let f2 = be.alloc(merged * cfg.vision_out_hidden);
            be.linear(&f1, 0, merged, &mg.fc2.w, mg.fc2.b.as_ref(), Act::None, &f2, 0);
            be.download(&f2)
        };
        for (bi, bl) in tower.blocks.iter().enumerate() {
            be.layer_norm(&h, n, vh, Some(&bl.norm1.0), Some(&bl.norm1.1), 1e-6, &normed);
            be.linear(&normed, 0, n, &bl.qkv.w, bl.qkv.b.as_ref(), Act::None, &qkv, 0);
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
            be.linear(&attn, 0, n, &bl.proj.w, bl.proj.b.as_ref(), Act::None, &tmp, 0);
            be.gate_residual(&h, &tmp, n, vh, Gate::None);
            be.layer_norm(&h, n, vh, Some(&bl.norm2.0), Some(&bl.norm2.1), 1e-6, &normed);
            be.linear(&normed, 0, n, &bl.fc1.w, bl.fc1.b.as_ref(), Act::GeluTanh, &hid, 0);
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
        let mut deep_rows: Vec<Vec<(usize, Vec<f32>)>> = vec![Vec::new(); cfg.vision_deepstack.len()];
        let mut blocks_by_input: Vec<Vec<&VisionBlock>> = Vec::new();
        {
            // Image inputs own one block; video inputs own `grid.0` blocks.
            let mut bi = 0;
            for inp in inputs {
                let count = if p.blocks.get(bi).is_some_and(|b| b.video) { inp.grid.0 } else { 1 };
                blocks_by_input.push(p.blocks[bi..bi + count].iter().collect());
                bi += count;
            }
            if bi != p.blocks.len() {
                return Err(Error::Geometry(format!("{} vision inputs for {} placeholder blocks", inputs.len(), p.blocks.len())));
            }
        }
        for (inp, blocks) in inputs.iter().zip(&blocks_by_input) {
            let (feat, deep) = self.vision_features(be, inp)?;
            let total: usize = blocks.iter().map(|b| b.tokens).sum();
            if total * h != feat.len() {
                return Err(Error::Geometry(format!("{} vision features for {total} placeholder tokens", feat.len() / h)));
            }
            let mut off = 0;
            for b in blocks {
                emb[b.start * h..(b.start + b.tokens) * h].copy_from_slice(&feat[off * h..(off + b.tokens) * h]);
                for (k, d) in deep.iter().enumerate() {
                    deep_rows[k].push((b.start, d[off * h..(off + b.tokens) * h].to_vec()));
                }
                off += b.tokens;
            }
        }
        let positions = self.rope_positions(p);
        let rope = qwen3_mrope(&positions, hd, cfg.rope_theta, cfg.mrope_section);
        let (rc, rs) = (be.upload_table(&rope.cos), be.upload_table(&rope.sin));
        let rp = Some(Rope { cos: &rc, sin: &rs, rot: hd });
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
        for l in 0..cfg.output_layer {
            let loaded;
            let layer = match &self.layers[l] {
                Some(x) => x,
                None => {
                    loaded = load_text_layer(be, &self.src, cfg, l)?;
                    &loaded
                }
            };
            be.rms_norm(&hb, n, h, Some(&layer.ln1), Modulation::None, cfg.rms_eps, &normed);
            be.linear(&normed, 0, n, &layer.qkv.w, None, Act::None, &qkv, 0);
            let qn = Some(HeadNorm { w: Some(&layer.q_norm), eps: cfg.rms_eps });
            let kn = Some(HeadNorm { w: Some(&layer.k_norm), eps: cfg.rms_eps });
            be.head_split(&qkv, n, qkv_w, 0, cfg.heads, hd, qn, rp, &q);
            be.head_split(&qkv, n, qkv_w, cfg.heads * hd, cfg.kv_heads, hd, kn, rp, &k);
            be.head_split(&qkv, n, qkv_w, (cfg.heads + cfg.kv_heads) * hd, cfg.kv_heads, hd, None, None, &v);
            be.attention(&q, &k, &v, cfg.heads, cfg.kv_heads, n, n, hd, true, &attn);
            be.linear(&attn, 0, n, &layer.o.w, None, Act::None, &tmp, 0);
            be.residual_rms_norm(&hb, &tmp, n, h, Gate::None, Some(&layer.ln2), Modulation::None, cfg.rms_eps, &normed);
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
            be.sync();
        }
        Ok(Encoded { embeds: be.download(&hb), rows: n })
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
        let m = Qwen3Vl::open(&be, cfg, src, false, false).unwrap();
        // 2 text, a 1×4×4 image block (4 merged tokens) wrapped in start/end, 3 text.
        let p = Presentation {
            ids: vec![5, 6, 1000, 7, 7, 7, 7, 1001, 8, 9, 10],
            tags: vec![1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            blocks: vec![VisionBlock { start: 3, tokens: 4, video: false, grid: (1, 4, 4) }],
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
        let streamed = Qwen3Vl::open(&be, cfg.clone(), tiny_checkpoint(&dir), false, true).unwrap();
        let resident = Qwen3Vl::open(&be, cfg.clone(), src, true, true).unwrap();
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
        let a = streamed.encode(&be, &p, &[&crate::vision::preprocess_image(&im).unwrap()]).unwrap();
        let b = resident.encode(&be, &p, &[&crate::vision::preprocess_image(&im).unwrap()]).unwrap();
        assert_eq!(a.rows, p.ids.len());
        assert_eq!(a.embeds.len(), p.ids.len() * cfg.hidden);
        assert!(a.embeds.iter().all(|v| v.is_finite()));
        assert_eq!(a.embeds, b.embeds);
        std::fs::remove_dir_all(&dir).ok();
    }
}
