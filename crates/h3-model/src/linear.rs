//! Device linears: load (optionally fused along the output axis), merge
//! adapters, quantize, or keep on the host for per-step streaming.

use h3_core::{
    Error, Result,
    backend::Backend,
    lora::Adapter,
    quant::{Mxfp8, Nvfp4, Q4, Q8, Stored, WeightQuant},
    weights::WeightSource,
};
use std::collections::HashMap;

pub type Shapes = HashMap<String, Vec<usize>>;

pub struct Linear<B: Backend> {
    pub w: B::Weight,
    pub b: Option<B::Buf>,
    pub rows: usize,
    pub cols: usize,
    /// Unmerged LoRA pairs ([`LowRank`]): `y = W x + b + B (A x)`.
    pub lora: Option<LowRank<B>>,
}

/// The LoRA pairs of one (possibly fused) linear, kept as factors: `a`
/// `[rank, cols]` (every pair's `A` stacked), `b` `[rows, rank]` (each pair's
/// `scale · B` in its rows and columns, zeros elsewhere, so a fused QKV keeps
/// one branch). Merging into the device weight would round most of a small
/// delta away ([`h3_core::lora::Adapter::lora_factors`]).
pub struct LowRank<B: Backend> {
    pub a: B::Weight,
    pub b: B::Weight,
    pub rank: usize,
}

impl<B: Backend> Linear<B> {
    /// `y[y_row + i] = x[x_row + i] · Wᵀ (+ b) (+ B·A·x)` for `i < rows`, no
    /// activation (the branch must land before one). `t` holds `rows × rank`
    /// when the linear has a branch ([`Linear::rank`]).
    #[allow(clippy::too_many_arguments)]
    pub fn apply(
        &self,
        be: &B,
        x: &B::Buf,
        x_row: usize,
        rows: usize,
        y: &B::Buf,
        y_row: usize,
        t: Option<&B::Buf>,
    ) {
        be.linear(
            x,
            x_row,
            rows,
            &self.w,
            self.b.as_ref(),
            h3_core::backend::Act::None,
            y,
            y_row,
        );
        if let Some(l) = &self.lora {
            let t = t.expect("a linear with a LoRA branch needs its rank scratch");
            be.linear(
                x,
                x_row,
                rows,
                &l.a,
                None,
                h3_core::backend::Act::None,
                t,
                0,
            );
            be.linear_add(t, 0, rows, &l.b, self.rows, y, y_row);
        }
    }

    pub fn rank(&self) -> usize {
        self.lora.as_ref().map_or(0, |l| l.rank)
    }
}

/// A linear's weights before they reach a device: the host side of a
/// streamed block, or the input of [`upload`].
pub enum HostWeight {
    F32(Vec<f32>),
    /// bf16 bits, `[rows][cols]`: the streamed form of a native-precision block
    /// ([`HostWeight::to_bf16`]), uploaded as bytes.
    Bf16(Vec<u16>),
    Q8(Q8),
    Q4(Q4),
    Nvfp4(Nvfp4),
    Mxfp8(Mxfp8),
}

impl HostWeight {
    pub fn bytes(&self) -> usize {
        match self {
            HostWeight::F32(v) => v.len() * 4,
            HostWeight::Bf16(v) => v.len() * 2,
            HostWeight::Q8(q) => q.bytes(),
            HostWeight::Q4(q) => q.bytes(),
            HostWeight::Nvfp4(q) => q.bytes(),
            HostWeight::Mxfp8(q) => q.bytes(),
        }
    }
}

pub struct HostLinear {
    pub w: HostWeight,
    pub b: Option<Vec<f32>>,
    pub rows: usize,
    pub cols: usize,
    pub lora: Option<HostLowRank>,
}

/// [`LowRank`] on the host: `a` `[rank, cols]`, `b` `[rows, rank]`, f32 or
/// (streamed) bf16.
pub struct HostLowRank {
    pub a: HostWeight,
    pub b: HostWeight,
    pub rank: usize,
}

/// Whether block linears keep their LoRA pairs as branches (`H3_LORA_MERGE=1`
/// merges them into the weight instead, as before 2026-11-15, to compare).
pub fn lora_branches() -> bool {
    !std::env::var("H3_LORA_MERGE").is_ok_and(|v| v == "1")
}

/// Whether stored NVFP4 weights run W4A4 (`H3_NVFP4_A4=1`): the activations
/// through NVFP4 as well, on the FP4 tensor cores on sm_100+ (ComfyUI's
/// recipe for these files), instead of bf16 activations against the weights
/// expanded per GEMM. Opt-in: it moves the numerics.
pub fn nvfp4_act4() -> bool {
    std::env::var("H3_NVFP4_A4").is_ok_and(|v| v == "1")
}

/// A page-lock guard ([`Backend::pin_host`]); dropped before its buffer.
pub type PinGuard = Box<dyn std::any::Any + Send>;

impl HostWeight {
    /// Page-lock the buffers a streamed upload reads ([`Backend::pin_host`]).
    pub fn pin<B: Backend>(&self, be: &B, out: &mut Vec<PinGuard>) {
        let bufs: Vec<&[u8]> = match self {
            // SAFETY (both casts): `f32` / `i8` reinterpreted as the bytes of the same allocation.
            HostWeight::F32(v) => {
                vec![unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }]
            }
            HostWeight::Bf16(v) => {
                vec![unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 2) }]
            }
            HostWeight::Q8(q) => {
                vec![unsafe { std::slice::from_raw_parts(q.q.as_ptr() as *const u8, q.q.len()) }]
            }
            HostWeight::Q4(_) => vec![],
            HostWeight::Nvfp4(q) => vec![&q.packed, &q.scales],
            HostWeight::Mxfp8(q) => vec![&q.codes, &q.scales],
        };
        out.extend(bufs.into_iter().filter_map(|b| be.pin_host(b)));
    }
}

impl HostWeight {
    /// A native-precision (f32) weight becomes bf16 bits, what the device
    /// holds anyway; the other forms are already byte-coded. For blocks that
    /// stream every step: half the host memory, and the upload is a plain copy.
    pub fn to_bf16(&mut self) {
        if let HostWeight::F32(v) = self {
            let h: Vec<u16> = v
                .iter()
                .map(|&x| half::bf16::from_f32(x).to_bits())
                .collect();
            *self = HostWeight::Bf16(h);
        }
    }
}

impl HostWeight {
    fn upload<B: Backend>(&self, be: &B, rows: usize, cols: usize) -> B::Weight {
        match self {
            HostWeight::F32(v) => be.weight(v, rows, cols),
            HostWeight::Bf16(v) => be.weight_bf16(v, rows, cols),
            HostWeight::Q8(q) => be.weight_q8(q),
            HostWeight::Q4(q) => be.weight_q4(q),
            HostWeight::Nvfp4(q) => be.weight_nvfp4(q),
            HostWeight::Mxfp8(q) => be.weight_mxfp8(q),
        }
    }
}

impl HostLinear {
    pub fn pin<B: Backend>(&self, be: &B, out: &mut Vec<PinGuard>) {
        self.w.pin(be, out);
        if let Some(l) = &self.lora {
            l.a.pin(be, out);
            l.b.pin(be, out);
        }
    }

    pub fn to_bf16(&mut self) {
        self.w.to_bf16();
        if let Some(l) = &mut self.lora {
            l.a.to_bf16();
            l.b.to_bf16();
        }
    }

    pub fn bytes(&self) -> usize {
        self.w.bytes() + self.lora.as_ref().map_or(0, |l| l.a.bytes() + l.b.bytes())
    }

    pub fn upload<B: Backend>(&self, be: &B) -> Linear<B> {
        Linear {
            w: self.w.upload(be, self.rows, self.cols),
            b: self.b.as_ref().map(|b| be.upload(b)),
            rows: self.rows,
            cols: self.cols,
            lora: self.lora.as_ref().map(|l| LowRank {
                a: l.a.upload(be, l.rank, self.cols),
                b: l.b.upload(be, self.rows, l.rank),
                rank: l.rank,
            }),
        }
    }
}

/// Read the linears `keys` (no `.weight`), stacked along the output axis
/// (fused QKV), merge `adapters`, and quantize as `quant.for_key(keys[0])`
/// says — unless the checkpoint stores them quantized already, in which case
/// the stored codes are used as they are (an adapter forces the f32 path).
/// `bias` says whether `<key>.bias` exists.
pub fn load_host(
    src: &dyn WeightSource,
    shapes: &Shapes,
    keys: &[&str],
    bias: bool,
    quant: WeightQuant,
    adapters: &[Adapter],
    branch: bool,
) -> Result<HostLinear> {
    let err = |m: String| Error::Weights(m);
    // With `branch`, LoRA pairs stay factors (`HostLowRank`) and only dense
    // deltas (set_weight / diff) force the f32 rebuild, so a stored quantized
    // file keeps its codes under a LoRA.
    let branch = branch && lora_branches();
    let touched = adapters.iter().any(|a| {
        keys.iter().any(|k| {
            if branch {
                a.touches_dense(k)
            } else {
                a.touches(k)
            }
        })
    });
    // (row offset, rows, A [r, cols], scale·B [rows, r], r) per key and adapter.
    let mut pairs: Vec<(usize, usize, Vec<f32>, Vec<f32>, usize)> = Vec::new();
    let (mut w, mut b, mut stored, mut rows, mut cols) =
        (Vec::new(), Vec::new(), Vec::<Stored>::new(), 0usize, 0usize);
    for k in keys {
        let shape = shapes
            .get(&format!("{k}.weight"))
            .ok_or_else(|| err(format!("`{k}` is not in the manifest")))?;
        let (o, i) = (shape[0], shape[1..].iter().product::<usize>());
        match src.stored(k)? {
            Some(s) if !touched => {
                if s.shape() != (o, i) {
                    return Err(err(format!(
                        "`{k}` is stored {:?}, expected {o}x{i}",
                        s.shape()
                    )));
                }
                stored.push(s);
            }
            Some(s) => w.extend(s.dequantize()),
            // A linear the base lacks but an adapter sets outright (the V1
            // LoRA's `to_gate_compress.set_weight` over MiniMax-H3) starts
            // from zeros.
            None if touched && !src.contains(&format!("{k}.weight")) => {
                w.resize(w.len() + o * i, 0.0)
            }
            None => w.extend(src.get(&format!("{k}.weight"), shape)?),
        }
        if touched {
            let start = w.len() - o * i;
            for a in adapters {
                if branch {
                    a.apply_dense(k, &mut w[start..], o, i)?;
                } else {
                    a.apply_weight(k, &mut w[start..], o, i)?;
                }
            }
        }
        if branch {
            for a in adapters {
                if let Some((fa, fb, r)) = a.lora_factors(k, o, i)? {
                    pairs.push((rows, o, fa, fb, r));
                }
            }
        }
        if bias {
            let mut bb = src.get(&format!("{k}.bias"), &[o])?;
            for a in adapters {
                a.apply_bias(k, &mut bb)?;
            }
            b.extend(bb);
        }
        rows += o;
        cols = i;
    }
    let w = if stored.is_empty() {
        match quant.for_key(keys[0]) {
            WeightQuant::Native => HostWeight::F32(w),
            WeightQuant::Int8 => HostWeight::Q8(Q8::quantize(&w, rows, cols)),
            WeightQuant::Int4 { group, hqq, .. } => {
                HostWeight::Q4(Q4::quantize(&w, rows, cols, group, hqq))
            }
            // NVFP4 is only ever read from a checkpoint; a dequantized merge is
            // re-quantized to int8 so the device cost stays small.
            WeightQuant::Nvfp4 => HostWeight::Q8(Q8::quantize(&w, rows, cols)),
            WeightQuant::Mxfp8 if cols.is_multiple_of(32) => {
                HostWeight::Mxfp8(Mxfp8::quantize(&w, rows, cols))
            }
            WeightQuant::Mxfp8 => HostWeight::F32(w),
        }
    } else if stored.len() != keys.len() {
        return Err(err(format!("{keys:?}: only some are stored quantized")));
    } else if stored.iter().all(|s| matches!(s, Stored::Q8(_))) {
        HostWeight::Q8(Q8::concat(
            stored
                .into_iter()
                .map(|s| {
                    if let Stored::Q8(q) = s {
                        q
                    } else {
                        unreachable!()
                    }
                })
                .collect(),
        ))
    } else if stored.iter().all(|s| matches!(s, Stored::Mxfp8(_))) {
        HostWeight::Mxfp8(Mxfp8::concat(
            stored
                .into_iter()
                .map(|s| {
                    if let Stored::Mxfp8(q) = s {
                        q
                    } else {
                        unreachable!()
                    }
                })
                .collect(),
        ))
    } else if stored.iter().all(|s| matches!(s, Stored::Q4(_))) {
        HostWeight::Q4(Q4::concat(
            stored
                .into_iter()
                .map(|s| {
                    if let Stored::Q4(q) = s {
                        q
                    } else {
                        unreachable!()
                    }
                })
                .collect(),
        ))
    } else if stored.iter().all(|s| matches!(s, Stored::Nvfp4(_))) {
        let parts: Vec<Nvfp4> = stored
            .into_iter()
            .map(|s| {
                if let Stored::Nvfp4(q) = s {
                    q
                } else {
                    unreachable!()
                }
            })
            .collect();
        match Nvfp4::concat(parts.clone()) {
            Some(q) => HostWeight::Nvfp4(Nvfp4 {
                act4: nvfp4_act4(),
                ..q
            }),
            None => {
                let mut w = Vec::new();
                for p in &parts {
                    w.extend(p.dequantize());
                }
                HostWeight::Q8(Q8::quantize(&w, rows, cols))
            }
        }
    } else {
        return Err(err(format!("{keys:?}: mixed stored formats")));
    };
    let lora = (!pairs.is_empty()).then(|| {
        let rank: usize = pairs.iter().map(|p| p.4).sum();
        let mut a = Vec::with_capacity(rank * cols);
        let mut b = vec![0f32; rows * rank];
        let mut c = 0;
        for (r0, o, fa, fb, r) in &pairs {
            a.extend_from_slice(fa);
            for i in 0..*o {
                b[(r0 + i) * rank + c..(r0 + i) * rank + c + r]
                    .copy_from_slice(&fb[i * r..(i + 1) * r]);
            }
            c += r;
        }
        HostLowRank {
            a: HostWeight::F32(a),
            b: HostWeight::F32(b),
            rank,
        }
    });
    Ok(HostLinear {
        w,
        b: bias.then_some(b),
        rows,
        cols,
        lora,
    })
}

pub fn load<B: Backend>(
    be: &B,
    src: &dyn WeightSource,
    shapes: &Shapes,
    keys: &[&str],
    bias: bool,
    quant: WeightQuant,
    adapters: &[Adapter],
) -> Result<Linear<B>> {
    Ok(load_host(src, shapes, keys, bias, quant, adapters, false)?.upload(be))
}

/// A plain f32 `[rows, cols]` matrix (+ bias) from the source, as a device linear.
pub fn load_plain<B: Backend>(
    be: &B,
    src: &dyn WeightSource,
    key: &str,
    rows: usize,
    cols: usize,
    bias: bool,
) -> Result<Linear<B>> {
    let w = src.get(&format!("{key}.weight"), &[rows, cols])?;
    let b = if bias {
        Some(be.upload(&src.get(&format!("{key}.bias"), &[rows])?))
    } else {
        None
    };
    Ok(Linear {
        w: be.weight(&w, rows, cols),
        b,
        rows,
        cols,
        lora: None,
    })
}

/// A 1-D vector from the source.
pub fn vector<B: Backend>(be: &B, src: &dyn WeightSource, key: &str, n: usize) -> Result<B::Buf> {
    Ok(be.upload(&src.get(key, &[n])?))
}
