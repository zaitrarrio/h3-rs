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
}

/// A linear's weights before they reach a device: the host side of a
/// streamed block, or the input of [`upload`].
pub enum HostWeight {
    F32(Vec<f32>),
    Q8(Q8),
    Q4(Q4),
    Nvfp4(Nvfp4),
    Mxfp8(Mxfp8),
}

impl HostWeight {
    pub fn bytes(&self) -> usize {
        match self {
            HostWeight::F32(v) => v.len() * 4,
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
}

impl HostWeight {
    /// Page-lock the buffers a streamed upload reads (see [`Backend::pin_host`]).
    pub fn pin<B: Backend>(&self, be: &B) {
        match self {
            // SAFETY (both casts): `f32` / `i8` reinterpreted as the bytes of the same allocation.
            HostWeight::F32(v) => be.pin_host(unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) }),
            HostWeight::Q8(q) => be.pin_host(unsafe { std::slice::from_raw_parts(q.q.as_ptr() as *const u8, q.q.len()) }),
            HostWeight::Q4(_) => {}
            HostWeight::Nvfp4(q) => {
                be.pin_host(&q.packed);
                be.pin_host(&q.scales);
            }
            HostWeight::Mxfp8(q) => {
                be.pin_host(&q.codes);
                be.pin_host(&q.scales);
            }
        }
    }
}

impl HostLinear {
    pub fn pin<B: Backend>(&self, be: &B) {
        self.w.pin(be);
    }

    pub fn upload<B: Backend>(&self, be: &B) -> Linear<B> {
        let w = match &self.w {
            HostWeight::F32(v) => be.weight(v, self.rows, self.cols),
            HostWeight::Q8(q) => be.weight_q8(q),
            HostWeight::Q4(q) => be.weight_q4(q),
            HostWeight::Nvfp4(q) => be.weight_nvfp4(q),
            HostWeight::Mxfp8(q) => be.weight_mxfp8(q),
        };
        Linear { w, b: self.b.as_ref().map(|b| be.upload(b)), rows: self.rows, cols: self.cols }
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
) -> Result<HostLinear> {
    let err = |m: String| Error::Weights(m);
    let touched = adapters.iter().any(|a| keys.iter().any(|k| a.touches(k)));
    let (mut w, mut b, mut stored, mut rows, mut cols) = (Vec::new(), Vec::new(), Vec::<Stored>::new(), 0usize, 0usize);
    for k in keys {
        let shape = shapes.get(&format!("{k}.weight")).ok_or_else(|| err(format!("`{k}` is not in the manifest")))?;
        let (o, i) = (shape[0], shape[1..].iter().product::<usize>());
        match src.stored(k)? {
            Some(s) if !touched => {
                if s.shape() != (o, i) {
                    return Err(err(format!("`{k}` is stored {:?}, expected {o}x{i}", s.shape())));
                }
                stored.push(s);
            }
            Some(s) => w.extend(s.dequantize()),
            // A linear the base lacks but an adapter sets outright (the V1
            // LoRA's `to_gate_compress.set_weight` over MiniMax-H3) starts
            // from zeros.
            None if touched && !src.contains(&format!("{k}.weight")) => w.resize(w.len() + o * i, 0.0),
            None => w.extend(src.get(&format!("{k}.weight"), shape)?),
        }
        if touched {
            let start = w.len() - o * i;
            for a in adapters {
                a.apply_weight(k, &mut w[start..], o, i)?;
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
            WeightQuant::Int4 { group, hqq, .. } => HostWeight::Q4(Q4::quantize(&w, rows, cols, group, hqq)),
            // NVFP4 is only ever read from a checkpoint; a dequantized merge is
            // re-quantized to int8 so the device cost stays small.
            WeightQuant::Nvfp4 => HostWeight::Q8(Q8::quantize(&w, rows, cols)),
            WeightQuant::Mxfp8 if cols.is_multiple_of(32) => HostWeight::Mxfp8(Mxfp8::quantize(&w, rows, cols)),
            WeightQuant::Mxfp8 => HostWeight::F32(w),
        }
    } else if stored.len() != keys.len() {
        return Err(err(format!("{keys:?}: only some are stored quantized")));
    } else if stored.iter().all(|s| matches!(s, Stored::Q8(_))) {
        HostWeight::Q8(Q8::concat(stored.into_iter().map(|s| if let Stored::Q8(q) = s { q } else { unreachable!() }).collect()))
    } else if stored.iter().all(|s| matches!(s, Stored::Mxfp8(_))) {
        HostWeight::Mxfp8(Mxfp8::concat(stored.into_iter().map(|s| if let Stored::Mxfp8(q) = s { q } else { unreachable!() }).collect()))
    } else if stored.iter().all(|s| matches!(s, Stored::Q4(_))) {
        HostWeight::Q4(Q4::concat(stored.into_iter().map(|s| if let Stored::Q4(q) = s { q } else { unreachable!() }).collect()))
    } else if stored.iter().all(|s| matches!(s, Stored::Nvfp4(_))) {
        let parts: Vec<Nvfp4> = stored.into_iter().map(|s| if let Stored::Nvfp4(q) = s { q } else { unreachable!() }).collect();
        match Nvfp4::concat(parts.clone()) {
            Some(q) => HostWeight::Nvfp4(q),
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
    Ok(HostLinear { w, b: bias.then_some(b), rows, cols })
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
    Ok(load_host(src, shapes, keys, bias, quant, adapters)?.upload(be))
}

/// A plain f32 `[rows, cols]` matrix (+ bias) from the source, as a device linear.
pub fn load_plain<B: Backend>(be: &B, src: &dyn WeightSource, key: &str, rows: usize, cols: usize, bias: bool) -> Result<Linear<B>> {
    let w = src.get(&format!("{key}.weight"), &[rows, cols])?;
    let b = if bias { Some(be.upload(&src.get(&format!("{key}.bias"), &[rows])?)) } else { None };
    Ok(Linear { w: be.weight(&w, rows, cols), b, rows, cols })
}

/// A 1-D vector from the source.
pub fn vector<B: Backend>(be: &B, src: &dyn WeightSource, key: &str, n: usize) -> Result<B::Buf> {
    Ok(be.upload(&src.get(key, &[n])?))
}
