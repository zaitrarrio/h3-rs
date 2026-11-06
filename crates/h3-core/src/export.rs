//! Quantized checkpoints: the DiT with its block linears stored as int8 or
//! 4-bit codes (fastwan-rs's `quant/1` format, one safetensors file), and
//! merged-adapter exports.
//!
//! A quantized linear `X` (`X.weight` in Diffusers) becomes, by
//! [`WeightQuant::for_key`]:
//!
//! | tensor | int8 | 4-bit |
//! |---|---|---|
//! | `X.weight.q8` / `X.weight.q4` | I8 `[out, in]` | U8 `[out, in/2]`, element `2i` low nibble |
//! | `X.weight.scale` | F32 `[out, in/group]` | F16 `[out, in/group]` |
//! | `X.weight.zero` | — | F16 `[out, in/group]` |
//!
//! Every other tensor keeps its name and is stored as F16 when its values fit
//! (F32 if not; BF16 / F16 sources copied). Metadata `h3.format = "quant/1"`,
//! `h3.weights = "<format>"`. A stored file decides its own format: the loader
//! reads the codes as they are ([`crate::weights::WeightSource::stored`]).

use crate::{
    Error, Result,
    config::H3Config,
    quant::{Q4, Q8, Stored, WeightQuant},
    weights::{SafetensorsDir, WeightSource, h3_manifest},
};
use safetensors::Dtype;
use std::collections::HashMap;
use std::path::Path;

/// One tensor to write: name, dtype, shape, little-endian bytes.
pub type Tensor = (String, Dtype, Vec<usize>, Vec<u8>);

/// A hook that can rewrite a weight before it is stored (adapter merging).
pub type Patch<'a> = &'a dyn Fn(&str, &mut [f32], &[usize]) -> Result<()>;

/// The DiT's tensors with the block linears quantized as `quant` says, after
/// `patch` (if any) has edited each f32 weight. Gate linears are included
/// when the source has them.
pub fn quantize_dit(src: &SafetensorsDir, cfg: &H3Config, quant: WeightQuant, patch: Option<Patch<'_>>, skip_adaln: bool) -> Result<Vec<Tensor>> {
    let with_gate = src.has_linear("transformer_blocks.0.attn.to_gate_compress");
    let names = h3_manifest(cfg, with_gate);
    let mut out = Vec::new();
    for (name, shape) in names {
        // With precomputed AdaLN tables (`h3 adaln-tables`) the 13 B modulation
        // weights are never read at generation time, so a consumer download
        // can leave them out.
        if skip_adaln && name.contains(".adaln_proj.") {
            continue;
        }
        let key = name.strip_suffix(".weight");
        let fmt = key.map_or(WeightQuant::Native, |k| quant.for_key(k));
        let mut w = match (key, src.stored(key.unwrap_or(""))?) {
            (Some(_), Some(stored)) => stored.dequantize(),
            _ => src.get(&name, &shape)?,
        };
        if let Some(p) = patch {
            p(&name, &mut w, &shape)?;
        }
        if fmt == WeightQuant::Native {
            out.push(plain(&name, &shape, &w, src.raw(&name).ok().map(|(dt, _, _)| dt))?);
            continue;
        }
        let (rows, cols) = (shape[0], shape[1..].iter().product::<usize>());
        match fmt {
            WeightQuant::Int8 => {
                let q = Q8::quantize(&w, rows, cols);
                let ng = cols / q.group;
                out.push((format!("{name}.q8"), Dtype::I8, vec![rows, cols], q.q.iter().map(|&v| v as u8).collect()));
                out.push((format!("{name}.scale"), Dtype::F32, vec![rows, ng], f32_bytes(&q.scales)));
            }
            WeightQuant::Int4 { group, hqq, .. } => {
                if cols % 2 != 0 {
                    return Err(Error::Weights(format!("`{name}`: 4-bit needs an even row length, got {cols}")));
                }
                let q = Q4::quantize(&w, rows, cols, group, hqq);
                let ng = cols / q.group;
                out.push((format!("{name}.q4"), Dtype::U8, vec![rows, cols / 2], q.packed()));
                out.push((format!("{name}.scale"), Dtype::F16, vec![rows, ng], f16_bytes(&q.scales)));
                out.push((format!("{name}.zero"), Dtype::F16, vec![rows, ng], f16_bytes(&q.zeros)));
            }
            WeightQuant::Nvfp4 => {
                return Err(Error::Weights("NVFP4 is a loaded format, not an export target".into()));
            }
            WeightQuant::Native => unreachable!(),
        }
    }
    Ok(out)
}

/// A tensor kept as is: F16 when the values fit, else F32; a BF16 source
/// stays BF16 when unpatched values are written (lossless).
fn plain(name: &str, shape: &[usize], w: &[f32], src_dtype: Option<Dtype>) -> Result<Tensor> {
    let fits = w.iter().all(|x| x.is_finite() && x.abs() <= 65504.0);
    Ok(match src_dtype {
        Some(Dtype::BF16) => (name.into(), Dtype::BF16, shape.to_vec(), bf16_bytes(w)),
        _ if fits => (name.into(), Dtype::F16, shape.to_vec(), f16_bytes(w)),
        _ => (name.into(), Dtype::F32, shape.to_vec(), f32_bytes(w)),
    })
}

pub fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn f16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|&x| half::f16::from_f32(x).to_le_bytes()).collect()
}

pub fn bf16_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|&x| half::bf16::from_f32(x).to_le_bytes()).collect()
}

/// Write `tensors` as one safetensors file with `metadata`.
pub fn write(path: &Path, tensors: &[Tensor], metadata: HashMap<String, String>) -> Result<()> {
    let views = tensors
        .iter()
        .map(|(n, dt, shape, data)| {
            safetensors::tensor::TensorView::new(*dt, shape.clone(), data)
                .map(|v| (n.clone(), v))
                .map_err(|e| Error::Weights(format!("`{n}`: {e}")))
        })
        .collect::<Result<Vec<_>>>()?;
    safetensors::serialize_to_file(views, Some(metadata), path).map_err(|e| Error::Weights(e.to_string()))
}

/// The metadata a quantized checkpoint carries.
pub fn metadata(quant: WeightQuant, variant: &str) -> HashMap<String, String> {
    HashMap::from([
        ("h3.format".to_string(), "quant/1".to_string()),
        ("h3.weights".to_string(), quant.to_string()),
        ("h3.variant".to_string(), variant.to_string()),
    ])
}

/// Re-read a stored linear to confirm it matches `quant` for `key`.
pub fn stored_matches(stored: &Option<Stored>, fmt: WeightQuant) -> bool {
    matches!(
        (stored, fmt),
        (None, WeightQuant::Native) | (Some(Stored::Q8(_)), WeightQuant::Int8) | (Some(Stored::Q4(_)), WeightQuant::Int4 { .. })
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::{Synthetic, missing};

    /// Synthetic tiny DiT → f32 safetensors → quantized file → `stored()`
    /// gives exactly the codes `--weights` would make from the source.
    #[test]
    fn stored_codes_equal_load_time_quantization() {
        let cfg = H3Config::tiny_vsa();
        let syn = Synthetic { seed: 5 };
        let names = h3_manifest(&cfg, true);
        let src_t: Vec<Tensor> =
            names.iter().map(|(n, s)| (n.clone(), Dtype::F32, s.clone(), f32_bytes(&syn.get(n, s).unwrap()))).collect();
        let dir = std::env::temp_dir().join(format!("h3-export-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f32_path = dir.join("src.safetensors");
        write(&f32_path, &src_t, HashMap::new()).unwrap();
        let src = SafetensorsDir::open(&f32_path).unwrap();
        for quant in [
            WeightQuant::Int8,
            WeightQuant::Int4 { group: 16, hqq: true, ffn_only: true },
            WeightQuant::Int4 { group: 32, hqq: false, ffn_only: false },
        ] {
            let path = dir.join(format!("{quant}.safetensors"));
            write(&path, &quantize_dit(&src, &cfg, quant, None, false).unwrap(), metadata(quant, "test")).unwrap();
            let q = SafetensorsDir::open(&path).unwrap();
            assert!(missing(&q, &h3_manifest(&cfg, true)).is_empty());
            assert_eq!(q.metadata("h3.weights").as_deref(), Some(quant.to_string().as_str()));
            for (n, s) in &names {
                let Some(key) = n.strip_suffix(".weight") else { continue };
                let (rows, cols) = (s[0], s[1..].iter().product::<usize>());
                let w = src.get(n, s).unwrap();
                let st = q.stored(key).unwrap();
                assert!(stored_matches(&st, quant.for_key(key)), "{n}: {:?}", st.as_ref().map(|s| s.shape()));
                match st {
                    None => {}
                    Some(Stored::Q8(got)) => {
                        let want = Q8::quantize(&w, rows, cols);
                        assert_eq!((got.q, got.scales, got.group), (want.q, want.scales, want.group), "{n}");
                    }
                    Some(Stored::Q4(got)) => {
                        let WeightQuant::Int4 { group, hqq, .. } = quant.for_key(key) else { unreachable!() };
                        let want = Q4::quantize(&w, rows, cols, group, hqq);
                        assert_eq!((got.q, got.scales, got.zeros, got.group), (want.q, want.scales, want.zeros, want.group), "{n}");
                    }
                    Some(Stored::Nvfp4(_)) => unreachable!(),
                }
            }
        }
        // A patch edits weights before they are stored.
        let patched = quantize_dit(&src, &cfg, WeightQuant::Native, Some(&|n, w, _| {
            if n == "proj_in.bias" {
                w.iter_mut().for_each(|x| *x = 7.0);
            }
            Ok(())
        }), true)
        .unwrap();
        assert!(!patched.iter().any(|t| t.0.contains("adaln_proj")));
        let (_, dt, _, bytes) = patched.iter().find(|t| t.0 == "proj_in.bias").unwrap();
        assert_eq!(*dt, Dtype::F16);
        assert_eq!(half::f16::from_le_bytes([bytes[0], bytes[1]]).to_f32(), 7.0);
        std::fs::remove_dir_all(&dir).ok();
    }
}
