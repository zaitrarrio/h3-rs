//! Adapter files merged into the base weights at load time.
//!
//! Every released H3 variant that is not a full checkpoint ships one of:
//!
//! * PEFT LoRA: `<module>.lora_A[.default].weight [r, in]`,
//!   `lora_B[.default].weight [out, r]`, `W += (alpha / r) · B·A`
//!   (Pika H3 Flex rank 128 / alpha 128; Open-H3-Regenerate-2K rank 576,
//!   keys prefixed `transformer.`).
//! * FastH3 V1 (`vsa-datafree/adapter_model.safetensors`): the same LoRA
//!   pairs (rank 64, scale 1) plus dense deltas `<module>.diff`
//!   (`W += diff`), `<module>.diff_b` (`b += diff_b`) and whole replacements
//!   `<module>.set_weight` (`W = set_weight`, the VSA gate the base lacks).
//!
//! Merging is exact in f32 and happens before the weight is quantized or
//! uploaded, so a merged adapter costs nothing per step.

use crate::{
    Error, Result,
    weights::{SafetensorsDir, WeightSource},
};

/// One adapter file and how its keys map onto the base model's.
pub struct Adapter {
    pub src: SafetensorsDir,
    /// `alpha / rank` for LoRA pairs; 1 when the file carries no alpha.
    pub scale: f32,
    /// Prefix the adapter adds to base keys (`"transformer."` for Regen-2K).
    pub prefix: String,
    /// The infix between the module and `lora_A` (`""` or `".default"`).
    infix: Option<String>,
}

impl Adapter {
    /// Open an adapter; `alpha` and `rank` come from its config when known
    /// (`None` → scale 1, the convention of every H3 release so far).
    pub fn open(path: &std::path::Path, alpha: Option<f32>, rank: Option<usize>) -> Result<Self> {
        let src = SafetensorsDir::open(path)?;
        let scale = match (alpha, rank) {
            (Some(a), Some(r)) if r > 0 => a / r as f32,
            _ => 1.0,
        };
        let mut prefix = String::new();
        let mut infix = None;
        for n in src.names() {
            if let Some(p) = n.find("lora_A") {
                let module = &n[..p];
                if module.starts_with("transformer.") {
                    prefix = "transformer.".into();
                }
                let rest = &n[p + "lora_A".len()..];
                infix = Some(if rest.starts_with(".default.") { ".default".to_string() } else { String::new() });
                break;
            }
            if n.ends_with(".diff") || n.ends_with(".set_weight") || n.ends_with(".diff_b") {
                if n.starts_with("transformer.") {
                    prefix = "transformer.".into();
                }
            }
        }
        Ok(Self { src, scale, prefix, infix })
    }

    fn key(&self, base: &str) -> String {
        format!("{}{base}", self.prefix)
    }

    /// Whether the adapter touches the linear `module` (no `.weight`) at all.
    pub fn touches(&self, module: &str) -> bool {
        let k = self.key(module);
        self.src.contains_raw(&format!("{k}.set_weight"))
            || self.src.contains_raw(&format!("{k}.diff"))
            || self.lora_names(module).is_some()
    }

    fn lora_names(&self, module: &str) -> Option<(String, String)> {
        let infix = self.infix.as_deref()?;
        let k = self.key(module);
        let a = format!("{k}.lora_A{infix}.weight");
        let b = format!("{k}.lora_B{infix}.weight");
        (self.src.contains_raw(&a) && self.src.contains_raw(&b)).then_some((a, b))
    }

    /// Apply every delta the adapter holds for `module` to `w` (`[out, in]`
    /// row-major). Returns how many tensors were used.
    pub fn apply_weight(&self, module: &str, w: &mut [f32], out: usize, inp: usize) -> Result<usize> {
        let mut used = 0;
        let k = self.key(module);
        let set = format!("{k}.set_weight");
        if self.src.contains_raw(&set) {
            let v = self.src.get(&set, &[out, inp])?;
            w.copy_from_slice(&v);
            used += 1;
        }
        let diff = format!("{k}.diff");
        if self.src.contains_raw(&diff) {
            let v = self.src.get(&diff, &[out, inp])?;
            for (a, b) in w.iter_mut().zip(v) {
                *a += b;
            }
            used += 1;
        }
        if let Some((an, bn)) = self.lora_names(module) {
            let (_, ashape, _) = self.src.raw(&an)?;
            if ashape.len() != 2 || ashape[1] != inp {
                return Err(Error::Weights(format!("`{an}` is {ashape:?}, expected [r, {inp}]")));
            }
            let r = ashape[0];
            let a = self.src.get(&an, &[r, inp])?;
            let b = self.src.get(&bn, &[out, r])?;
            merge_lora(w, &a, &b, out, inp, r, self.scale);
            used += 1;
        }
        Ok(used)
    }

    /// Apply a bias delta (`diff_b`) if present.
    pub fn apply_bias(&self, module: &str, b: &mut [f32]) -> Result<bool> {
        let n = format!("{}.diff_b", self.key(module));
        if !self.src.contains_raw(&n) {
            return Ok(false);
        }
        let v = self.src.get(&n, &[b.len()])?;
        for (x, d) in b.iter_mut().zip(v) {
            *x += d;
        }
        Ok(true)
    }

    /// Apply a 1-D delta to a norm weight (`.diff` on `norm1`, `norm2`,
    /// `norm_out.norm`).
    pub fn apply_vector(&self, module: &str, w: &mut [f32]) -> Result<bool> {
        let n = format!("{}.diff", self.key(module));
        if !self.src.contains_raw(&n) {
            return Ok(false);
        }
        let v = self.src.get(&n, &[w.len()])?;
        for (x, d) in w.iter_mut().zip(v) {
            *x += d;
        }
        Ok(true)
    }

    /// Modules (base keys, no `.weight`) the adapter carries LoRA pairs or
    /// dense deltas for.
    pub fn modules(&self) -> Vec<String> {
        let mut v: Vec<String> = self
            .src
            .names()
            .filter_map(|n| {
                let n = n.strip_prefix(self.prefix.as_str()).unwrap_or(n);
                n.find(".lora_A")
                    .or_else(|| n.find(".set_weight"))
                    .or_else(|| n.find(".diff"))
                    .map(|p| n[..p].to_string())
            })
            .collect();
        v.sort();
        v.dedup();
        v
    }
}

/// `w += scale · B·A` with `a` `[r, inp]`, `b` `[out, r]`, blocked over rows
/// across threads (a 28672×5376 rank-576 merge is 89 GFLOP).
pub fn merge_lora(w: &mut [f32], a: &[f32], b: &[f32], out: usize, inp: usize, r: usize, scale: f32) {
    assert_eq!(w.len(), out * inp);
    assert_eq!(a.len(), r * inp);
    assert_eq!(b.len(), out * r);
    let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
    let per = out.div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        for (chunk_i, wc) in w.chunks_mut(per * inp).enumerate() {
            sc.spawn(move || {
                let row0 = chunk_i * per;
                for (i, wr) in wc.chunks_mut(inp).enumerate() {
                    let brow = &b[(row0 + i) * r..(row0 + i + 1) * r];
                    for (k, &bk) in brow.iter().enumerate() {
                        if bk == 0.0 {
                            continue;
                        }
                        let s = bk * scale;
                        let ar = &a[k * inp..(k + 1) * inp];
                        for (x, &av) in wr.iter_mut().zip(ar) {
                            *x += s * av;
                        }
                    }
                }
            });
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::export::{Tensor, f32_bytes, write};
    use safetensors::Dtype;
    use std::collections::HashMap;

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("h3-lora-{}-{name}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn merge_matches_naive() {
        let (out, inp, r) = (5, 7, 3);
        let a: Vec<f32> = (0..r * inp).map(|i| i as f32 * 0.1).collect();
        let b: Vec<f32> = (0..out * r).map(|i| 1.0 - i as f32 * 0.05).collect();
        let mut w = vec![0.5f32; out * inp];
        merge_lora(&mut w, &a, &b, out, inp, r, 2.0);
        for i in 0..out {
            for j in 0..inp {
                let mut s = 0.5;
                for k in 0..r {
                    s += 2.0 * b[i * r + k] * a[k * inp + j];
                }
                assert!((w[i * inp + j] - s).abs() < 1e-5);
            }
        }
    }

    #[test]
    fn adapter_formats() {
        let dir = tmp("fmt");
        // Pika-style `.default` LoRA with a prefix-free key, plus a V1-style
        // set_weight / diff / diff_b.
        let t: Vec<Tensor> = vec![
            ("transformer_blocks.0.attn.to_q.lora_A.default.weight".into(), Dtype::F32, vec![2, 4], f32_bytes(&[1., 0., 0., 0., 0., 1., 0., 0.])),
            ("transformer_blocks.0.attn.to_q.lora_B.default.weight".into(), Dtype::F32, vec![3, 2], f32_bytes(&[1., 0., 0., 1., 1., 1.])),
            ("proj_in.diff".into(), Dtype::F32, vec![2, 2], f32_bytes(&[1., 1., 1., 1.])),
            ("proj_in.diff_b".into(), Dtype::F32, vec![2], f32_bytes(&[0.5, -0.5])),
            ("transformer_blocks.0.attn.to_gate_compress.set_weight".into(), Dtype::F32, vec![2, 2], f32_bytes(&[9., 8., 7., 6.])),
        ];
        write(&dir.join("adapter.safetensors"), &t, HashMap::new()).unwrap();
        let ad = Adapter::open(&dir.join("adapter.safetensors"), Some(128.0), Some(128)).unwrap();
        assert_eq!(ad.scale, 1.0);
        assert!(ad.touches("transformer_blocks.0.attn.to_q"));
        assert!(!ad.touches("transformer_blocks.0.attn.to_k"));
        let mut w = vec![0.0f32; 3 * 4];
        assert_eq!(ad.apply_weight("transformer_blocks.0.attn.to_q", &mut w, 3, 4).unwrap(), 1);
        // B·A: rows [1,0,0,0], [0,1,0,0], [1,1,0,0]
        assert_eq!(w, vec![1., 0., 0., 0., 0., 1., 0., 0., 1., 1., 0., 0.]);
        let mut p = vec![1.0f32; 4];
        ad.apply_weight("proj_in", &mut p, 2, 2).unwrap();
        assert_eq!(p, vec![2.0; 4]);
        let mut b = vec![0.0f32; 2];
        assert!(ad.apply_bias("proj_in", &mut b).unwrap());
        assert_eq!(b, vec![0.5, -0.5]);
        let mut g = vec![0.0f32; 4];
        ad.apply_weight("transformer_blocks.0.attn.to_gate_compress", &mut g, 2, 2).unwrap();
        assert_eq!(g, vec![9., 8., 7., 6.]);
        let mods = ad.modules();
        assert!(mods.contains(&"proj_in".to_string()) && mods.contains(&"transformer_blocks.0.attn.to_q".to_string()));

        // Regen-style `transformer.` prefix, plain lora_A/B.
        let t2: Vec<Tensor> = vec![
            ("transformer.transformer_blocks.1.ff.net.2.lora_A.weight".into(), Dtype::F32, vec![1, 2], f32_bytes(&[1., 2.])),
            ("transformer.transformer_blocks.1.ff.net.2.lora_B.weight".into(), Dtype::F32, vec![2, 1], f32_bytes(&[1., -1.])),
        ];
        write(&dir.join("regen.safetensors"), &t2, HashMap::new()).unwrap();
        let ad2 = Adapter::open(&dir.join("regen.safetensors"), None, None).unwrap();
        assert_eq!(ad2.prefix, "transformer.");
        let mut w2 = vec![0.0f32; 4];
        ad2.apply_weight("transformer_blocks.1.ff.net.2", &mut w2, 2, 2).unwrap();
        assert_eq!(w2, vec![1., 2., -1., -2.]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
