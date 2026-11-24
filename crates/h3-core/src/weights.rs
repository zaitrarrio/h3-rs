//! Checkpoint access: safetensors (mmap'd, any of F32/F16/BF16, any number
//! of shards across any number of directories) or deterministic synthetic
//! weights of the same names and shapes (benchmarks and conformance tests
//! without a download).
//!
//! Every model asks for every tensor by its exact Diffusers key and shape. A
//! missing key or a shape mismatch is an error, never a default.

use crate::{
    Error, Result,
    config::{
        AudioVaeConfig, H3Config, Qwen3VlConfig, TaeH3Config, UpscalerConfig, VideoVaeConfig,
    },
    quant::{Nvfp4, Q4, Q8, Stored},
    rng::Rng,
};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// ComfyUI's single-file H3 checkpoints carry `adaln_t_table [grid, rank]`
/// instead of a time embedder (see [`ComfySource`]).
pub const ADALN_TABLE_KEY: &str = "adaln_t_table";

pub trait WeightSource {
    /// The tensor as f32, checked against `shape` (size-1 dims are ignored,
    /// so a Conv3d `[o, i, 1, 1, 1]` may be asked for as `[o, i]`).
    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>>;
    fn contains(&self, name: &str) -> bool;
    /// The stored shape of a tensor whose shape the model does not know in
    /// advance (`adaln_t_table`).
    fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Err(Error::Weights(format!(
            "`{name}`: this source cannot report shapes"
        )))
    }
    /// The linear `key` (no `.weight`) if the checkpoint stores it quantized
    /// (`<key>.weight.q8` / `.q4` + `.scale` [+ `.zero`], see [`crate::export`],
    /// or the Trim NVFP4 triple `<nvfp4 module>::_nvfp4_weight` / `_scale` /
    /// `_alpha`); `None` for a plain tensor.
    fn stored(&self, _key: &str) -> Result<Option<Stored>> {
        Ok(None)
    }
    /// Whether the linear `key` (no `.weight`) is present, plain or quantized.
    fn has_linear(&self, key: &str) -> bool {
        self.contains(&format!("{key}.weight"))
            || self.contains(&format!("{key}.weight.q8"))
            || self.contains(&format!("{key}.weight.q4"))
            || self.contains(&format!("{key}.weight.mx8"))
            || self.contains(&format!("{}::_nvfp4_weight", nvfp4_module(key)))
    }
    /// A convolution weight that may be stored weight-normalised
    /// (`<key>.weight_g [cout, 1, 1]`, `<key>.weight_v [cout, …]`):
    /// `w = g · v / ‖v‖` with the norm over every dim but the first.
    fn conv_weight(&self, key: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let plain = format!("{key}.weight");
        if self.contains(&plain) {
            return self.get(&plain, shape);
        }
        let g = self.get(&format!("{key}.weight_g"), &[shape[0]])?;
        let v = self.get(&format!("{key}.weight_v"), shape)?;
        Ok(fold_weight_norm(&g, &v, shape[0]))
    }
    fn has_conv_weight(&self, key: &str) -> bool {
        self.contains(&format!("{key}.weight"))
            || (self.contains(&format!("{key}.weight_g"))
                && self.contains(&format!("{key}.weight_v")))
    }
}

/// `w[o, :] = g[o] · v[o, :] / ‖v[o, :]‖₂`.
pub fn fold_weight_norm(g: &[f32], v: &[f32], cout: usize) -> Vec<f32> {
    let per = v.len() / cout;
    let mut w = Vec::with_capacity(v.len());
    for o in 0..cout {
        let row = &v[o * per..(o + 1) * per];
        let n = row
            .iter()
            .map(|x| (*x as f64) * (*x as f64))
            .sum::<f64>()
            .sqrt() as f32;
        let s = if n > 0.0 { g[o] / n } else { 0.0 };
        w.extend(row.iter().map(|x| x * s));
    }
    w
}

/// The Trim NVFP4 file names modules without Diffusers' `.net.N` / `.0`
/// infixes: `ff.net.0.proj → ff.fc_in`, `ff.net.2 → ff.fc_out`,
/// `attn.to_out.0 → attn.to_out`.
pub fn nvfp4_module(key: &str) -> String {
    key.replace(".ff.net.0.proj", ".ff.fc_in")
        .replace(".ff.net.2", ".ff.fc_out")
        .replace(".attn.to_out.0", ".attn.to_out")
}

fn squeeze(s: &[usize]) -> Vec<usize> {
    s.iter().copied().filter(|&d| d != 1).collect()
}

/// All `*.safetensors` under one or more directories (or files). Later
/// entries shadow earlier ones, so a variant directory can overlay the base.
pub struct SafetensorsDir {
    maps: Vec<memmap2::Mmap>,
    files: Vec<PathBuf>,
    index: HashMap<String, usize>,
}

impl SafetensorsDir {
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_many(&[path.to_path_buf()])
    }

    pub fn open_many(paths: &[PathBuf]) -> Result<Self> {
        let mut files: Vec<PathBuf> = Vec::new();
        for path in paths {
            if path.is_dir() {
                let mut v: Vec<PathBuf> = std::fs::read_dir(path)?
                    .filter_map(|e| e.ok().map(|e| e.path()))
                    .filter(|p| p.extension().is_some_and(|e| e == "safetensors"))
                    .collect();
                v.sort();
                files.extend(v);
            } else if path.is_file() {
                files.push(path.clone());
            } else {
                return Err(Error::Weights(format!("{} does not exist", path.display())));
            }
        }
        if files.is_empty() {
            return Err(Error::Weights(format!("no .safetensors in {:?}", paths)));
        }
        let mut maps = Vec::new();
        let mut index = HashMap::new();
        // `H3_MADVISE=1`: MADV_SEQUENTIAL on each map and MADV_WILLNEED per tensor.
        // Measured slower from RunPod's FUSE network volume (load 147-167 s vs
        // 118 s without; `docs/runs/2026-11-13-runpod-pro6000-stage`), so it is
        // opt-in; staging the files onto local disk is what pays there.
        let advise = std::env::var_os("H3_MADVISE").is_some();
        for (i, f) in files.iter().enumerate() {
            let file = std::fs::File::open(f)?;
            let map = map_readonly(&file)?;
            if advise {
                let _ = map.advise(memmap2::Advice::Sequential);
            }
            let st = safetensors::SafeTensors::deserialize(&map)
                .map_err(|e| Error::Weights(format!("{}: {e}", f.display())))?;
            for name in st.names() {
                index.insert(name.to_string(), i);
            }
            maps.push(map);
        }
        Ok(Self { maps, files, index })
    }

    /// The stored bytes of a tensor, without conversion: `(dtype, shape, data)`.
    pub fn raw(&self, name: &str) -> Result<(safetensors::Dtype, Vec<usize>, &[u8])> {
        let i = *self
            .index
            .get(name)
            .ok_or_else(|| Error::Weights(format!("missing tensor `{name}`")))?;
        let st = safetensors::SafeTensors::deserialize(&self.maps[i])
            .map_err(|e| Error::Weights(e.to_string()))?;
        let t = st.tensor(name).map_err(|e| Error::Weights(e.to_string()))?;
        let data = t.data();
        if std::env::var_os("H3_MADVISE").is_some() {
            let off = data.as_ptr() as usize - self.maps[i].as_ptr() as usize;
            let _ = self.maps[i].advise_range(memmap2::Advice::WillNeed, off, data.len());
        }
        Ok((t.dtype(), t.shape().to_vec(), data))
    }

    /// A tensor's bytes for conversion, borrowed from the map; with
    /// `H3_PREAD=1`, 1 MiB and up through [`Self::read_into`] (parallel
    /// `pread`s) instead. That did not pay from RunPod's FUSE volume: a cold
    /// 25.6 GB load took 107.9 s through pread and 89.1 s through the map, on
    /// two pods (`docs/runs/2026-11-24-runpod-parity`); staging the files is
    /// what pays there.
    pub fn bytes(
        &self,
        name: &str,
    ) -> Result<(safetensors::Dtype, Vec<usize>, std::borrow::Cow<'_, [u8]>)> {
        let (dt, shape, data) = self.raw(name)?;
        if data.len() < (1 << 20) || !std::env::var("H3_PREAD").is_ok_and(|v| v == "1") {
            return Ok((dt, shape, std::borrow::Cow::Borrowed(data)));
        }
        let mut buf = vec![0u8; data.len()];
        self.read_into(name, &mut buf)?;
        Ok((dt, shape, std::borrow::Cow::Owned(buf)))
    }

    pub fn contains_raw(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    /// Copy a tensor's stored bytes into `out` (exactly its length) with
    /// parallel `pread`s from the file, not through the memory map: one copy
    /// from the page cache, no page faults, several reads in flight on a
    /// network or cold disk (what fastvideo-rs's read-ahead pool does).
    pub fn read_into(&self, name: &str, out: &mut [u8]) -> Result<()> {
        use rayon::prelude::*;
        use std::os::unix::fs::FileExt;
        let i = *self
            .index
            .get(name)
            .ok_or_else(|| Error::Weights(format!("missing tensor `{name}`")))?;
        let (_, _, data) = self.raw(name)?;
        if data.len() != out.len() {
            return Err(Error::Weights(format!(
                "`{name}` holds {} bytes, the buffer {}",
                data.len(),
                out.len()
            )));
        }
        let off = (data.as_ptr() as usize - self.maps[i].as_ptr() as usize) as u64;
        let file = std::fs::File::open(&self.files[i])?;
        const CHUNK: usize = 8 << 20;
        out.par_chunks_mut(CHUNK)
            .enumerate()
            .try_for_each(|(k, c)| file.read_exact_at(c, off + (k * CHUNK) as u64))?;
        Ok(())
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.index.keys().map(String::as_str)
    }

    pub fn files(&self) -> &[PathBuf] {
        &self.files
    }

    /// Metadata of the first file that has the key.
    pub fn metadata(&self, key: &str) -> Option<String> {
        for m in &self.maps {
            if let Ok((_, meta)) = safetensors::SafeTensors::read_metadata(m) {
                if let Some(v) = meta.metadata().as_ref().and_then(|m| m.get(key)) {
                    return Some(v.clone());
                }
            }
        }
        None
    }

    /// A scalar (shape `[]` or `[1]`) as f32.
    pub fn scalar(&self, name: &str) -> Result<f32> {
        let (_, shape, _) = self.raw(name)?;
        let v = self.get(name, &shape)?;
        v.first()
            .copied()
            .ok_or_else(|| Error::Weights(format!("`{name}` is empty")))
    }
}

#[allow(unsafe_code)]
fn map_readonly(file: &std::fs::File) -> Result<memmap2::Mmap> {
    // SAFETY: weights are read-only inputs; concurrent truncation by another
    // process is outside this program's contract (as for every mmap loader).
    Ok(unsafe { memmap2::Mmap::map(file)? })
}

pub fn bytes_to_f32(dt: safetensors::Dtype, b: &[u8], name: &str) -> Result<Vec<f32>> {
    use safetensors::Dtype;
    Ok(match dt {
        Dtype::F32 => b
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::F16 => b
            .chunks_exact(2)
            .map(|c| half::f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::BF16 => b
            .chunks_exact(2)
            .map(|c| half::bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::F64 => b
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        Dtype::I8 => b.iter().map(|&x| x as i8 as f32).collect(),
        Dtype::U8 => b.iter().map(|&x| x as f32).collect(),
        Dtype::I32 => b
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        Dtype::I64 => b
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes(c.try_into().unwrap()) as f32)
            .collect(),
        other => {
            return Err(Error::Weights(format!(
                "`{name}`: unsupported dtype {other:?}"
            )));
        }
    })
}

impl WeightSource for SafetensorsDir {
    fn contains(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }

    fn shape(&self, name: &str) -> Result<Vec<usize>> {
        Ok(self.raw(name)?.1)
    }

    fn stored(&self, key: &str) -> Result<Option<Stored>> {
        use safetensors::Dtype;
        let f32s = |name: &str| -> Result<(Vec<usize>, Vec<f32>)> {
            let (_, shape, _) = self.raw(name)?;
            let v = self.get(name, &shape)?;
            Ok((shape, v))
        };
        let bad = |m: String| Error::Weights(format!("`{key}`: {m}"));
        let q8 = format!("{key}.weight.q8");
        let q4 = format!("{key}.weight.q4");
        let mx8 = format!("{key}.weight.mx8");
        let nv = format!("{}::_nvfp4_weight", nvfp4_module(key));
        if self.contains_raw(&mx8) {
            // MXFP8: E4M3 codes [rows, cols] + the swizzled E8M0 scales as stored (crate::export).
            let (dt, shape, _) = self.raw(&mx8)?;
            if dt != Dtype::U8 || shape.len() != 2 {
                return Err(bad(format!("mx8 is {dt:?} {shape:?}")));
            }
            let (sdt, sshape, sdata) = self.raw(&format!("{key}.weight.mxs"))?;
            if sdt != Dtype::U8 || sdata.len() != crate::quant::mx_scale_len(shape[0], shape[1]) {
                return Err(bad(format!(
                    "mxs is {sdt:?} {sshape:?} for codes {shape:?}"
                )));
            }
            let data = self.bytes(&mx8)?.2.into_owned();
            return Ok(Some(Stored::Mxfp8(crate::quant::Mxfp8::from_stored(
                data,
                sdata.to_vec(),
                shape[0],
                shape[1],
            ))));
        }
        if self.contains_raw(&q8) {
            let (dt, shape, data) = self.raw(&q8)?;
            if dt != Dtype::I8 || shape.len() != 2 {
                return Err(bad(format!("q8 is {dt:?} {shape:?}")));
            }
            let (sshape, scales) = f32s(&format!("{key}.weight.scale"))?;
            if sshape.len() != 2 || sshape[0] != shape[0] || shape[1] % sshape[1] != 0 {
                return Err(bad(format!("scale {sshape:?} for codes {shape:?}")));
            }
            let mut codes = vec![0u8; data.len()];
            self.read_into(&q8, &mut codes)?;
            // SAFETY: `u8` and `i8` share size and alignment; the Vec keeps its allocation.
            let codes: Vec<i8> = unsafe {
                let mut c = std::mem::ManuallyDrop::new(codes);
                Vec::from_raw_parts(c.as_mut_ptr() as *mut i8, c.len(), c.capacity())
            };
            return Ok(Some(Stored::Q8(Q8 {
                q: codes,
                scales,
                rows: shape[0],
                cols: shape[1],
                group: shape[1] / sshape[1],
            })));
        }
        if self.contains_raw(&q4) {
            let (dt, shape, data) = self.bytes(&q4)?;
            if dt != Dtype::U8 || shape.len() != 2 {
                return Err(bad(format!("q4 is {dt:?} {shape:?}")));
            }
            let (rows, cols) = (shape[0], 2 * shape[1]);
            let (sshape, scales) = f32s(&format!("{key}.weight.scale"))?;
            let (zshape, zeros) = f32s(&format!("{key}.weight.zero"))?;
            if sshape != zshape || sshape.len() != 2 || sshape[0] != rows || cols % sshape[1] != 0 {
                return Err(bad(format!(
                    "scale {sshape:?} / zero {zshape:?} for codes {shape:?}"
                )));
            }
            return Ok(Some(Stored::Q4(Q4::from_packed(
                &data, scales, zeros, rows, cols,
            ))));
        }
        if self.contains_raw(&nv) {
            let m = nvfp4_module(key);
            let (dt, shape, data) = self.bytes(&nv)?;
            if dt != Dtype::U8 || shape.len() != 2 {
                return Err(bad(format!("nvfp4 codes are {dt:?} {shape:?}")));
            }
            let (rows, cols) = (shape[0], 2 * shape[1]);
            let (sdt, sshape, sdata) = self.raw(&format!("{m}::_nvfp4_weight_scale"))?;
            if sdt != Dtype::U8 || sshape.len() != 2 || sshape[0] != rows.div_ceil(128) * 128 {
                return Err(bad(format!(
                    "nvfp4 scales are {sdt:?} {sshape:?} for codes {shape:?}"
                )));
            }
            let alpha = self.scalar(&format!("{m}::_nvfp4_alpha"))?;
            return Ok(Some(Stored::Nvfp4(Nvfp4::from_stored(
                &data, sdata, alpha, rows, cols, false,
            ))));
        }
        Ok(None)
    }

    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let (dt, stored, data) = self.bytes(name)?;
        if squeeze(&stored) != squeeze(shape) {
            return Err(Error::Weights(format!(
                "`{name}` is {stored:?}, expected {shape:?}"
            )));
        }
        bytes_to_f32(dt, &data, name)
    }
}

/// A source whose keys are rewritten before lookup (`f(diffusers key) →
/// stored key`): the regen fast decoder stores the VAE decoder in its
/// original layout; a `transformer.`-prefixed file is a Diffusers one with a
/// prefix.
pub struct Remapped<S: WeightSource> {
    pub inner: S,
    pub map: Box<dyn Fn(&str) -> String + Send + Sync>,
}

impl<S: WeightSource> WeightSource for Remapped<S> {
    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        self.inner.get(&(self.map)(name), shape)
    }
    fn contains(&self, name: &str) -> bool {
        self.inner.contains(&(self.map)(name))
    }
    fn shape(&self, name: &str) -> Result<Vec<usize>> {
        self.inner.shape(&(self.map)(name))
    }
    fn stored(&self, key: &str) -> Result<Option<Stored>> {
        self.inner.stored(&(self.map)(key))
    }
}

/// Which part of a ComfyUI tensor a Diffusers key names.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComfyPart {
    Whole,
    /// Row block `i` of 3 (`qkv_proj` → `to_q` / `to_k` / `to_v`).
    Third(usize),
    /// The two row halves exchanged (`mlp.fc1` is `[gate | value]`,
    /// Diffusers' `ff.net.0.proj` is `[value | gate]`).
    SwapHalves,
}

/// Diffusers DiT key → ComfyUI single-file key (`comfy/ldm/minimax/model.py`
/// names) and the part of that tensor. `None` for keys the file never
/// carries (`time_embedder.*`, `adaln_basis.weight`: the file holds
/// `adaln_t_table` instead).
pub fn comfy_h3_key(name: &str) -> Option<(String, ComfyPart)> {
    use ComfyPart::*;
    let whole = |k: String| Some((k, Whole));
    if name.starts_with("time_embedder.") || name == "adaln_basis.weight" {
        return None;
    }
    let fixed = [
        ("proj_in.", "video_patch_proj."),
        ("audio_proj_in.", "audio_patch_proj."),
        ("context_embedder.", "condition_proj."),
        ("proj_out.", "final_layer.video_out."),
        ("audio_proj_out.", "final_layer.audio_out."),
        ("norm_out.linear.", "final_layer.adaln_proj.linear."),
        ("norm_out.norm.", "final_layer.norm."),
    ];
    for (d, c) in fixed {
        if let Some(rest) = name.strip_prefix(d) {
            return whole(format!("{c}{rest}"));
        }
    }
    let (prefix, rest) = if let Some(r) = name.strip_prefix("transformer_blocks.") {
        ("blocks.", r)
    } else if let Some(r) = name.strip_prefix("token_refiner.refiner_blocks.") {
        ("token_refiner.blocks.", r)
    } else {
        return whole(name.to_string());
    };
    let (n, field) = rest.split_once('.')?;
    let p = format!("{prefix}{n}.");
    Some(match field {
        "attn.to_q.weight" => (format!("{p}attn.qkv_proj.weight"), Third(0)),
        "attn.to_k.weight" => (format!("{p}attn.qkv_proj.weight"), Third(1)),
        "attn.to_v.weight" => (format!("{p}attn.qkv_proj.weight"), Third(2)),
        "attn.to_out.0.weight" => (format!("{p}attn.out_proj.weight"), Whole),
        "attn.norm_q.weight" => (format!("{p}attn.q_norm.weight"), Whole),
        "attn.norm_k.weight" => (format!("{p}attn.k_norm.weight"), Whole),
        "ff.net.0.proj.weight" => (format!("{p}mlp.fc1.weight"), SwapHalves),
        "ff.net.2.weight" => (format!("{p}mlp.fc2.weight"), Whole),
        other => (format!("{p}{other}"), Whole),
    })
}

/// A ComfyUI single-file H3 checkpoint (`video_patch_proj`, fused
/// `qkv_proj`, `mlp.fc1/fc2`, optional `comfy_quant` NVFP4 triples `weight`
/// U8 `[out, in/2]` first-code-high + `weight_scale` E4M3 swizzled 128×4 +
/// `weight_scale_2` f32, `adaln_t_table`) read through Diffusers keys.
pub struct ComfySource {
    pub inner: SafetensorsDir,
}

impl ComfySource {
    /// Whether a checkpoint is in this layout.
    pub fn detect(src: &SafetensorsDir) -> bool {
        src.contains("video_patch_proj.weight") && src.contains("blocks.0.attn.qkv_proj.weight")
    }

    fn is_nvfp4(&self, comfy_weight: &str) -> bool {
        let base = comfy_weight.strip_suffix(".weight").unwrap_or(comfy_weight);
        self.inner.contains_raw(&format!("{base}.weight_scale"))
            && self.inner.contains_raw(&format!("{base}.comfy_quant"))
    }

    /// The whole NVFP4 tensor behind `<base>.weight`.
    fn nvfp4(&self, comfy_weight: &str) -> Result<Nvfp4> {
        use safetensors::Dtype;
        let base = comfy_weight.strip_suffix(".weight").unwrap_or(comfy_weight);
        let bad = |m: String| Error::Weights(format!("`{comfy_weight}`: {m}"));
        let (dt, shape, data) = self.inner.raw(comfy_weight)?;
        if dt != Dtype::U8 || shape.len() != 2 {
            return Err(bad(format!("nvfp4 codes are {dt:?} {shape:?}")));
        }
        let (rows, cols) = (shape[0], 2 * shape[1]);
        let (sdt, sshape, sdata) = self.inner.raw(&format!("{base}.weight_scale"))?;
        if sdt != Dtype::F8_E4M3
            || sshape.len() != 2
            || sshape[0] != rows.div_ceil(128) * 128
            || sshape[1] != (cols / 16).div_ceil(4) * 4
        {
            return Err(bad(format!(
                "nvfp4 scales are {sdt:?} {sshape:?} for codes {shape:?}"
            )));
        }
        let fmt = self.inner.raw(&format!("{base}.comfy_quant"))?.2;
        if !std::str::from_utf8(fmt).unwrap_or("").contains("\"nvfp4\"") {
            return Err(bad(format!(
                "comfy_quant is {:?}, not nvfp4",
                String::from_utf8_lossy(fmt)
            )));
        }
        let alpha = self.inner.scalar(&format!("{base}.weight_scale_2"))?;
        Ok(Nvfp4::from_stored(data, sdata, alpha, rows, cols, true))
    }

    fn part_of(q: Nvfp4, part: ComfyPart) -> Nvfp4 {
        match part {
            ComfyPart::Whole => q,
            ComfyPart::Third(i) => {
                let n = q.rows / 3;
                q.rows(i * n, (i + 1) * n)
            }
            ComfyPart::SwapHalves => q.swap_halves(),
        }
    }

    fn part_rows(v: &[f32], rows: usize, part: ComfyPart) -> Vec<f32> {
        let per = v.len() / rows;
        match part {
            ComfyPart::Whole => v.to_vec(),
            ComfyPart::Third(i) => {
                let n = rows / 3;
                v[i * n * per..(i + 1) * n * per].to_vec()
            }
            ComfyPart::SwapHalves => {
                let h = rows / 2;
                let mut out = v[h * per..].to_vec();
                out.extend_from_slice(&v[..h * per]);
                out
            }
        }
    }
}

impl WeightSource for ComfySource {
    fn contains(&self, name: &str) -> bool {
        comfy_h3_key(name).is_some_and(|(k, _)| self.inner.contains(&k))
    }

    fn shape(&self, name: &str) -> Result<Vec<usize>> {
        let (k, part) = comfy_h3_key(name)
            .ok_or_else(|| Error::Weights(format!("`{name}` has no ComfyUI counterpart")))?;
        let mut shape = self.inner.shape(&k)?;
        if self.is_nvfp4(&k) && shape.len() == 2 {
            shape[1] *= 2;
        }
        if let ComfyPart::Third(_) = part {
            shape[0] /= 3;
        }
        Ok(shape)
    }

    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let (k, part) = comfy_h3_key(name)
            .ok_or_else(|| Error::Weights(format!("`{name}` has no ComfyUI counterpart")))?;
        let stored = self.shape(name)?;
        if squeeze(&stored) != squeeze(shape) {
            return Err(Error::Weights(format!(
                "`{name}` (`{k}`) is {stored:?}, expected {shape:?}"
            )));
        }
        if self.is_nvfp4(&k) {
            return Ok(Self::part_of(self.nvfp4(&k)?, part).dequantize());
        }
        let full = self.inner.shape(&k)?;
        let v = self.inner.get(&k, &full)?;
        Ok(Self::part_rows(&v, full[0], part))
    }

    fn stored(&self, key: &str) -> Result<Option<Stored>> {
        let Some((k, part)) = comfy_h3_key(&format!("{key}.weight")) else {
            return Ok(None);
        };
        if !self.is_nvfp4(&k) {
            return Ok(None);
        }
        Ok(Some(Stored::Nvfp4(Self::part_of(self.nvfp4(&k)?, part))))
    }

    fn has_linear(&self, key: &str) -> bool {
        self.contains(&format!("{key}.weight"))
    }
}

/// Diffusers VAE-decoder key → the original (`vae_source` / `vae_fast`)
/// layout: `transformer_blocks.N.attn.to_q` has no counterpart (the source
/// fuses `to_qkv`), so callers read `to_qkv` through [`WeightSource::get`]
/// with the fused shape; everything else maps one to one.
pub fn vae_source_key(name: &str) -> String {
    let n = name.strip_prefix("decoder.").unwrap_or(name);
    let n = n
        .replace("transformer_blocks.", "blocks.")
        .replace(".attn.to_out.0.", ".attn.to_out.")
        .replace(".ff.net.0.proj.", ".ff.w1.")
        .replace(".ff.net.2.", ".ff.w2.")
        .replace("proj_in.", "x_embedder.");
    if name.starts_with("decoder.") && n.starts_with("blocks.") {
        n
    } else {
        n
    }
}

/// Deterministic weights for every name a model asks for: `U(−a, a)` with
/// `a = 1/√fan_in` for matrices, `1 + U(−0.1, 0.1)` for norm weights, small
/// values for biases and modulation tables. Same name → same values on
/// every backend.
pub struct Synthetic {
    pub seed: u64,
}

impl WeightSource for Synthetic {
    /// Everything but the ComfyUI table: the synthetic model follows the
    /// Diffusers layout (time embedder + basis).
    fn contains(&self, name: &str) -> bool {
        name != ADALN_TABLE_KEY
    }

    fn get(&self, name: &str, shape: &[usize]) -> Result<Vec<f32>> {
        let n: usize = shape.iter().product();
        let mut h = self.seed;
        for b in name.bytes() {
            h = (h ^ b as u64).wrapping_mul(0x100_0000_01B3);
        }
        let mut rng = Rng::new(h);
        let is_norm = (name.contains("norm") && name.ends_with(".weight"))
            || name.ends_with("scale1")
            || name.ends_with("scale2");
        Ok(if is_norm && shape.len() == 1 {
            rng.uniform(n, 0.1).into_iter().map(|v| 1.0 + v).collect()
        } else if name.ends_with(".bias")
            || name.ends_with("alpha")
            || name.ends_with("beta")
            || name.contains("register_tokens")
        {
            rng.uniform(n, 0.1)
        } else if name.contains("adaln_proj") || name.contains("norm_out.linear") {
            // Modulation rows: keep `1 + scale` positive and gates small.
            rng.uniform(n, 0.02)
        } else if name.contains("filter") {
            // Sinc filters are normalised; a flat kernel stands in.
            vec![1.0 / n as f32; n]
        } else {
            let fan_in: usize = shape[1..].iter().product::<usize>().max(1);
            rng.uniform(n, 1.0 / (fan_in as f32).sqrt())
        })
    }
}

pub type Manifest = Vec<(String, Vec<usize>)>;

fn attn_ff(v: &mut Manifest, p: &str, d: usize, qkv: usize, head_dim: usize, ffn: usize) {
    for m in ["to_q", "to_k", "to_v"] {
        v.push((format!("{p}.attn.{m}.weight"), vec![qkv, d]));
    }
    v.push((format!("{p}.attn.to_out.0.weight"), vec![d, qkv]));
    v.push((format!("{p}.attn.norm_q.weight"), vec![head_dim]));
    v.push((format!("{p}.attn.norm_k.weight"), vec![head_dim]));
    v.push((format!("{p}.norm1.weight"), vec![d]));
    v.push((format!("{p}.norm2.weight"), vec![d]));
    v.push((format!("{p}.ff.net.0.proj.weight"), vec![2 * ffn, d]));
    v.push((format!("{p}.ff.net.2.weight"), vec![d, ffn]));
}

/// Every tensor the H3 DiT reads, with its Diffusers shape. `with_gate` adds
/// the VSA gate linears (FastH3 students carry them, the base does not).
pub fn h3_manifest(cfg: &H3Config, with_gate: bool) -> Manifest {
    let d = cfg.hidden;
    let qkv = cfg.dim();
    let adaln_in = cfg.adaln_rank.unwrap_or(cfg.time_embed_dim);
    let mods = H3Config::MOD_SLOTS * H3Config::MODALITIES * d;
    let heads = cfg.pdd_steps.unwrap_or(1);
    let mut v: Manifest = vec![
        ("proj_in.weight".into(), vec![d, cfg.video_patch_dim()]),
        ("proj_in.bias".into(), vec![d]),
        (
            "audio_proj_in.weight".into(),
            vec![d, cfg.audio_in_channels],
        ),
        ("audio_proj_in.bias".into(), vec![d]),
        ("context_embedder.weight".into(), vec![d, cfg.text_dim]),
        ("context_embedder.bias".into(), vec![d]),
        (
            "time_embedder.linear_1.weight".into(),
            vec![cfg.time_embed_hidden, cfg.freq_dim],
        ),
        (
            "time_embedder.linear_1.bias".into(),
            vec![cfg.time_embed_hidden],
        ),
        (
            "time_embedder.linear_2.weight".into(),
            vec![cfg.time_embed_dim, cfg.time_embed_hidden],
        ),
        (
            "time_embedder.linear_2.bias".into(),
            vec![cfg.time_embed_dim],
        ),
        ("norm_out.linear.weight".into(), vec![2 * d, adaln_in]),
        ("norm_out.linear.bias".into(), vec![2 * d]),
        ("norm_out.norm.weight".into(), vec![d]),
        (
            "proj_out.weight".into(),
            vec![heads * cfg.video_patch_dim(), d],
        ),
        ("proj_out.bias".into(), vec![heads * cfg.video_patch_dim()]),
        (
            "audio_proj_out.weight".into(),
            vec![heads * cfg.audio_in_channels, d],
        ),
        (
            "audio_proj_out.bias".into(),
            vec![heads * cfg.audio_in_channels],
        ),
        ("token_refiner.final_norm.weight".into(), vec![d]),
    ];
    if let Some(r) = cfg.adaln_rank {
        v.push(("adaln_basis.weight".into(), vec![r, cfg.time_embed_dim]));
    }
    for l in 0..cfg.refiner_layers {
        attn_ff(
            &mut v,
            &format!("token_refiner.refiner_blocks.{l}"),
            d,
            qkv,
            cfg.head_dim,
            cfg.ffn_dim,
        );
    }
    for l in 0..cfg.layers {
        let p = format!("transformer_blocks.{l}");
        attn_ff(&mut v, &p, d, qkv, cfg.head_dim, cfg.ffn_dim);
        v.push((
            format!("{p}.adaln_proj.linear.weight"),
            vec![mods, adaln_in],
        ));
        v.push((format!("{p}.adaln_proj.linear.bias"), vec![mods]));
        if with_gate {
            v.push((format!("{p}.attn.to_gate_compress.weight"), vec![qkv, d]));
        }
    }
    v
}

/// Every tensor the video VAE reads.
pub fn video_vae_manifest(cfg: &VideoVaeConfig) -> Manifest {
    let mut v: Manifest = Vec::new();
    let c0 = cfg.block_out_channels[0];
    v.push((
        "encoder.conv_in.weight".into(),
        vec![c0, cfg.in_channels, 3, 3, 3],
    ));
    v.push(("encoder.conv_in.bias".into(), vec![c0]));
    let mut cin = c0;
    for (i, &cout) in cfg.block_out_channels.iter().enumerate() {
        for j in 0..cfg.layers_per_block {
            let p = format!("encoder.down_blocks.{i}.resnets.{j}");
            let ci = if j == 0 { cin } else { cout };
            v.push((format!("{p}.norm1.weight"), vec![ci]));
            v.push((format!("{p}.norm1.bias"), vec![ci]));
            v.push((format!("{p}.conv1.weight"), vec![cout, ci, 3, 3, 3]));
            v.push((format!("{p}.conv1.bias"), vec![cout]));
            v.push((format!("{p}.norm2.weight"), vec![cout]));
            v.push((format!("{p}.norm2.bias"), vec![cout]));
            v.push((format!("{p}.conv2.weight"), vec![cout, cout, 3, 3, 3]));
            v.push((format!("{p}.conv2.bias"), vec![cout]));
            if ci != cout {
                v.push((format!("{p}.conv_shortcut.weight"), vec![cout, ci, 1, 1, 1]));
                v.push((format!("{p}.conv_shortcut.bias"), vec![cout]));
            }
        }
        if cfg.spatial_down[i] > 1 || cfg.temporal_down[i] > 1 {
            v.push((
                format!("encoder.down_blocks.{i}.downsamplers.0.conv.weight"),
                vec![cout, cout, 3, 3, 3],
            ));
            v.push((
                format!("encoder.down_blocks.{i}.downsamplers.0.conv.bias"),
                vec![cout],
            ));
        }
        cin = cout;
    }
    let lz = cfg.latent_channels;
    v.push(("encoder.norm_out.weight".into(), vec![cin]));
    v.push(("encoder.norm_out.bias".into(), vec![cin]));
    v.push(("encoder.conv_out.weight".into(), vec![2 * lz, cin, 3, 3, 3]));
    v.push(("encoder.conv_out.bias".into(), vec![2 * lz]));
    v.push(("quant_conv.weight".into(), vec![2 * lz, 2 * lz, 1, 1, 1]));
    v.push(("quant_conv.bias".into(), vec![2 * lz]));
    v.push(("post_quant_conv.weight".into(), vec![lz, lz, 1, 1, 1]));
    v.push(("post_quant_conv.bias".into(), vec![lz]));
    v.extend(vae_decoder_manifest(cfg));
    v
}

/// The ViT decoder's tensors (`decoder.` prefix, Diffusers layout).
pub fn vae_decoder_manifest(cfg: &VideoVaeConfig) -> Manifest {
    let dd = cfg.decoder_dim();
    let ff = cfg.decoder_ffn_mult * dd;
    let mut v: Manifest = vec![
        (
            "decoder.proj_in.weight".into(),
            vec![dd, cfg.latent_channels],
        ),
        ("decoder.proj_in.bias".into(), vec![dd]),
        (
            "decoder.register_tokens".into(),
            vec![1, cfg.decoder_register_tokens, dd],
        ),
        ("decoder.norm_out.weight".into(), vec![dd]),
        ("decoder.norm_out.bias".into(), vec![dd]),
        (
            "decoder.proj_out.weight".into(),
            vec![
                cfg.in_channels * cfg.temporal_ratio() * cfg.spatial_ratio() * cfg.spatial_ratio(),
                dd,
            ],
        ),
        (
            "decoder.proj_out.bias".into(),
            vec![
                cfg.in_channels * cfg.temporal_ratio() * cfg.spatial_ratio() * cfg.spatial_ratio(),
            ],
        ),
    ];
    for l in 0..cfg.decoder_layers {
        let p = format!("decoder.transformer_blocks.{l}");
        for m in ["to_q", "to_k", "to_v", "to_out.0"] {
            v.push((format!("{p}.attn.{m}.weight"), vec![dd, dd]));
            v.push((format!("{p}.attn.{m}.bias"), vec![dd]));
        }
        v.push((format!("{p}.ff.net.0.proj.weight"), vec![2 * ff, dd]));
        v.push((format!("{p}.ff.net.0.proj.bias"), vec![2 * ff]));
        v.push((format!("{p}.ff.net.2.weight"), vec![dd, ff]));
        v.push((format!("{p}.ff.net.2.bias"), vec![dd]));
        v.push((format!("{p}.norm1.weight"), vec![dd]));
        v.push((format!("{p}.norm2.weight"), vec![dd]));
        v.push((format!("{p}.scale1"), vec![dd]));
        v.push((format!("{p}.scale2"), vec![dd]));
    }
    v
}

/// Every tensor the audio VAE reads. Weight-normalised convolutions are
/// listed by their folded `<key>.weight` name (read with
/// [`WeightSource::conv_weight`]); the fixed anti-aliasing filters are
/// tensors too.
pub fn audio_vae_manifest(cfg: &AudioVaeConfig) -> Manifest {
    let mut v: Manifest = Vec::new();
    let d0 = cfg.encoder_dim;
    v.push(("encoder.block.0.weight".into(), vec![d0, 1, 7]));
    v.push(("encoder.block.0.bias".into(), vec![d0]));
    let mut c = d0;
    for (i, &stride) in cfg.encoder_rates.iter().enumerate() {
        let p = format!("encoder.block.{}", i + 1);
        for (j, _dil) in [1usize, 3, 9].iter().enumerate() {
            let q = format!("{p}.block.{j}");
            v.push((format!("{q}.block.0.alpha"), vec![1, c, 1]));
            v.push((format!("{q}.block.1.weight"), vec![c, c, 7]));
            v.push((format!("{q}.block.1.bias"), vec![c]));
            v.push((format!("{q}.block.2.alpha"), vec![1, c, 1]));
            v.push((format!("{q}.block.3.weight"), vec![c, c, 1]));
            v.push((format!("{q}.block.3.bias"), vec![c]));
        }
        v.push((format!("{p}.block.3.alpha"), vec![1, c, 1]));
        v.push((format!("{p}.block.4.weight"), vec![2 * c, c, 2 * stride]));
        v.push((format!("{p}.block.4.bias"), vec![2 * c]));
        c *= 2;
    }
    let n = cfg.encoder_rates.len() + 1;
    v.push((format!("encoder.block.{n}.alpha"), vec![1, c, 1]));
    v.push((
        format!("encoder.block.{}.weight", n + 1),
        vec![cfg.latent_dim, c, 3],
    ));
    v.push((
        format!("encoder.block.{}.bias", n + 1),
        vec![cfg.latent_dim],
    ));
    let (ld, lc) = (cfg.latent_dim, cfg.latent_channels);
    for (k, s) in [
        ("pre_block.norm1.weight", vec![ld]),
        ("pre_block.norm1.bias", vec![ld]),
        ("pre_block.attn.qkv.weight", vec![3 * ld, ld]),
        ("pre_block.attn.q_bias", vec![ld]),
        ("pre_block.attn.v_bias", vec![ld]),
        ("pre_block.attn.zero_k_bias", vec![ld]),
        ("pre_block.attn.proj.weight", vec![lc, lc]),
        ("pre_block.attn.proj.bias", vec![lc]),
        ("pre_block.proj.weight", vec![lc, ld]),
        ("pre_block.proj.bias", vec![lc]),
        ("pre_block.norm2.weight", vec![lc]),
        ("pre_block.norm2.bias", vec![lc]),
        ("pre_block.norm3.weight", vec![ld]),
        ("pre_block.norm3.bias", vec![ld]),
        ("pre_block.mlp.norm.weight", vec![lc]),
        ("pre_block.mlp.norm.bias", vec![lc]),
        ("pre_block.mlp.w0.weight", vec![2 * lc, lc]),
        ("pre_block.mlp.w0.bias", vec![2 * lc]),
        ("pre_block.mlp.w1.weight", vec![2 * lc, lc]),
        ("pre_block.mlp.w1.bias", vec![2 * lc]),
        ("pre_block.mlp.w2.weight", vec![lc, 2 * lc]),
        ("pre_block.mlp.w2.bias", vec![lc]),
        ("mean_proj.weight", vec![lc, lc, 1]),
        ("mean_proj.bias", vec![lc]),
        ("logs_proj.weight", vec![lc, lc, 1]),
        ("logs_proj.bias", vec![lc]),
        ("dec_in_proj.weight", vec![ld, lc, 1]),
        ("dec_in_proj.bias", vec![ld]),
    ] {
        v.push((k.to_string(), s));
    }
    let dd = cfg.decoder_dim;
    v.push(("decoder.conv_pre.weight".into(), vec![dd, ld, 7]));
    v.push(("decoder.conv_pre.bias".into(), vec![dd]));
    let mut c = dd;
    let nk = cfg.resblock_kernels.len();
    for (i, (&rate, &k)) in cfg
        .decoder_rates
        .iter()
        .zip(&cfg.decoder_kernels)
        .enumerate()
    {
        v.push((format!("decoder.ups.{i}.0.weight"), vec![c, c / 2, k]));
        v.push((format!("decoder.ups.{i}.0.bias"), vec![c / 2]));
        let _ = rate;
        c /= 2;
        for (j, &rk) in cfg.resblock_kernels.iter().enumerate() {
            let p = format!("decoder.resblocks.{}", i * nk + j);
            for d in 0..cfg.resblock_dilations[j].len() {
                v.push((format!("{p}.convs1.{d}.weight"), vec![c, c, rk]));
                v.push((format!("{p}.convs1.{d}.bias"), vec![c]));
                v.push((format!("{p}.convs2.{d}.weight"), vec![c, c, rk]));
                v.push((format!("{p}.convs2.{d}.bias"), vec![c]));
            }
            for a in 0..2 * cfg.resblock_dilations[j].len() {
                v.push((format!("{p}.activations.{a}.act.alpha"), vec![c]));
                v.push((format!("{p}.activations.{a}.act.beta"), vec![c]));
                v.push((
                    format!("{p}.activations.{a}.upsample.filter"),
                    vec![1, 1, 12],
                ));
                v.push((
                    format!("{p}.activations.{a}.downsample.lowpass.filter"),
                    vec![1, 1, 12],
                ));
            }
        }
    }
    v.push(("decoder.activation_post.act.alpha".into(), vec![c]));
    v.push(("decoder.activation_post.act.beta".into(), vec![c]));
    v.push((
        "decoder.activation_post.upsample.filter".into(),
        vec![1, 1, 12],
    ));
    v.push((
        "decoder.activation_post.downsample.lowpass.filter".into(),
        vec![1, 1, 12],
    ));
    v.push(("decoder.conv_post.weight".into(), vec![1, c, 7]));
    v
}

/// Every tensor the TAEH3 decoder reads (positional `decoder.N` keys, as
/// `taehv.py`'s `nn.Sequential` numbers them).
pub fn taeh3_manifest(cfg: &TaeH3Config) -> Manifest {
    let nf = cfg.widths;
    let mut v = vec![
        (
            "decoder.1.weight".to_string(),
            vec![nf[0], cfg.latent_channels, 3, 3],
        ),
        ("decoder.1.bias".to_string(), vec![nf[0]]),
    ];
    for s in 0..3 {
        let base = 3 + 6 * s;
        for m in 0..cfg.mem_blocks {
            let p = format!("decoder.{}", base + m);
            v.push((format!("{p}.conv.0.weight"), vec![nf[s], 2 * nf[s], 3, 3]));
            v.push((format!("{p}.conv.0.bias"), vec![nf[s]]));
            v.push((format!("{p}.conv.2.weight"), vec![nf[s], nf[s], 3, 3]));
            v.push((format!("{p}.conv.2.bias"), vec![nf[s]]));
            v.push((format!("{p}.conv.4.weight"), vec![nf[s], nf[s], 3, 3]));
            v.push((format!("{p}.conv.4.bias"), vec![nf[s]]));
        }
        v.push((
            format!("decoder.{}.conv.weight", base + 4),
            vec![nf[s] * cfg.tgrow[s], nf[s], 1, 1],
        ));
        v.push((
            format!("decoder.{}.weight", base + 5),
            vec![nf[s + 1], nf[s], 3, 3],
        ));
    }
    v.push((
        "decoder.22.weight".into(),
        vec![cfg.out_channels(), nf[3], 3, 3],
    ));
    v.push(("decoder.22.bias".into(), vec![cfg.out_channels()]));
    v
}

/// Every tensor the Regen-2K latent upscaler reads.
pub fn upscaler_manifest(cfg: &UpscalerConfig) -> Manifest {
    let c = cfg.channels;
    let mut v: Manifest = vec![
        ("conv_in.weight".into(), vec![c, cfg.in_channels, 3, 3, 3]),
        ("conv_in.bias".into(), vec![c]),
        ("embed.0.weight".into(), vec![cfg.emb_dim, 1]),
        ("embed.0.bias".into(), vec![cfg.emb_dim]),
        ("embed.2.weight".into(), vec![cfg.emb_dim, cfg.emb_dim]),
        ("embed.2.bias".into(), vec![cfg.emb_dim]),
        ("norm_out.weight".into(), vec![c]),
        ("norm_out.bias".into(), vec![c]),
        ("conv_out.weight".into(), vec![cfg.in_channels, c, 3, 3, 3]),
        ("conv_out.bias".into(), vec![cfg.in_channels]),
    ];
    for (side, count) in [("in_blocks", cfg.in_blocks), ("out_blocks", cfg.out_blocks)] {
        for (i, temporal) in cfg.block_layout(count).into_iter().enumerate() {
            let p = format!("{side}.{i}");
            if temporal {
                v.push((format!("{p}.norm.weight"), vec![c]));
                v.push((format!("{p}.norm.bias"), vec![c]));
                v.push((
                    format!("{p}.dwconv.weight"),
                    vec![c, 1, cfg.temporal_kernel, 1, 1],
                ));
                v.push((format!("{p}.dwconv.bias"), vec![c]));
                v.push((format!("{p}.pwconv.weight"), vec![c, c, 1, 1, 1]));
                v.push((format!("{p}.pwconv.bias"), vec![c]));
            } else {
                v.push((format!("{p}.in_layers.0.weight"), vec![c]));
                v.push((format!("{p}.in_layers.0.bias"), vec![c]));
                v.push((format!("{p}.in_layers.2.weight"), vec![c, c, 3, 3, 3]));
                v.push((format!("{p}.in_layers.2.bias"), vec![c]));
                v.push((format!("{p}.emb_layers.1.weight"), vec![2 * c, cfg.emb_dim]));
                v.push((format!("{p}.emb_layers.1.bias"), vec![2 * c]));
                v.push((format!("{p}.out_norm.weight"), vec![c]));
                v.push((format!("{p}.out_norm.bias"), vec![c]));
                v.push((format!("{p}.out_layers.2.weight"), vec![c, c, 3, 3, 3]));
                v.push((format!("{p}.out_layers.2.bias"), vec![c]));
            }
        }
    }
    v
}

/// Every tensor the Qwen3-VL conditioner reads: the language stack up to the
/// output layer, and the vision tower with its DeepStack mergers.
pub fn qwen3vl_manifest(cfg: &Qwen3VlConfig) -> Manifest {
    let h = cfg.hidden;
    let mut v: Manifest = vec![(
        "model.language_model.embed_tokens.weight".into(),
        vec![cfg.vocab, h],
    )];
    for l in 0..cfg.output_layer {
        let p = format!("model.language_model.layers.{l}");
        v.push((format!("{p}.input_layernorm.weight"), vec![h]));
        v.push((format!("{p}.post_attention_layernorm.weight"), vec![h]));
        v.push((
            format!("{p}.self_attn.q_proj.weight"),
            vec![cfg.heads * cfg.head_dim, h],
        ));
        v.push((
            format!("{p}.self_attn.k_proj.weight"),
            vec![cfg.kv_heads * cfg.head_dim, h],
        ));
        v.push((
            format!("{p}.self_attn.v_proj.weight"),
            vec![cfg.kv_heads * cfg.head_dim, h],
        ));
        v.push((
            format!("{p}.self_attn.o_proj.weight"),
            vec![h, cfg.heads * cfg.head_dim],
        ));
        v.push((format!("{p}.self_attn.q_norm.weight"), vec![cfg.head_dim]));
        v.push((format!("{p}.self_attn.k_norm.weight"), vec![cfg.head_dim]));
        v.push((
            format!("{p}.mlp.gate_proj.weight"),
            vec![cfg.intermediate, h],
        ));
        v.push((format!("{p}.mlp.up_proj.weight"), vec![cfg.intermediate, h]));
        v.push((
            format!("{p}.mlp.down_proj.weight"),
            vec![h, cfg.intermediate],
        ));
    }
    let vh = cfg.vision_hidden;
    let vm = vh * cfg.vision_merge * cfg.vision_merge;
    v.push((
        "model.visual.patch_embed.proj.weight".into(),
        vec![
            vh,
            3,
            cfg.vision_temporal_patch,
            cfg.vision_patch,
            cfg.vision_patch,
        ],
    ));
    v.push(("model.visual.patch_embed.proj.bias".into(), vec![vh]));
    v.push((
        "model.visual.pos_embed.weight".into(),
        vec![cfg.vision_num_position_embeddings, vh],
    ));
    for b in 0..cfg.vision_depth {
        let p = format!("model.visual.blocks.{b}");
        for n in ["norm1", "norm2"] {
            v.push((format!("{p}.{n}.weight"), vec![vh]));
            v.push((format!("{p}.{n}.bias"), vec![vh]));
        }
        v.push((format!("{p}.attn.qkv.weight"), vec![3 * vh, vh]));
        v.push((format!("{p}.attn.qkv.bias"), vec![3 * vh]));
        v.push((format!("{p}.attn.proj.weight"), vec![vh, vh]));
        v.push((format!("{p}.attn.proj.bias"), vec![vh]));
        v.push((
            format!("{p}.mlp.linear_fc1.weight"),
            vec![cfg.vision_intermediate, vh],
        ));
        v.push((
            format!("{p}.mlp.linear_fc1.bias"),
            vec![cfg.vision_intermediate],
        ));
        v.push((
            format!("{p}.mlp.linear_fc2.weight"),
            vec![vh, cfg.vision_intermediate],
        ));
        v.push((format!("{p}.mlp.linear_fc2.bias"), vec![vh]));
    }
    v.push(("model.visual.merger.norm.weight".into(), vec![vh]));
    v.push(("model.visual.merger.norm.bias".into(), vec![vh]));
    v.push(("model.visual.merger.linear_fc1.weight".into(), vec![vm, vm]));
    v.push(("model.visual.merger.linear_fc1.bias".into(), vec![vm]));
    v.push((
        "model.visual.merger.linear_fc2.weight".into(),
        vec![cfg.vision_out_hidden, vm],
    ));
    v.push((
        "model.visual.merger.linear_fc2.bias".into(),
        vec![cfg.vision_out_hidden],
    ));
    for i in 0..cfg.vision_deepstack.len() {
        let p = format!("model.visual.deepstack_merger_list.{i}");
        v.push((format!("{p}.norm.weight"), vec![vm]));
        v.push((format!("{p}.norm.bias"), vec![vm]));
        v.push((format!("{p}.linear_fc1.weight"), vec![vm, vm]));
        v.push((format!("{p}.linear_fc1.bias"), vec![vm]));
        v.push((
            format!("{p}.linear_fc2.weight"),
            vec![cfg.vision_out_hidden, vm],
        ));
        v.push((format!("{p}.linear_fc2.bias"), vec![cfg.vision_out_hidden]));
    }
    v
}

/// Names in `manifest` that `src` lacks (plain, quantized or weight-normed).
pub fn missing(src: &dyn WeightSource, manifest: &[(String, Vec<usize>)]) -> Vec<String> {
    let table = src.contains(ADALN_TABLE_KEY);
    manifest
        .iter()
        .filter(|(n, _)| {
            if src.contains(n) {
                return false;
            }
            if table && (n.starts_with("time_embedder.") || n == "adaln_basis.weight") {
                return false;
            }
            if let Some(k) = n.strip_suffix(".weight") {
                return !(src.has_linear(k) || src.has_conv_weight(k));
            }
            true
        })
        .map(|(n, _)| n.clone())
        .collect()
}

pub fn manifest_params(m: &Manifest) -> usize {
    m.iter().map(|(_, s)| s.iter().product::<usize>()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h3_manifest_matches_param_count() {
        let cfg = H3Config::h3();
        assert_eq!(manifest_params(&h3_manifest(&cfg, false)), cfg.params());
        let trim = H3Config::fasth3_trim();
        let m = h3_manifest(&trim, true);
        assert!(
            m.iter()
                .any(|(n, s)| n == "adaln_basis.weight" && s == &vec![16, 2688])
        );
        assert!(m.iter().any(
            |(n, s)| n == "transformer_blocks.41.adaln_proj.linear.weight" && s == &vec![96768, 16]
        ));
        assert!(m.iter().any(
            |(n, s)| n == "transformer_blocks.0.attn.to_gate_compress.weight"
                && s == &vec![7168, 5376]
        ));
        let omni = H3Config::fasth3_omni();
        let m = h3_manifest(&omni, true);
        assert!(
            m.iter()
                .any(|(n, s)| n == "proj_out.weight" && s == &vec![32 * 96, 5376])
        );
        assert!(
            m.iter()
                .any(|(n, s)| n == "audio_proj_out.bias" && s == &vec![32 * 32])
        );
    }

    #[test]
    fn vae_manifests_have_the_checkpoint_shapes() {
        let m = video_vae_manifest(&VideoVaeConfig::h3());
        let has = |n: &str, s: &[usize]| m.iter().any(|(k, v)| k == n && v == s);
        assert!(has(
            "encoder.down_blocks.1.resnets.0.conv_shortcut.weight",
            &[256, 128, 1, 1, 1]
        ));
        assert!(has(
            "encoder.down_blocks.3.downsamplers.0.conv.weight",
            &[512, 512, 3, 3, 3]
        ));
        assert!(
            !m.iter()
                .any(|(k, _)| k.starts_with("encoder.down_blocks.4.downsamplers"))
        );
        assert!(has("encoder.conv_out.weight", &[48, 1024, 3, 3, 3]));
        assert!(has("decoder.proj_out.weight", &[3072, 2048]));
        assert!(has(
            "decoder.transformer_blocks.35.ff.net.0.proj.weight",
            &[16384, 2048]
        ));
        assert!(has(
            "decoder.transformer_blocks.35.ff.net.2.weight",
            &[2048, 8192]
        ));
        assert!(
            !m.iter()
                .any(|(k, _)| k.starts_with("decoder.transformer_blocks.36."))
        );

        let a = audio_vae_manifest(&AudioVaeConfig::h3());
        let has = |n: &str, s: &[usize]| a.iter().any(|(k, v)| k == n && v == s);
        assert!(has("encoder.block.1.block.0.block.1.weight", &[64, 64, 7]));
        assert!(has("encoder.block.5.block.4.weight", &[2048, 1024, 10]));
        assert!(has("encoder.block.7.weight", &[2048, 2048, 3]));
        assert!(has("decoder.ups.0.0.weight", &[1024, 512, 9]));
        assert!(has("decoder.ups.6.0.weight", &[16, 8, 4]));
        assert!(has("decoder.resblocks.20.convs1.0.weight", &[8, 8, 11]));
        assert!(has("decoder.resblocks.0.activations.5.act.beta", &[512]));
        assert!(has("decoder.conv_post.weight", &[1, 8, 7]));
        assert!(has("pre_block.attn.proj.weight", &[32, 32]));

        let t = taeh3_manifest(&TaeH3Config::taeh3());
        assert!(
            t.iter()
                .any(|(n, s)| n == "decoder.22.weight" && s == &vec![12, 64, 3, 3])
        );
        assert!(
            t.iter()
                .any(|(n, s)| n == "decoder.13.conv.weight" && s == &vec![256, 128, 1, 1])
        );

        let u = upscaler_manifest(&UpscalerConfig::regen_2k());
        assert!(
            u.iter()
                .any(|(n, s)| n == "in_blocks.1.dwconv.weight" && s == &vec![512, 1, 5, 1, 1])
        );
        assert!(u.iter().any(
            |(n, s)| n == "out_blocks.17.out_layers.2.weight" && s == &vec![512, 512, 3, 3, 3]
        ));
        assert!(!u.iter().any(|(n, _)| n.starts_with("in_blocks.18.")));
        assert!(
            u.iter()
                .any(|(n, s)| n == "in_blocks.0.emb_layers.1.weight" && s == &vec![1024, 64])
        );

        let q = qwen3vl_manifest(&Qwen3VlConfig::h3());
        assert!(q.iter().any(
            |(n, s)| n == "model.language_model.layers.49.mlp.down_proj.weight"
                && s == &vec![5120, 25600]
        ));
        assert!(
            !q.iter()
                .any(|(n, _)| n.starts_with("model.language_model.layers.50."))
        );
        assert!(q.iter().any(
            |(n, s)| n == "model.visual.deepstack_merger_list.2.norm.weight" && s == &vec![4608]
        ));
        assert!(
            q.iter()
                .any(|(n, s)| n == "model.visual.patch_embed.proj.weight"
                    && s == &vec![1152, 3, 2, 16, 16])
        );
    }

    #[test]
    fn comfy_keys_map_to_the_single_file_layout() {
        use ComfyPart::*;
        let k = |n: &str| comfy_h3_key(n);
        assert_eq!(
            k("transformer_blocks.3.attn.to_k.weight"),
            Some(("blocks.3.attn.qkv_proj.weight".into(), Third(1)))
        );
        assert_eq!(
            k("transformer_blocks.3.ff.net.0.proj.weight"),
            Some(("blocks.3.mlp.fc1.weight".into(), SwapHalves))
        );
        assert_eq!(
            k("transformer_blocks.3.ff.net.2.weight"),
            Some(("blocks.3.mlp.fc2.weight".into(), Whole))
        );
        assert_eq!(
            k("transformer_blocks.3.attn.to_out.0.weight"),
            Some(("blocks.3.attn.out_proj.weight".into(), Whole))
        );
        assert_eq!(
            k("transformer_blocks.3.attn.norm_q.weight"),
            Some(("blocks.3.attn.q_norm.weight".into(), Whole))
        );
        assert_eq!(
            k("transformer_blocks.3.adaln_proj.linear.bias"),
            Some(("blocks.3.adaln_proj.linear.bias".into(), Whole))
        );
        assert_eq!(
            k("token_refiner.refiner_blocks.1.attn.to_v.weight"),
            Some((
                "token_refiner.blocks.1.attn.qkv_proj.weight".into(),
                Third(2)
            ))
        );
        assert_eq!(
            k("token_refiner.final_norm.weight"),
            Some(("token_refiner.final_norm.weight".into(), Whole))
        );
        assert_eq!(
            k("proj_in.bias"),
            Some(("video_patch_proj.bias".into(), Whole))
        );
        assert_eq!(
            k("audio_proj_out.weight"),
            Some(("final_layer.audio_out.weight".into(), Whole))
        );
        assert_eq!(
            k("norm_out.linear.weight"),
            Some(("final_layer.adaln_proj.linear.weight".into(), Whole))
        );
        assert_eq!(
            k("norm_out.norm.weight"),
            Some(("final_layer.norm.weight".into(), Whole))
        );
        assert_eq!(
            k("context_embedder.weight"),
            Some(("condition_proj.weight".into(), Whole))
        );
        assert_eq!(k(ADALN_TABLE_KEY), Some((ADALN_TABLE_KEY.into(), Whole)));
        assert_eq!(k("time_embedder.linear_1.weight"), None);
        assert_eq!(k("adaln_basis.weight"), None);
    }

    #[test]
    fn nvfp4_module_names() {
        assert_eq!(
            nvfp4_module("transformer_blocks.3.ff.net.0.proj"),
            "transformer_blocks.3.ff.fc_in"
        );
        assert_eq!(
            nvfp4_module("transformer_blocks.3.ff.net.2"),
            "transformer_blocks.3.ff.fc_out"
        );
        assert_eq!(
            nvfp4_module("transformer_blocks.3.attn.to_out.0"),
            "transformer_blocks.3.attn.to_out"
        );
        assert_eq!(
            nvfp4_module("transformer_blocks.3.attn.to_q"),
            "transformer_blocks.3.attn.to_q"
        );
        assert_eq!(
            vae_source_key("decoder.transformer_blocks.2.ff.net.2.weight"),
            "blocks.2.ff.w2.weight"
        );
        assert_eq!(vae_source_key("decoder.proj_in.bias"), "x_embedder.bias");
        assert_eq!(
            vae_source_key("post_quant_conv.weight"),
            "post_quant_conv.weight"
        );
    }

    #[test]
    fn weight_norm_folds() {
        let w = fold_weight_norm(&[2.0, 3.0], &[3.0, 4.0, 0.0, 1.0], 2);
        assert_eq!(w, vec![1.2, 1.6, 0.0, 3.0]);
    }

    #[test]
    fn synthetic_is_deterministic() {
        let s = Synthetic { seed: 1 };
        assert_eq!(
            s.get("a.weight", &[4, 4]).unwrap(),
            s.get("a.weight", &[4, 4]).unwrap()
        );
        assert_ne!(
            s.get("a.weight", &[4, 4]).unwrap(),
            s.get("b.weight", &[4, 4]).unwrap()
        );
        assert!(
            s.get("x.norm1.weight", &[4])
                .unwrap()
                .iter()
                .all(|v| (v - 1.0).abs() <= 0.1)
        );
    }

    #[test]
    fn safetensors_roundtrip_and_nvfp4_stored() {
        let dir = std::env::temp_dir().join(format!("h3-st-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let vals = [1.0f32, -2.5, 0.125, 3.0];
        let bytes: Vec<u8> = vals
            .iter()
            .flat_map(|&v| half::bf16::from_f32(v).to_le_bytes())
            .collect();
        let view =
            safetensors::tensor::TensorView::new(safetensors::Dtype::BF16, vec![2, 2], &bytes)
                .unwrap();
        // An NVFP4 linear: 128 rows × 32 cols, scales padded [128, 4].
        let codes = vec![0x21u8; 128 * 16]; // cols 2i → code 1 (0.5), 2i+1 → code 2 (1.0)
        let scales = vec![0x38u8; 128 * 4]; // 1.0
        let alpha = 2.0f32.to_le_bytes().to_vec();
        let cview =
            safetensors::tensor::TensorView::new(safetensors::Dtype::U8, vec![128, 16], &codes)
                .unwrap();
        let sview =
            safetensors::tensor::TensorView::new(safetensors::Dtype::U8, vec![128, 4], &scales)
                .unwrap();
        let aview =
            safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![], &alpha).unwrap();
        safetensors::serialize_to_file(
            [
                ("w", view),
                ("transformer_blocks.0.ff.fc_in::_nvfp4_weight", cview),
                ("transformer_blocks.0.ff.fc_in::_nvfp4_weight_scale", sview),
                ("transformer_blocks.0.ff.fc_in::_nvfp4_alpha", aview),
            ],
            None,
            &dir.join("m.safetensors"),
        )
        .unwrap();
        let src = SafetensorsDir::open(&dir).unwrap();
        assert_eq!(src.get("w", &[2, 2]).unwrap(), vals);
        assert!(src.get("w", &[4]).is_err());
        assert!(src.get("nope", &[1]).is_err());
        assert!(src.has_linear("transformer_blocks.0.ff.net.0.proj"));
        match src.stored("transformer_blocks.0.ff.net.0.proj").unwrap() {
            Some(Stored::Nvfp4(q)) => {
                assert_eq!((q.rows, q.cols, q.alpha), (128, 32, 2.0));
                let d = q.dequantize();
                assert_eq!(d[0], 1.0);
                assert_eq!(d[1], 2.0);
            }
            other => panic!("{other:?}"),
        }
        assert!(
            missing(
                &src,
                &[(
                    "transformer_blocks.0.ff.net.0.proj.weight".into(),
                    vec![128, 32]
                )]
            )
            .is_empty()
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
