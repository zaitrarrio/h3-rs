//! Weight-only quantization of the block linears: int8 ([`Q8`]), 4-bit
//! ([`Q4`]) — fastwan-rs's formats — and NVIDIA's NVFP4 ([`Nvfp4`]) as the
//! FastH3 Trim checkpoint stores it. [`WeightQuant::for_key`] is the one rule
//! for which linears get which format.
//!
//! Q8: symmetric round-to-nearest, one f32 scale per group of [`Q8_GROUP`]
//! consecutive input columns (`w ≈ q·s`, `q ∈ [-127, 127]`). Q4: unsigned
//! codes with an affine scale/zero per group, optionally HQQ-refined. The CPU
//! backend dequantizes with [`Q8::dequantize`] / [`Q4::dequantize`] /
//! [`Nvfp4::dequantize`] and is the reference the GPU kernels are checked
//! against.

use rayon::prelude::*;

pub const Q8_GROUP: usize = 128;

/// How the DiT's transformer-block weights are stored on the device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WeightQuant {
    /// The backend's own precision (bf16 on CUDA, f16 pairs on WebGPU).
    #[default]
    Native,
    Int8,
    /// 4-bit with `group` columns per scale/zero; `ffn_only` keeps attention
    /// and gate linears at int8.
    Int4 { group: usize, hqq: bool, ffn_only: bool },
    /// NVFP4 as stored by the Trim checkpoint (FP4 E2M1 codes, 16-column
    /// E4M3 scales, one f32 alpha per tensor). Only loaded, never made here.
    Nvfp4,
    /// MXFP8 W8A8 ([`Mxfp8`]): E4M3 weights *and* activations with one E8M0
    /// scale per 32 input columns, the block GEMMs on FP8 tensor cores
    /// (cuBLASLt `VEC32_UE8M0`, sm_100+; emulated elsewhere). Sol-H3 /
    /// fastvideo-rs's recipe: see [`WeightQuant::for_block`].
    Mxfp8,
}

impl WeightQuant {
    pub fn attn(self) -> Self {
        match self {
            Self::Int4 { ffn_only: true, .. } => Self::Int8,
            q => q,
        }
    }

    /// The format of the linear `key` (no `.weight`): the block FFN gets
    /// `self`, block attention and gate get [`Self::attn`], and everything
    /// else — token refiner, boundary projections, AdaLN — stays `Native`.
    /// Exporter and loader both decide with this.
    pub fn for_key(self, key: &str) -> Self {
        let Some(role) = key
            .strip_prefix("transformer_blocks.")
            .and_then(|r| r.split_once('.'))
            .map(|(_, r)| r)
        else {
            return Self::Native;
        };
        match role {
            "ff.net.0.proj" | "ff.net.2" => self,
            "attn.to_q" | "attn.to_k" | "attn.to_v" | "attn.to_out.0" | "attn.to_gate_compress" => self.attn(),
            _ => Self::Native,
        }
    }

    pub fn is_quantized(self) -> bool {
        self != Self::Native
    }

    /// The format for transformer block `block` of `layers`: MXFP8 leaves the
    /// first two and the last three blocks in the backend's precision (Sol-H3
    /// `compute_quant.py`, blocks `[2, 46]` of 50; the 42-block Trim keeps the
    /// same edges). Every other format applies to every block.
    pub fn for_block(self, block: usize, layers: usize) -> Self {
        match self {
            Self::Mxfp8 if block < MXFP8_FIRST_BLOCK || block + MXFP8_BF16_TAIL >= layers => Self::Native,
            q => q,
        }
    }

    /// [`Self::for_block`] then [`Self::for_key`] for a key of a `layers`-deep
    /// DiT (the exporter's rule; the loader knows the block it is loading).
    pub fn for_layer_key(self, key: &str, layers: usize) -> Self {
        let block = key.strip_prefix("transformer_blocks.").and_then(|r| r.split_once('.')).and_then(|(i, _)| i.parse().ok());
        match block {
            Some(b) => self.for_block(b, layers).for_key(key),
            None => self.for_key(key),
        }
    }
}

/// Blocks before this one stay in the backend's precision under MXFP8.
pub const MXFP8_FIRST_BLOCK: usize = 2;
/// Blocks at the end of the stack that stay in the backend's precision under MXFP8.
pub const MXFP8_BF16_TAIL: usize = 3;

impl std::fmt::Display for WeightQuant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Native => f.write_str("native"),
            Self::Int8 => f.write_str("int8"),
            Self::Nvfp4 => f.write_str("nvfp4"),
            Self::Mxfp8 => f.write_str("mxfp8"),
            Self::Int4 { group, hqq, ffn_only } => write!(
                f,
                "int4{}-g{group}{}",
                if *hqq { "-hqq" } else { "" },
                if *ffn_only { "-ffn" } else { "" }
            ),
        }
    }
}

impl std::str::FromStr for WeightQuant {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "native" | "bf16" | "f16" => Ok(Self::Native),
            "int8" | "q8" => Ok(Self::Int8),
            "nvfp4" | "fp4" => Ok(Self::Nvfp4),
            "mxfp8" | "fp8" => Ok(Self::Mxfp8),
            _ => {
                let (s0, ffn_only) = match s.strip_suffix("-ffn") {
                    Some(r) => (r, true),
                    None => (s, false),
                };
                let rest = s0.strip_prefix("int4").ok_or_else(|| {
                    format!("unknown weight format `{s}` (native | int8 | int4[-hqq][-gN][-ffn] | nvfp4 | mxfp8)")
                })?;
                let (hqq, rest) = match rest.strip_prefix("-hqq") {
                    Some(r) => (true, r),
                    None => (false, rest),
                };
                let group = match rest.strip_prefix("-g") {
                    Some(g) => g.parse().map_err(|_| format!("bad group in `{s}`"))?,
                    None if rest.is_empty() => 64,
                    None => return Err(format!("bad weight format `{s}`")),
                };
                if group == 0 || group % 8 != 0 {
                    return Err(format!("int4 group must be a multiple of 8, got {group}"));
                }
                Ok(Self::Int4 { group, hqq, ffn_only })
            }
        }
    }
}

/// A linear as stored in a quantized checkpoint ([`crate::weights::WeightSource::stored`]).
#[derive(Debug, Clone)]
pub enum Stored {
    Q8(Q8),
    Q4(Q4),
    Nvfp4(Nvfp4),
    Mxfp8(Mxfp8),
}

impl Stored {
    pub fn dequantize(&self) -> Vec<f32> {
        match self {
            Stored::Q8(q) => q.dequantize(),
            Stored::Q4(q) => q.dequantize(),
            Stored::Nvfp4(q) => q.dequantize(),
            Stored::Mxfp8(q) => q.dequantize(),
        }
    }
    pub fn shape(&self) -> (usize, usize) {
        match self {
            Stored::Q8(q) => (q.rows, q.cols),
            Stored::Q4(q) => (q.rows, q.cols),
            Stored::Nvfp4(q) => (q.rows, q.cols),
            Stored::Mxfp8(q) => (q.rows, q.cols),
        }
    }
}

/// A `[rows, cols]` matrix as int8 codes and per-group scales.
#[derive(Debug, Clone)]
pub struct Q8 {
    pub q: Vec<i8>,
    /// `[rows, cols / group]`.
    pub scales: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub group: usize,
}

impl Q8 {
    pub fn quantize(data: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        let group = if cols.is_multiple_of(Q8_GROUP) { Q8_GROUP } else { cols };
        let mut q = vec![0i8; data.len()];
        let mut scales = Vec::with_capacity(data.len() / group);
        for (src, dst) in data.chunks(group).zip(q.chunks_mut(group)) {
            let amax = src.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let s = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            for (d, v) in dst.iter_mut().zip(src) {
                *d = (v / s).round().clamp(-127.0, 127.0) as i8;
            }
            scales.push(s);
        }
        Self { q, scales, rows, cols, group }
    }

    pub fn dequantize(&self) -> Vec<f32> {
        self.q.iter().enumerate().map(|(i, &q)| q as f32 * self.scales[i / self.group]).collect()
    }

    /// Stack matrices with the same columns and group by rows (fused QKV).
    pub fn concat(parts: Vec<Q8>) -> Self {
        let mut it = parts.into_iter();
        let mut a = it.next().expect("at least one matrix");
        for b in it {
            assert_eq!((a.cols, a.group), (b.cols, b.group));
            a.q.extend(b.q);
            a.scales.extend(b.scales);
            a.rows += b.rows;
        }
        a
    }

    pub fn bytes(&self) -> usize {
        self.q.len() + 4 * self.scales.len()
    }
}

/// A `[rows, cols]` matrix as unsigned 4-bit codes with an affine scale/zero
/// per group of columns: `w ≈ (q − z)·s`, `q ∈ [0, 15]`. Scales and zeros are
/// rounded to f16 when made (how they are stored), so a quantized file and an
/// in-memory quantization are identical.
#[derive(Debug, Clone)]
pub struct Q4 {
    /// One code per byte (packing is a storage concern; see [`Q4::packed`]).
    pub q: Vec<u8>,
    pub scales: Vec<f32>,
    pub zeros: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    pub group: usize,
}

impl Q4 {
    /// Min/max round-to-nearest; with `hqq`, refine each group's zero with
    /// HQQ's half-quadratic proximal solver (lp-norm 0.7, β 10, κ 1.01, ≤ 20
    /// iterations).
    pub fn quantize(data: &[f32], rows: usize, cols: usize, group: usize, hqq: bool) -> Self {
        assert_eq!(data.len(), rows * cols);
        let group = if cols.is_multiple_of(group) { group } else { cols };
        let n = data.len() / group;
        let mut q = vec![0u8; data.len()];
        let mut sz = vec![(0.0f32, 0.0f32); n];
        let threads = std::thread::available_parallelism().map_or(1, |t| t.get());
        let per = n.div_ceil(threads).max(1);
        std::thread::scope(|sc| {
            for ((w, qc), szc) in data.chunks(per * group).zip(q.chunks_mut(per * group)).zip(sz.chunks_mut(per)) {
                sc.spawn(move || {
                    for ((w, qg), sz) in w.chunks(group).zip(qc.chunks_mut(group)).zip(szc) {
                        let (lo, hi) = w.iter().fold((f32::MAX, f32::MIN), |(a, b), &v| (a.min(v), b.max(v)));
                        let is = if hi > lo { 15.0 / (hi - lo) } else { 1.0 };
                        let mut z = -lo * is;
                        let code = |v: f32, z: f32| (v * is + z).round().clamp(0.0, 15.0);
                        if hqq {
                            let (p, mut beta, kappa) = (0.7f32, 10.0f32, 1.01f32);
                            let (mut best, mut best_z) = (f32::MAX, z);
                            for _ in 0..20 {
                                let mut err = 0.0;
                                let mut zsum = 0.0;
                                for &v in w {
                                    let c = code(v, z);
                                    let r = (c - z) / is;
                                    let e = v - r;
                                    err += e.abs();
                                    let a = e.abs();
                                    let sh = if a > 0.0 { e.signum() * (a - a.powf(p - 1.0) / beta).max(0.0) } else { 0.0 };
                                    zsum += c - (v - sh) * is;
                                }
                                let err = err / w.len() as f32;
                                if err >= best {
                                    break;
                                }
                                (best, best_z) = (err, z);
                                z = zsum / w.len() as f32;
                                beta *= kappa;
                            }
                            z = best_z;
                        }
                        for (d, &v) in qg.iter_mut().zip(w) {
                            *d = code(v, z) as u8;
                        }
                        let f = |v: f32| half::f16::from_f32(v).to_f32();
                        *sz = (f(1.0 / is), f(z));
                    }
                });
            }
        });
        let (scales, zeros) = sz.into_iter().unzip();
        Self { q, scales, zeros, rows, cols, group }
    }

    pub fn dequantize(&self) -> Vec<f32> {
        self.q
            .iter()
            .enumerate()
            .map(|(i, &c)| {
                let g = i / self.group;
                (c as f32 - self.zeros[g]) * self.scales[g]
            })
            .collect()
    }

    /// Two codes per byte, element `2i` in the low nibble.
    pub fn packed(&self) -> Vec<u8> {
        self.q.chunks(2).map(|c| c[0] | (c.get(1).copied().unwrap_or(0) << 4)).collect()
    }

    pub fn from_packed(packed: &[u8], scales: Vec<f32>, zeros: Vec<f32>, rows: usize, cols: usize) -> Self {
        let n = rows * cols;
        assert_eq!(packed.len(), n.div_ceil(2));
        assert!(!scales.is_empty() && scales.len() == zeros.len());
        assert_eq!(n % scales.len(), 0);
        let q = (0..n).map(|i| (packed[i / 2] >> (4 * (i % 2))) & 15).collect();
        let group = n / scales.len();
        Self { q, scales, zeros, rows, cols, group }
    }

    pub fn concat(parts: Vec<Q4>) -> Self {
        let mut it = parts.into_iter();
        let mut a = it.next().expect("at least one matrix");
        for b in it {
            assert_eq!((a.cols, a.group), (b.cols, b.group));
            a.q.extend(b.q);
            a.scales.extend(b.scales);
            a.zeros.extend(b.zeros);
            a.rows += b.rows;
        }
        a
    }

    pub fn bytes(&self) -> usize {
        self.q.len().div_ceil(2) + 4 * self.scales.len()
    }
}

/// FP4 E2M1 values by code (sign bit 3).
pub const E2M1: [f32; 16] =
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

/// Decode an FP8 E4M3 (finite-only, "fn") byte.
pub fn e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let exp = ((b >> 3) & 0xF) as i32;
    let man = (b & 7) as f32;
    if exp == 0 {
        sign * man / 8.0 * 2f32.powi(-6)
    } else if exp == 15 && man == 7.0 {
        f32::NAN
    } else {
        sign * (1.0 + man / 8.0) * 2f32.powi(exp - 7)
    }
}

/// NVFP4 block scaling: 16 columns share one E4M3 scale.
pub const NVFP4_GROUP: usize = 16;

/// A `[rows, cols]` matrix as NVFP4: one E2M1 code per weight, an E4M3 scale
/// per 16 columns, one f32 `alpha` per tensor. `w = code · e4m3(scale) · alpha`.
#[derive(Debug, Clone)]
pub struct Nvfp4 {
    /// Two codes per byte, `[rows, cols / 2]`: column `2i` in the low nibble
    /// (FlashInfer / ModelOpt, `first_high == false`) or in the high nibble
    /// (ComfyUI's `float.py`, `first_high == true`).
    pub packed: Vec<u8>,
    /// Row-major `[rows, cols / 16]` E4M3 bytes (unswizzled).
    pub scales: Vec<u8>,
    pub alpha: f32,
    pub rows: usize,
    pub cols: usize,
    pub first_high: bool,
}

impl Nvfp4 {
    /// Undo NVIDIA's 128×4 block-scale swizzle (torchao `to_blocked`,
    /// FlashInfer, ComfyUI's `TensorCoreNVFP4Layout`): the stored tensor is
    /// `[rows padded to 128, groups padded to 4]` viewed as `(rows/128, 4, 32,
    /// groups/4, 4)` and permuted to `(0, 3, 2, 1, 4)`. Returns the row-major
    /// scales. Verified against the bf16 twin of the FastH3-Trim NVFP4 file
    /// (`docs/runs/2026-11-07-vast-nvfp4-vsa2`).
    pub fn unswizzle_scales(swizzled: &[u8], rows: usize, groups: usize) -> Vec<u8> {
        let rp = rows.div_ceil(128) * 128;
        let gp = groups.div_ceil(4) * 4;
        assert_eq!(swizzled.len(), rp * gp, "swizzled scale tensor is {} bytes, expected {}", swizzled.len(), rp * gp);
        let nc = gp / 4;
        let mut out = vec![0u8; rows * groups];
        for r in 0..rows {
            let (i0, rem) = (r / 128, r % 128);
            let (i3, i2) = (rem / 32, rem % 32);
            for g in 0..groups {
                let (i1, i4) = (g / 4, g % 4);
                out[r * groups + g] = swizzled[Self::swizzled_index(i0, i1, i2, i3, i4, nc)];
            }
        }
        out
    }

    /// Where the scale of logical row `i0·128 + i3·32 + i2`, group `i1·4 + i4`
    /// sits in the blocked tensor (`nc` = padded groups / 4).
    pub fn swizzled_index(i0: usize, i1: usize, i2: usize, i3: usize, i4: usize, nc: usize) -> usize {
        (((i0 * nc + i1) * 32 + i2) * 4 + i3) * 4 + i4
    }

    pub fn from_stored(packed: &[u8], swizzled_scales: &[u8], alpha: f32, rows: usize, cols: usize, first_high: bool) -> Self {
        assert_eq!(packed.len(), rows * cols / 2);
        assert_eq!(cols % NVFP4_GROUP, 0);
        let scales = Self::unswizzle_scales(swizzled_scales, rows, cols / NVFP4_GROUP);
        Self { packed: packed.to_vec(), scales, alpha, rows, cols, first_high }
    }

    pub fn code(&self, r: usize, c: usize) -> u8 {
        let b = self.packed[(r * self.cols + c) / 2];
        if (c % 2 == 0) != self.first_high { b & 15 } else { b >> 4 }
    }

    /// Rows `[start, end)` as their own matrix (a part of a fused QKV).
    pub fn rows(&self, start: usize, end: usize) -> Self {
        let groups = self.cols / NVFP4_GROUP;
        Self {
            packed: self.packed[start * self.cols / 2..end * self.cols / 2].to_vec(),
            scales: self.scales[start * groups..end * groups].to_vec(),
            alpha: self.alpha,
            rows: end - start,
            cols: self.cols,
            first_high: self.first_high,
        }
    }

    /// The two row halves exchanged (a `[gate | value]` SwiGLU input
    /// projection becoming `[value | gate]`).
    pub fn swap_halves(&self) -> Self {
        let h = self.rows / 2;
        let mut lo = self.rows(h, self.rows);
        let hi = self.rows(0, h);
        lo.packed.extend(hi.packed);
        lo.scales.extend(hi.scales);
        lo.rows = self.rows;
        lo
    }

    pub fn dequantize(&self) -> Vec<f32> {
        let groups = self.cols / NVFP4_GROUP;
        let mut out = Vec::with_capacity(self.rows * self.cols);
        for r in 0..self.rows {
            for c in 0..self.cols {
                let s = e4m3_to_f32(self.scales[r * groups + c / NVFP4_GROUP]);
                out.push(E2M1[self.code(r, c) as usize] * s * self.alpha);
            }
        }
        out
    }

    /// Scales as f32, `[rows, cols/16]`, for kernels that take them as numbers.
    pub fn scales_f32(&self) -> Vec<f32> {
        self.scales.iter().map(|&b| e4m3_to_f32(b) * self.alpha).collect()
    }

    /// Stack by rows (fused QKV) when the alphas agree; otherwise the caller
    /// dequantizes.
    pub fn concat(parts: Vec<Nvfp4>) -> Option<Self> {
        let mut it = parts.into_iter();
        let mut a = it.next()?;
        for b in it {
            if a.cols != b.cols || a.alpha != b.alpha || a.first_high != b.first_high {
                return None;
            }
            a.packed.extend(b.packed);
            a.scales.extend(b.scales);
            a.rows += b.rows;
        }
        Some(a)
    }

    pub fn bytes(&self) -> usize {
        self.packed.len() + self.scales.len()
    }
}

// ------------------------------------------------------------------ MXFP8

/// Values per MX scale (OCP microscaling block).
pub const MX_BLOCK: usize = 32;
/// Largest finite E4M3 value.
pub const E4M3_MAX: f32 = 448.0;

/// f32 → bf16 → f32, round to nearest even (what the device sees).
pub fn bf16_round(v: f32) -> f32 {
    half::bf16::from_f32(v).to_f32()
}

/// f32 → E4M3 as `cvt.rn.satfinite.e4m3x2.f32`: round to nearest even,
/// finite values past 448 saturate to ±448, NaN → 0x7F. Twin of the CUDA
/// `mx_e4m3` and the WGSL `e4m3_sat`.
pub fn e4m3_satfinite(x: f32) -> u8 {
    let u = x.to_bits();
    let sign = ((u >> 24) & 0x80) as u8;
    let a = u & 0x7FFF_FFFF;
    if a > 0x7F80_0000 {
        return sign | 0x7F;
    }
    if a >= 0x43E0_0000 {
        return sign | 0x7E;
    }
    if a < 0x3C80_0000 {
        // Below 2^-6: subnormal codes with quantum 2^-9; the product is exact.
        let m = (f32::from_bits(a) * 512.0).round_ties_even() as u32;
        return sign | m as u8;
    }
    let mut e = (a >> 23) as i32 - 127;
    let mut m = (a >> 20) & 7;
    let rem = a & 0xF_FFFF;
    if rem > 0x8_0000 || (rem == 0x8_0000 && (m & 1) == 1) {
        m += 1;
        if m == 8 {
            m = 0;
            e += 1;
        }
    }
    sign | (((e + 7) as u32) << 3) as u8 | m as u8
}

/// E4M3 (fn variant: no infinities, 0x7F / 0xFF are NaN) → f32.
pub fn e4m3_decode(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((b >> 3) & 0xF);
    let m = f32::from(b & 7);
    if e == 15 && (b & 7) == 7 {
        return f32::NAN;
    }
    if e == 0 {
        return sign * m / 512.0;
    }
    sign * (1.0 + m / 8.0) * 2f32.powi(e - 7)
}

/// The E8M0 block exponent for a 32-value block with maximum magnitude
/// `amax`: the smallest power of two that keeps `amax` within 448 after
/// scaling (Sol-H3 `mxfp8.py::_mx_e8m0_from_amax`). Returns the biased byte
/// and `2^-exponent`.
pub fn mx_e8m0(amax: f32) -> (u8, f32) {
    let bits = amax.to_bits() as i32;
    let e0 = ((bits >> 23) & 0xFF) - 135;
    let threshold = f32::from_bits((((e0 + 135) << 23) | 0x60_0000) as u32);
    let mut e = e0 + i32::from(amax > threshold);
    e = e.max(-127);
    let inv = f32::from_bits(((127 - e) << 23) as u32);
    ((e + 127) as u8, inv)
}

/// Position of the scale of `(row, group)` in cuBLASLt's `VEC32_UE8M0`
/// layout (torch `SWIZZLE_32_4_4`): tiles of 128 rows × 4 groups, each 512
/// bytes, 32 × 4 × 4 inside.
pub fn mx_scale_offset(row: usize, group: usize, column_blocks: usize) -> usize {
    let tile = (row / 128) * column_blocks + group / 4;
    tile * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 + group % 4
}

/// Scale bytes of a `[rows, k]` operand: rows padded to 128, groups to 4.
pub fn mx_scale_len(rows: usize, k: usize) -> usize {
    rows.div_ceil(128) * 128 * (k / MX_BLOCK).div_ceil(4) * 4
}

/// A `[rows, cols]` matrix as E4M3 codes with one E8M0 scale per 32 columns,
/// the scales in the `VEC32_UE8M0` swizzle so the device uses them as they
/// are. Activations get the same treatment per GEMM ([`Mxfp8::round_rows`]
/// is the host form of that step).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Mxfp8 {
    /// `[rows, cols]` E4M3 bytes.
    pub codes: Vec<u8>,
    /// [`mx_scale_len`] swizzled E8M0 bytes.
    pub scales: Vec<u8>,
    pub rows: usize,
    pub cols: usize,
}

impl Mxfp8 {
    /// Quantize `data` (`cols % 32 == 0`); one task per 128-row scale tile.
    pub fn quantize(data: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        assert!(cols.is_multiple_of(MX_BLOCK), "MXFP8 needs cols % 32 == 0, got {cols}");
        let cb = (cols / MX_BLOCK).div_ceil(4);
        let tile = cb * 512;
        let mut codes = vec![0u8; rows * cols];
        let mut scales = vec![0u8; mx_scale_len(rows, cols)];
        codes.par_chunks_mut(128 * cols).zip(scales.par_chunks_mut(tile)).enumerate().for_each(|(t, (cq, sq))| {
            let nrows = cq.len() / cols;
            for r in 0..nrows {
                let row = t * 128 + r;
                for g in 0..cols / MX_BLOCK {
                    let blk = &data[row * cols + g * MX_BLOCK..][..MX_BLOCK];
                    let (byte, inv) = mx_e8m0(amax_abs(blk));
                    sq[mx_scale_offset(r, g, cb)] = byte;
                    for (j, &v) in blk.iter().enumerate() {
                        cq[r * cols + g * MX_BLOCK + j] = e4m3_satfinite(v * inv);
                    }
                }
            }
        });
        Self { codes, scales, rows, cols }
    }

    /// From stored codes `[rows, cols]` and swizzled scales ([`mx_scale_len`] bytes).
    pub fn from_stored(codes: Vec<u8>, scales: Vec<u8>, rows: usize, cols: usize) -> Self {
        assert_eq!(codes.len(), rows * cols);
        assert_eq!(scales.len(), mx_scale_len(rows, cols));
        Self { codes, scales, rows, cols }
    }

    /// Stack matrices with the same columns by rows (fused QKV): the scale
    /// tiles are re-laid for the combined row count.
    pub fn concat(parts: Vec<Mxfp8>) -> Self {
        let cols = parts[0].cols;
        let rows: usize = parts.iter().map(|p| p.rows).sum();
        let cb = (cols / MX_BLOCK).div_ceil(4);
        let mut codes = Vec::with_capacity(rows * cols);
        let mut scales = vec![0u8; mx_scale_len(rows, cols)];
        let mut r0 = 0;
        for p in &parts {
            assert_eq!(p.cols, cols);
            codes.extend_from_slice(&p.codes);
            for r in 0..p.rows {
                for g in 0..cols / MX_BLOCK {
                    scales[mx_scale_offset(r0 + r, g, cb)] = p.scales[mx_scale_offset(r, g, cb)];
                }
            }
            r0 += p.rows;
        }
        Self { codes, scales, rows, cols }
    }

    pub fn scale_of(&self, row: usize, group: usize) -> f32 {
        let cb = (self.cols / MX_BLOCK).div_ceil(4);
        2f32.powi(i32::from(self.scales[mx_scale_offset(row, group, cb)]) - 127)
    }

    pub fn dequantize(&self) -> Vec<f32> {
        (0..self.rows * self.cols)
            .map(|i| {
                let (r, c) = (i / self.cols, i % self.cols);
                e4m3_decode(self.codes[i]) * self.scale_of(r, c / MX_BLOCK)
            })
            .collect()
    }

    pub fn bytes(&self) -> usize {
        self.codes.len() + self.scales.len()
    }

    /// The activation side of W8A8: every 32-column block of every row of `x`
    /// (`[rows, k]`, `k % 32 == 0`) rounded to bf16 (the device's storage),
    /// quantized to E4M3 with its E8M0 block scale and decoded again. The CPU
    /// reference feeds this to the f32 GEMM; the GPU paths produce the same
    /// codes and scales for the FP8 tensor cores.
    pub fn round_rows(x: &[f32], k: usize) -> Vec<f32> {
        assert!(k.is_multiple_of(MX_BLOCK) && x.len().is_multiple_of(k));
        let mut out = Vec::with_capacity(x.len());
        for blk in x.chunks_exact(MX_BLOCK) {
            let b: Vec<f32> = blk.iter().map(|&v| bf16_round(v)).collect();
            let (byte, inv) = mx_e8m0(amax_abs(&b));
            let s = 2f32.powi(i32::from(byte) - 127);
            out.extend(b.iter().map(|&v| e4m3_decode(e4m3_satfinite(v * inv)) * s));
        }
        out
    }
}

/// max |x|; a NaN wins (the kernels propagate it too).
fn amax_abs(x: &[f32]) -> f32 {
    x.iter().fold(0.0f32, |a, &v| {
        let v = v.abs();
        if v > a || v.is_nan() { v } else { a }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_round_trip_error_is_half_a_step() {
        let (rows, cols) = (8, 256);
        let w: Vec<f32> =
            (0..rows * cols).map(|i| ((i * 7919 % 1000) as f32 / 500.0 - 1.0) * (1 + i / cols) as f32).collect();
        let q = Q8::quantize(&w, rows, cols);
        assert_eq!(q.group, 128);
        let d = q.dequantize();
        for (i, (a, b)) in w.iter().zip(&d).enumerate() {
            assert!((a - b).abs() <= q.scales[i / 128] * 0.5 + 1e-6);
        }
    }

    #[test]
    fn int4_round_trip_and_hqq_does_not_hurt() {
        let (rows, cols) = (16, 256);
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| {
                let u = (i * 7919 % 1009) as f32 / 1009.0 - 0.5;
                if i % 97 == 0 { u * 8.0 } else { u * 0.3 }
            })
            .collect();
        let err = |q: &Q4| -> f32 { w.iter().zip(q.dequantize()).map(|(a, b)| (a - b).powi(2)).sum() };
        let rtn = Q4::quantize(&w, rows, cols, 64, false);
        for (i, (a, b)) in w.iter().zip(rtn.dequantize()).enumerate() {
            assert!((a - b).abs() <= rtn.scales[i / 64] * 0.52 + 1e-5);
        }
        let hqq = Q4::quantize(&w, rows, cols, 64, true);
        assert!(err(&hqq) <= err(&rtn) * 1.05);
        let back = Q4::from_packed(&rtn.packed(), rtn.scales.clone(), rtn.zeros.clone(), rows, cols);
        assert_eq!(back.q, rtn.q);
        assert_eq!("int4-hqq-g32".parse(), Ok(WeightQuant::Int4 { group: 32, hqq: true, ffn_only: false }));
        assert_eq!("int4-hqq-g64-ffn".parse::<WeightQuant>().map(|q| q.attn()), Ok(WeightQuant::Int8));
        assert_eq!("nvfp4".parse(), Ok(WeightQuant::Nvfp4));
    }

    #[test]
    fn mxfp8_codes_round_trip_and_swizzle_is_a_bijection() {
        // Every E4M3 code decodes and re-encodes to itself (NaN codes aside).
        for b in 0..=255u8 {
            if (b & 0x7F) == 0x7F {
                continue;
            }
            assert_eq!(e4m3_satfinite(e4m3_decode(b)), b, "code {b:#x}");
        }
        // Saturation and rounding: 448 is the max, 464 rounds to 448, 2^-9 is the smallest subnormal.
        assert_eq!(e4m3_decode(e4m3_satfinite(1000.0)), 448.0);
        assert_eq!(e4m3_decode(e4m3_satfinite(-1000.0)), -448.0);
        assert_eq!(e4m3_satfinite(1.0 / 512.0), 1);
        assert_eq!(e4m3_satfinite(1.0 / 1024.0), 0, "ties to even: half a quantum rounds to 0");
        assert_eq!(e4m3_satfinite(3.0 / 1024.0), 2);
        // E8M0: a block max of 448 scales by 1, of 449 by 1/2, of 1 by 2^-... keeps the max ≤ 448.
        assert_eq!(mx_e8m0(448.0), (127, 1.0));
        assert_eq!(mx_e8m0(449.0), (128, 0.5));
        for amax in [1e-30f32, 0.003, 0.9, 1.0, 7.5, 300.0, 448.0, 1e4, 3e38] {
            let (byte, inv) = mx_e8m0(amax);
            assert!(amax * inv <= 448.0, "{amax}");
            assert_eq!(inv, 2f32.powi(127 - i32::from(byte)));
        }
        // The swizzle covers every byte of a padded [rows, k] scale tensor exactly once.
        let (rows, k) = (300usize, 7usize * 32);
        let cb = (k / 32).div_ceil(4);
        let mut seen = vec![false; mx_scale_len(rows.div_ceil(128) * 128, k)];
        for r in 0..rows.div_ceil(128) * 128 {
            for g in 0..cb * 4 {
                let o = mx_scale_offset(r, g, cb);
                assert!(!seen[o], "({r}, {g}) collides");
                seen[o] = true;
            }
        }
        assert!(seen.iter().all(|&s| s));
        // Quantize → dequantize is within half an E4M3 ulp (1/16 relative, or half a
        // subnormal quantum of the block scale).
        let mut rng = crate::rng::Rng::new(3);
        let (rows, cols) = (37, 96);
        let w: Vec<f32> = rng.normal(rows * cols).iter().map(|v| v * 0.3).collect();
        let q = Mxfp8::quantize(&w, rows, cols);
        assert_eq!(q.scales.len(), 128 * 4);
        let d = q.dequantize();
        for r in 0..rows {
            for c in 0..cols {
                let s = q.scale_of(r, c / 32);
                let v = w[r * cols + c];
                assert!((d[r * cols + c] - v).abs() <= v.abs() / 16.0 + s / 1024.0 + 1e-7, "({r}, {c})");
            }
        }
        // concat re-lays the scale tiles: equal to quantizing the stacked matrix.
        let (a, b) = (rng.normal(37 * 96), rng.normal(100 * 96));
        let stacked: Vec<f32> = a.iter().chain(&b).copied().collect();
        assert_eq!(Mxfp8::concat(vec![Mxfp8::quantize(&a, 37, 96), Mxfp8::quantize(&b, 100, 96)]), Mxfp8::quantize(&stacked, 137, 96));
        assert_eq!(WeightQuant::Mxfp8.for_layer_key("transformer_blocks.1.ff.net.2", 50), WeightQuant::Native);
        assert_eq!(WeightQuant::Mxfp8.for_layer_key("transformer_blocks.2.ff.net.2", 50), WeightQuant::Mxfp8);
        assert_eq!(WeightQuant::Mxfp8.for_layer_key("transformer_blocks.2.norm1", 50), WeightQuant::Native);
        // round_rows is quantize ∘ dequantize of the bf16-rounded rows.
        let x: Vec<f32> = rng.normal(5 * 64);
        let rr = Mxfp8::round_rows(&x, 64);
        let xb: Vec<f32> = x.iter().map(|&v| bf16_round(v)).collect();
        assert_eq!(rr, Mxfp8::quantize(&xb, 5, 64).dequantize());
        // Block routing: the first two and the last three blocks stay native.
        assert_eq!(WeightQuant::Mxfp8.for_block(1, 50), WeightQuant::Native);
        assert_eq!(WeightQuant::Mxfp8.for_block(2, 50), WeightQuant::Mxfp8);
        assert_eq!(WeightQuant::Mxfp8.for_block(46, 50), WeightQuant::Mxfp8);
        assert_eq!(WeightQuant::Mxfp8.for_block(47, 50), WeightQuant::Native);
        assert_eq!(WeightQuant::Mxfp8.for_block(38, 42), WeightQuant::Mxfp8);
        assert_eq!(WeightQuant::Mxfp8.for_block(39, 42), WeightQuant::Native);
        assert_eq!(WeightQuant::Int8.for_block(0, 50), WeightQuant::Int8);
        assert_eq!("mxfp8".parse::<WeightQuant>(), Ok(WeightQuant::Mxfp8));
        assert_eq!(WeightQuant::Mxfp8.to_string(), "mxfp8");
    }

    #[test]
    fn for_key_routes_block_linears_only() {
        let q = WeightQuant::Int8;
        assert_eq!(q.for_key("transformer_blocks.3.ff.net.0.proj"), WeightQuant::Int8);
        assert_eq!(q.for_key("transformer_blocks.3.attn.to_gate_compress"), WeightQuant::Int8);
        assert_eq!(q.for_key("transformer_blocks.3.adaln_proj.linear"), WeightQuant::Native);
        assert_eq!(q.for_key("token_refiner.refiner_blocks.0.attn.to_q"), WeightQuant::Native);
        assert_eq!(q.for_key("proj_in"), WeightQuant::Native);
    }

    #[test]
    fn e4m3_and_e2m1_decode() {
        assert_eq!(e4m3_to_f32(0x38), 1.0);
        assert_eq!(e4m3_to_f32(0x40), 2.0);
        assert_eq!(e4m3_to_f32(0xC0), -2.0);
        assert_eq!(e4m3_to_f32(0x7E), 448.0);
        assert_eq!(e4m3_to_f32(0x01), 2f32.powi(-9));
        assert_eq!(E2M1[7], 6.0);
        assert_eq!(E2M1[9], -0.5);
    }

    #[test]
    fn nvfp4_unswizzle_is_a_permutation_and_dequantizes() {
        // 256 rows × 32 cols → 2 groups (padded to 4), 2 row blocks.
        let (rows, cols) = (256usize, 32usize);
        let groups = cols / 16;
        let (rp, gp) = (256, 4);
        // Fill the swizzled buffer from the logical (row, group) so the
        // inverse must recover them.
        let mut sw = vec![0u8; rp * gp];
        for r in 0..rows {
            let (i0, rem) = (r / 128, r % 128);
            let (i3, i2) = (rem / 32, rem % 32);
            for g in 0..groups {
                let (i1, i4) = (g / 4, g % 4);
                let idx = Nvfp4::swizzled_index(i0, i1, i2, i3, i4, gp / 4);
                sw[idx] = ((r * 3 + g * 7) % 127) as u8 | 0x38; // ≈ 1..2 range
            }
        }
        // Not the identity: row 32's scales sit after the first 32 rows' first
        // group block, not at 32·groups.
        assert_ne!(Nvfp4::swizzled_index(0, 0, 0, 1, 0, gp / 4), 32 * gp);
        let scales = Nvfp4::unswizzle_scales(&sw, rows, groups);
        for r in 0..rows {
            for g in 0..groups {
                assert_eq!(scales[r * groups + g], ((r * 3 + g * 7) % 127) as u8 | 0x38);
            }
        }
        let packed: Vec<u8> = (0..rows * cols / 2).map(|i| (i % 256) as u8).collect();
        let q = Nvfp4::from_stored(&packed, &sw, 0.5, rows, cols, false);
        let d = q.dequantize();
        assert_eq!(d.len(), rows * cols);
        // Column 1 of row 0: code = high nibble of byte 0 = 0 → 0.
        assert_eq!(d[1], 0.0);
        // Column 3 of row 0: byte 1 = 0x01 → high nibble 0; column 2 → low nibble 1 → 0.5.
        assert_eq!(d[2], 0.5 * e4m3_to_f32(q.scales[0]) * 0.5);
        // ComfyUI order: the same bytes read first-high put the 1 in column 3.
        let qh = Nvfp4::from_stored(&packed, &sw, 0.5, rows, cols, true);
        let dh = qh.dequantize();
        assert_eq!(dh[2], 0.0);
        assert_eq!(dh[3], 0.5 * e4m3_to_f32(q.scales[0]) * 0.5);
        // Row slices and the half swap keep every row's bytes together.
        // (compared as bits: the synthetic scale bytes include NaN patterns)
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        let part = q.rows(100, 140);
        assert_eq!(bits(&part.dequantize()), bits(&d[100 * cols..140 * cols]));
        let sw2 = q.swap_halves();
        assert_eq!(bits(&sw2.dequantize()[..cols]), bits(&d[(rows / 2) * cols..(rows / 2 + 1) * cols]));
    }
}
