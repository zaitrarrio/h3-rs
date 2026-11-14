//! The op vocabulary every backend implements.
//!
//! Fused and small, as in fastwan-rs: each op is one kernel on a GPU, and the
//! set is exactly what an H3 denoising step, the two VAEs, TAEH3, the latent
//! upscaler and the Qwen3-VL conditioner need. Shapes are passed as integers,
//! never stored; buffers are flat and contiguous.
//!
//! Activation precision is the backend's business (`f32` on CPU and WebGPU,
//! `bf16` on CUDA): [`Backend::upload`] / [`Backend::download`] convert at the
//! boundary. Lookup tables ([`Backend::upload_table`]) stay f32 everywhere.
//!
//! Aliasing rule: an output buffer never aliases an input, except the
//! arguments documented as *in place*.

use crate::quant::{Mxfp8, Nvfp4, Q4, Q8};
use crate::vsa::VsaSpec;

/// Epilogue activation of [`Backend::linear`] and [`Backend::group_norm`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Act {
    None,
    /// `0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))` (`gelu(approximate="tanh")`).
    GeluTanh,
    /// `0.5·x·(1 + erf(x/√2))` (exact GELU; the VAE decoder FFN and Qwen3-VL merger).
    GeluErf,
    /// `x·σ(x)`.
    Silu,
}

/// A per-row modulation table: `table` is `[entries, slots, dim]` and
/// `idx[row]` (u32) picks the entry of a row. One table serves every row of a
/// packed sequence whatever its timestep or modality.
#[derive(Debug)]
pub struct ModTable<'a, T> {
    pub table: &'a T,
    pub idx: &'a T,
    pub slots: usize,
}

/// `ModTable` + the slot read by a gate.
#[derive(Debug)]
pub struct ModRef<'a, T> {
    pub table: ModTable<'a, T>,
    pub slot: usize,
}

/// How a residual is scaled before it is added.
#[derive(Debug)]
pub enum Gate<'a, T> {
    /// `h += a`.
    None,
    /// `h += a · g[c]` for a `[dim]` vector (the VAE decoder's `scale1` /
    /// `scale2`).
    Vector(&'a T),
    /// `h += a · table[idx[row]][slot][c]` (AdaLN gate).
    Table(ModRef<'a, T>),
}

/// The modulation applied after an RMSNorm.
#[derive(Debug)]
pub enum Modulation<'a, T> {
    None,
    /// `y · (1 + table[idx][scale]) + table[idx][shift]`.
    Table { m: ModTable<'a, T>, scale: usize, shift: usize },
}

/// Per-head normalisation inside [`Backend::head_split`]: RMSNorm over the
/// `head_dim` channels of every head, optionally with a `[head_dim]` weight.
#[derive(Debug)]
pub struct HeadNorm<'a, T> {
    pub w: Option<&'a T>,
    pub eps: f32,
}

/// Rotary tables for [`Backend::head_split`]: `cos` / `sin` are
/// `[rows, rot / 2]` and the first `rot` channels of a head are rotated
/// (rotate-half: `(x₁, x₂) → (x₁·c − x₂·s, x₂·c + x₁·s)` with `x₂ = x₁ + rot/2`).
#[derive(Debug)]
pub struct Rope<'a, T> {
    pub cos: &'a T,
    pub sin: &'a T,
    pub rot: usize,
}

/// One flow-matching update ([`crate::schedule`]). `v` is the model's
/// velocity, `x0 = x + σ·v`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FlowStep {
    /// `x = x + (σ − σ')·v`.
    Euler { sigma: f32, next: f32 },
    /// `x = (1 − σ')·x0 + σ'·noise` (Pika's renoise; `noise` required).
    Renoise { sigma: f32, next: f32 },
    /// `x = x0`.
    Terminal { sigma: f32 },
}

/// Epilogue of a convolution, after the bias.
#[derive(Debug)]
pub enum ConvEpilogue<'a, T> {
    None,
    Relu,
    /// `relu(y + r)` with `r` shaped like the output (TAEH3 MemBlock skip).
    AddRelu(&'a T),
    /// `y + r` (ResNet skip).
    Add(&'a T),
}

/// A 2-D convolution over `frames` independent images, NCHW, stride 1, "same"
/// padding (`k / 2`) — fastwan-rs's TAEHV op, reused for TAEH3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvShape {
    pub frames: usize,
    /// Channels read from the first input.
    pub cin_a: usize,
    /// Channels read from the optional second input, concatenated after the
    /// first (`cat([a, b], dim=1)` without materialising it).
    pub cin_b: usize,
    pub cout: usize,
    pub h: usize,
    pub w: usize,
    /// Read the inputs through a nearest ×2 upsample (output is `2h × 2w`).
    pub upsample: bool,
    /// 1 or 3.
    pub k: usize,
}

impl ConvShape {
    pub fn cin(&self) -> usize {
        self.cin_a + self.cin_b
    }
    pub fn out_hw(&self) -> (usize, usize) {
        let s = if self.upsample { 2 } else { 1 };
        (self.h * s, self.w * s)
    }
    pub fn out_len(&self) -> usize {
        let (h, w) = self.out_hw();
        self.frames * self.cout * h * w
    }
    pub fn k_len(&self) -> usize {
        self.cin() * self.k * self.k
    }
}

/// Spatial padding of a 3-D convolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pad {
    Zero,
    /// PyTorch `reflect`: the edge is not repeated.
    Reflect,
}

/// A 3-D convolution over one `[cin, t, h, w]` volume: weight `[cout, cin /
/// groups, kt, kh, kw]`, explicit padding (temporal front/back, spatial
/// top/bottom/left/right), strides. H3's causal VAE pads the time axis with
/// zeros in front only; the spatial padding reflects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv3dShape {
    pub cin: usize,
    pub cout: usize,
    pub groups: usize,
    pub t: usize,
    pub h: usize,
    pub w: usize,
    pub k: (usize, usize, usize),
    pub stride: (usize, usize, usize),
    /// `(front, back)`; always zero-padded.
    pub pad_t: (usize, usize),
    /// `(top, bottom)`.
    pub pad_h: (usize, usize),
    /// `(left, right)`.
    pub pad_w: (usize, usize),
    pub spatial_pad: Pad,
}

impl Conv3dShape {
    /// A `k³` convolution, stride 1, H3-VAE padding (causal temporal, reflect spatial).
    pub fn causal(cin: usize, cout: usize, t: usize, h: usize, w: usize, k: usize) -> Self {
        Self {
            cin,
            cout,
            groups: 1,
            t,
            h,
            w,
            k: (k, k, k),
            stride: (1, 1, 1),
            pad_t: (k - 1, 0),
            pad_h: (k / 2, k / 2),
            pad_w: (k / 2, k / 2),
            spatial_pad: Pad::Reflect,
        }
    }
    /// A `k³` convolution, stride 1, symmetric zero padding (the latent upscaler).
    pub fn same(cin: usize, cout: usize, t: usize, h: usize, w: usize, k: usize) -> Self {
        Self {
            pad_t: (k / 2, k / 2),
            spatial_pad: Pad::Zero,
            ..Self::causal(cin, cout, t, h, w, k)
        }
    }
    pub fn out_dims(&self) -> (usize, usize, usize) {
        let o = |n: usize, k: usize, s: usize, p: (usize, usize)| (n + p.0 + p.1 - k) / s + 1;
        (
            o(self.t, self.k.0, self.stride.0, self.pad_t),
            o(self.h, self.k.1, self.stride.1, self.pad_h),
            o(self.w, self.k.2, self.stride.2, self.pad_w),
        )
    }
    pub fn out_len(&self) -> usize {
        let (t, h, w) = self.out_dims();
        self.cout * t * h * w
    }
    pub fn in_len(&self) -> usize {
        self.cin * self.t * self.h * self.w
    }
    pub fn weight_len(&self) -> usize {
        self.cout * (self.cin / self.groups) * self.k.0 * self.k.1 * self.k.2
    }
    /// Reduction length of the implicit GEMM.
    pub fn k_len(&self) -> usize {
        (self.cin / self.groups) * self.k.0 * self.k.1 * self.k.2
    }
}

/// A 1-D convolution over one `[cin, len]` signal (weight `[cout, cin / groups, k]`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Conv1dShape {
    pub cin: usize,
    pub cout: usize,
    pub groups: usize,
    pub len: usize,
    pub k: usize,
    pub stride: usize,
    pub dilation: usize,
    pub pad: (usize, usize),
}

impl Conv1dShape {
    pub fn out_len_1d(&self) -> usize {
        (self.len + self.pad.0 + self.pad.1 - self.dilation * (self.k - 1) - 1) / self.stride + 1
    }
    pub fn out_len(&self) -> usize {
        self.cout * self.out_len_1d()
    }
    pub fn weight_len(&self) -> usize {
        self.cout * (self.cin / self.groups) * self.k
    }
}

/// A transposed 1-D convolution (PyTorch `ConvTranspose1d`, weight
/// `[cin, cout / groups, k]`, output `(len − 1)·stride − 2·pad + k`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConvT1dShape {
    pub cin: usize,
    pub cout: usize,
    pub groups: usize,
    pub len: usize,
    pub k: usize,
    pub stride: usize,
    pub pad: usize,
}

impl ConvT1dShape {
    pub fn out_len_1d(&self) -> usize {
        (self.len - 1) * self.stride + self.k - 2 * self.pad
    }
    pub fn out_len(&self) -> usize {
        self.cout * self.out_len_1d()
    }
    pub fn weight_len(&self) -> usize {
        self.cin * (self.cout / self.groups) * self.k
    }
}

/// Snake activations of the audio VAE.
#[derive(Debug)]
pub enum Snake<'a, T> {
    /// DAC `Snake1d`: `x + sin²(α·x) / (α + 1e-9)`, `α` `[c]`.
    Alpha(&'a T),
    /// BigVGAN `SnakeBeta` (log scale): `x + sin²(e^α·x) / (e^β + 1e-9)`.
    AlphaBeta { alpha: &'a T, beta: &'a T },
}

impl<T> Clone for ModTable<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for ModTable<'_, T> {}
impl<T> Clone for ModRef<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for ModRef<'_, T> {}
impl<T> Clone for Gate<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Gate<'_, T> {}
impl<T> Clone for Modulation<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Modulation<'_, T> {}
impl<T> Clone for HeadNorm<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for HeadNorm<'_, T> {}
impl<T> Clone for Rope<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Rope<'_, T> {}
impl<T> Clone for ConvEpilogue<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for ConvEpilogue<'_, T> {}
impl<T> Clone for Snake<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Snake<'_, T> {}

pub trait Backend {
    /// A flat device buffer of activations.
    type Buf;
    /// Device state of [`Backend::vsa_attention`] for one layout.
    type Vsa;
    /// A `[rows, cols]` row-major matrix (`nn.Linear` layout `[out, in]`), or
    /// a convolution weight flattened to `[cout, k_len]`, in the backend's
    /// storage precision.
    type Weight;

    fn name(&self) -> String;

    /// The most f32 elements one buffer may hold (WebGPU caps a buffer at
    /// 4 GiB; the model chunks its FFN hidden to stay under it).
    fn max_buffer_elems(&self) -> usize {
        usize::MAX
    }

    // ---- memory ----------------------------------------------------------
    /// Zero-filled.
    fn alloc(&self, len: usize) -> Self::Buf;
    fn upload(&self, data: &[f32]) -> Self::Buf;
    /// A table the kernels read in full precision even on a backend whose
    /// activations are narrower (RoPE cos/sin, modulation tables).
    fn upload_table(&self, data: &[f32]) -> Self::Buf {
        self.upload(data)
    }
    /// Row indices and other integer side tables.
    fn upload_u32(&self, data: &[u32]) -> Self::Buf;
    /// Overwrite the first `data.len()` elements.
    fn write(&self, buf: &Self::Buf, data: &[f32]);
    /// Blocking read of the whole buffer.
    fn download(&self, buf: &Self::Buf) -> Vec<f32>;
    fn buf_len(&self, buf: &Self::Buf) -> usize;
    /// `dst[dst_off..][..len] = src[src_off..][..len]`.
    fn copy(&self, src: &Self::Buf, src_off: usize, dst: &Self::Buf, dst_off: usize, len: usize);
    /// `buf[off..off + len] = value`.
    fn fill(&self, buf: &Self::Buf, off: usize, len: usize, value: f32);
    fn weight(&self, data: &[f32], rows: usize, cols: usize) -> Self::Weight;
    /// [`Backend::weight`] from bf16 bits: streamed bf16 blocks keep their
    /// host copy in this form so every step's upload is a byte copy (a
    /// page-locked DMA on CUDA) instead of an f32 → bf16 pass plus a
    /// synchronous pageable copy. The default converts and calls `weight`.
    fn weight_bf16(&self, data: &[u16], rows: usize, cols: usize) -> Self::Weight {
        let f: Vec<f32> = data.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
        self.weight(&f, rows, cols)
    }
    /// Whether [`Backend::weight`] stores bf16 anyway, so a streamed block
    /// loses nothing by keeping its host copy as bf16 bits. False for the
    /// f32 reference (its streamed and resident paths stay bit-identical).
    fn bf16_weights(&self) -> bool {
        false
    }
    /// Int8 weight-only matrix for [`Backend::linear`]; must equal `linear`
    /// with `weight(&q.dequantize())` up to the backend's activation rounding.
    fn weight_q8(&self, q: &Q8) -> Self::Weight;
    /// 4-bit weight-only matrix, same contract.
    fn weight_q4(&self, q: &Q4) -> Self::Weight;
    /// NVFP4 (E2M1 codes, E4M3 block scales); backends without FP4 tensor
    /// cores dequantize at load.
    fn weight_nvfp4(&self, q: &Nvfp4) -> Self::Weight {
        let w = q.dequantize();
        self.weight(&w, q.rows, q.cols)
    }
    /// MXFP8 W8A8 ([`Mxfp8`]): `linear` with this weight must equal the f32
    /// GEMM of `q.dequantize()` against [`Mxfp8::round_rows`] of the bf16
    /// activations (the CPU reference does exactly that; CUDA runs the FP8
    /// tensor cores on sm_100+ and emulates below). Backends without an FP8
    /// path dequantize the weight at load and round the activations per GEMM.
    fn weight_mxfp8(&self, q: &Mxfp8) -> Self::Weight;
    /// Bytes a weight occupies on the device.
    fn weight_bytes(&self, w: &Self::Weight) -> usize;
    // ---- streamed weights ----------------------------------------------
    /// Uploads between `prefetch_begin` and `prefetch_end` may land on a side
    /// queue that overlaps the compute queue; they are usable only after
    /// [`Backend::prefetch_wait`]. Backends with one queue do nothing.
    fn prefetch_begin(&self) {}
    fn prefetch_end(&self) {}
    /// The compute queue waits for the last prefetch.
    fn prefetch_wait(&self) {}
    /// The side queue waits for the compute queue's work so far: call before
    /// dropping prefetched weights the compute queue may still be reading.
    fn prefetch_release(&self) {}
    /// Page-lock host memory that streamed uploads will read every step, so
    /// the copies run as DMA without a host-side wait. The returned guard
    /// unlocks on drop and must be dropped before the memory is freed (the
    /// caller keeps it next to the buffer). Best effort; `None` by default.
    fn pin_host(&self, _bytes: &[u8]) -> Option<Box<dyn std::any::Any + Send>> {
        None
    }
    /// Wait for queued work (timing only; downloads synchronise themselves).
    fn sync(&self);
    /// Per-kernel device time since the last [`Backend::profile_reset`], when
    /// profiling is on (`H3_PROFILE=1`).
    fn profile_report(&self) -> Option<String> {
        None
    }
    fn profile_reset(&self) {}

    // ---- transformer -----------------------------------------------------
    /// `y[y_row + i, :] = act(x[x_row + i, :] · Wᵀ + b)` for `i < rows`.
    #[allow(clippy::too_many_arguments)]
    fn linear(
        &self,
        x: &Self::Buf,
        x_row: usize,
        rows: usize,
        w: &Self::Weight,
        bias: Option<&Self::Buf>,
        act: Act,
        y: &Self::Buf,
        y_row: usize,
    );

    /// `out[i, :dim] = x[i, :dim] · silu(x[i, dim:2·dim])` (H3 / Qwen3-VL
    /// `value, gate` halves; `x` rows are `2·dim` wide).
    fn swiglu(&self, x: &Self::Buf, rows: usize, dim: usize, out: &Self::Buf);

    /// `out = rms(x) · w`, then [`Modulation`]. `rms(x) = x · rsqrt(mean(x²) + eps)`.
    #[allow(clippy::too_many_arguments)]
    fn rms_norm(
        &self,
        x: &Self::Buf,
        rows: usize,
        dim: usize,
        w: Option<&Self::Buf>,
        modulation: Modulation<'_, Self::Buf>,
        eps: f32,
        out: &Self::Buf,
    );

    /// In place on `h`: `h += gate(a)`, then `normed = rms_norm(h)` with `w`
    /// and `modulation` as in [`Backend::rms_norm`]. The norm reads the
    /// unrounded sum.
    #[allow(clippy::too_many_arguments)]
    fn residual_rms_norm(
        &self,
        h: &Self::Buf,
        a: &Self::Buf,
        rows: usize,
        dim: usize,
        gate: Gate<'_, Self::Buf>,
        w: Option<&Self::Buf>,
        modulation: Modulation<'_, Self::Buf>,
        eps: f32,
        normed: &Self::Buf,
    );

    /// In place on `h`: `h += gate(a)`.
    fn gate_residual(&self, h: &Self::Buf, a: &Self::Buf, rows: usize, dim: usize, gate: Gate<'_, Self::Buf>);

    /// `out = LN(x) · w + b` over rows of width `dim`.
    #[allow(clippy::too_many_arguments)]
    fn layer_norm(
        &self,
        x: &Self::Buf,
        rows: usize,
        dim: usize,
        w: Option<&Self::Buf>,
        b: Option<&Self::Buf>,
        eps: f32,
        out: &Self::Buf,
    );

    /// Columns `[col_off, col_off + heads·head_dim)` of `src` (`rows` rows of
    /// width `src_stride`) → `out` as `[heads, rows, head_dim]`, optionally
    /// RMS-normalised per head and rotated.
    #[allow(clippy::too_many_arguments)]
    fn head_split(
        &self,
        src: &Self::Buf,
        rows: usize,
        src_stride: usize,
        col_off: usize,
        heads: usize,
        head_dim: usize,
        norm: Option<HeadNorm<'_, Self::Buf>>,
        rope: Option<Rope<'_, Self::Buf>>,
        out: &Self::Buf,
    );

    /// Softmax attention, scale `1/√head_dim`. `q` is `[heads, sq, d]`,
    /// `k`/`v` `[kv_heads, sk, d]` (query head `i` reads KV head
    /// `i / (heads / kv_heads)`); `causal` masks `j > i + (sk − sq)`. `out` is
    /// `[sq, heads·d]`, ready for `to_out`.
    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        q: &Self::Buf,
        k: &Self::Buf,
        v: &Self::Buf,
        heads: usize,
        kv_heads: usize,
        sq: usize,
        sk: usize,
        head_dim: usize,
        causal: bool,
        out: &Self::Buf,
    );
    /// [`Backend::attention`] with INT8 Q·Kᵀ: must equal `attention` of
    /// [`crate::vsa::attn_qk8_round`]'s copies of Q and K up to the backend's
    /// rounding (CUDA runs int8 tensor cores on the same codes; the CPU and
    /// WebGPU round and run the exact kernel). With `pv8` the P·V half runs
    /// in FP8 too and the oracle is [`crate::vsa::attention_pv8_reference`].
    #[allow(clippy::too_many_arguments)]
    fn attention_qk8(
        &self,
        q: &Self::Buf,
        k: &Self::Buf,
        v: &Self::Buf,
        heads: usize,
        kv_heads: usize,
        sq: usize,
        sk: usize,
        head_dim: usize,
        causal: bool,
        pv8: bool,
        out: &Self::Buf,
    );

    /// Whether [`Backend::vsa_attention`] can run at this head dim.
    fn vsa_supported(&self, _head_dim: usize) -> bool {
        true
    }

    /// Build the VSA state for one layout ([`crate::vsa`]).
    fn vsa_prepare(&self, spec: &VsaSpec) -> Self::Vsa;

    /// Video sparse attention ([`crate::vsa`]). `q`, `k`, `v`, `gate` are
    /// `[heads, seq, d]`; `out` is `[seq, heads·d]`. Without `gate` the
    /// pooled branch is skipped.
    fn vsa_attention(
        &self,
        q: &Self::Buf,
        k: &Self::Buf,
        v: &Self::Buf,
        gate: Option<&Self::Buf>,
        vsa: &Self::Vsa,
        out: &Self::Buf,
    );

    /// Latent `[c, t, h, w]` → rows `[(t/pt)·(h/ph)·(w/pw), c·pt·ph·pw]`,
    /// feature `((c·pt + kt)·ph + kh)·pw + kw` (channel-major: a checkpoint
    /// contract of H3). Rows are written at `out_row`.
    #[allow(clippy::too_many_arguments)]
    fn patchify(
        &self,
        latent: &Self::Buf,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        patch: (usize, usize, usize),
        out: &Self::Buf,
        out_row: usize,
    );

    /// Rows `x[x_row..]` (features channel-major as above, `x_stride` wide,
    /// read from column `col`) → latent `[c, t, h, w]`.
    #[allow(clippy::too_many_arguments)]
    fn unpatchify(
        &self,
        x: &Self::Buf,
        x_row: usize,
        x_stride: usize,
        col: usize,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        patch: (usize, usize, usize),
        out: &Self::Buf,
    );

    /// In place on `x[off..off + n]` with `v[v_off..]` and `noise[off..]`.
    #[allow(clippy::too_many_arguments)]
    fn flow_step(
        &self,
        x: &Self::Buf,
        off: usize,
        v: &Self::Buf,
        v_off: usize,
        noise: Option<&Self::Buf>,
        n: usize,
        step: FlowStep,
    );

    /// In place: `y[off + i] = a·y[off + i] + b·x[x_off + i]` for `i < n`.
    #[allow(clippy::too_many_arguments)]
    fn axpby(&self, y: &Self::Buf, off: usize, a: f32, x: &Self::Buf, x_off: usize, b: f32, n: usize);

    /// `out[c, i] = x[c, i] · scale[c] + shift[c]` over `[c, inner]`.
    #[allow(clippy::too_many_arguments)]
    fn channel_affine(
        &self,
        x: &Self::Buf,
        c: usize,
        inner: usize,
        scale: &Self::Buf,
        shift: Option<&Self::Buf>,
        out: &Self::Buf,
    );

    // ---- convolutions ------------------------------------------------------
    /// `out = epilogue(conv2d(cat([a, b]), W) + bias)`; `W` is `[cout, cin·k²]`.
    #[allow(clippy::too_many_arguments)]
    fn conv2d(
        &self,
        a: &Self::Buf,
        b: Option<&Self::Buf>,
        shape: &ConvShape,
        w: &Self::Weight,
        bias: Option<&Self::Buf>,
        epilogue: ConvEpilogue<'_, Self::Buf>,
        out: &Self::Buf,
    );

    /// `out = epilogue(conv3d(x, W) + bias)`; `W` is `[cout, k_len]`.
    fn conv3d(
        &self,
        x: &Self::Buf,
        shape: &Conv3dShape,
        w: &Self::Weight,
        bias: Option<&Self::Buf>,
        epilogue: ConvEpilogue<'_, Self::Buf>,
        out: &Self::Buf,
    );

    /// `out = epilogue(conv1d(x, W) + bias)`; `W` is `[cout, (cin/groups)·k]`.
    fn conv1d(
        &self,
        x: &Self::Buf,
        shape: &Conv1dShape,
        w: &Self::Weight,
        bias: Option<&Self::Buf>,
        epilogue: ConvEpilogue<'_, Self::Buf>,
        out: &Self::Buf,
    );

    /// `out[c, i] = x[c, clamp(i − left, 0, len − 1)]` over `[c, len + left + right]`
    /// (PyTorch `replicate` padding; the audio VAE's anti-aliasing filters).
    fn replicate_pad1d(&self, x: &Self::Buf, c: usize, len: usize, left: usize, right: usize, out: &Self::Buf);

    /// `out = conv_transpose1d(x, W) + bias`; `W` is `[cin, (cout/groups)·k]`
    /// (PyTorch layout, rows = input channels).
    fn conv_transpose1d(
        &self,
        x: &Self::Buf,
        shape: &ConvT1dShape,
        w: &Self::Weight,
        bias: Option<&Self::Buf>,
        out: &Self::Buf,
    );

    /// GroupNorm over `[c, t, hw]`: with `per_frame`, statistics per `(frame,
    /// group)` over `(c/groups)·hw` (H3's time-isolated encoder norm); else
    /// per group over `(c/groups)·t·hw`. Then `· w[c] + b[c]`, then `act`.
    #[allow(clippy::too_many_arguments)]
    fn group_norm(
        &self,
        x: &Self::Buf,
        c: usize,
        t: usize,
        hw: usize,
        groups: usize,
        per_frame: bool,
        w: &Self::Buf,
        b: &Self::Buf,
        eps: f32,
        act: Act,
        out: &Self::Buf,
    );

    /// In place over `[c, len]`.
    fn snake(&self, x: &Self::Buf, c: usize, len: usize, snake: Snake<'_, Self::Buf>);

    /// Trilinear resize of `[c, t, h, w]` to `[c, t2, h2, w2]`
    /// (`align_corners=False`).
    #[allow(clippy::too_many_arguments)]
    fn trilinear_resize(
        &self,
        x: &Self::Buf,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        t2: usize,
        h2: usize,
        w2: usize,
        out: &Self::Buf,
    );

    /// `[frames, c·r², h, w]` → `[frames, c, h·r, w·r]` (PyTorch `pixel_shuffle`).
    #[allow(clippy::too_many_arguments)]
    fn pixel_shuffle(&self, x: &Self::Buf, frames: usize, c: usize, h: usize, w: usize, r: usize, out: &Self::Buf);

    /// In place: `x = tanh(x / s) · s` over `n` elements.
    fn tanh_clamp(&self, x: &Self::Buf, n: usize, s: f32);

    /// In place: `x = clamp(x, lo, hi)` over `n` elements.
    fn clamp(&self, x: &Self::Buf, n: usize, lo: f32, hi: f32);
}
