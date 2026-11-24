//! CUDA backend: cudarc (driver loaded at run time, so this builds on a
//! machine without CUDA), kernels compiled by NVRTC at start-up for the
//! device's own SM, GEMMs on cuBLAS / cuBLASLt in bf16 with f32 accumulation
//! (bf16 tensor cores need Ampere, sm_80+).
//!
//! Activations and weights are bf16; RoPE / modulation tables, VSA state and
//! row indices stay 32-bit ([`Backend::upload_table`], [`Backend::upload_u32`]).
//! Every kernel is the CUDA twin of an h3-cpu op; `h3 check --backend cuda`
//! runs [`h3_model::check::run_all`] against the CPU reference on a GPU.
//!
//! Environment: `H3_PROFILE=1` per-kernel times, `H3_CUDA_NO_LT=1` plain
//! cuBLAS + `fw_bias_act` instead of cuBLASLt epilogues, `H3_CUDA_SCALAR_ATTN=1`
//! / `H3_CUDA_SCALAR_CONV=1` the scalar kernels (for A/B against the mma ones).

#![allow(clippy::too_many_arguments)]

use cudarc::cublas::{CudaBlas, sys as blas_sys};
use cudarc::cublaslt::{result as lt_result, sys as lt_sys};
use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig,
    PushKernelArg,
};
use h3_core::backend::{
    Act, Backend, BoxRows, Conv1dShape, Conv3dShape, ConvEpilogue, ConvShape, ConvT1dShape,
    FlowStep, Gate, HeadNorm, Modulation, Pad, Rope, Snake,
};
use h3_core::quant::{Mxfp8, Nvfp4, Q4, Q8, mx_scale_len};
use h3_core::vsa::VsaSpec;
use half::bf16;
use rayon::prelude::*;
use std::sync::Arc;

const KERNELS: &str = include_str!("kernels.cu");

#[derive(Clone)]
pub struct CuBuf {
    s: Arc<CudaSlice<u16>>,
    len: usize,
    /// Stored as 32-bit words (2 u16 per element): f32 tables and u32 indices.
    f32: bool,
}

/// Token chunks of the two-stage per-head K mean (`fw_vsa_kmean` / `fw_vsa_kmean2`).
const KMEAN_CHUNKS: usize = 128;

/// A page-locked host range (`pin_host`); unlocked and forgotten on drop.
struct PinGuard {
    start: usize,
    end: usize,
    ranges: Arc<std::sync::Mutex<Vec<(usize, usize)>>>,
}

impl Drop for PinGuard {
    fn drop(&mut self) {
        // SAFETY: the range was registered by `pin_host` and is still mapped
        // (the guard lives next to its buffer and drops first).
        unsafe { cudarc::driver::sys::cuMemHostUnregister(self.start as *mut _) };
        self.ranges
            .lock()
            .unwrap()
            .retain(|&(s, e)| (s, e) != (self.start, self.end));
    }
}

/// VSA state: tile tables (u32), the region table, f32 pooled means and
/// pooled-branch output, selections and counts.
pub struct CuVsa {
    slot: CuBuf,
    bsz: CuBuf,
    reg: CuBuf,
    pooled: CuBuf,
    pooled_out: CuBuf,
    sel: CuBuf,
    nsel: CuBuf,
    /// Paired-tile unions (`fw_vsa_pair`): `[heads][npairs][ustride]`, counts.
    usel: CuBuf,
    nusel: CuBuf,
    npairs: usize,
    ustride: usize,
    nb: usize,
    np: usize,
    nreg: usize,
    dense_all: bool,
    /// `slot_of_row`: each token's padded slot (tile = slot / te), for the
    /// dense-plan path's gate add.
    rowslot: CuBuf,
    maxsel: usize,
    te: usize,
    heads: usize,
    d: usize,
    seq: usize,
    scale: f32,
    /// INT8 Q·Kᵀ: codes `[heads][seq][d]` (bytes in u16 words), per-(head,
    /// tile) scales, the per-head K mean.
    qk8: Option<(CuBuf, CuBuf, CuBuf, CuBuf, CuBuf, CuBuf)>,
    /// FP8 P·V: V codes transposed per head to `[d][nb·te]` in slot order
    /// (bytes in u16 words), the per-head V mean, chunk maxima, per-head scale.
    pv8: Option<(CuBuf, CuBuf, CuBuf, CuBuf)>,
    /// FP8 P·V asked for a head size without an FP8 fine kernel (only 64 and
    /// 128 have one), or with the fine3 path switched off: the call runs the
    /// host oracle, as the dense path does. Pre-sm_89 cards warn instead and
    /// run bf16 P·V (a real clip through the oracle would take hours).
    host_pv8: Option<VsaSpec>,
}

pub struct CuWeight {
    buf: CuBuf,
    rows: usize,
    cols: usize,
    /// Weight-only quantized: `buf` holds the codes (int8: 2 per word, int4:
    /// 4 per word), this table the f32 per-group scales (int8) or (scale,
    /// zero) pairs (int4), then the group. [`Backend::linear`] expands them
    /// into the shared bf16 scratch before the GEMM.
    quant: Option<(Quant, CuBuf, usize)>,
}

#[derive(Clone, Copy, PartialEq)]
enum Quant {
    Q8,
    Q4,
    /// E2M1 codes (2 per byte) with E4M3 per-16 scales in the table buffer.
    /// `act4`: W4A4, the codes and scales in cuBLASLt's FP4 layout (even
    /// column low, `VEC16_UE4M3` scale tiles) for the FP4 tensor cores.
    Nvfp4 {
        alpha: f32,
        first_high: bool,
        act4: bool,
    },
    /// E4M3 codes (1 per byte) with E8M0 per-32 scales in cuBLASLt's
    /// `VEC32_UE8M0` swizzle in the table buffer: the FP8 tensor-core GEMM on
    /// sm_100+ (activations quantized per call), expanded to bf16 elsewhere.
    Mxfp8,
}

#[derive(Default)]
struct Stall {
    waits: Vec<(cudarc::driver::CudaEvent, cudarc::driver::CudaEvent)>,
    copies: Vec<(
        cudarc::driver::CudaEvent,
        Arc<cudarc::driver::CudaEvent>,
        usize,
    )>,
    open: Option<cudarc::driver::CudaEvent>,
    bytes: usize,
}

pub struct Cuda {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    /// Non-blocking side stream for prefetched (streamed block) weights.
    copy: Arc<CudaStream>,
    /// Uploads go to `copy` between `prefetch_begin` and `prefetch_end`.
    staging: std::sync::atomic::AtomicBool,
    /// The ends of the prefetches whose blocks have not run yet, oldest
    /// first: `prefetch_wait` pops the front (the block about to run). A
    /// first version of this FIFO kept the front when it was the only entry,
    /// so with one prefetch in flight every block after the first waited on
    /// the previous block's fence and could run before its own weights had
    /// landed (clips that differed between depth 1 and depth 2, 3eb1d82).
    fence: std::sync::Mutex<std::collections::VecDeque<Arc<cudarc::driver::CudaEvent>>>,
    /// The end of the newest prefetch, whatever `prefetch_wait` has popped:
    /// `prefetch_begin` synchronises it on the host before the bounce buffer
    /// is reused (every earlier prefetch's copies are behind it).
    last_fence: std::sync::Mutex<Option<Arc<cudarc::driver::CudaEvent>>>,
    /// `H3_STREAM_STRESS=1`: every prefetch copy lands in zeroed memory after
    /// a ~5 ms spin on the copy stream (`stress_delay`), so a missing or wrong
    /// fence wait shows up as a wrong result in the streamed checks rather
    /// than passing by timing or by address reuse.
    stream_stress: bool,
    /// `H3_STALL=1`: event pairs around every `prefetch_wait` on the compute
    /// stream and around every prefetch's copies on the copy stream, with the
    /// bytes copied, so the per-forward summary can say how long the GPU sat
    /// waiting for streamed weights and at what rate they arrived.
    stall: Option<std::sync::Mutex<Stall>>,
    /// Host ranges page-locked by `pin_host` (the interior pages of streamed
    /// weight buffers): copies from inside them are DMA with no host wait.
    /// Shared with the guards, which remove their range when dropped.
    pinned: Arc<std::sync::Mutex<Vec<(usize, usize)>>>,
    /// Page-lock streamed weights (default; `H3_CUDA_NO_PIN=1` compares):
    /// 127 / 127 checks with it on a 5090, V2 forward 7.2 → 6.45 s there.
    use_pin: bool,
    /// Pinned bounce buffer for the small uploads of a prefetch (norms, the
    /// unlocked edge pages): copied in, then DMA'd with no host wait. Reset
    /// at `prefetch_begin`, once the previous prefetch's copies are done.
    bounce: std::sync::Mutex<Option<(cudarc::driver::PinnedHostSlice<u8>, usize)>>,
    pin_warned: std::sync::atomic::AtomicBool,
    pv8_warned: std::sync::atomic::AtomicBool,
    blas: CudaBlas,
    /// cuBLASLt with the bias (+ tanh-GELU) epilogue fused into the GEMM;
    /// `None` when unavailable or `H3_CUDA_NO_LT=1`.
    lt: Option<Lt>,
    #[allow(dead_code)]
    module: Arc<CudaModule>,
    f: Kernels,
    name: String,
    /// Compute capability major ≥ 8: bf16 mma.sync is available.
    ampere: bool,
    /// sm_89+: FP8 tensor cores (`mma.e4m3`).
    ada: bool,
    q8_scratch: std::sync::Mutex<Option<CuBuf>>,
    /// INT8 dense attention scratch for the last shape: (heads, kv_heads, sq,
    /// sk, d) and (q8, k8, qsc, ksc, kmean, kpart).
    #[allow(clippy::type_complexity)]
    attn_qk8_scratch: std::sync::Mutex<
        Option<(
            (usize, usize, usize, usize, usize),
            (CuBuf, CuBuf, CuBuf, CuBuf, CuBuf, CuBuf),
        )>,
    >,
    /// FP8 P·V scratch keyed by (kv_heads, sk, d): transposed V codes, V mean, partial maxima, scales.
    attn_pv8_scratch:
        std::sync::Mutex<Option<((usize, usize, usize), (CuBuf, CuBuf, CuBuf, CuBuf))>>,
    /// MXFP8 activations per GEMM: (E4M3 codes, swizzled E8M0 scales, bf16
    /// rounded copy for the emulation path), grown on demand.
    mx_scratch: std::sync::Mutex<Option<(CuBuf, CuBuf, CuBuf)>>,
    /// NVFP4 W4A4 per call: (state: maximum / alpha / beta, codes, scales,
    /// the emulation's rounded activations).
    nv_scratch: std::sync::Mutex<Option<(CuBuf, CuBuf, CuBuf, CuBuf)>>,
    /// Compute capability major; block-scaled FP8 GEMMs need 10+.
    sm_major: u32,
    /// `H3_CUDA_NO_FP8=1`: emulate MXFP8 in bf16 even on sm_100+.
    no_fp8: bool,
    /// `H3_CUDA_LT_NTH=i`: use cuBLASLt's i-th heuristic candidate for the
    /// MXFP8 GEMMs instead of timing them (experiments).
    lt_nth: Option<usize>,
    /// MXFP8 GEMMs may run as `mx_split` GEMMs over row slices of the weight
    /// (`H3_CUDA_MX_SPLIT=p`): cuBLASLt's FP8 kernels for the 28672-row FFN-up
    /// ran at 355 TFLOP/s on an RTX PRO 6000 (the QKV shape at 720, no gain
    /// from splitting), while on an RTX 5090 splitting the QKV shape too took
    /// the V2 forward 5.70 → 5.32 s. The slices round like the whole GEMM
    /// (candidate pinned, clips byte-identical), so the choice is pure speed.
    mx_split: usize,
    /// `H3_CUDA_MX_SPLIT_MIN=n`: split every shape with `n >= min`. Unset:
    /// decide per shape on its first call (`mx_choice`).
    mx_split_min: Option<usize>,
    /// Auto mode's decision per (n, k, rows): parts (1 = whole).
    mx_choice: std::sync::Mutex<std::collections::HashMap<(usize, usize, usize), usize>>,
    scalar_conv: bool,
    scalar_attn: bool,
    /// `H3_CUDA_ATTN_V1=1`: the 16-rows-per-warp kernels for dense attention
    /// too (v2, 32 rows per warp and 128-row blocks, is the dense default:
    /// 11 % faster on an L40S).
    attn_v1: bool,
    /// `H3_CUDA_VSA_V2=1`: the v2 body for the VSA fine stage. Measured on an
    /// L40S (`docs/runs/2026-11-07-vast-l40s-nvfp4-vsa2`): 29 % slower than v1
    /// on 64-token tiles (2 warps per block), equal on 128-token tiles, so v1
    /// stays the VSA default.
    vsa_v2: bool,
    /// `H3_CUDA_VSA_PAIR=1`: two 64-token query tiles per block over the union
    /// of their selections (measured equal to one tile per block on an L40S,
    /// so off by default).
    vsa_pair: bool,
    /// `H3_CUDA_VSA_COARSE1=1`: the one-query-tile coarse kernel.
    vsa_coarse1: bool,
    /// `H3_CUDA_VSA_DBG=<bits>`: timing experiments in the fine kernel (see
    /// `attn_mma_body`); the output is wrong, only `bench` should use it.
    vsa_dbg: i32,
    /// `H3_CUDA_VSA_FINE=v1|a|b|c|d`: the VSA fine body. Default d (32-key
    /// chunks, K/V double-buffered a chunk ahead, Q resident, swizzled 32 KB
    /// staging for three blocks per SM: 66.5 ms per call on an L40S at
    /// 1344×768×124 vs 72–75 for b and 87 for the original v1); a and c
    /// re-read Q from L1 per mma and are 2.5× slower, kept for the record.
    vsa_fine3: Option<usize>,
    prof: Option<std::sync::Mutex<std::collections::HashMap<String, (f64, usize)>>>,
}

struct Kernels {
    bias_act: CudaFunction,
    swiglu: CudaFunction,
    gate_residual: CudaFunction,
    vsa_gate_add: CudaFunction,
    rms: CudaFunction,
    layer_norm: CudaFunction,
    head_split: CudaFunction,
    head_split_warp: CudaFunction,
    patch: CudaFunction,
    flow_step: CudaFunction,
    axpby: CudaFunction,
    channel_affine: CudaFunction,
    fill: CudaFunction,
    box_rows: CudaFunction,
    box_rows16: CudaFunction,
    spin: CudaFunction,
    tanh_clamp: CudaFunction,
    clamp: CudaFunction,
    snake: CudaFunction,
    pixel_shuffle: CudaFunction,
    trilinear: CudaFunction,
    gn_stats: CudaFunction,
    gn_apply: CudaFunction,
    replicate_pad1d: CudaFunction,
    conv1d: CudaFunction,
    conv_transpose1d: CudaFunction,
    attention: CudaFunction,
    attention_mma64: CudaFunction,
    attention_mma128: CudaFunction,
    attention_mma2_64: CudaFunction,
    attention_mma2_128: CudaFunction,
    attention_qk8_64: CudaFunction,
    attention_qk8_128: CudaFunction,
    attn_qk8_quant: CudaFunction,
    attention_pv8_64: CudaFunction,
    attention_pv8_128: CudaFunction,
    attn_v8_max: CudaFunction,
    attn_v8: CudaFunction,
    vsa_fine_mma2_64_w2: CudaFunction,
    vsa_fine_mma2_64_w4: CudaFunction,
    vsa_fine_mma2_128_w2: CudaFunction,
    vsa_fine_mma2_128_w4: CudaFunction,
    vsa_pool: CudaFunction,
    vsa_coarse: CudaFunction,
    vsa_fine: CudaFunction,
    vsa_fine_mma64: CudaFunction,
    vsa_fine_mma128: CudaFunction,
    vsa_fine_pair_mma64: CudaFunction,
    vsa_fine_pair_mma128: CudaFunction,
    vsa_pair: CudaFunction,
    vsa_coarse4: CudaFunction,
    vsa_pool_scalar: CudaFunction,
    vsa_fine3: [[CudaFunction; 7]; 2],
    vsa_kmean: CudaFunction,
    vsa_kmean2: CudaFunction,
    vsa_qk8: CudaFunction,
    dequant_q8: CudaFunction,
    dequant_q4: CudaFunction,
    dequant_nvfp4: CudaFunction,
    nv_prep: CudaFunction,
    nv_amax: CudaFunction,
    nv_quant: CudaFunction,
    mx_quant: CudaFunction,
    mx_quant8: CudaFunction,
    dequant_mxfp8: CudaFunction,
    conv2d: CudaFunction,
    conv2d_mma: CudaFunction,
    conv3d: CudaFunction,
    conv3d_mma: CudaFunction,
}

#[derive(Debug)]
pub struct CudaError(pub String);
impl std::fmt::Display for CudaError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for CudaError {}

fn err(e: impl std::fmt::Debug) -> CudaError {
    CudaError(format!("{e:?}"))
}

/// Compile the kernels to PTX for `arch` (e.g. `compute_89`). Needs only
/// libnvrtc, not a GPU — the test keeps `kernels.cu` compiling.
pub fn compile_kernels(arch: &'static str) -> Result<cudarc::nvrtc::Ptx, CudaError> {
    cudarc::nvrtc::compile_ptx_with_opts(
        KERNELS,
        cudarc::nvrtc::CompileOptions {
            arch: Some(arch),
            options: vec!["--std=c++17".into()],
            ..Default::default()
        },
    )
    .map_err(err)
}

/// The real SM target (`sm_XY`) NVRTC can emit SASS for, or `None` for an SM
/// it cannot target directly (then the `compute_XY` PTX path is used).
pub fn sass_arch(maj: i32, min: i32) -> Option<&'static str> {
    Some(match (maj, min) {
        (7, 5) => "sm_75",
        (8, 0) => "sm_80",
        (8, 6) => "sm_86",
        (8, 7) => "sm_87",
        (8, 9) => "sm_89",
        (9, 0) => "sm_90",
        (10, 0) => "sm_100",
        (10, 3) => "sm_103",
        (11, 0) => "sm_110",
        (12, 0) => "sm_120",
        (12, 1) => "sm_121",
        _ => return None,
    })
}

/// Compile the kernels to SASS (a cubin) for a real target (`sm_120`), the
/// way fastvideo-rs's `nvrtc_sass` does: a cubin needs no PTX JIT, so an
/// NVRTC newer than the driver (13.4 against R595 rejected the PTX with
/// CUDA_ERROR_UNSUPPORTED_PTX_VERSION) still loads. Needs only libnvrtc.
pub fn compile_cubin(arch: &str) -> Result<Vec<u8>, CudaError> {
    use cudarc::nvrtc::{result, sys};
    use std::ffi::{CStr, CString};
    let src = CString::new(KERNELS).map_err(|e| CudaError(format!("nvrtc source: {e}")))?;
    let prog = result::create_program(&src, None)
        .map_err(|e| CudaError(format!("nvrtc create: {e:?}")))?;
    let options = vec![
        "--std=c++17".to_string(),
        format!("--gpu-architecture={arch}"),
    ];
    // SAFETY: `prog` comes from `create_program`, `src` outlives it, and it is
    // destroyed exactly once below whatever happens in between.
    let out = unsafe {
        match result::compile_program(prog, &options) {
            Err(e) => {
                let log = result::get_program_log(prog)
                    .ok()
                    .map(|l| CStr::from_ptr(l.as_ptr()).to_string_lossy().into_owned())
                    .unwrap_or_default();
                Err(CudaError(format!("nvrtc {arch}: {e:?}\n{log}")))
            }
            Ok(()) => {
                let mut size = 0usize;
                match sys::nvrtcGetCUBINSize(prog, &mut size).result() {
                    Err(e) => Err(CudaError(format!("nvrtcGetCUBINSize {arch}: {e:?}"))),
                    Ok(()) if size == 0 => Err(CudaError(format!("nvrtc {arch}: empty cubin"))),
                    Ok(()) => {
                        let mut buf = vec![0u8; size];
                        sys::nvrtcGetCUBIN(prog, buf.as_mut_ptr().cast())
                            .result()
                            .map(|()| buf)
                            .map_err(|e| CudaError(format!("nvrtcGetCUBIN {arch}: {e:?}")))
                    }
                }
            }
        }
    };
    // SAFETY: created above, not destroyed yet.
    let _ = unsafe { result::destroy_program(prog) };
    out
}

/// One cached cuBLASLt plan per (n, k, rows, epilogue).
struct LtPlan {
    desc: lt_sys::cublasLtMatmulDesc_t,
    a: lt_sys::cublasLtMatrixLayout_t,
    b: lt_sys::cublasLtMatrixLayout_t,
    c: lt_sys::cublasLtMatrixLayout_t,
    algo: lt_sys::cublasLtMatmulAlgo_t,
    /// The heuristic's other candidates (MXFP8 plans): the first use of the
    /// shape times each once and keeps the fastest in `algo`. cuBLASLt's
    /// first pick for the FFN-up GEMM (n = 28672) on an RTX PRO 6000 ran at
    /// 355 TFLOP/s, slower than bf16, while the other shapes ran at 780.
    candidates: Vec<lt_sys::cublasLtMatmulAlgo_t>,
    tuned: bool,
}

/// (n, k, rows, epilogue) → plan, `None` when cuBLASLt declined the shape.
type PlanCache = std::collections::HashMap<(usize, usize, usize, u8), Option<LtPlan>>;

struct Lt {
    handle: lt_sys::cublasLtHandle_t,
    workspace: CudaSlice<u8>,
    plans: std::sync::Mutex<PlanCache>,
    /// MXFP8 plans: (n, k, rows, ldc, epilogue 0 none / 1 bias / 2 bias + tanh-GELU).
    mx_plans: std::sync::Mutex<
        std::collections::HashMap<(usize, usize, usize, usize, u8), Option<LtPlan>>,
    >,
    /// NVFP4 W4A4 plans by `(n, k, rows, epi)`.
    fp4_plans:
        std::sync::Mutex<std::collections::HashMap<(usize, usize, usize, u8), Option<LtPlan>>>,
}

// SAFETY: the handles are only used from the thread that owns the backend,
// on its one stream; the Mutex serialises the plan cache.
unsafe impl Send for Lt {}
unsafe impl Sync for Lt {}

const LT_WORKSPACE: usize = 32 << 20;

impl Lt {
    fn new(stream: &Arc<CudaStream>) -> Result<Self, CudaError> {
        let handle = lt_result::create_handle().map_err(err)?;
        let workspace = stream.alloc_zeros::<u8>(LT_WORKSPACE).map_err(err)?;
        Ok(Self {
            handle,
            workspace,
            plans: Default::default(),
            mx_plans: Default::default(),
            fp4_plans: Default::default(),
        })
    }

    /// Column-major yᵀ[n, rows] = Wᵀ…: A = W (op T, stored k × n), B = x
    /// (k × rows), C = D = y (n × rows), all bf16, f32 accumulate. `epi` 0 =
    /// bias, 1 = tanh-GELU(·+bias). `None` = not supported.
    fn plan(&self, n: usize, k: usize, rows: usize, epi: u8) -> Option<LtPlan> {
        use lt_sys::cudaDataType_t::{CUDA_R_16BF, CUDA_R_32F};
        let a = lt_result::create_matrix_layout(CUDA_R_16BF, k as u64, n as u64, k as i64).ok()?;
        let b =
            lt_result::create_matrix_layout(CUDA_R_16BF, k as u64, rows as u64, k as i64).ok()?;
        let c =
            lt_result::create_matrix_layout(CUDA_R_16BF, n as u64, rows as u64, n as i64).ok()?;
        let desc = lt_result::create_matmul_desc(
            lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            CUDA_R_32F,
        )
        .ok()?;
        let t: i32 = 1;
        let epilogue = if epi == 1 {
            lt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS
        } else {
            lt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS
        };
        let pref = lt_result::create_matmul_pref().ok()?;
        let ws = LT_WORKSPACE as u64;
        // SAFETY: freshly created descriptors; attribute buffers outlive the calls.
        unsafe {
            lt_result::set_matmul_desc_attribute(
                desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA,
                (&t) as *const i32 as *const _,
                4,
            )
            .ok()?;
            lt_result::set_matmul_desc_attribute(
                desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE,
                (&epilogue) as *const _ as *const _,
                std::mem::size_of::<lt_sys::cublasLtEpilogue_t>(),
            )
            .ok()?;
            lt_result::set_matmul_pref_attribute(
                pref,
                lt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&ws) as *const u64 as *const _,
                8,
            )
            .ok()?;
            let dummy: u64 = self.workspace_ptr();
            lt_result::set_matmul_desc_attribute(
                desc,
                lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                (&dummy) as *const u64 as *const _,
                8,
            )
            .ok()?;
            let h = lt_result::get_matmul_algo_heuristic(self.handle, desc, a, b, c, c, pref);
            let _ = lt_result::destroy_matmul_pref(pref);
            Some(LtPlan {
                desc,
                a,
                b,
                c,
                algo: h.ok()?.algo,
                candidates: Vec::new(),
                tuned: true,
            })
        }
    }

    /// The MXFP8 twin of [`Lt::plan`]: A = weight codes (E4M3, `k × n`, op T),
    /// B = activation codes (E4M3, `k × rows`), both with one E8M0 scale per
    /// 32 of K in cuBLASLt's `VEC32_UE8M0` layout; C = D = y (bf16, `n ×
    /// rows`). `epi` 0 = none, 1 = bias, 2 = bias + tanh-GELU. `None` when
    /// cuBLASLt has no kernel (pre-Blackwell, or a shape it declines).
    fn plan_mx(&self, n: usize, k: usize, rows: usize, ldc: usize, epi: u8) -> Option<LtPlan> {
        use lt_sys::cudaDataType_t::{CUDA_R_8F_E4M3, CUDA_R_16BF, CUDA_R_32F};
        let a =
            lt_result::create_matrix_layout(CUDA_R_8F_E4M3, k as u64, n as u64, k as i64).ok()?;
        let b = lt_result::create_matrix_layout(CUDA_R_8F_E4M3, k as u64, rows as u64, k as i64)
            .ok()?;
        let c =
            lt_result::create_matrix_layout(CUDA_R_16BF, n as u64, rows as u64, ldc as i64).ok()?;
        let desc = lt_result::create_matmul_desc(
            lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            CUDA_R_32F,
        )
        .ok()?;
        let t: i32 = 1;
        let epilogue = match epi {
            0 => lt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
            1 => lt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
            _ => lt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS,
        };
        let mode = lt_sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0;
        let bias_ty = CUDA_R_16BF;
        let pref = lt_result::create_matmul_pref().ok()?;
        let ws = LT_WORKSPACE as u64;
        use lt_sys::cublasLtMatmulDescAttributes_t as A;
        // SAFETY: freshly created descriptors; attribute buffers outlive the calls.
        unsafe {
            let set = |attr: A, p: *const std::ffi::c_void, sz: usize| {
                lt_result::set_matmul_desc_attribute(desc, attr, p, sz).ok()
            };
            set(
                A::CUBLASLT_MATMUL_DESC_TRANSA,
                (&t) as *const i32 as *const _,
                4,
            )?;
            set(
                A::CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                (&mode) as *const _ as *const _,
                std::mem::size_of_val(&mode),
            )?;
            set(
                A::CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                (&mode) as *const _ as *const _,
                std::mem::size_of_val(&mode),
            )?;
            let dummy: u64 = self.workspace_ptr();
            set(
                A::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                (&dummy) as *const u64 as *const _,
                8,
            )?;
            set(
                A::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                (&dummy) as *const u64 as *const _,
                8,
            )?;
            if epi > 0 {
                set(
                    A::CUBLASLT_MATMUL_DESC_EPILOGUE,
                    (&epilogue) as *const _ as *const _,
                    std::mem::size_of_val(&epilogue),
                )?;
                set(
                    A::CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                    (&bias_ty) as *const _ as *const _,
                    std::mem::size_of_val(&bias_ty),
                )?;
                set(
                    A::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                    (&dummy) as *const u64 as *const _,
                    8,
                )?;
            }
            lt_result::set_matmul_pref_attribute(
                pref,
                lt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&ws) as *const u64 as *const _,
                8,
            )
            .ok()?;
            // Up to 8 candidates; the first call on the shape times them.
            let mut hs: Vec<lt_sys::cublasLtMatmulHeuristicResult_t> = vec![std::mem::zeroed(); 8];
            let mut found: i32 = 0;
            let st = lt_sys::cublasLtMatmulAlgoGetHeuristic(
                self.handle,
                desc,
                a,
                b,
                c,
                c,
                pref,
                8,
                hs.as_mut_ptr(),
                &mut found,
            );
            let _ = lt_result::destroy_matmul_pref(pref);
            if st != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS || found <= 0 {
                return None;
            }
            hs.truncate(found as usize);
            let candidates: Vec<_> = hs.iter().map(|h| h.algo).collect();
            Some(LtPlan {
                desc,
                a,
                b,
                c,
                algo: candidates[0],
                tuned: candidates.len() == 1,
                candidates,
            })
        }
    }

    /// The NVFP4 W4A4 twin of [`Lt::plan_mx`]: A = weight codes (E2M1, two per
    /// byte, `k × n`, op T), B = activation codes (E2M1, `k × rows`), both
    /// with one E4M3 scale per 16 of K in cuBLASLt's `VEC16_UE4M3` layout;
    /// alpha and beta are device pointers (the activations' tensor scale is
    /// computed on the device); C = D = y (bf16, `n ×
    /// rows`). `epi` 0 = none, 1 = bias, 2 = bias + tanh-GELU. `None` when
    /// cuBLASLt has no kernel (pre-Blackwell, or a shape it declines).
    fn plan_fp4(&self, n: usize, k: usize, rows: usize, ldc: usize, epi: u8) -> Option<LtPlan> {
        use lt_sys::cudaDataType_t::{CUDA_R_4F_E2M1, CUDA_R_16BF, CUDA_R_32F};
        let a =
            lt_result::create_matrix_layout(CUDA_R_4F_E2M1, k as u64, n as u64, k as i64).ok()?;
        let b = lt_result::create_matrix_layout(CUDA_R_4F_E2M1, k as u64, rows as u64, k as i64)
            .ok()?;
        let c =
            lt_result::create_matrix_layout(CUDA_R_16BF, n as u64, rows as u64, ldc as i64).ok()?;
        let desc = lt_result::create_matmul_desc(
            lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            CUDA_R_32F,
        )
        .ok()?;
        let t: i32 = 1;
        let epilogue = match epi {
            0 => lt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_DEFAULT,
            1 => lt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS,
            _ => lt_sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_GELU_BIAS,
        };
        let mode = lt_sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        let bias_ty = CUDA_R_16BF;
        let pref = lt_result::create_matmul_pref().ok()?;
        let ws = LT_WORKSPACE as u64;
        use lt_sys::cublasLtMatmulDescAttributes_t as A;
        // SAFETY: freshly created descriptors; attribute buffers outlive the calls.
        unsafe {
            let set = |attr: A, p: *const std::ffi::c_void, sz: usize| {
                lt_result::set_matmul_desc_attribute(desc, attr, p, sz).ok()
            };
            set(
                A::CUBLASLT_MATMUL_DESC_TRANSA,
                (&t) as *const i32 as *const _,
                4,
            )?;
            let pm = lt_sys::cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_DEVICE;
            set(
                A::CUBLASLT_MATMUL_DESC_POINTER_MODE,
                (&pm) as *const _ as *const _,
                std::mem::size_of_val(&pm),
            )?;
            set(
                A::CUBLASLT_MATMUL_DESC_A_SCALE_MODE,
                (&mode) as *const _ as *const _,
                std::mem::size_of_val(&mode),
            )?;
            set(
                A::CUBLASLT_MATMUL_DESC_B_SCALE_MODE,
                (&mode) as *const _ as *const _,
                std::mem::size_of_val(&mode),
            )?;
            let dummy: u64 = self.workspace_ptr();
            set(
                A::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                (&dummy) as *const u64 as *const _,
                8,
            )?;
            set(
                A::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                (&dummy) as *const u64 as *const _,
                8,
            )?;
            if epi > 0 {
                set(
                    A::CUBLASLT_MATMUL_DESC_EPILOGUE,
                    (&epilogue) as *const _ as *const _,
                    std::mem::size_of_val(&epilogue),
                )?;
                set(
                    A::CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                    (&bias_ty) as *const _ as *const _,
                    std::mem::size_of_val(&bias_ty),
                )?;
                set(
                    A::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                    (&dummy) as *const u64 as *const _,
                    8,
                )?;
            }
            lt_result::set_matmul_pref_attribute(
                pref,
                lt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                (&ws) as *const u64 as *const _,
                8,
            )
            .ok()?;
            // Up to 8 candidates; the first call on the shape times them.
            let mut hs: Vec<lt_sys::cublasLtMatmulHeuristicResult_t> = vec![std::mem::zeroed(); 8];
            let mut found: i32 = 0;
            let st = lt_sys::cublasLtMatmulAlgoGetHeuristic(
                self.handle,
                desc,
                a,
                b,
                c,
                c,
                pref,
                8,
                hs.as_mut_ptr(),
                &mut found,
            );
            let _ = lt_result::destroy_matmul_pref(pref);
            if st != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS || found <= 0 {
                return None;
            }
            hs.truncate(found as usize);
            let candidates: Vec<_> = hs.iter().map(|h| h.algo).collect();
            Some(LtPlan {
                desc,
                a,
                b,
                c,
                algo: candidates[0],
                tuned: candidates.len() == 1,
                candidates,
            })
        }
    }

    /// A plain FP8 plan (per-tensor scales, no block scaling) of the same
    /// shape, for timing only: how much of an MXFP8 GEMM's time is the
    /// `VEC32_UE8M0` scaling (item 2 of the 5090 round).
    fn plan_fp8_plain(&self, n: usize, k: usize, rows: usize, ldc: usize) -> Option<LtPlan> {
        use lt_sys::cudaDataType_t::{CUDA_R_8F_E4M3, CUDA_R_16BF, CUDA_R_32F};
        let a =
            lt_result::create_matrix_layout(CUDA_R_8F_E4M3, k as u64, n as u64, k as i64).ok()?;
        let b = lt_result::create_matrix_layout(CUDA_R_8F_E4M3, k as u64, rows as u64, k as i64)
            .ok()?;
        let c =
            lt_result::create_matrix_layout(CUDA_R_16BF, n as u64, rows as u64, ldc as i64).ok()?;
        let desc = lt_result::create_matmul_desc(
            lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            CUDA_R_32F,
        )
        .ok()?;
        let t: i32 = 1;
        let pref = lt_result::create_matmul_pref().ok()?;
        let ws = LT_WORKSPACE as u64;
        use lt_sys::cublasLtMatmulDescAttributes_t as A;
        // SAFETY: freshly created descriptors; attribute buffers outlive the calls.
        unsafe {
            lt_result::set_matmul_desc_attribute(
                desc,
                A::CUBLASLT_MATMUL_DESC_TRANSA,
                (&t) as *const i32 as *const _,
                4,
            )
            .ok()?;
            lt_result::set_matmul_pref_attribute(pref, lt_sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, (&ws) as *const u64 as *const _, 8).ok()?;
            let mut hs: Vec<lt_sys::cublasLtMatmulHeuristicResult_t> = vec![std::mem::zeroed(); 8];
            let mut found: i32 = 0;
            let st = lt_sys::cublasLtMatmulAlgoGetHeuristic(
                self.handle,
                desc,
                a,
                b,
                c,
                c,
                pref,
                8,
                hs.as_mut_ptr(),
                &mut found,
            );
            let _ = lt_result::destroy_matmul_pref(pref);
            if st != lt_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS || found <= 0 {
                return None;
            }
            hs.truncate(found as usize);
            let candidates: Vec<_> = hs.iter().map(|h| h.algo).collect();
            Some(LtPlan {
                desc,
                a,
                b,
                c,
                algo: candidates[0],
                tuned: candidates.len() == 1,
                candidates,
            })
        }
    }

    fn workspace_ptr(&self) -> u64 {
        let (p, _sync) = self.workspace.device_ptr(self.workspace.stream());
        p
    }
}

enum Arg {
    P(u64),
    I(i32),
    L(i64),
    F(f32),
}

fn env1(name: &str) -> bool {
    std::env::var(name).is_ok_and(|v| v == "1")
}

impl Cuda {
    pub fn new(ordinal: usize) -> Result<Self, CudaError> {
        // cudarc panics on the first driver call when libcuda is missing (a
        // container started without the NVIDIA runtime): an error instead,
        // so `--backend auto` falls back.
        // SAFETY: only probes whether the driver library loads.
        if !unsafe { cudarc::driver::sys::is_culib_present() } {
            return Err(CudaError(
                "no CUDA driver (libcuda.so.1 not found): no NVIDIA GPU, or the container was started without the NVIDIA runtime".into(),
            ));
        }
        let ctx = CudaContext::new(ordinal).map_err(err)?;
        let (maj, min) = ctx.compute_capability().map_err(err)?;
        // SASS for the device's own SM first (no PTX JIT: an NVRTC newer than
        // the driver still loads); PTX for `compute_XY` when NVRTC has no
        // real target for it or the cubin failed.
        let module = match sass_arch(maj, min).map(compile_cubin) {
            Some(Ok(cubin)) => ctx
                .load_module(cudarc::nvrtc::Ptx::from_binary(cubin))
                .map_err(err)?,
            other => {
                if let Some(Err(e)) = other {
                    eprintln!("cuda: SASS compile failed ({}); falling back to PTX", e.0);
                }
                let arch: &'static str = Box::leak(format!("compute_{maj}{min}").into_boxed_str());
                ctx.load_module(compile_kernels(arch)?).map_err(err)?
            }
        };
        let get = |n: &str| module.load_function(n).map_err(err);
        let f = Kernels {
            bias_act: get("fw_bias_act")?,
            swiglu: get("fw_swiglu")?,
            gate_residual: get("fw_gate_residual")?,
            vsa_gate_add: get("fw_vsa_gate_add")?,
            rms: get("fw_rms")?,
            layer_norm: get("fw_layer_norm")?,
            head_split: get("fw_head_split")?,
            head_split_warp: get("fw_head_split_warp")?,
            patch: get("fw_patch")?,
            flow_step: get("fw_flow_step")?,
            axpby: get("fw_axpby")?,
            channel_affine: get("fw_channel_affine")?,
            fill: get("fw_fill")?,
            box_rows: get("fw_box_rows")?,
            box_rows16: get("fw_box_rows16")?,
            spin: get("fw_spin")?,
            tanh_clamp: get("fw_tanh_clamp")?,
            clamp: get("fw_clamp")?,
            snake: get("fw_snake")?,
            pixel_shuffle: get("fw_pixel_shuffle")?,
            trilinear: get("fw_trilinear")?,
            gn_stats: get("fw_group_norm_stats")?,
            gn_apply: get("fw_group_norm_apply")?,
            replicate_pad1d: get("fw_replicate_pad1d")?,
            conv1d: get("fw_conv1d")?,
            conv_transpose1d: get("fw_conv_transpose1d")?,
            attention: get("fw_attention")?,
            attention_mma64: get("fw_attention_mma64")?,
            attention_mma128: get("fw_attention_mma128")?,
            attention_mma2_64: get("fw_attention_mma2_64")?,
            attention_mma2_128: get("fw_attention_mma2_128")?,
            attention_qk8_64: get("fw_attention_qk8_64")?,
            attention_qk8_128: get("fw_attention_qk8_128")?,
            attn_qk8_quant: get("fw_attn_qk8_quant")?,
            attention_pv8_64: get("fw_attention_pv8_64")?,
            attention_pv8_128: get("fw_attention_pv8_128")?,
            attn_v8_max: get("fw_attn_v8_max")?,
            attn_v8: get("fw_attn_v8")?,
            vsa_fine_mma2_64_w2: get("fw_vsa_fine_mma2_64_w2")?,
            vsa_fine_mma2_64_w4: get("fw_vsa_fine_mma2_64_w4")?,
            vsa_fine_mma2_128_w2: get("fw_vsa_fine_mma2_128_w2")?,
            vsa_fine_mma2_128_w4: get("fw_vsa_fine_mma2_128_w4")?,
            vsa_pool: get("fw_vsa_pool")?,
            vsa_coarse: get("fw_vsa_coarse")?,
            vsa_fine: get("fw_vsa_fine")?,
            vsa_fine_mma64: get("fw_vsa_fine_mma64")?,
            vsa_fine_mma128: get("fw_vsa_fine_mma128")?,
            vsa_fine_pair_mma64: get("fw_vsa_fine_pair_mma64")?,
            vsa_fine_pair_mma128: get("fw_vsa_fine_pair_mma128")?,
            vsa_pair: get("fw_vsa_pair")?,
            vsa_coarse4: get("fw_vsa_coarse4")?,
            vsa_pool_scalar: get("fw_vsa_pool_scalar")?,
            vsa_fine3: [
                [
                    get("fw_vsa_fine3a_mma64")?,
                    get("fw_vsa_fine3b_mma64")?,
                    get("fw_vsa_fine3c_mma64")?,
                    get("fw_vsa_fine3d_mma64")?,
                    get("fw_vsa_fine3e_mma64")?,
                    get("fw_vsa_fine3f_mma64")?,
                    get("fw_vsa_fine3g_mma64")?,
                ],
                [
                    get("fw_vsa_fine3a_mma128")?,
                    get("fw_vsa_fine3b_mma128")?,
                    get("fw_vsa_fine3c_mma128")?,
                    get("fw_vsa_fine3d_mma128")?,
                    get("fw_vsa_fine3e_mma128")?,
                    get("fw_vsa_fine3f_mma128")?,
                    get("fw_vsa_fine3g_mma128")?,
                ],
            ],
            vsa_kmean: get("fw_vsa_kmean")?,
            vsa_kmean2: get("fw_vsa_kmean2")?,
            vsa_qk8: get("fw_vsa_qk8")?,
            dequant_q8: get("fw_dequant_q8")?,
            dequant_q4: get("fw_dequant_q4")?,
            dequant_nvfp4: get("fw_dequant_nvfp4")?,
            nv_prep: get("fw_nv_prep")?,
            nv_amax: get("fw_nv_amax")?,
            nv_quant: get("fw_nv_quant")?,
            mx_quant: get("fw_mx_quant")?,
            mx_quant8: get("fw_mx_quant8")?,
            dequant_mxfp8: get("fw_dequant_mxfp8")?,
            conv2d: get("fw_conv2d")?,
            conv2d_mma: get("fw_conv2d_mma")?,
            conv3d: get("fw_conv3d")?,
            conv3d_mma: get("fw_conv3d_mma")?,
        };
        let stream = ctx.default_stream();
        let copy = ctx.new_stream().map_err(err)?;
        let blas = CudaBlas::new(stream.clone()).map_err(err)?;
        let lt = if env1("H3_CUDA_NO_LT") {
            None
        } else {
            Lt::new(&stream).ok()
        };
        let name = format!("cuda {} (sm_{maj}{min})", ctx.name().unwrap_or_default());
        let me = Self {
            ctx,
            stream,
            copy,
            staging: std::sync::atomic::AtomicBool::new(false),
            fence: std::sync::Mutex::new(std::collections::VecDeque::new()),
            last_fence: std::sync::Mutex::new(None),
            stream_stress: env1("H3_STREAM_STRESS"),
            stall: env1("H3_STALL").then(Default::default),
            pinned: Arc::new(std::sync::Mutex::new(Vec::new())),
            use_pin: !env1("H3_CUDA_NO_PIN"),
            bounce: std::sync::Mutex::new(None),
            pin_warned: std::sync::atomic::AtomicBool::new(false),
            pv8_warned: std::sync::atomic::AtomicBool::new(false),
            blas,
            lt,
            module,
            f,
            name,
            ampere: maj >= 8,
            ada: maj > 8 || (maj == 8 && min >= 9),
            q8_scratch: std::sync::Mutex::new(None),
            attn_qk8_scratch: std::sync::Mutex::new(None),
            attn_pv8_scratch: std::sync::Mutex::new(None),
            mx_scratch: std::sync::Mutex::new(None),
            nv_scratch: std::sync::Mutex::new(None),
            sm_major: maj as u32,
            no_fp8: env1("H3_CUDA_NO_FP8"),
            lt_nth: std::env::var("H3_CUDA_LT_NTH")
                .ok()
                .and_then(|v| v.parse().ok()),
            mx_split: std::env::var("H3_CUDA_MX_SPLIT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(2)
                .max(1),
            mx_split_min: std::env::var("H3_CUDA_MX_SPLIT_MIN")
                .ok()
                .and_then(|v| v.parse().ok()),
            mx_choice: std::sync::Mutex::new(std::collections::HashMap::new()),
            scalar_conv: env1("H3_CUDA_SCALAR_CONV"),
            scalar_attn: env1("H3_CUDA_SCALAR_ATTN"),
            attn_v1: env1("H3_CUDA_ATTN_V1"),
            vsa_v2: env1("H3_CUDA_VSA_V2"),
            vsa_pair: env1("H3_CUDA_VSA_PAIR"),
            vsa_coarse1: env1("H3_CUDA_VSA_COARSE1"),
            vsa_dbg: std::env::var("H3_CUDA_VSA_DBG")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0),
            vsa_fine3: match std::env::var("H3_CUDA_VSA_FINE").as_deref() {
                Ok("a") => Some(0),
                Ok("c") => Some(2),
                Ok("b") => Some(1),
                Ok("e") => Some(4),
                Ok("f") => Some(5),
                Ok("g") => Some(6),
                Ok("v1") => None,
                _ => Some(3),
            },
            prof: env1("H3_PROFILE").then(Default::default),
        };
        if let Some(line) = me.h2d_probe() {
            eprintln!("{line}");
        }
        Ok(me)
    }

    fn ptr(&self, b: &CuBuf) -> u64 {
        let (p, _sync) = b.s.device_ptr(&self.stream);
        p
    }

    /// The stream uploads run on: the side stream while a prefetch is open.
    fn ustream(&self) -> &Arc<CudaStream> {
        if self.staging.load(std::sync::atomic::Ordering::Relaxed) {
            &self.copy
        } else {
            &self.stream
        }
    }

    /// `data` uploaded as is (bf16 or f32 pairs, as `f32` says). Allocated on
    /// the upload stream; the copy is complete when this returns.
    fn upload_u16(&self, data: &[u16], f32: bool, len: usize) -> CuBuf {
        let st = self.ustream();
        let n = data.len().max(1);
        // SAFETY: fully overwritten below (or a one-word pad nothing reads).
        let s = if self.stream_stress {
            st.alloc_zeros::<u16>(n).expect("cuda alloc")
        } else {
            unsafe { st.alloc::<u16>(n) }.expect("cuda alloc")
        };
        if !data.is_empty() {
            self.stress_delay(st);
            let (p, _g) = s.device_ptr(st);
            // SAFETY: `u16` reinterpreted as its bytes.
            let bytes =
                unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
            self.htod(p, bytes, st);
        }
        CuBuf {
            s: Arc::new(s),
            len,
            f32,
        }
    }

    /// Copy `src` to the device at `dst` on the upload stream. During a
    /// prefetch with page-locking on, small sources go through the pinned
    /// bounce buffer and the call returns without waiting; otherwise the
    /// pageable copy is waited for.
    fn htod(&self, dst: u64, src: &[u8], st: &Arc<CudaStream>) {
        const BOUNCE: usize = 64 << 20;
        self.count_upload(src.len());
        if self.use_pin && self.staging.load(std::sync::atomic::Ordering::Relaxed) {
            let mut g = self.bounce.lock().unwrap();
            if g.is_none() {
                // SAFETY: written before it is read.
                if let Ok(b) = unsafe { self.ctx.alloc_pinned_with_flags::<u8>(BOUNCE, 0) } {
                    *g = Some((b, 0));
                }
            }
            if let Some((buf, used)) = g.as_mut() {
                let off = *used;
                if off + src.len() <= BOUNCE {
                    let slot = &mut buf.as_mut_slice().expect("pinned")[off..off + src.len()];
                    slot.copy_from_slice(src);
                    *used = (off + src.len()).div_ceil(256) * 256;
                    // SAFETY: the bounce region stays untouched until the next
                    // `prefetch_begin`, which waits for this prefetch's copies.
                    unsafe {
                        cudarc::driver::result::memcpy_htod_async(dst, &*slot, st.cu_stream())
                            .expect("upload")
                    };
                    return;
                }
            }
        }
        // SAFETY: `src` outlives the copy: the stream is synchronised before returning.
        unsafe {
            cudarc::driver::result::memcpy_htod_async(dst, src, st.cu_stream()).expect("upload")
        };
        st.synchronize().expect("sync");
    }

    /// An event with timing enabled (cudarc's default disables it), recorded on `st`.
    fn timed_event(&self, st: &Arc<CudaStream>) -> cudarc::driver::CudaEvent {
        st.record_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event")
    }

    /// `H3_STREAM_STRESS=1`, during a prefetch: the destination is already
    /// zero (allocated zeroed), and a ~5 ms spin on the upload stream goes
    /// ahead of the copy, so a compute stream that does not wait on this
    /// prefetch's fence reads zeros, deterministically, instead of whatever
    /// the stream-ordered allocator left at that address (a stale fence had
    /// passed the streamed tiny-DiT checks that way).
    fn stress_delay(&self, st: &Arc<CudaStream>) {
        if !(self.stream_stress && self.staging.load(std::sync::atomic::Ordering::Relaxed)) {
            return;
        }
        let mut lb = st.launch_builder(&self.f.spin);
        let ns: i64 = 5_000_000;
        lb.arg(&ns);
        // SAFETY: a kernel with one scalar argument and no memory access.
        unsafe {
            lb.launch(LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            })
        }
        .expect("spin");
    }

    /// Bytes of an upload issued during a prefetch, for the stall summary.
    fn count_upload(&self, n: usize) {
        if let Some(st) = &self.stall {
            if self.staging.load(std::sync::atomic::Ordering::Relaxed) {
                st.lock().unwrap().bytes += n;
            }
        }
    }

    /// `H3_STALL=1`: one-off host-to-device rate probe, 256 MB pinned and
    /// pageable, so a stalled forward can be read against the link.
    fn h2d_probe(&self) -> Option<String> {
        self.stall.as_ref()?;
        const N: usize = 256 << 20;
        let dst = self.stream.alloc_zeros::<u8>(N).ok()?;
        let (p, _g) = dst.device_ptr(&self.stream);
        let mut out = String::from("h2d probe (256 MB):");
        let pageable = vec![1u8; N];
        // SAFETY: `pinned` is written before it is read; both sources outlive their copies (synchronised below).
        let pinned = unsafe { self.ctx.alloc_pinned_with_flags::<u8>(N, 0) }.ok();
        for (name, src) in [
            ("pageable", Some(&pageable[..])),
            ("pinned", pinned.as_ref().and_then(|b| b.as_slice().ok())),
        ] {
            let Some(src) = src else { continue };
            let mut best = f32::INFINITY;
            for _ in 0..3 {
                let e0 = self.timed_event(&self.stream);
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(p, src, self.stream.cu_stream())
                        .ok()?
                };
                let e1 = self.timed_event(&self.stream);
                self.stream.synchronize().ok()?;
                best = best.min(e0.elapsed_ms(&e1).ok()?);
            }
            out += &format!(" {name} {:.1} GB/s", N as f64 / 1e9 / (best as f64 / 1e3));
        }
        Some(out)
    }

    /// The part of `bytes` that lies in a page-locked range, as (start, end)
    /// offsets into `bytes`; empty when none does.
    fn pinned_part(&self, bytes: &[u8]) -> (usize, usize) {
        let (a, b) = (
            bytes.as_ptr() as usize,
            bytes.as_ptr() as usize + bytes.len(),
        );
        for &(s, e) in self.pinned.lock().unwrap().iter() {
            if s < b && e > a {
                return (s.max(a) - a, e.min(b) - a);
            }
        }
        (0, 0)
    }

    /// Byte-coded weights (int8 / int4 / NVFP4 / MXFP8 codes and scales) as a
    /// 16-bit buffer, copied straight from the host bytes: a streamed block's
    /// weights used to go through a `u8 → u16` pass on the CPU every step.
    /// While prefetching, the bytes go through pinned staging so the DMA runs
    /// without a host-side wait; otherwise the pageable copy is waited for.
    fn upload_bytes(&self, bytes: &[u8]) -> CuBuf {
        let st = self.ustream();
        let words = bytes.len().div_ceil(2).max(1);
        // Zeroed so an odd trailing byte (or the empty pad) reads as zero.
        let s = st.alloc_zeros::<u16>(words).expect("cuda alloc");
        if !bytes.is_empty() {
            self.stress_delay(st);
            let (p, _g) = s.device_ptr(st);
            let (i0, i1) = self.pinned_part(bytes);
            if i1 > i0 {
                // The unlocked edge pages (at most one each) through the bounce
                // buffer, the locked interior as a DMA with no wait.
                if i0 > 0 {
                    self.htod(p, &bytes[..i0], st);
                }
                if i1 < bytes.len() {
                    self.htod(p + i1 as u64, &bytes[i1..], st);
                }
                // SAFETY: the locked pages stay mapped and locked while the DiT
                // that owns them (and their guard) lives.
                self.count_upload(i1 - i0);
                unsafe {
                    cudarc::driver::result::memcpy_htod_async(
                        p + i0 as u64,
                        &bytes[i0..i1],
                        st.cu_stream(),
                    )
                    .expect("upload")
                };
            } else {
                self.htod(p, bytes, st);
            }
        }
        CuBuf {
            s: Arc::new(s),
            len: words,
            f32: false,
        }
    }

    fn opt(&self, b: Option<&CuBuf>) -> u64 {
        b.map_or(0, |b| self.ptr(b))
    }

    /// Element offset into a bf16 buffer, in bytes.
    fn ptr_at(&self, b: &CuBuf, elems: usize) -> u64 {
        self.ptr(b) + (elems * 2) as u64
    }

    fn launch(
        &self,
        f: &CudaFunction,
        grid: (u32, u32, u32),
        block: (u32, u32, u32),
        args: &[Arg],
    ) {
        let mut lb = self.stream.launch_builder(f);
        for a in args {
            match a {
                Arg::P(v) => lb.arg(v),
                Arg::I(v) => lb.arg(v),
                Arg::L(v) => lb.arg(v),
                Arg::F(v) => lb.arg(v),
            };
        }
        let cfg = LaunchConfig {
            grid_dim: grid,
            block_dim: block,
            shared_mem_bytes: 0,
        };
        // SAFETY: every kernel's parameter list is matched by the `args`
        // built next to its launch, and every pointer is a live allocation of
        // at least the extent the kernel indexes (lengths asserted by callers).
        self.timed(
            || self.kernel_name(f).to_string(),
            || unsafe { lb.launch(cfg) }.expect("kernel launch"),
        );
    }

    /// Run `op`; with profiling on, wall time between stream syncs under `tag`.
    fn timed<R>(&self, tag: impl FnOnce() -> String, op: impl FnOnce() -> R) -> R {
        let Some(prof) = &self.prof else {
            return op();
        };
        self.stream.synchronize().expect("sync");
        let t = std::time::Instant::now();
        let r = op();
        self.stream.synchronize().expect("sync");
        let dt = t.elapsed().as_secs_f64();
        let mut m = prof.lock().unwrap();
        let e = m.entry(tag()).or_default();
        e.0 += dt;
        e.1 += 1;
        r
    }

    fn kernel_name(&self, f: &CudaFunction) -> &'static str {
        let k = &self.f;
        let all: [(&CudaFunction, &'static str); 82] = [
            (&k.bias_act, "bias_act"),
            (&k.swiglu, "swiglu"),
            (&k.gate_residual, "gate_residual"),
            (&k.vsa_gate_add, "vsa gate add (dense plan)"),
            (&k.rms, "rms_norm (+residual, modulation)"),
            (&k.layer_norm, "layer_norm"),
            (&k.head_split, "head_split (qk-norm + rope)"),
            (&k.head_split_warp, "head_split (qk-norm + rope)"),
            (&k.patch, "patchify / unpatchify"),
            (&k.flow_step, "flow_step"),
            (&k.axpby, "axpby"),
            (&k.channel_affine, "channel_affine"),
            (&k.fill, "fill"),
            (&k.box_rows, "box_rows"),
            (&k.box_rows16, "box_rows (16-byte)"),
            (&k.spin, "stream stress spin"),
            (&k.tanh_clamp, "tanh_clamp"),
            (&k.clamp, "clamp"),
            (&k.snake, "snake"),
            (&k.pixel_shuffle, "pixel_shuffle"),
            (&k.trilinear, "trilinear"),
            (&k.gn_stats, "group_norm stats"),
            (&k.gn_apply, "group_norm apply"),
            (&k.replicate_pad1d, "replicate_pad1d"),
            (&k.conv1d, "conv1d"),
            (&k.conv_transpose1d, "conv_transpose1d"),
            (&k.attention, "attention (scalar)"),
            (&k.attention_mma64, "attention (mma, d64)"),
            (&k.attention_mma128, "attention (mma, d128)"),
            (&k.attention_mma2_64, "attention (mma2, d64)"),
            (&k.attention_mma2_128, "attention (mma2, d128)"),
            (&k.attention_qk8_64, "attention (int8 qk, d64)"),
            (&k.attention_qk8_128, "attention (int8 qk, d128)"),
            (&k.attn_qk8_quant, "attention int8 q/k quant"),
            (&k.attention_pv8_64, "attention (int8 qk + fp8 pv, d64)"),
            (&k.attention_pv8_128, "attention (int8 qk + fp8 pv, d128)"),
            (&k.attn_v8_max, "attention fp8 v max"),
            (&k.attn_v8, "attention fp8 v quant"),
            (&k.vsa_fine_mma2_64_w2, "vsa fine (mma2, d64)"),
            (&k.vsa_fine_mma2_64_w4, "vsa fine (mma2, d64)"),
            (&k.vsa_fine_mma2_128_w2, "vsa fine (mma2, d128)"),
            (&k.vsa_fine_mma2_128_w4, "vsa fine (mma2, d128)"),
            (&k.vsa_pool, "vsa pool"),
            (&k.vsa_coarse, "vsa coarse + top-k"),
            (&k.vsa_fine, "vsa fine (scalar)"),
            (&k.vsa_fine_mma64, "vsa fine (mma, d64)"),
            (&k.vsa_fine_mma128, "vsa fine (mma, d128)"),
            (&k.vsa_fine_pair_mma64, "vsa fine (paired tiles, d64)"),
            (&k.vsa_fine_pair_mma128, "vsa fine (paired tiles, d128)"),
            (&k.vsa_pair, "vsa pair union"),
            (&k.vsa_coarse4, "vsa coarse + top-k (4 query tiles)"),
            (&k.vsa_pool_scalar, "vsa pool (scalar)"),
            (&k.vsa_fine3[0][0], "vsa fine (3a, d64)"),
            (&k.vsa_fine3[0][1], "vsa fine (3b, d64)"),
            (&k.vsa_fine3[0][2], "vsa fine (3c, d64)"),
            (&k.vsa_fine3[1][0], "vsa fine (3a, d128)"),
            (&k.vsa_fine3[1][1], "vsa fine (3b, d128)"),
            (&k.vsa_fine3[1][2], "vsa fine (3c, d128)"),
            (&k.vsa_fine3[0][3], "vsa fine (3d, d64)"),
            (&k.vsa_fine3[1][3], "vsa fine (3d, d128)"),
            (&k.vsa_fine3[0][4], "vsa fine (3e int8 qk, d64)"),
            (&k.vsa_fine3[1][4], "vsa fine (3e int8 qk, d128)"),
            (&k.vsa_fine3[0][5], "vsa fine (3f int8 qk, d64)"),
            (&k.vsa_fine3[1][5], "vsa fine (3f int8 qk, d128)"),
            (&k.vsa_fine3[0][6], "vsa fine (3g int8 qk + fp8 pv, d64)"),
            (&k.vsa_fine3[1][6], "vsa fine (3g int8 qk + fp8 pv, d128)"),
            (&k.vsa_kmean, "vsa k mean"),
            (&k.vsa_kmean2, "vsa k mean (fold)"),
            (&k.vsa_qk8, "vsa int8 q/k quant"),
            (&k.dequant_q8, "dequant int8"),
            (&k.dequant_q4, "dequant int4"),
            (&k.dequant_nvfp4, "dequant nvfp4"),
            (&k.nv_prep, "nvfp4 act state"),
            (&k.nv_amax, "nvfp4 act max"),
            (&k.nv_quant, "nvfp4 act quant"),
            (&k.mx_quant, "mx quant (activations, scalar)"),
            (&k.mx_quant8, "mx quant (activations)"),
            (&k.dequant_mxfp8, "dequant mxfp8"),
            (&k.conv2d, "conv2d (scalar)"),
            (&k.conv2d_mma, "conv2d (mma)"),
            (&k.conv3d, "conv3d (scalar)"),
            (&k.conv3d_mma, "conv3d (mma)"),
        ];
        all.iter()
            .find(|(g, _)| std::ptr::eq(*g, f))
            .map_or("kernel", |(_, n)| n)
    }

    fn elems(n: usize) -> (u32, u32, u32) {
        ((n.div_ceil(256)) as u32, 1, 1)
    }

    /// MXFP8 activation quantization of `total` bf16 values at `xp` (`k`
    /// columns per row): `codes` = (E4M3 codes, swizzled E8M0 scales) for the
    /// FP8 GEMM, else the decoded bf16 copy into `out` (emulation). The
    /// eight-per-thread kernel needs a 16-byte-aligned `xp`.
    fn mx_quantize(&self, xp: u64, total: usize, k: usize, codes: Option<(u64, u64)>, out: u64) {
        let (cp, sp, mode) = match codes {
            Some((c, s)) => (c, s, 0),
            None => (0, 0, 1),
        };
        let vec = xp % 16 == 0 && total % 8 == 0;
        let f = if vec {
            &self.f.mx_quant8
        } else {
            &self.f.mx_quant
        };
        let threads = if vec { total / 8 } else { total };
        self.launch(
            f,
            Self::elems(threads),
            (256, 1, 1),
            &[
                Arg::P(xp),
                Arg::L(if vec { total } else { total / 32 } as i64),
                Arg::I(k as i32),
                Arg::P(cp),
                Arg::P(sp),
                Arg::P(out),
                Arg::I(mode),
            ],
        );
    }

    /// Time every heuristic candidate of an MXFP8 plan once on the real
    /// operands (the output is recomputed by the winner's later calls) and
    /// keep the fastest; `H3_CUDA_LT_NTH=i` picks candidate i instead.
    fn tune_mx(&self, lt: &Lt, p: &mut LtPlan, ap: u64, bp: u64, yp: u64) {
        p.tuned = true;
        if let Some(i) = self.lt_nth {
            if let Some(a) = p.candidates.get(i) {
                p.algo = *a;
            }
            return;
        }
        let (alpha, beta) = (1.0f32, 0.0f32);
        let ws = lt.workspace_ptr();
        let mut best: Option<(f64, usize)> = None;
        for (i, algo) in p.candidates.iter().enumerate() {
            self.stream.synchronize().expect("sync");
            let t = std::time::Instant::now();
            // SAFETY: the same pointers and layouts the caller just ran.
            let ok = unsafe {
                lt_result::matmul(
                    lt.handle,
                    p.desc,
                    (&alpha) as *const f32 as *const _,
                    (&beta) as *const f32 as *const _,
                    ap as *const _,
                    p.a,
                    bp as *const _,
                    p.b,
                    yp as *const _,
                    p.c,
                    yp as *mut _,
                    p.c,
                    algo,
                    ws as *mut _,
                    LT_WORKSPACE,
                    self.stream.cu_stream() as *mut _,
                )
                .is_ok()
            };
            self.stream.synchronize().expect("sync");
            let dt = t.elapsed().as_secs_f64();
            if ok && best.is_none_or(|(b, _)| dt < b) {
                best = Some((dt, i));
            }
        }
        if let Some((_, i)) = best {
            p.algo = p.candidates[i];
        }
    }

    /// 32-bit words (u32 tables or f32 scratch) in a table buffer.
    fn words(&self, bits: &[u32]) -> CuBuf {
        let v: Vec<f32> = bits.iter().map(|&b| f32::from_bits(b)).collect();
        Backend::upload_table(self, if v.is_empty() { &[0.0][..] } else { &v })
    }

    fn profile_table(&self) -> Option<String> {
        let prof = self.prof.as_ref()?;
        let m = prof.lock().unwrap();
        let flops = m.get("gemm FLOP (total)").map_or(0.0, |e| e.0);
        let mut rows: Vec<(&String, &(f64, usize))> = m
            .iter()
            .filter(|(k, _)| k.as_str() != "gemm FLOP (total)")
            .collect();
        rows.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
        let total: f64 = rows.iter().map(|r| r.1.0).sum();
        let gemm: f64 = rows
            .iter()
            .filter(|r| r.0.starts_with("gemm"))
            .map(|r| r.1.0)
            .sum();
        let mut out = format!(
            "profile (CUDA, synced per op): {:.1} ms in ops; GEMMs {:.1} ms at {:.1} TFLOP/s\n",
            total * 1e3,
            gemm * 1e3,
            if gemm > 0.0 { flops / gemm / 1e12 } else { 0.0 }
        );
        for (k, (t, n)) in rows {
            out += &format!(
                "  {k:40} {:9.1} ms {:5.1}%  {n:6} calls  {:8.3} ms/call\n",
                t * 1e3,
                100.0 * t / total.max(1e-12),
                t * 1e3 / *n as f64
            );
        }
        Some(out)
    }

    fn count_flops(&self, rows: usize, n: usize, k: usize) {
        if let Some(p) = &self.prof {
            let mut m = p.lock().unwrap();
            let e = m.entry("gemm FLOP (total)".into()).or_default();
            e.0 += 2.0 * rows as f64 * n as f64 * k as f64;
            e.1 += 1;
        }
    }

    /// Gate arguments: kind, vector, table, idx, slots, slot.
    fn gate_args(&self, g: &Gate<'_, CuBuf>) -> [Arg; 6] {
        match g {
            Gate::None => [
                Arg::I(0),
                Arg::P(0),
                Arg::P(0),
                Arg::P(0),
                Arg::I(0),
                Arg::I(0),
            ],
            Gate::Vector(v) => [
                Arg::I(1),
                Arg::P(self.ptr(v)),
                Arg::P(0),
                Arg::P(0),
                Arg::I(0),
                Arg::I(0),
            ],
            Gate::Table(m) => {
                assert!(
                    m.table.table.f32 && m.table.idx.f32,
                    "modulation tables come from upload_table / upload_u32"
                );
                [
                    Arg::I(2),
                    Arg::P(0),
                    Arg::P(self.ptr(m.table.table)),
                    Arg::P(self.ptr(m.table.idx)),
                    Arg::I(m.table.slots as i32),
                    Arg::I(m.slot as i32),
                ]
            }
        }
    }

    /// Modulation arguments: has_mod, table, idx, slots, scale, shift.
    fn mod_args(&self, m: &Modulation<'_, CuBuf>) -> [Arg; 6] {
        match m {
            Modulation::None => [
                Arg::I(0),
                Arg::P(0),
                Arg::P(0),
                Arg::I(0),
                Arg::I(0),
                Arg::I(0),
            ],
            Modulation::Table { m, scale, shift } => {
                assert!(
                    m.table.f32 && m.idx.f32,
                    "modulation tables come from upload_table / upload_u32"
                );
                [
                    Arg::I(1),
                    Arg::P(self.ptr(m.table)),
                    Arg::P(self.ptr(m.idx)),
                    Arg::I(m.slots as i32),
                    Arg::I(*scale as i32),
                    Arg::I(*shift as i32),
                ]
            }
        }
    }

    fn rms_impl(
        &self,
        h: &CuBuf,
        a: Option<&CuBuf>,
        rows: usize,
        dim: usize,
        gate: Gate<'_, CuBuf>,
        w: Option<&CuBuf>,
        m: Modulation<'_, CuBuf>,
        eps: f32,
        out: &CuBuf,
    ) {
        assert!(rows * dim <= h.len && rows * dim <= out.len);
        let mut args = vec![
            Arg::P(self.ptr(h)),
            Arg::P(self.opt(a)),
            Arg::P(self.ptr(out)),
            Arg::I(rows as i32),
            Arg::I(dim as i32),
            Arg::I(a.is_some() as i32),
        ];
        args.extend(self.gate_args(&gate));
        args.push(Arg::I(w.is_some() as i32));
        args.push(Arg::P(self.opt(w)));
        args.extend(self.mod_args(&m));
        args.push(Arg::F(eps));
        self.launch(&self.f.rms, (rows as u32, 1, 1), (256, 1, 1), &args);
    }

    fn epilogue_code<'a>(ep: &ConvEpilogue<'a, CuBuf>) -> (i32, Option<&'a CuBuf>) {
        match ep {
            ConvEpilogue::None => (0, None),
            ConvEpilogue::Relu => (1, None),
            ConvEpilogue::AddRelu(r) => (2, Some(r)),
            ConvEpilogue::Add(r) => (3, Some(r)),
            ConvEpilogue::Prelu(slope) => (4, Some(slope)),
        }
    }

    /// Grid (y, z) split for `mt` tiles along M.
    fn split_grid(mt: u32) -> (u32, u32) {
        if mt <= 65535 {
            (mt.max(1), 1)
        } else {
            let z = mt.div_ceil(65535);
            (mt.div_ceil(z), z)
        }
    }
}

/// Run the backend conformance checks on device `ordinal`.
pub fn conformance(
    ordinal: usize,
    tol: f32,
) -> Result<Vec<h3_model::check::CheckResult>, CudaError> {
    let be = Cuda::new(ordinal)?;
    Ok(h3_model::check::run_all(&be, tol))
}

impl Backend for Cuda {
    type Buf = CuBuf;
    type Vsa = CuVsa;
    type Weight = CuWeight;

    fn name(&self) -> String {
        self.name.clone()
    }

    fn alloc(&self, len: usize) -> CuBuf {
        CuBuf {
            s: Arc::new(
                self.stream
                    .alloc_zeros::<u16>(len.max(1))
                    .expect("cuda alloc"),
            ),
            len,
            f32: false,
        }
    }

    fn upload(&self, data: &[f32]) -> CuBuf {
        let h: Vec<u16> = data.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
        self.upload_u16(&h, false, data.len())
    }

    fn upload_table(&self, data: &[f32]) -> CuBuf {
        let h: Vec<u16> = data
            .iter()
            .flat_map(|v| {
                let b = v.to_bits();
                [(b & 0xffff) as u16, (b >> 16) as u16]
            })
            .collect();
        if h.is_empty() {
            return self.upload_u16(&[0, 0], true, 0);
        }
        self.upload_u16(&h, true, data.len())
    }

    fn upload_u32(&self, data: &[u32]) -> CuBuf {
        self.words(data)
    }

    fn write(&self, buf: &CuBuf, data: &[f32]) {
        assert!(data.len() <= buf.len);
        if buf.f32 {
            let h: Vec<u16> = data
                .iter()
                .flat_map(|v| {
                    let b = v.to_bits();
                    [(b & 0xffff) as u16, (b >> 16) as u16]
                })
                .collect();
            // SAFETY: `h` holds 2·data.len() u16 and the destination at least that many.
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    self.ptr(buf),
                    &h,
                    self.stream.cu_stream(),
                )
                .expect("write")
            };
        } else {
            let h: Vec<u16> = data.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
            // SAFETY: `h` is a host slice of `data.len()` u16s and the
            // destination holds at least that many; the copy is ordered on this
            // stream and `h` is kept alive until the stream is synchronised.
            unsafe {
                cudarc::driver::result::memcpy_htod_async(
                    self.ptr(buf),
                    &h,
                    self.stream.cu_stream(),
                )
                .expect("write")
            };
        }
        self.stream.synchronize().expect("sync");
    }

    fn download(&self, buf: &CuBuf) -> Vec<f32> {
        let h = self.stream.clone_dtoh(&*buf.s).expect("download");
        if buf.f32 {
            h[..2 * buf.len]
                .par_chunks_exact(2)
                .map(|p| f32::from_bits(p[0] as u32 | (p[1] as u32) << 16))
                .collect()
        } else {
            h[..buf.len]
                .par_iter()
                .map(|&b| bf16::from_bits(b).to_f32())
                .collect()
        }
    }

    fn buf_len(&self, buf: &CuBuf) -> usize {
        buf.len
    }

    fn copy(&self, src: &CuBuf, so: usize, dst: &CuBuf, d: usize, len: usize) {
        if len == 0 {
            return;
        }
        assert!(src.f32 == dst.f32 && so + len <= src.len && d + len <= dst.len);
        let es = if src.f32 { 4 } else { 2 };
        // SAFETY: both ranges are inside their allocations (asserted above).
        self.timed(
            || "copy (dtod)".into(),
            || unsafe {
                cudarc::driver::result::memcpy_dtod_async(
                    self.ptr(dst) + (d * es) as u64,
                    self.ptr(src) + (so * es) as u64,
                    len * es,
                    self.stream.cu_stream(),
                )
                .expect("dtod");
            },
        );
    }

    fn box_rows(&self, grid: &CuBuf, packed: &CuBuf, b: &BoxRows, scatter: bool) {
        let n = b.len();
        if n == 0 {
            return;
        }
        assert!(!grid.f32 && !packed.f32, "box_rows moves bf16 activations");
        // Eight bf16 per thread when the rows are whole 16-byte units (every
        // index is a multiple of `width`, so the 16-byte alignment holds).
        let vec = if b.width.is_multiple_of(8) { 8 } else { 1 };
        let (n, width) = (n / vec, b.width / vec);
        let l = |v: usize| Arg::L(v as i64);
        let i = |v: usize| Arg::I(v as i32);
        self.launch(
            if vec == 8 {
                &self.f.box_rows16
            } else {
                &self.f.box_rows
            },
            Self::elems(n),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(grid)),
                Arg::P(self.ptr(packed)),
                l(n),
                i(width),
                l(b.rows()),
                i(b.heads),
                i(b.count.1),
                i(b.grid.1),
                i(b.grid.2),
                i(b.t.0),
                i(b.y.0),
                i(b.x.0),
                i(b.y.1 - b.y.0),
                i(b.x.1 - b.x.0),
                i(b.step.0),
                i(b.step.1),
                l(b.grid_off),
                l(b.grid_head),
                l(b.grid_row),
                l(b.packed_off),
                l(b.packed_win),
                l(b.packed_head),
                l(b.packed_row),
                i(scatter as usize),
            ],
        );
    }

    fn fill(&self, buf: &CuBuf, off: usize, len: usize, value: f32) {
        if len == 0 {
            return;
        }
        assert!(!buf.f32 && off + len <= buf.len);
        self.launch(
            &self.f.fill,
            Self::elems(len),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(buf)),
                Arg::L(off as i64),
                Arg::L(len as i64),
                Arg::F(value),
            ],
        );
    }

    fn weight(&self, data: &[f32], rows: usize, cols: usize) -> CuWeight {
        assert_eq!(data.len(), rows * cols);
        CuWeight {
            buf: self.upload(data),
            rows,
            cols,
            quant: None,
        }
    }

    fn bf16_weights(&self) -> bool {
        true
    }

    fn weight_bf16(&self, data: &[u16], rows: usize, cols: usize) -> CuWeight {
        assert_eq!(data.len(), rows * cols);
        // SAFETY: the bf16 bits reinterpreted as their bytes; `upload_bytes` keeps the
        // page-locked interior as a DMA (streamed edge blocks: 770 MB each, no host pass).
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
        let mut buf = self.upload_bytes(bytes);
        buf.len = data.len();
        CuWeight {
            buf,
            rows,
            cols,
            quant: None,
        }
    }

    fn weight_q8(&self, q: &Q8) -> CuWeight {
        // SAFETY: `i8` and `u8` share size and alignment; the device reads the bit pattern.
        let bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(q.q.as_ptr() as *const u8, q.q.len()) };
        CuWeight {
            buf: self.upload_bytes(bytes),
            rows: q.rows,
            cols: q.cols,
            quant: Some((Quant::Q8, Backend::upload_table(self, &q.scales), q.group)),
        }
    }

    fn weight_q4(&self, q: &Q4) -> CuWeight {
        let bytes = q.packed();
        let sz: Vec<f32> = q
            .scales
            .iter()
            .zip(&q.zeros)
            .flat_map(|(&s, &z)| [s, z])
            .collect();
        CuWeight {
            buf: self.upload_bytes(&bytes),
            rows: q.rows,
            cols: q.cols,
            quant: Some((Quant::Q4, Backend::upload_table(self, &sz), q.group)),
        }
    }

    /// NVFP4 stays 4-bit on the device (codes + E4M3 scales, 0.53 B per
    /// weight) and is expanded into the shared bf16 scratch before each GEMM
    /// like int8 / int4 (a Blackwell FP4 tensor-core GEMM is future work).
    /// Expanding at load instead costs 2 B per weight: 36 GB for the 42-block
    /// Trim, which is what ran the L40S out of memory.
    fn weight_nvfp4(&self, q: &Nvfp4) -> CuWeight {
        // W4A4 weights go up in cuBLASLt's FP4 layout (the dequant fallback
        // reads it too); W4A16 ones as stored.
        let (gc, gs, first_high) = if q.act4 {
            let (packed, scales) = q.cublas_layout();
            (
                self.upload_bytes(&packed),
                self.upload_bytes(&scales),
                false,
            )
        } else {
            (
                self.upload_bytes(&q.packed),
                self.upload_bytes(&q.scales),
                q.first_high,
            )
        };
        CuWeight {
            buf: gc,
            rows: q.rows,
            cols: q.cols,
            quant: Some((
                Quant::Nvfp4 {
                    alpha: q.alpha,
                    first_high,
                    act4: q.act4,
                },
                gs,
                16,
            )),
        }
    }

    fn weight_mxfp8(&self, q: &Mxfp8) -> CuWeight {
        CuWeight {
            buf: self.upload_bytes(&q.codes),
            rows: q.rows,
            cols: q.cols,
            quant: Some((Quant::Mxfp8, self.upload_bytes(&q.scales), 32)),
        }
    }

    fn prefetch_begin(&self) {
        if let Some(ev) = self.last_fence.lock().unwrap().as_ref() {
            // Every earlier prefetch has run its copies (some from the bounce
            // buffer) before the buffer is reused: the host blocks here until
            // the copy stream has run the newest, at most two blocks behind.
            ev.synchronize().expect("event");
        }
        if let Some((_, used)) = self.bounce.lock().unwrap().as_mut() {
            *used = 0;
        }
        if let Some(st) = &self.stall {
            let mut st = st.lock().unwrap();
            st.bytes = 0;
            // Recorded behind the release wait already queued on the copy
            // stream, so it marks when this prefetch's copies actually start.
            st.open = Some(self.timed_event(&self.copy));
        }
        self.staging
            .store(true, std::sync::atomic::Ordering::Relaxed);
    }

    fn prefetch_end(&self) {
        self.staging
            .store(false, std::sync::atomic::Ordering::Relaxed);
        let ev = Arc::new(if self.stall.is_some() {
            self.timed_event(&self.copy)
        } else {
            self.copy.record_event(None).expect("event")
        });
        if let Some(st) = &self.stall {
            let mut st = st.lock().unwrap();
            if let Some(e0) = st.open.take() {
                let bytes = st.bytes;
                st.copies.push((e0, Arc::clone(&ev), bytes));
            }
        }
        *self.last_fence.lock().unwrap() = Some(Arc::clone(&ev));
        self.fence.lock().unwrap().push_back(ev);
    }

    /// Page-lock the whole pages inside `bytes` (never the partial edge pages:
    /// an allocation can share those with its neighbours, and a copy from a
    /// buffer that straddles a locked page fails with INVALID_VALUE, which is
    /// what registering page-rounded ranges did). `H3_CUDA_NO_PIN=1` disables.
    fn pin_host(&self, bytes: &[u8]) -> Option<Box<dyn std::any::Any + Send>> {
        const PAGE: usize = 4096;
        if !self.use_pin || bytes.is_empty() {
            return None;
        }
        let a = bytes.as_ptr() as usize;
        let start = a.div_ceil(PAGE) * PAGE;
        let end = ((a + bytes.len()) / PAGE) * PAGE;
        if end <= start {
            return None;
        }
        let mut reg = self.pinned.lock().unwrap();
        if reg.iter().any(|&(s, e)| start < e && end > s) {
            return None;
        }
        // SAFETY: whole pages owned by this allocation, readable; the guard
        // unregisters them before the allocation is freed (it is dropped first).
        let r =
            unsafe { cudarc::driver::sys::cuMemHostRegister_v2(start as *mut _, end - start, 0) };
        if r == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            reg.push((start, end));
            Some(Box::new(PinGuard {
                start,
                end,
                ranges: Arc::clone(&self.pinned),
            }))
        } else {
            if !self
                .pin_warned
                .swap(true, std::sync::atomic::Ordering::Relaxed)
            {
                eprintln!(
                    "warning: cuMemHostRegister failed ({r:?}); streamed weights copy from pageable memory"
                );
            }
            None
        }
    }

    fn prefetch_wait(&self) {
        // The front is the block about to run; each fence is waited on once.
        // The host-side sync before the bounce buffer is reused is
        // `prefetch_begin`'s, on `last_fence`: taking the only fence there
        // (29317cb) let the next prefetch overwrite bounce slots whose copies
        // were still queued behind a release wait, and keeping the front here
        // when it was the only entry (3eb1d82) made every block after the
        // first wait on the previous block's fence.
        let ev = self.fence.lock().unwrap().pop_front();
        if let Some(ev) = ev {
            if let Some(st) = &self.stall {
                let w0 = self.timed_event(&self.stream);
                self.stream.wait(&*ev).expect("wait");
                let w1 = self.timed_event(&self.stream);
                st.lock().unwrap().waits.push((w0, w1));
            } else {
                self.stream.wait(&*ev).expect("wait");
            }
        }
    }

    fn prefetch_release(&self) {
        let ev = self.stream.record_event(None).expect("event");
        self.copy.wait(&ev).expect("wait");
    }

    fn weight_bytes(&self, w: &CuWeight) -> usize {
        w.buf.len * 2
            + w.quant
                .as_ref()
                .map_or(0, |(_, s, _)| s.len * if s.f32 { 4 } else { 2 })
    }

    fn memory_free(&self) -> Option<usize> {
        self.ctx.mem_get_info().ok().map(|(free, _)| free)
    }

    fn sync(&self) {
        self.stream.synchronize().expect("sync");
    }

    fn profile_report(&self) -> Option<String> {
        let mut out = self.profile_table().unwrap_or_default();
        if let Some(st) = &self.stall {
            self.stream.synchronize().expect("sync");
            self.copy.synchronize().expect("sync");
            let mut st = st.lock().unwrap();
            if !st.waits.is_empty() || !st.copies.is_empty() {
                let wait: f64 = st
                    .waits
                    .iter()
                    .map(|(a, b)| a.elapsed_ms(b).unwrap_or(0.0) as f64)
                    .sum();
                let worst = st
                    .waits
                    .iter()
                    .map(|(a, b)| a.elapsed_ms(b).unwrap_or(0.0))
                    .fold(0f32, f32::max);
                let copy: f64 = st
                    .copies
                    .iter()
                    .map(|(a, b, _)| a.elapsed_ms(b).unwrap_or(0.0) as f64)
                    .sum();
                let bytes: usize = st.copies.iter().map(|c| c.2).sum();
                out += &format!(
                    "prefetch: {} blocks; compute stream waited {:.0} ms total, {:.1} ms/block, worst {:.1} ms; copies {:.0} ms for {:.2} GB ({:.1} GB/s while copying, {:.2} GB/block)\n",
                    st.waits.len(),
                    wait,
                    wait / st.waits.len().max(1) as f64,
                    worst,
                    copy,
                    bytes as f64 / 1e9,
                    if copy > 0.0 {
                        bytes as f64 / 1e9 / (copy / 1e3)
                    } else {
                        0.0
                    },
                    bytes as f64 / 1e9 / st.copies.len().max(1) as f64,
                );
                st.waits.clear();
                st.copies.clear();
            }
        }
        (!out.is_empty()).then_some(out)
    }

    fn profile_reset(&self) {
        if let Some(p) = &self.prof {
            self.stream.synchronize().expect("sync");
            p.lock().unwrap().clear();
        }
        if let Some(st) = &self.stall {
            self.stream.synchronize().expect("sync");
            self.copy.synchronize().expect("sync");
            let mut st = st.lock().unwrap();
            st.waits.clear();
            st.copies.clear();
        }
    }

    fn linear(
        &self,
        x: &CuBuf,
        x_row: usize,
        rows: usize,
        w: &CuWeight,
        bias: Option<&CuBuf>,
        act: Act,
        y: &CuBuf,
        y_row: usize,
    ) {
        let (n, k) = (w.rows, w.cols);
        assert!((x_row + rows) * k <= x.len && (y_row + rows) * n <= y.len);
        if rows == 0 {
            return;
        }
        let act_code = match act {
            Act::None => 0,
            Act::GeluTanh => 1,
            Act::GeluErf => 2,
            Act::Silu => 3,
        };
        let yp = self.ptr_at(y, y_row * n);
        let mut xp = self.ptr_at(x, x_row * k);
        let mut scratch = self.q8_scratch.lock().unwrap();
        let mut mx_scratch = self.mx_scratch.lock().unwrap();
        let mut nv_scratch = self.nv_scratch.lock().unwrap();
        if let Some((
            Quant::Nvfp4 {
                alpha: walpha,
                act4: true,
                ..
            },
            wscales,
            _,
        )) = &w.quant
        {
            // NVFP4 W4A4: the call's maximum, then the FP4 tensor cores
            // (sm_100+) on E2M1 codes with VEC16_UE4M3 scales and the tensor
            // scale as a device-side alpha; elsewhere the activations are
            // rounded into bf16 (`Nvfp4::round_rows`) and the GEMM below runs
            // on the expanded weight.
            let codes_len = (rows * k).div_ceil(2);
            let sc_len = h3_core::quant::nvfp4_scale_len(rows, k);
            if nv_scratch.as_ref().is_none_or(|(_, c, s, a)| {
                c.len < codes_len.div_ceil(2) || s.len < sc_len.div_ceil(2) || a.len < rows * k
            }) {
                *nv_scratch = Some((
                    self.alloc(8),
                    self.alloc(codes_len.div_ceil(2)),
                    self.alloc(sc_len.div_ceil(2)),
                    self.alloc(rows * k),
                ));
            }
            let (state, codes, ascales, xround) = nv_scratch.as_ref().unwrap();
            let stp = self.ptr(state);
            self.launch(&self.f.nv_prep, (1, 1, 1), (1, 1, 1), &[Arg::P(stp)]);
            let total = rows * k;
            self.launch(
                &self.f.nv_amax,
                ((total.div_ceil(256)).clamp(1, 1024) as u32, 1, 1),
                (256, 1, 1),
                &[Arg::P(xp), Arg::L(total as i64), Arg::P(stp)],
            );
            let blocks = total / 16;
            let quant = |mode: i32, out: u64| {
                self.launch(
                    &self.f.nv_quant,
                    Self::elems(blocks),
                    (256, 1, 1),
                    &[
                        Arg::P(xp),
                        Arg::L(blocks as i64),
                        Arg::I(k as i32),
                        Arg::P(stp),
                        Arg::F(*walpha),
                        Arg::P(self.ptr(codes)),
                        Arg::P(self.ptr(ascales)),
                        Arg::P(out),
                        Arg::I(mode),
                    ],
                )
            };
            let aligned = (y_row * n) % 8 == 0 && n % 32 == 0;
            if let (Some(lt), true, true) = (
                &self.lt,
                self.sm_major >= 10 && !self.no_fp8 && aligned,
                k % 32 == 0,
            ) {
                let epi: u8 = match (bias.is_some(), act) {
                    (false, _) => 0,
                    (true, Act::GeluTanh) => 2,
                    (true, _) => 1,
                };
                let mut plans = lt.fp4_plans.lock().unwrap();
                let epi = if epi == 2
                    && plans
                        .entry((n, k, rows, 2))
                        .or_insert_with(|| lt.plan_fp4(n, k, rows, n, 2))
                        .is_none()
                {
                    1
                } else {
                    epi
                };
                if let Some(p) = plans
                    .entry((n, k, rows, epi))
                    .or_insert_with(|| lt.plan_fp4(n, k, rows, n, epi))
                    .as_ref()
                {
                    quant(0, 0);
                    let (ap, bp) = (self.ptr(&w.buf), self.ptr(codes));
                    let (asp, bsp) = (self.ptr(wscales), self.ptr(ascales));
                    let (alpha_p, beta_p) = (stp + 4, stp + 8);
                    let ws = lt.workspace_ptr();
                    // SAFETY: the plan's layouts match these extents; the scale
                    // tensors are VEC16_UE4M3 of the two operands; alpha / beta
                    // are f32 in the state buffer (POINTER_MODE_DEVICE); bias
                    // holds n bf16 values; workspace is LT_WORKSPACE.
                    let tag = || {
                        format!(
                            "gemm nvfp4 w4a4{} n={n} k={k}{}",
                            if epi == 2 { "+gelu" } else { "" },
                            if rows < 1024 { " (small)" } else { "" }
                        )
                    };
                    self.count_flops(rows, n, k);
                    self.timed(tag, || unsafe {
                        use lt_sys::cublasLtMatmulDescAttributes_t as A;
                        for (attr, v) in [
                            (A::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, asp),
                            (A::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, bsp),
                        ] {
                            lt_result::set_matmul_desc_attribute(
                                p.desc,
                                attr,
                                (&v) as *const u64 as *const _,
                                8,
                            )
                            .expect("lt fp4 scale");
                        }
                        if let Some(b) = bias {
                            let bp = self.ptr(b);
                            lt_result::set_matmul_desc_attribute(
                                p.desc,
                                A::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                (&bp) as *const u64 as *const _,
                                8,
                            )
                            .expect("lt fp4 bias");
                        }
                        lt_result::matmul(
                            lt.handle,
                            p.desc,
                            alpha_p as *const _,
                            beta_p as *const _,
                            ap as *const _,
                            p.a,
                            bp as *const _,
                            p.b,
                            yp as *const _,
                            p.c,
                            yp as *mut _,
                            p.c,
                            &p.algo,
                            ws as *mut _,
                            LT_WORKSPACE,
                            self.stream.cu_stream() as *mut _,
                        )
                        .expect("cublasLt fp4 matmul");
                    });
                    let left = if act == Act::GeluTanh && epi == 2 {
                        0
                    } else {
                        act_code
                    };
                    if left != 0 || bias.is_some() && epi == 0 {
                        let total = rows * n;
                        self.launch(
                            &self.f.bias_act,
                            Self::elems(total),
                            (256, 1, 1),
                            &[
                                Arg::P(yp),
                                Arg::P(0),
                                Arg::L(total as i64),
                                Arg::I(n as i32),
                                Arg::I(0),
                                Arg::I(left),
                            ],
                        );
                    }
                    return;
                }
            }
            quant(1, self.ptr(xround));
            xp = self.ptr(xround);
        }
        if let Some((Quant::Mxfp8, wscales, _)) = &w.quant {
            // W8A8 on the FP8 tensor cores (sm_100+): quantize the activation
            // rows to E4M3 codes + swizzled E8M0 scales, then cuBLASLt with
            // VEC32_UE8M0 block scaling. Anything else emulates in bf16 below.
            if mx_scratch.as_ref().is_none_or(|(c, s, a)| {
                c.len < (rows * k).div_ceil(2)
                    || s.len < mx_scale_len(rows, k).div_ceil(2)
                    || a.len < rows * k
            }) {
                *mx_scratch = Some((
                    self.alloc((rows * k).div_ceil(2)),
                    self.alloc(mx_scale_len(rows, k).div_ceil(2)),
                    self.alloc(rows * k),
                ));
            }
            let (codes, ascales, xround) = mx_scratch.as_ref().unwrap();
            let aligned = (y_row * n) % 8 == 0 && (x_row * k) % 8 == 0 && n % 8 == 0;
            if let (Some(lt), true, true) = (
                &self.lt,
                self.sm_major >= 10 && !self.no_fp8 && aligned,
                k % 32 == 0,
            ) {
                let epi: u8 = match (bias.is_some(), act) {
                    (false, _) => 0,
                    (true, Act::GeluTanh) => 2,
                    (true, _) => 1,
                };
                let mut plans = lt.mx_plans.lock().unwrap();
                // Row slices of the weight when the shape is one cuBLASLt runs
                // badly whole. `H3_CUDA_MX_SPLIT_MIN=n` forces the rule; by
                // default each large shape times whole against split once, on
                // its first call, and keeps the faster (`mx_choice`).
                let splittable = n % (128 * self.mx_split) == 0 && n >= 16384 && rows >= 1024;
                let key = (n, k, rows);
                let parts = match self.mx_split_min {
                    Some(min) => {
                        if n >= min && n % (128 * self.mx_split) == 0 {
                            self.mx_split
                        } else {
                            1
                        }
                    }
                    None => self
                        .mx_choice
                        .lock()
                        .unwrap()
                        .get(&key)
                        .copied()
                        .unwrap_or(1),
                };
                let np = n / parts;
                // A GELU epilogue cuBLASLt declines falls back to bias-only + a pass.
                let epi = if epi == 2
                    && plans
                        .entry((np, k, rows, n, 2))
                        .or_insert_with(|| lt.plan_mx(np, k, rows, n, 2))
                        .is_none()
                {
                    1
                } else {
                    epi
                };
                if plans
                    .entry((np, k, rows, n, epi))
                    .or_insert_with(|| lt.plan_mx(np, k, rows, n, epi))
                    .is_some()
                {
                    self.mx_quantize(
                        xp,
                        rows * k,
                        k,
                        Some((self.ptr(codes), self.ptr(ascales))),
                        0,
                    );
                    let (ap0, bp) = (self.ptr(&w.buf), self.ptr(codes));
                    let (asp0, bsp) = (self.ptr(wscales), self.ptr(ascales));
                    let biasp0 = bias.map(|b| self.ptr(b));
                    let ws = lt.workspace_ptr();
                    let (alpha, beta) = (1.0f32, 0.0f32);
                    // Part i of `parts`: weight rows [i·np, (i+1)·np) (codes contiguous,
                    // scales whole 128-row tiles), output columns i·np.. of the n-wide y.
                    let slice = |i: usize, np: usize| {
                        (
                            ap0 + (i * np * k) as u64,
                            asp0 + (i * mx_scale_len(np, k)) as u64,
                            yp + (i * np * 2) as u64,
                            biasp0.map(|b| b + (i * np * 2) as u64),
                        )
                    };
                    // SAFETY (the closure): the plan's layouts match these extents;
                    // the scale tensors are VEC32_UE8M0 of the two operands; bias
                    // holds n bf16 values; workspace is LT_WORKSPACE.
                    let run = |p: &LtPlan, parts: usize| unsafe {
                        use lt_sys::cublasLtMatmulDescAttributes_t as A;
                        let np = n / parts;
                        for i in 0..parts {
                            let (ap, asp, dp, bias_p) = slice(i, np);
                            lt_result::set_matmul_desc_attribute(
                                p.desc,
                                A::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                (&asp) as *const u64 as *const _,
                                8,
                            )
                            .expect("lt a scale");
                            lt_result::set_matmul_desc_attribute(
                                p.desc,
                                A::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                (&bsp) as *const u64 as *const _,
                                8,
                            )
                            .expect("lt b scale");
                            if let Some(bp) = bias_p {
                                lt_result::set_matmul_desc_attribute(
                                    p.desc,
                                    A::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                    (&bp) as *const u64 as *const _,
                                    8,
                                )
                                .expect("lt bias pointer");
                            }
                            lt_result::matmul(
                                lt.handle,
                                p.desc,
                                (&alpha) as *const f32 as *const _,
                                (&beta) as *const f32 as *const _,
                                ap as *const _,
                                p.a,
                                bp as *const _,
                                p.b,
                                dp as *const _,
                                p.c,
                                dp as *mut _,
                                p.c,
                                &p.algo,
                                ws as *mut _,
                                LT_WORKSPACE,
                                self.stream.cu_stream() as *mut _,
                            )
                            .expect("cublasLt fp8 matmul");
                        }
                    };
                    // Tune a plan's candidates on part 0 of its own slicing.
                    let tune = |this: &Self, p: &mut LtPlan, parts: usize| {
                        let (ap, asp, dp, bias_p) = slice(0, n / parts);
                        // SAFETY: the pointers of part 0, as `run` uses them.
                        unsafe {
                            use lt_sys::cublasLtMatmulDescAttributes_t as A;
                            lt_result::set_matmul_desc_attribute(
                                p.desc,
                                A::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                (&asp) as *const u64 as *const _,
                                8,
                            )
                            .expect("lt a scale");
                            lt_result::set_matmul_desc_attribute(
                                p.desc,
                                A::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                (&bsp) as *const u64 as *const _,
                                8,
                            )
                            .expect("lt b scale");
                            if let Some(bp) = bias_p {
                                lt_result::set_matmul_desc_attribute(
                                    p.desc,
                                    A::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                    (&bp) as *const u64 as *const _,
                                    8,
                                )
                                .expect("lt bias pointer");
                            }
                        }
                        this.tune_mx(lt, p, ap, bp, dp);
                    };
                    {
                        let p = plans
                            .get_mut(&(np, k, rows, n, epi))
                            .unwrap()
                            .as_mut()
                            .unwrap();
                        self.timed(
                            || {
                                format!(
                                    "gemm mxfp8{} n={n} k={k}{}{}",
                                    if epi == 2 { "+gelu" } else { "" },
                                    if parts > 1 {
                                        format!(" (×{parts})")
                                    } else {
                                        String::new()
                                    },
                                    if rows < 1024 { " (small)" } else { "" }
                                )
                            },
                            || run(p, parts),
                        );
                        self.count_flops(rows, n, k);
                        if !p.tuned {
                            tune(self, p, parts);
                        }
                    }
                    if self.mx_split_min.is_none()
                        && splittable
                        && !self.mx_choice.lock().unwrap().contains_key(&key)
                    {
                        // First call on this shape: the split plan, tuned, against the whole one.
                        let sp = self.mx_split;
                        let sk = (n / sp, k, rows, n, epi);
                        if let Some(p) = plans
                            .entry(sk)
                            .or_insert_with(|| lt.plan_mx(n / sp, k, rows, n, epi))
                            .as_mut()
                        {
                            if !p.tuned {
                                tune(self, p, sp);
                            }
                        }
                        let mut choice = 1;
                        // The plain-FP8 probe below writes y too (unscaled codes: garbage).
                        let mut probed = false;
                        if let (Some(Some(whole)), Some(Some(split))) =
                            (plans.get(&(n, k, rows, n, epi)), plans.get(&sk))
                        {
                            let time = |p: &LtPlan, parts: usize| {
                                self.stream.synchronize().expect("sync");
                                let t = std::time::Instant::now();
                                run(p, parts);
                                self.stream.synchronize().expect("sync");
                                t.elapsed().as_secs_f64()
                            };
                            let (tw, ts) = (time(whole, 1), time(split, sp));
                            // Diagnostics only: the same shape as plain FP8 (per-tensor scales).
                            let plain = if self.prof.is_some() || self.stall.is_some() {
                                lt.plan_fp8_plain(n, k, rows, n).map(|mut pp| {
                                    let (ap, _asp, dp, _b) = slice(0, n);
                                    self.tune_mx(lt, &mut pp, ap, bp, dp);
                                    // SAFETY: the plan's layouts match the operands' extents; the
                                    // E4M3 codes are read as plain FP8, the output is scratch.
                                    let mut best = f32::INFINITY;
                                    for _ in 0..2 {
                                        self.stream.synchronize().expect("sync");
                                        let t = std::time::Instant::now();
                                        unsafe {
                                            lt_result::matmul(
                                                lt.handle,
                                                pp.desc,
                                                (&alpha) as *const f32 as *const _,
                                                (&beta) as *const f32 as *const _,
                                                ap as *const _,
                                                pp.a,
                                                bp as *const _,
                                                pp.b,
                                                dp as *const _,
                                                pp.c,
                                                dp as *mut _,
                                                pp.c,
                                                &pp.algo,
                                                ws as *mut _,
                                                LT_WORKSPACE,
                                                self.stream.cu_stream() as *mut _,
                                            )
                                            .expect("plain fp8");
                                        }
                                        self.stream.synchronize().expect("sync");
                                        best = best.min(t.elapsed().as_secs_f32());
                                    }
                                    best as f64
                                })
                            } else {
                                None
                            };
                            // Only a clear win: the two big shapes split 1.75× faster on a
                            // 5090, while a 0.2 ms shape measured the same either way and a
                            // noise-driven split there changed the clip's bits (the slices'
                            // rounding matches the whole GEMM's only when cuBLASLt picks
                            // the same kind of kernel for both).
                            if ts < tw * 0.9 {
                                choice = sp;
                            }
                            probed = plain.is_some();
                            if self.prof.is_some() || self.stall.is_some() {
                                eprintln!(
                                    "mxfp8 n={n} k={k} rows={rows}: whole {:.2} ms, ×{sp} {:.2} ms → {}{}",
                                    tw * 1e3,
                                    ts * 1e3,
                                    if choice > 1 {
                                        format!("×{sp}")
                                    } else {
                                        "whole".into()
                                    },
                                    plain.map_or(String::new(), |t| format!(
                                        "; plain fp8 (per-tensor scale) {:.2} ms",
                                        t * 1e3
                                    ))
                                );
                            }
                        }
                        self.mx_choice.lock().unwrap().insert(key, choice);
                        if choice != sp || probed {
                            // y must come from the chosen plan: the timing runs left the
                            // split's output, the plain-FP8 probe its unscaled one. The first
                            // version recomputed only when the choice differed from the
                            // initial `parts`, so a shape that kept whole returned the split's
                            // rounding on its first call (the clip moved into the usual band
                            // against the previous binary's on a Server Edition, where the QKV
                            // shape stays whole) and, under `H3_STALL=1` / `H3_PROFILE=1`, the
                            // probe's garbage (every diagnostic clip of that run, and a decoder
                            // tile on the 5090; `docs/runs/2026-11-14-runpod-6000s-probe-bug`).
                            let p = plans
                                .get(&(n / choice, k, rows, n, epi))
                                .unwrap()
                                .as_ref()
                                .unwrap();
                            run(p, choice);
                        }
                    }
                    // The epilogue did the bias (and tanh-GELU when epi == 2); apply what is left.
                    let left = if act == Act::GeluTanh && epi == 2 {
                        0
                    } else {
                        act_code
                    };
                    if left != 0 || bias.is_some() && epi == 0 {
                        let total = rows * n;
                        self.launch(
                            &self.f.bias_act,
                            Self::elems(total),
                            (256, 1, 1),
                            &[
                                Arg::P(yp),
                                Arg::P(0),
                                Arg::L(total as i64),
                                Arg::I(n as i32),
                                Arg::I(0),
                                Arg::I(left),
                            ],
                        );
                    }
                    return;
                }
            }
            // Emulation: round the activations through E4M3 / E8M0 into bf16 (exact), then the bf16 GEMM below.
            self.mx_quantize(xp, rows * k, k, None, self.ptr(xround));
            xp = self.ptr(xround);
        }
        let wp = match &w.quant {
            None => self.ptr(&w.buf),
            Some((kind, scales, group)) => {
                if scratch.as_ref().is_none_or(|b| b.len < n * k) {
                    *scratch = Some(self.alloc(n * k));
                }
                let dst = scratch.as_ref().unwrap();
                match kind {
                    Quant::Q8 | Quant::Q4 => self.launch(
                        if *kind == Quant::Q8 {
                            &self.f.dequant_q8
                        } else {
                            &self.f.dequant_q4
                        },
                        Self::elems(n * k),
                        (256, 1, 1),
                        &[
                            Arg::P(self.ptr(dst)),
                            Arg::P(self.ptr(&w.buf)),
                            Arg::P(self.ptr(scales)),
                            Arg::L((n * k) as i64),
                            Arg::I(*group as i32),
                        ],
                    ),
                    Quant::Nvfp4 {
                        alpha,
                        first_high,
                        act4,
                    } => self.launch(
                        &self.f.dequant_nvfp4,
                        Self::elems(n * k),
                        (256, 1, 1),
                        &[
                            Arg::P(self.ptr(dst)),
                            Arg::P(self.ptr(&w.buf)),
                            Arg::P(self.ptr(scales)),
                            Arg::L((n * k) as i64),
                            Arg::I(k as i32),
                            Arg::F(*alpha),
                            Arg::I(*first_high as i32),
                            Arg::I(*act4 as i32),
                        ],
                    ),
                    Quant::Mxfp8 => self.launch(
                        &self.f.dequant_mxfp8,
                        Self::elems(n * k),
                        (256, 1, 1),
                        &[
                            Arg::P(self.ptr(dst)),
                            Arg::P(self.ptr(&w.buf)),
                            Arg::P(self.ptr(scales)),
                            Arg::L((n * k) as i64),
                            Arg::I(k as i32),
                        ],
                    ),
                }
                self.ptr(dst)
            }
        };
        let (alpha, beta) = (1.0f32, 0.0f32);
        // cuBLASLt with the bias (+ tanh-GELU) in the epilogue: no second pass.
        if let (Some(lt), Some(b)) = (&self.lt, bias) {
            let epi = (act == Act::GeluTanh) as u8;
            let mut plans = lt.plans.lock().unwrap();
            let plan = plans
                .entry((n, k, rows, epi))
                .or_insert_with(|| lt.plan(n, k, rows, epi));
            if let Some(p) = plan {
                let bp = self.ptr(b);
                let ws = lt.workspace_ptr();
                self.timed(
                    || format!("gemm+bias{} n={n} k={k}{}", if epi == 1 { "+gelu" } else { "" }, if rows < 1024 { " (small)" } else { "" }),
                    // SAFETY: the plan's layouts match these extents (checked
                    // above); bias holds n values; workspace is LT_WORKSPACE.
                    || unsafe {
                        lt_result::set_matmul_desc_attribute(
                            p.desc,
                            lt_sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                            (&bp) as *const u64 as *const _,
                            8,
                        )
                        .expect("lt bias pointer");
                        lt_result::matmul(
                            lt.handle,
                            p.desc,
                            (&alpha) as *const f32 as *const _,
                            (&beta) as *const f32 as *const _,
                            wp as *const _,
                            p.a,
                            xp as *const _,
                            p.b,
                            yp as *const _,
                            p.c,
                            yp as *mut _,
                            p.c,
                            &p.algo,
                            ws as *mut _,
                            LT_WORKSPACE,
                            self.stream.cu_stream() as *mut _,
                        )
                        .expect("cublasLt matmul");
                    },
                );
                self.count_flops(rows, n, k);
                if act_code >= 2 {
                    let total = rows * n;
                    self.launch(
                        &self.f.bias_act,
                        Self::elems(total),
                        (256, 1, 1),
                        &[
                            Arg::P(yp),
                            Arg::P(0),
                            Arg::L(total as i64),
                            Arg::I(n as i32),
                            Arg::I(0),
                            Arg::I(act_code),
                        ],
                    );
                }
                return;
            }
        }
        // Row-major y[m,n] = x[m,k]·W[n,k]ᵀ is column-major yᵀ = W·xᵀ:
        // A = W (transposed), B = x, C = y, all with leading dims k, k, n.
        // SAFETY: pointers and extents checked above; alpha/beta outlive the call.
        self.timed(
            || {
                format!(
                    "gemm n={n} k={k}{}",
                    if rows < 1024 { " (small)" } else { "" }
                )
            },
            || unsafe {
                cudarc::cublas::result::gemm_ex(
                    *self.blas.handle(),
                    blas_sys::cublasOperation_t::CUBLAS_OP_T,
                    blas_sys::cublasOperation_t::CUBLAS_OP_N,
                    n as i32,
                    rows as i32,
                    k as i32,
                    (&alpha) as *const f32 as *const _,
                    wp as *const _,
                    blas_sys::cudaDataType_t::CUDA_R_16BF,
                    k as i32,
                    xp as *const _,
                    blas_sys::cudaDataType_t::CUDA_R_16BF,
                    k as i32,
                    (&beta) as *const f32 as *const _,
                    yp as *mut _,
                    blas_sys::cudaDataType_t::CUDA_R_16BF,
                    n as i32,
                    blas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    blas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
                .expect("cublas gemm");
            },
        );
        self.count_flops(rows, n, k);
        if bias.is_some() || act != Act::None {
            let total = rows * n;
            self.launch(
                &self.f.bias_act,
                Self::elems(total),
                (256, 1, 1),
                &[
                    Arg::P(yp),
                    Arg::P(self.opt(bias)),
                    Arg::L(total as i64),
                    Arg::I(n as i32),
                    Arg::I(bias.is_some() as i32),
                    Arg::I(act_code),
                ],
            );
        }
    }

    fn linear_add(
        &self,
        x: &CuBuf,
        x_row: usize,
        rows: usize,
        w: &CuWeight,
        out: usize,
        y: &CuBuf,
        y_row: usize,
    ) {
        let (n, k) = (w.rows, w.cols);
        assert_eq!(n, out, "linear_add: weight rows {n} != output width {out}");
        assert!((x_row + rows) * k <= x.len && (y_row + rows) * n <= y.len);
        if rows == 0 {
            return;
        }
        if w.quant.is_some() {
            // LoRA factors are plain bf16; anything else takes the composite.
            let chunk = ((1usize << 24) / n.max(1)).clamp(1, rows);
            let scratch = self.alloc(chunk * n);
            let mut r = 0;
            while r < rows {
                let m = chunk.min(rows - r);
                self.linear(x, x_row + r, m, w, None, Act::None, &scratch, 0);
                self.axpby(y, (y_row + r) * n, 1.0, &scratch, 0, 1.0, m * n);
                r += m;
            }
            return;
        }
        let (yp, xp, wp) = (
            self.ptr_at(y, y_row * n),
            self.ptr_at(x, x_row * k),
            self.ptr(&w.buf),
        );
        let (alpha, beta) = (1.0f32, 1.0f32);
        // As in `linear`'s fallback (column-major yᵀ = W·xᵀ), with beta = 1:
        // the GEMM reads y and writes y + x·Wᵀ in one pass.
        // SAFETY: pointers and extents checked above; alpha/beta outlive the call.
        self.timed(
            || {
                format!(
                    "gemm+acc n={n} k={k}{}",
                    if rows < 1024 { " (small)" } else { "" }
                )
            },
            || unsafe {
                cudarc::cublas::result::gemm_ex(
                    *self.blas.handle(),
                    blas_sys::cublasOperation_t::CUBLAS_OP_T,
                    blas_sys::cublasOperation_t::CUBLAS_OP_N,
                    n as i32,
                    rows as i32,
                    k as i32,
                    (&alpha) as *const f32 as *const _,
                    wp as *const _,
                    blas_sys::cudaDataType_t::CUDA_R_16BF,
                    k as i32,
                    xp as *const _,
                    blas_sys::cudaDataType_t::CUDA_R_16BF,
                    k as i32,
                    (&beta) as *const f32 as *const _,
                    yp as *mut _,
                    blas_sys::cudaDataType_t::CUDA_R_16BF,
                    n as i32,
                    blas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                    blas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
                )
                .expect("cublas gemm (accumulate)");
            },
        );
        self.count_flops(rows, n, k);
    }

    fn swiglu(&self, x: &CuBuf, rows: usize, dim: usize, out: &CuBuf) {
        let n = rows * dim;
        assert!(2 * n <= x.len && n <= out.len);
        self.launch(
            &self.f.swiglu,
            Self::elems(n),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(x)),
                Arg::P(self.ptr(out)),
                Arg::L(n as i64),
                Arg::I(dim as i32),
            ],
        );
    }

    fn rms_norm(
        &self,
        x: &CuBuf,
        rows: usize,
        dim: usize,
        w: Option<&CuBuf>,
        modulation: Modulation<'_, CuBuf>,
        eps: f32,
        out: &CuBuf,
    ) {
        self.rms_impl(x, None, rows, dim, Gate::None, w, modulation, eps, out);
    }

    fn residual_rms_norm(
        &self,
        h: &CuBuf,
        a: &CuBuf,
        rows: usize,
        dim: usize,
        gate: Gate<'_, CuBuf>,
        w: Option<&CuBuf>,
        modulation: Modulation<'_, CuBuf>,
        eps: f32,
        normed: &CuBuf,
    ) {
        self.rms_impl(h, Some(a), rows, dim, gate, w, modulation, eps, normed);
    }

    fn gate_residual(&self, h: &CuBuf, a: &CuBuf, rows: usize, dim: usize, gate: Gate<'_, CuBuf>) {
        let n = rows * dim;
        assert!(n <= h.len && n <= a.len);
        let mut args = vec![
            Arg::P(self.ptr(h)),
            Arg::P(self.ptr(a)),
            Arg::L(n as i64),
            Arg::I(dim as i32),
        ];
        args.extend(self.gate_args(&gate));
        self.launch(&self.f.gate_residual, Self::elems(n), (256, 1, 1), &args);
    }

    fn layer_norm(
        &self,
        x: &CuBuf,
        rows: usize,
        dim: usize,
        w: Option<&CuBuf>,
        b: Option<&CuBuf>,
        eps: f32,
        out: &CuBuf,
    ) {
        assert!(rows * dim <= x.len && rows * dim <= out.len);
        self.launch(
            &self.f.layer_norm,
            (rows as u32, 1, 1),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(x)),
                Arg::P(self.ptr(out)),
                Arg::I(rows as i32),
                Arg::I(dim as i32),
                Arg::I(w.is_some() as i32),
                Arg::P(self.opt(w)),
                Arg::I(b.is_some() as i32),
                Arg::P(self.opt(b)),
                Arg::F(eps),
            ],
        );
    }

    fn head_split(
        &self,
        src: &CuBuf,
        rows: usize,
        stride: usize,
        col_off: usize,
        heads: usize,
        hd: usize,
        norm: Option<HeadNorm<'_, CuBuf>>,
        rope: Option<Rope<'_, CuBuf>>,
        out: &CuBuf,
    ) {
        assert!(
            heads <= 256,
            "head_split keeps per-head statistics for up to 256 heads"
        );
        assert!(heads * rows * hd <= out.len);
        if let Some(r) = &rope {
            assert!(
                r.cos.f32 && r.sin.f32,
                "RoPE tables must come from upload_table"
            );
            assert!(r.rot.is_multiple_of(2) && r.rot <= hd);
        }
        let w = norm.as_ref().and_then(|n| n.w);
        let warp_kernel = hd.is_multiple_of(32) && hd <= 128;
        self.launch(
            if warp_kernel {
                &self.f.head_split_warp
            } else {
                &self.f.head_split
            },
            (rows as u32, 1, 1),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(src)),
                Arg::P(self.ptr(out)),
                Arg::I(rows as i32),
                Arg::I(stride as i32),
                Arg::I(col_off as i32),
                Arg::I(heads as i32),
                Arg::I(hd as i32),
                Arg::I(norm.is_some() as i32),
                Arg::I(w.is_some() as i32),
                Arg::P(self.opt(w)),
                Arg::F(norm.as_ref().map_or(0.0, |n| n.eps)),
                Arg::I(rope.is_some() as i32),
                Arg::P(self.opt(rope.as_ref().map(|r| r.cos))),
                Arg::P(self.opt(rope.as_ref().map(|r| r.sin))),
                Arg::I(rope.as_ref().map_or(0, |r| r.rot as i32)),
            ],
        );
    }

    fn attention(
        &self,
        q: &CuBuf,
        k: &CuBuf,
        v: &CuBuf,
        heads: usize,
        kv_heads: usize,
        sq: usize,
        sk: usize,
        d: usize,
        causal: bool,
        out: &CuBuf,
    ) {
        assert!(d <= 128 && heads.is_multiple_of(kv_heads));
        assert!(
            heads * sq * d <= q.len
                && kv_heads * sk * d <= k.len
                && kv_heads * sk * d <= v.len
                && sq * heads * d <= out.len
        );
        let scale = 1.0 / (d as f32).sqrt();
        let mma = match (d, self.attn_v1) {
            (128, true) => Some((&self.f.attention_mma128, 64)),
            (64, true) => Some((&self.f.attention_mma64, 64)),
            (128, false) => Some((&self.f.attention_mma2_128, 128)),
            (64, false) => Some((&self.f.attention_mma2_64, 128)),
            _ => None,
        };
        if let (Some((f, qr)), true, false) = (mma, self.ampere, self.scalar_attn) {
            self.launch(
                f,
                (sq.div_ceil(qr) as u32, heads as u32, 1),
                (128, 1, 1),
                &[
                    Arg::P(self.ptr(q)),
                    Arg::P(self.ptr(k)),
                    Arg::P(self.ptr(v)),
                    Arg::P(self.ptr(out)),
                    Arg::I(heads as i32),
                    Arg::I(kv_heads as i32),
                    Arg::I(sq as i32),
                    Arg::I(sk as i32),
                    Arg::I(causal as i32),
                    Arg::F(scale),
                ],
            );
            return;
        }
        self.launch(
            &self.f.attention,
            (sq.div_ceil(64) as u32, heads as u32, 1),
            (64, 1, 1),
            &[
                Arg::P(self.ptr(q)),
                Arg::P(self.ptr(k)),
                Arg::P(self.ptr(v)),
                Arg::P(self.ptr(out)),
                Arg::I(heads as i32),
                Arg::I(kv_heads as i32),
                Arg::I(sq as i32),
                Arg::I(sk as i32),
                Arg::I(d as i32),
                Arg::I(causal as i32),
                Arg::F(scale),
            ],
        );
    }

    fn vsa_supported(&self, head_dim: usize) -> bool {
        head_dim <= 128 && head_dim.is_multiple_of(4)
    }

    fn attention_qk8(
        &self,
        q: &CuBuf,
        k: &CuBuf,
        v: &CuBuf,
        heads: usize,
        kv_heads: usize,
        sq: usize,
        sk: usize,
        d: usize,
        causal: bool,
        pv8: bool,
        out: &CuBuf,
    ) {
        assert!(heads.is_multiple_of(kv_heads));
        assert!(
            heads * sq * d <= q.len
                && kv_heads * sk * d <= k.len
                && kv_heads * sk * d <= v.len
                && sq * heads * d <= out.len
        );
        if pv8 && !(self.ada && !self.scalar_attn && (d == 64 || d == 128)) {
            // No FP8 tensor cores (or no kernel for this head dim): the oracle on the host.
            let (qh, kh, vh) = (self.download(q), self.download(k), self.download(v));
            let o = h3_core::vsa::attention_pv8_reference(
                &qh[..heads * sq * d],
                &kh[..kv_heads * sk * d],
                &vh[..kv_heads * sk * d],
                heads,
                kv_heads,
                sq,
                sk,
                d,
                causal,
            );
            self.write(out, &o);
            return;
        }
        let f = match d {
            128 if pv8 => &self.f.attention_pv8_128,
            64 if pv8 => &self.f.attention_pv8_64,
            128 if self.ampere && !self.scalar_attn => &self.f.attention_qk8_128,
            64 if self.ampere && !self.scalar_attn => &self.f.attention_qk8_64,
            _ => {
                // No int8 kernel for this shape: round on the device as the
                // oracle does and run the exact kernel (a conformance path).
                let (qh, kh) = (self.download(q), self.download(k));
                let (q8, k8) = h3_core::vsa::attn_qk8_round(
                    &qh[..heads * sq * d],
                    &kh[..kv_heads * sk * d],
                    heads,
                    kv_heads,
                    sq,
                    sk,
                    d,
                );
                let (qb, kb) = (self.upload(&q8), self.upload(&k8));
                return self.attention(&qb, &kb, v, heads, kv_heads, sq, sk, d, causal, out);
            }
        };
        let key = (heads, kv_heads, sq, sk, d);
        let mut guard = self.attn_qk8_scratch.lock().unwrap();
        if guard.as_ref().is_none_or(|(k, _)| *k != key) {
            let (nqt, nkt) = (sq.div_ceil(64), sk.div_ceil(64));
            *guard = Some((
                key,
                (
                    self.alloc((heads * sq * d).div_ceil(2)),
                    self.alloc((kv_heads * sk * d).div_ceil(2)),
                    Backend::upload_table(self, &vec![0.0; heads * nqt]),
                    Backend::upload_table(self, &vec![0.0; kv_heads * nkt]),
                    Backend::upload_table(self, &vec![0.0; kv_heads * d]),
                    Backend::upload_table(self, &vec![0.0; kv_heads * KMEAN_CHUNKS * d]),
                ),
            ));
        }
        let (_, (q8, k8, qsc, ksc, kmean, kpart)) = guard.as_ref().unwrap();
        let nch = KMEAN_CHUNKS as i32;
        self.launch(
            &self.f.vsa_kmean,
            (KMEAN_CHUNKS as u32, kv_heads as u32, 1),
            (128, 1, 1),
            &[
                Arg::P(self.ptr(k)),
                Arg::P(self.ptr(kpart)),
                Arg::I(sk as i32),
                Arg::I(d as i32),
                Arg::I(nch),
            ],
        );
        self.launch(
            &self.f.vsa_kmean2,
            (kv_heads as u32, 1, 1),
            (128, 1, 1),
            &[
                Arg::P(self.ptr(kpart)),
                Arg::P(self.ptr(kmean)),
                Arg::I(sk as i32),
                Arg::I(d as i32),
                Arg::I(nch),
            ],
        );
        self.launch(
            &self.f.attn_qk8_quant,
            (sq.div_ceil(64) as u32, heads as u32, 1),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(q)),
                Arg::P(self.ptr(kmean)),
                Arg::P(self.ptr(q8)),
                Arg::P(self.ptr(qsc)),
                Arg::I(sq as i32),
                Arg::I(d as i32),
                Arg::I(0),
            ],
        );
        self.launch(
            &self.f.attn_qk8_quant,
            (sk.div_ceil(64) as u32, kv_heads as u32, 1),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(k)),
                Arg::P(self.ptr(kmean)),
                Arg::P(self.ptr(k8)),
                Arg::P(self.ptr(ksc)),
                Arg::I(sk as i32),
                Arg::I(d as i32),
                Arg::I(1),
            ],
        );
        let mut args = vec![
            Arg::P(self.ptr(q)),
            Arg::P(self.ptr(k)),
            Arg::P(self.ptr(v)),
            Arg::P(self.ptr(out)),
            Arg::I(heads as i32),
            Arg::I(kv_heads as i32),
            Arg::I(sq as i32),
            Arg::I(sk as i32),
            Arg::I(causal as i32),
            Arg::F(1.0 / (d as f32).sqrt()),
            Arg::P(self.ptr(q8)),
            Arg::P(self.ptr(k8)),
            Arg::P(self.ptr(qsc)),
            Arg::P(self.ptr(ksc)),
        ];
        let mut pv8_guard = self.attn_pv8_scratch.lock().unwrap();
        if pv8 {
            // V: mean (the K-mean kernels on V), chunk maxima, then the
            // transposed E4M3 codes in mma fragment order (fw_attn_v8).
            let key = (kv_heads, sk, d);
            let skp = sk.div_ceil(32) * 32;
            let nparts = sk.div_ceil(64);
            if pv8_guard.as_ref().is_none_or(|(k, _)| *k != key) {
                *pv8_guard = Some((
                    key,
                    (
                        self.alloc((kv_heads * d * skp).div_ceil(2)),
                        Backend::upload_table(self, &vec![0.0; kv_heads * d]),
                        Backend::upload_table(self, &vec![0.0; kv_heads * nparts]),
                        Backend::upload_table(self, &vec![0.0; kv_heads]),
                    ),
                ));
            }
            let (_, (v8t, vmean, vpart, vsc)) = pv8_guard.as_ref().unwrap();
            self.launch(
                &self.f.vsa_kmean,
                (KMEAN_CHUNKS as u32, kv_heads as u32, 1),
                (128, 1, 1),
                &[
                    Arg::P(self.ptr(v)),
                    Arg::P(self.ptr(kpart)),
                    Arg::I(sk as i32),
                    Arg::I(d as i32),
                    Arg::I(nch),
                ],
            );
            self.launch(
                &self.f.vsa_kmean2,
                (kv_heads as u32, 1, 1),
                (128, 1, 1),
                &[
                    Arg::P(self.ptr(kpart)),
                    Arg::P(self.ptr(vmean)),
                    Arg::I(sk as i32),
                    Arg::I(d as i32),
                    Arg::I(nch),
                ],
            );
            self.launch(
                &self.f.attn_v8_max,
                (nparts as u32, kv_heads as u32, 1),
                (256, 1, 1),
                &[
                    Arg::P(self.ptr(v)),
                    Arg::P(self.ptr(vmean)),
                    Arg::P(self.ptr(vpart)),
                    Arg::I(sk as i32),
                    Arg::I(d as i32),
                ],
            );
            self.launch(
                &self.f.attn_v8,
                ((skp / 32) as u32, kv_heads as u32, 1),
                (128, 1, 1),
                &[
                    Arg::P(self.ptr(v)),
                    Arg::P(self.ptr(vmean)),
                    Arg::P(self.ptr(vpart)),
                    Arg::I(nparts as i32),
                    Arg::P(self.ptr(vsc)),
                    Arg::P(self.ptr(v8t)),
                    Arg::I(sk as i32),
                    Arg::I(d as i32),
                    Arg::P(0),
                ],
            );
            args.extend([
                Arg::P(self.ptr(v8t)),
                Arg::P(self.ptr(vmean)),
                Arg::P(self.ptr(vsc)),
            ]);
        }
        self.launch(
            f,
            (sq.div_ceil(128) as u32, heads as u32, 1),
            (128, 1, 1),
            &args,
        );
    }

    fn vsa_prepare(&self, spec: &VsaSpec) -> CuVsa {
        let plan = &spec.plan;
        let (nb, h, d) = (plan.num_tiles(), spec.heads, spec.head_dim);
        assert!(
            nb <= 4096,
            "VSA selection holds at most 4096 tiles per block"
        );
        assert!(d <= 128 && d.is_multiple_of(4));
        let mut reg: Vec<u32> = Vec::new();
        let mut maxsel = plan.num_prefix_tiles;
        for r in &plan.regions {
            reg.extend([r.start as u32, r.end as u32, r.topk as u32]);
            maxsel += r.topk.min(r.num_tiles());
        }
        let maxsel = maxsel.max(1);
        CuVsa {
            slot: self.words(&plan.slot_src),
            bsz: self.words(&plan.block_sizes),
            reg: self.words(&reg),
            pooled: self.words(&vec![0; 3 * h * nb * d]),
            pooled_out: self.words(&vec![0; h * nb * d]),
            sel: self.words(&vec![0; h * nb * maxsel]),
            nsel: self.words(&vec![0; h * nb]),
            usel: self.words(&vec![0; h * nb.div_ceil(2) * nb.max(2 * maxsel)]),
            nusel: self.words(&vec![0; h * nb.div_ceil(2)]),
            npairs: nb.div_ceil(2),
            ustride: nb.max(2 * maxsel),
            nb,
            np: plan.num_prefix_tiles,
            nreg: plan.regions.len(),
            dense_all: plan.is_dense(),
            rowslot: self.words(&plan.slot_of_row()),
            maxsel,
            te: plan.tile_elems,
            heads: h,
            qk8: spec.qk8.then(|| {
                assert!(
                    plan.tile_elems <= 128 && d <= 128,
                    "fw_vsa_qk8 holds a tile of ≤ 128 rows × ≤ 128 channels in registers"
                );
                let codes = (h * spec.plan.seq * d).div_ceil(2);
                (
                    self.alloc(codes),
                    self.alloc(codes),
                    Backend::upload_table(self, &vec![0.0; h * nb]),
                    Backend::upload_table(self, &vec![0.0; h * nb]),
                    Backend::upload_table(self, &vec![0.0; h * d]),
                    Backend::upload_table(self, &vec![0.0; h * KMEAN_CHUNKS * d]),
                )
            }),
            pv8: spec.pv8.then(|| {
                (
                    self.alloc((h * d * nb * plan.tile_elems).div_ceil(2)),
                    Backend::upload_table(self, &vec![0.0; h * d]),
                    Backend::upload_table(self, &vec![0.0; h * plan.seq.div_ceil(64)]),
                    Backend::upload_table(self, &vec![0.0; h]),
                )
            }),
            host_pv8: (spec.pv8
                && self.ada
                && !(self.vsa_fine3.is_some() && !self.scalar_attn && (d == 64 || d == 128)))
                .then(|| spec.clone()),
            d,
            seq: plan.seq,
            scale: spec.scale,
        }
    }

    fn vsa_attention(
        &self,
        q: &CuBuf,
        k: &CuBuf,
        v: &CuBuf,
        gate: Option<&CuBuf>,
        s: &CuVsa,
        out: &CuBuf,
    ) {
        let n = s.heads * s.seq * s.d;
        assert!(n <= q.len && n <= k.len && n <= v.len && n <= out.len);
        if let Some(g) = gate {
            assert!(n <= g.len);
        }
        if let Some(spec) = &s.host_pv8 {
            let (qh, kh, vh) = (self.download(q), self.download(k), self.download(v));
            let gh = gate.map(|g| self.download(g));
            let mut o = vec![0f32; n];
            h3_cpu::vsa_reference(
                &qh[..n],
                &kh[..n],
                &vh[..n],
                gh.as_deref().map(|g| &g[..n]),
                spec,
                &mut o,
            );
            self.write(out, &o);
            return;
        }
        let grid = (s.nb as u32, s.heads as u32, 1);
        let vec_pool = s.d.is_multiple_of(4) && s.d <= 128;
        self.launch(
            if vec_pool {
                &self.f.vsa_pool
            } else {
                &self.f.vsa_pool_scalar
            },
            grid,
            (128, 1, 1),
            &[
                Arg::P(self.ptr(q)),
                Arg::P(self.ptr(k)),
                Arg::P(self.ptr(v)),
                Arg::P(self.ptr(&s.slot)),
                Arg::P(self.ptr(&s.bsz)),
                Arg::P(self.ptr(&s.pooled)),
                Arg::I(s.heads as i32),
                Arg::I(s.seq as i32),
                Arg::I(s.d as i32),
                Arg::I(s.nb as i32),
                Arg::I(s.te as i32),
            ],
        );
        // Four query tiles per block share the pooled K̄ / V̄ reads (nb ≤ 2048;
        // `H3_CUDA_VSA_COARSE1=1` keeps one per block).
        let coarse4 = s.nb <= 2048 && !self.vsa_coarse1;
        self.launch(
            if coarse4 {
                &self.f.vsa_coarse4
            } else {
                &self.f.vsa_coarse
            },
            if coarse4 {
                (s.nb.div_ceil(4) as u32, s.heads as u32, 1)
            } else {
                grid
            },
            (256, 1, 1),
            &[
                Arg::P(self.ptr(&s.pooled)),
                Arg::P(self.ptr(&s.pooled_out)),
                Arg::P(self.ptr(&s.sel)),
                Arg::P(self.ptr(&s.nsel)),
                Arg::P(self.ptr(&s.reg)),
                Arg::I(s.heads as i32),
                Arg::I(s.d as i32),
                Arg::I(s.nb as i32),
                Arg::I(s.np as i32),
                Arg::I(s.nreg as i32),
                Arg::I(s.dense_all as i32),
                Arg::I(s.maxsel as i32),
                Arg::I(gate.is_some() as i32),
                Arg::F(s.scale),
            ],
        );
        // A dense plan: the dense attention kernel over the natural order (its
        // INT8 / FP8 recipe when the spec asks) plus the gated pooled branch.
        // The all-tiles fine stage computed the same softmax at ~2× the cost
        // (a 5090: 9.8 s against 5.3 s per V2 step).
        if s.dense_all
            && !self.scalar_attn
            && (s.d == 64 || s.d == 128)
            && (s.scale - 1.0 / (s.d as f32).sqrt()).abs() <= 1e-6 * s.scale
        {
            if s.qk8.is_some() || s.pv8.is_some() {
                self.attention_qk8(
                    q,
                    k,
                    v,
                    s.heads,
                    s.heads,
                    s.seq,
                    s.seq,
                    s.d,
                    false,
                    s.pv8.is_some(),
                    out,
                );
            } else {
                self.attention(q, k, v, s.heads, s.heads, s.seq, s.seq, s.d, false, out);
            }
            if let Some(g) = gate {
                self.launch(
                    &self.f.vsa_gate_add,
                    Self::elems(n),
                    (256, 1, 1),
                    &[
                        Arg::P(self.ptr(out)),
                        Arg::P(self.ptr(&s.pooled_out)),
                        Arg::P(self.ptr(g)),
                        Arg::P(self.ptr(&s.rowslot)),
                        Arg::L(n as i64),
                        Arg::I(s.heads as i32),
                        Arg::I(s.seq as i32),
                        Arg::I(s.d as i32),
                        Arg::I(s.nb as i32),
                        Arg::I(s.te as i32),
                    ],
                );
            }
            return;
        }
        let tail = |a: &mut Vec<Arg>| {
            a.extend([
                Arg::P(self.ptr(&s.slot)),
                Arg::P(self.ptr(&s.sel)),
                Arg::P(self.ptr(&s.nsel)),
                Arg::P(self.ptr(&s.pooled_out)),
                Arg::P(self.opt(gate)),
                Arg::I(gate.is_some() as i32),
                Arg::I(s.nb as i32),
                Arg::I(s.np as i32),
                Arg::I(s.dense_all as i32),
                Arg::I(s.te as i32),
                Arg::I(s.maxsel as i32),
            ]);
        };
        let mut args = vec![
            Arg::P(self.ptr(q)),
            Arg::P(self.ptr(k)),
            Arg::P(self.ptr(v)),
            Arg::P(self.ptr(out)),
            Arg::I(s.heads as i32),
            Arg::I(s.seq as i32),
        ];
        // Paired tiles (64-token tiles, d 64 / 128): two query tiles per
        // 256-thread block over the union of their selections.
        let pair = match s.d {
            128 => Some(&self.f.vsa_fine_pair_mma128),
            64 => Some(&self.f.vsa_fine_pair_mma64),
            _ => None,
        };
        if let (Some(f), true, false, true, true) = (
            pair,
            self.ampere,
            self.scalar_attn,
            self.vsa_pair,
            s.te == 64,
        ) {
            self.launch(
                &self.f.vsa_pair,
                (s.npairs as u32, s.heads as u32, 1),
                (32, 1, 1),
                &[
                    Arg::P(self.ptr(&s.sel)),
                    Arg::P(self.ptr(&s.nsel)),
                    Arg::P(self.ptr(&s.usel)),
                    Arg::P(self.ptr(&s.nusel)),
                    Arg::I(s.nb as i32),
                    Arg::I(s.np as i32),
                    Arg::I(s.dense_all as i32),
                    Arg::I(s.maxsel as i32),
                    Arg::I(s.ustride as i32),
                    Arg::I(s.npairs as i32),
                ],
            );
            args.push(Arg::F(s.scale));
            tail(&mut args);
            args.extend([
                Arg::P(self.ptr(&s.usel)),
                Arg::P(self.ptr(&s.nusel)),
                Arg::I(s.npairs as i32),
                Arg::I(s.ustride as i32),
                Arg::I(self.vsa_dbg),
            ]);
            self.launch(f, (s.npairs as u32, s.heads as u32, 1), (256, 1, 1), &args);
            return;
        }
        if let (Some(var), true, false, true) = (
            self.vsa_fine3,
            self.ampere,
            self.scalar_attn,
            s.d == 64 || s.d == 128,
        ) {
            args.push(Arg::F(s.scale));
            tail(&mut args);
            // INT8 Q·Kᵀ: the per-head K mean, then codes + scales per (head, tile), then the e / f body.
            let mut var = match (&s.qk8, var) {
                (Some(_), 4 | 5) => var,
                (Some(_), _) => 4,
                (None, 4 | 5 | 6) => 3,
                (None, v) => v,
            };
            if s.pv8.is_some() {
                if self.ada {
                    var = 6;
                } else if !self
                    .pv8_warned
                    .swap(true, std::sync::atomic::Ordering::Relaxed)
                {
                    eprintln!(
                        "cuda: no FP8 tensor cores before sm_89; the VSA fine stage runs INT8 Q·Kᵀ with bf16 P·V (pv8 ignored)"
                    );
                }
            }
            if let Some((q8, k8, qsc, ksc, kmean, kpart)) = &s.qk8 {
                let nch = KMEAN_CHUNKS as i32;
                self.launch(
                    &self.f.vsa_kmean,
                    (KMEAN_CHUNKS as u32, s.heads as u32, 1),
                    (128, 1, 1),
                    &[
                        Arg::P(self.ptr(k)),
                        Arg::P(self.ptr(kpart)),
                        Arg::I(s.seq as i32),
                        Arg::I(s.d as i32),
                        Arg::I(nch),
                    ],
                );
                self.launch(
                    &self.f.vsa_kmean2,
                    (s.heads as u32, 1, 1),
                    (128, 1, 1),
                    &[
                        Arg::P(self.ptr(kpart)),
                        Arg::P(self.ptr(kmean)),
                        Arg::I(s.seq as i32),
                        Arg::I(s.d as i32),
                        Arg::I(nch),
                    ],
                );
                self.launch(
                    &self.f.vsa_qk8,
                    (s.nb as u32, s.heads as u32, 1),
                    (256, 1, 1),
                    &[
                        Arg::P(self.ptr(q)),
                        Arg::P(self.ptr(k)),
                        Arg::P(self.ptr(&s.slot)),
                        Arg::P(self.ptr(&s.bsz)),
                        Arg::P(self.ptr(kmean)),
                        Arg::P(self.ptr(q8)),
                        Arg::P(self.ptr(k8)),
                        Arg::P(self.ptr(qsc)),
                        Arg::P(self.ptr(ksc)),
                        Arg::I(s.heads as i32),
                        Arg::I(s.seq as i32),
                        Arg::I(s.d as i32),
                        Arg::I(s.nb as i32),
                        Arg::I(s.te as i32),
                    ],
                );
                args.extend([
                    Arg::P(self.ptr(q8)),
                    Arg::P(self.ptr(k8)),
                    Arg::P(self.ptr(qsc)),
                    Arg::P(self.ptr(ksc)),
                ]);
                if let (Some((v8t, vmean, vpart, vsc)), 6) = (&s.pv8, var) {
                    // V: per-head mean (the K-mean kernels on V), chunk maxima,
                    // then the transposed codes in slot order (fw_attn_v8 with slot).
                    let nparts = s.seq.div_ceil(64);
                    self.launch(
                        &self.f.vsa_kmean,
                        (KMEAN_CHUNKS as u32, s.heads as u32, 1),
                        (128, 1, 1),
                        &[
                            Arg::P(self.ptr(v)),
                            Arg::P(self.ptr(kpart)),
                            Arg::I(s.seq as i32),
                            Arg::I(s.d as i32),
                            Arg::I(nch),
                        ],
                    );
                    self.launch(
                        &self.f.vsa_kmean2,
                        (s.heads as u32, 1, 1),
                        (128, 1, 1),
                        &[
                            Arg::P(self.ptr(kpart)),
                            Arg::P(self.ptr(vmean)),
                            Arg::I(s.seq as i32),
                            Arg::I(s.d as i32),
                            Arg::I(nch),
                        ],
                    );
                    self.launch(
                        &self.f.attn_v8_max,
                        (nparts as u32, s.heads as u32, 1),
                        (256, 1, 1),
                        &[
                            Arg::P(self.ptr(v)),
                            Arg::P(self.ptr(vmean)),
                            Arg::P(self.ptr(vpart)),
                            Arg::I(s.seq as i32),
                            Arg::I(s.d as i32),
                        ],
                    );
                    self.launch(
                        &self.f.attn_v8,
                        ((s.nb * s.te / 32) as u32, s.heads as u32, 1),
                        (128, 1, 1),
                        &[
                            Arg::P(self.ptr(v)),
                            Arg::P(self.ptr(vmean)),
                            Arg::P(self.ptr(vpart)),
                            Arg::I(nparts as i32),
                            Arg::P(self.ptr(vsc)),
                            Arg::P(self.ptr(v8t)),
                            Arg::I(s.seq as i32),
                            Arg::I(s.d as i32),
                            Arg::P(self.ptr(&s.slot)),
                        ],
                    );
                    args.extend([
                        Arg::P(self.ptr(v8t)),
                        Arg::P(self.ptr(vmean)),
                        Arg::P(self.ptr(vsc)),
                    ]);
                }
            }
            let f = &self.f.vsa_fine3[if s.d == 128 { 1 } else { 0 }][var];
            self.launch(
                f,
                (s.nb as u32 * (s.te / 64) as u32, s.heads as u32, 1),
                (128, 1, 1),
                &args,
            );
            return;
        }
        // v1: 64 query rows per block (te/64 blocks per tile), 128 threads.
        // v2: 64 rows (2 warps) for 64-token tiles, 128 rows (4 warps) and
        // te/128 blocks per tile otherwise.
        let mma = match (s.d, !self.vsa_v2, s.te >= 128) {
            (128, true, _) => Some((&self.f.vsa_fine_mma128, (s.te / 64) as u32, 128)),
            (64, true, _) => Some((&self.f.vsa_fine_mma64, (s.te / 64) as u32, 128)),
            (128, false, false) => Some((&self.f.vsa_fine_mma2_128_w2, 1, 64)),
            (64, false, false) => Some((&self.f.vsa_fine_mma2_64_w2, 1, 64)),
            (128, false, true) => Some((&self.f.vsa_fine_mma2_128_w4, (s.te / 128) as u32, 128)),
            (64, false, true) => Some((&self.f.vsa_fine_mma2_64_w4, (s.te / 128) as u32, 128)),
            _ => None,
        };
        if let (Some((f, sub, threads)), true, false) = (mma, self.ampere, self.scalar_attn) {
            args.push(Arg::F(s.scale));
            tail(&mut args);
            if !self.vsa_v2 {
                args.push(Arg::I(self.vsa_dbg));
            }
            self.launch(
                f,
                (s.nb as u32 * sub, s.heads as u32, 1),
                (threads, 1, 1),
                &args,
            );
        } else {
            args.push(Arg::I(s.d as i32));
            args.push(Arg::F(s.scale));
            tail(&mut args);
            self.launch(&self.f.vsa_fine, grid, (64, 1, 1), &args);
        }
    }

    fn patchify(
        &self,
        latent: &CuBuf,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        patch: (usize, usize, usize),
        out: &CuBuf,
        out_row: usize,
    ) {
        let n = c * t * h * w;
        let f = c * patch.0 * patch.1 * patch.2;
        assert!(n <= latent.len && (out_row * f + n) <= out.len);
        self.launch(
            &self.f.patch,
            Self::elems(n),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(out)),
                Arg::P(self.ptr(latent)),
                Arg::L(n as i64),
                Arg::I(c as i32),
                Arg::I(t as i32),
                Arg::I(h as i32),
                Arg::I(w as i32),
                Arg::I(patch.0 as i32),
                Arg::I(patch.1 as i32),
                Arg::I(patch.2 as i32),
                Arg::I(0),
                Arg::L(out_row as i64),
                Arg::I(f as i32),
                Arg::I(0),
            ],
        );
    }

    fn unpatchify(
        &self,
        x: &CuBuf,
        x_row: usize,
        x_stride: usize,
        col: usize,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        patch: (usize, usize, usize),
        out: &CuBuf,
    ) {
        let n = c * t * h * w;
        assert!(n <= out.len);
        self.launch(
            &self.f.patch,
            Self::elems(n),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(out)),
                Arg::P(self.ptr(x)),
                Arg::L(n as i64),
                Arg::I(c as i32),
                Arg::I(t as i32),
                Arg::I(h as i32),
                Arg::I(w as i32),
                Arg::I(patch.0 as i32),
                Arg::I(patch.1 as i32),
                Arg::I(patch.2 as i32),
                Arg::I(1),
                Arg::L(x_row as i64),
                Arg::I(x_stride as i32),
                Arg::I(col as i32),
            ],
        );
    }

    fn flow_step(
        &self,
        x: &CuBuf,
        off: usize,
        v: &CuBuf,
        v_off: usize,
        noise: Option<&CuBuf>,
        n: usize,
        step: FlowStep,
    ) {
        assert!(off + n <= x.len && v_off + n <= v.len);
        let (mode, sigma, next) = match step {
            FlowStep::Euler { sigma, next } => (0, sigma, next),
            FlowStep::Renoise { sigma, next } => {
                assert!(
                    noise.is_some_and(|z| off + n <= z.len),
                    "renoise needs noise"
                );
                (1, sigma, next)
            }
            FlowStep::Terminal { sigma } => (2, sigma, 0.0),
        };
        self.launch(
            &self.f.flow_step,
            Self::elems(n),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(x)),
                Arg::L(off as i64),
                Arg::P(self.ptr(v)),
                Arg::L(v_off as i64),
                Arg::P(self.opt(noise)),
                Arg::L(n as i64),
                Arg::I(mode),
                Arg::F(sigma),
                Arg::F(next),
            ],
        );
    }

    fn axpby(&self, y: &CuBuf, off: usize, a: f32, x: &CuBuf, x_off: usize, b: f32, n: usize) {
        assert!(off + n <= y.len && x_off + n <= x.len);
        self.launch(
            &self.f.axpby,
            Self::elems(n),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(y)),
                Arg::L(off as i64),
                Arg::F(a),
                Arg::P(self.ptr(x)),
                Arg::L(x_off as i64),
                Arg::F(b),
                Arg::L(n as i64),
            ],
        );
    }

    fn channel_affine(
        &self,
        x: &CuBuf,
        c: usize,
        inner: usize,
        scale: &CuBuf,
        shift: Option<&CuBuf>,
        out: &CuBuf,
    ) {
        let n = c * inner;
        assert!(n <= x.len && n <= out.len && c <= scale.len);
        self.launch(
            &self.f.channel_affine,
            Self::elems(n),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(x)),
                Arg::P(self.ptr(out)),
                Arg::L(n as i64),
                Arg::I(inner as i32),
                Arg::P(self.ptr(scale)),
                Arg::I(shift.is_some() as i32),
                Arg::P(self.opt(shift)),
            ],
        );
    }

    fn conv2d(
        &self,
        a: &CuBuf,
        b: Option<&CuBuf>,
        s: &ConvShape,
        w: &CuWeight,
        bias: Option<&CuBuf>,
        ep: ConvEpilogue<'_, CuBuf>,
        out: &CuBuf,
    ) {
        assert_eq!((w.rows, w.cols), (s.cout, s.k_len()));
        assert!(w.quant.is_none(), "conv weights are not quantized");
        assert!(s.out_len() <= out.len);
        let (epc, res) = Self::epilogue_code(&ep);
        let (oh, ow) = s.out_hw();
        let mma = self.ampere && !self.scalar_conv;
        let tile = if mma { 128 } else { 64 };
        let mt = (s.frames * oh * ow).div_ceil(tile) as u32;
        let (gy, gz) = Self::split_grid(mt);
        self.launch(
            if mma {
                &self.f.conv2d_mma
            } else {
                &self.f.conv2d
            },
            (s.cout.div_ceil(64) as u32, gy, gz),
            if mma { (128, 1, 1) } else { (16, 16, 1) },
            &[
                Arg::P(self.ptr(a)),
                Arg::P(self.opt(b)),
                Arg::P(self.ptr(&w.buf)),
                Arg::P(self.opt(bias)),
                Arg::P(self.opt(res)),
                Arg::P(self.ptr(out)),
                Arg::I(s.frames as i32),
                Arg::I(s.cin_a as i32),
                Arg::I(s.cin_b as i32),
                Arg::I(s.cout as i32),
                Arg::I(s.h as i32),
                Arg::I(s.w as i32),
                Arg::I(if s.upsample { 2 } else { 1 }),
                Arg::I(s.k as i32),
                Arg::I(bias.is_some() as i32),
                Arg::I(epc),
            ],
        );
    }

    fn conv3d(
        &self,
        x: &CuBuf,
        s: &Conv3dShape,
        w: &CuWeight,
        bias: Option<&CuBuf>,
        ep: ConvEpilogue<'_, CuBuf>,
        out: &CuBuf,
    ) {
        assert!(
            !matches!(ep, ConvEpilogue::Prelu(_)),
            "PReLU is a conv2d epilogue"
        );
        assert_eq!((w.rows, w.cols), (s.cout, s.k_len()), "conv3d weight shape");
        assert!(w.quant.is_none(), "conv weights are not quantized");
        assert!(s.in_len() <= x.len && s.out_len() <= out.len);
        assert!(s.cin.is_multiple_of(s.groups) && s.cout.is_multiple_of(s.groups));
        let (epc, res) = Self::epilogue_code(&ep);
        let (ot, oh, ow) = s.out_dims();
        let reflect = (s.spatial_pad == Pad::Reflect) as i32;
        let common = |args: &mut Vec<Arg>| {
            args.extend([
                Arg::I(s.k.0 as i32),
                Arg::I(s.k.1 as i32),
                Arg::I(s.k.2 as i32),
                Arg::I(s.stride.0 as i32),
                Arg::I(s.stride.1 as i32),
                Arg::I(s.stride.2 as i32),
                Arg::I(s.pad_t.0 as i32),
                Arg::I(s.pad_h.0 as i32),
                Arg::I(s.pad_w.0 as i32),
                Arg::I(reflect),
                Arg::I(ot as i32),
                Arg::I(oh as i32),
                Arg::I(ow as i32),
                Arg::I(bias.is_some() as i32),
                Arg::I(epc),
            ]);
        };
        let mma = self.ampere && !self.scalar_conv && s.groups == 1;
        if mma {
            let mt = (ot * oh * ow).div_ceil(128) as u32;
            let (gy, gz) = Self::split_grid(mt);
            let mut args = vec![
                Arg::P(self.ptr(x)),
                Arg::P(self.ptr(&w.buf)),
                Arg::P(self.opt(bias)),
                Arg::P(self.opt(res)),
                Arg::P(self.ptr(out)),
                Arg::I(s.cin as i32),
                Arg::I(s.cout as i32),
                Arg::I(s.t as i32),
                Arg::I(s.h as i32),
                Arg::I(s.w as i32),
            ];
            common(&mut args);
            self.launch(
                &self.f.conv3d_mma,
                (s.cout.div_ceil(64) as u32, gy, gz),
                (128, 1, 1),
                &args,
            );
        } else {
            let n = s.out_len();
            let mut args = vec![
                Arg::P(self.ptr(x)),
                Arg::P(self.ptr(&w.buf)),
                Arg::P(self.opt(bias)),
                Arg::P(self.opt(res)),
                Arg::P(self.ptr(out)),
                Arg::I(s.cin as i32),
                Arg::I(s.cout as i32),
                Arg::I(s.groups as i32),
                Arg::I(s.t as i32),
                Arg::I(s.h as i32),
                Arg::I(s.w as i32),
            ];
            common(&mut args);
            self.launch(&self.f.conv3d, Self::elems(n), (256, 1, 1), &args);
        }
    }

    fn conv1d(
        &self,
        x: &CuBuf,
        s: &Conv1dShape,
        w: &CuWeight,
        bias: Option<&CuBuf>,
        ep: ConvEpilogue<'_, CuBuf>,
        out: &CuBuf,
    ) {
        assert!(
            !matches!(ep, ConvEpilogue::Prelu(_)),
            "PReLU is a conv2d epilogue"
        );
        assert_eq!((w.rows, w.cols), (s.cout, (s.cin / s.groups) * s.k));
        assert!(w.quant.is_none());
        assert!(s.cin * s.len <= x.len && s.out_len() <= out.len);
        let (epc, res) = Self::epilogue_code(&ep);
        let ol = s.out_len_1d();
        self.launch(
            &self.f.conv1d,
            Self::elems(s.cout * ol),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(x)),
                Arg::P(self.ptr(&w.buf)),
                Arg::P(self.opt(bias)),
                Arg::P(self.opt(res)),
                Arg::P(self.ptr(out)),
                Arg::I(s.cin as i32),
                Arg::I(s.cout as i32),
                Arg::I(s.groups as i32),
                Arg::I(s.len as i32),
                Arg::I(s.k as i32),
                Arg::I(s.stride as i32),
                Arg::I(s.dilation as i32),
                Arg::I(s.pad.0 as i32),
                Arg::I(ol as i32),
                Arg::I(bias.is_some() as i32),
                Arg::I(epc),
            ],
        );
    }

    fn replicate_pad1d(
        &self,
        x: &CuBuf,
        c: usize,
        len: usize,
        left: usize,
        right: usize,
        out: &CuBuf,
    ) {
        let n = c * (len + left + right);
        assert!(c * len <= x.len && n <= out.len && len > 0);
        self.launch(
            &self.f.replicate_pad1d,
            Self::elems(n),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(x)),
                Arg::P(self.ptr(out)),
                Arg::L(n as i64),
                Arg::I(len as i32),
                Arg::I(left as i32),
                Arg::I(right as i32),
            ],
        );
    }

    fn conv_transpose1d(
        &self,
        x: &CuBuf,
        s: &ConvT1dShape,
        w: &CuWeight,
        bias: Option<&CuBuf>,
        out: &CuBuf,
    ) {
        assert_eq!((w.rows, w.cols), (s.cin, (s.cout / s.groups) * s.k));
        assert!(w.quant.is_none());
        assert!(s.cin * s.len <= x.len && s.out_len() <= out.len);
        let ol = s.out_len_1d();
        self.launch(
            &self.f.conv_transpose1d,
            Self::elems(s.cout * ol),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(x)),
                Arg::P(self.ptr(&w.buf)),
                Arg::P(self.opt(bias)),
                Arg::P(self.ptr(out)),
                Arg::I(s.cin as i32),
                Arg::I(s.cout as i32),
                Arg::I(s.groups as i32),
                Arg::I(s.len as i32),
                Arg::I(s.k as i32),
                Arg::I(s.stride as i32),
                Arg::I(s.pad as i32),
                Arg::I(ol as i32),
                Arg::I(bias.is_some() as i32),
            ],
        );
    }

    fn group_norm(
        &self,
        x: &CuBuf,
        c: usize,
        t: usize,
        hw: usize,
        groups: usize,
        per_frame: bool,
        w: &CuBuf,
        b: &CuBuf,
        eps: f32,
        act: Act,
        out: &CuBuf,
    ) {
        let n = c * t * hw;
        assert!(n <= x.len && n <= out.len && c.is_multiple_of(groups));
        let fg = if per_frame { t * groups } else { groups };
        let stats = self.words(&vec![0; 2 * fg]);
        self.launch(
            &self.f.gn_stats,
            (fg as u32, 1, 1),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(x)),
                Arg::P(self.ptr(&stats)),
                Arg::I(c as i32),
                Arg::I(t as i32),
                Arg::I(hw as i32),
                Arg::I(groups as i32),
                Arg::I(per_frame as i32),
                Arg::F(eps),
            ],
        );
        let act_code = match act {
            Act::None => 0,
            Act::GeluTanh => 1,
            Act::GeluErf => 2,
            Act::Silu => 3,
        };
        self.launch(
            &self.f.gn_apply,
            Self::elems(n),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(x)),
                Arg::P(self.ptr(out)),
                Arg::P(self.ptr(&stats)),
                Arg::L(n as i64),
                Arg::I(c as i32),
                Arg::I(t as i32),
                Arg::I(hw as i32),
                Arg::I(groups as i32),
                Arg::I(per_frame as i32),
                Arg::P(self.ptr(w)),
                Arg::P(self.ptr(b)),
                Arg::I(act_code),
            ],
        );
    }

    fn snake(&self, x: &CuBuf, c: usize, len: usize, snake: Snake<'_, CuBuf>) {
        let n = c * len;
        assert!(n <= x.len);
        let (alpha, beta, kind) = match snake {
            Snake::Alpha(a) => (self.ptr(a), 0, 0),
            Snake::AlphaBeta { alpha, beta } => (self.ptr(alpha), self.ptr(beta), 1),
        };
        self.launch(
            &self.f.snake,
            Self::elems(n),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(x)),
                Arg::L(n as i64),
                Arg::I(len as i32),
                Arg::P(alpha),
                Arg::P(beta),
                Arg::I(kind),
            ],
        );
    }

    fn trilinear_resize(
        &self,
        x: &CuBuf,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        t2: usize,
        h2: usize,
        w2: usize,
        out: &CuBuf,
    ) {
        let n = c * t2 * h2 * w2;
        assert!(c * t * h * w <= x.len && n <= out.len);
        self.launch(
            &self.f.trilinear,
            Self::elems(n),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(x)),
                Arg::P(self.ptr(out)),
                Arg::L(n as i64),
                Arg::I(c as i32),
                Arg::I(t as i32),
                Arg::I(h as i32),
                Arg::I(w as i32),
                Arg::I(t2 as i32),
                Arg::I(h2 as i32),
                Arg::I(w2 as i32),
            ],
        );
    }

    fn pixel_shuffle(
        &self,
        x: &CuBuf,
        frames: usize,
        c: usize,
        h: usize,
        w: usize,
        r: usize,
        out: &CuBuf,
    ) {
        let n = frames * c * h * w * r * r;
        assert!(n <= x.len && n <= out.len);
        self.launch(
            &self.f.pixel_shuffle,
            Self::elems(n),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(x)),
                Arg::P(self.ptr(out)),
                Arg::L(n as i64),
                Arg::I(c as i32),
                Arg::I(h as i32),
                Arg::I(w as i32),
                Arg::I(r as i32),
            ],
        );
    }

    fn tanh_clamp(&self, x: &CuBuf, n: usize, s: f32) {
        assert!(n <= x.len);
        self.launch(
            &self.f.tanh_clamp,
            Self::elems(n),
            (256, 1, 1),
            &[Arg::P(self.ptr(x)), Arg::L(n as i64), Arg::F(s)],
        );
    }

    fn clamp(&self, x: &CuBuf, n: usize, lo: f32, hi: f32) {
        assert!(n <= x.len);
        self.launch(
            &self.f.clamp,
            Self::elems(n),
            (256, 1, 1),
            &[
                Arg::P(self.ptr(x)),
                Arg::L(n as i64),
                Arg::F(lo),
                Arg::F(hi),
            ],
        );
    }
}

#[cfg(test)]
mod tests {
    /// Needs libnvrtc only (no GPU): `pip install nvidia-cuda-nvrtc-cu12` and
    /// point `LD_LIBRARY_PATH` at its `lib/`. Skipped when NVRTC cannot be loaded.
    #[test]
    /// `H3_PTX_OUT=<file> cargo test -p h3-cuda dump_ptx` writes the sm_89
    /// PTX for `ptxas -v` (registers, spills, shared memory per kernel).
    #[test]
    fn dump_ptx() {
        let Ok(path) = std::env::var("H3_PTX_OUT") else {
            return;
        };
        match std::panic::catch_unwind(|| super::compile_kernels("compute_89")) {
            Ok(Ok(ptx)) => std::fs::write(&path, ptx.to_src()).unwrap(),
            Ok(Err(e)) => panic!("{e}"),
            Err(_) => eprintln!("skipping: libnvrtc not loadable"),
        }
    }

    #[test]
    fn kernels_compile_to_sass_for_ada_and_blackwell() {
        for arch in ["sm_89", "sm_120"] {
            match std::panic::catch_unwind(|| super::compile_cubin(arch)) {
                Ok(Ok(cubin)) => assert!(cubin.len() > 1024, "{arch}: cubin too small"),
                Ok(Err(e)) => panic!("{arch}: {}", e.0),
                Err(_) => {
                    eprintln!("skipping: libnvrtc not loadable");
                    return;
                }
            }
        }
    }

    #[test]
    fn kernels_compile_for_ampere_ada_and_blackwell() {
        for arch in ["compute_80", "compute_89", "compute_100", "compute_120"] {
            match std::panic::catch_unwind(|| super::compile_kernels(arch)) {
                Ok(Ok(_)) => {}
                Ok(Err(e)) => panic!("{arch}: {e}"),
                Err(_) => {
                    eprintln!("skipping: libnvrtc not loadable");
                    return;
                }
            }
        }
    }
}
