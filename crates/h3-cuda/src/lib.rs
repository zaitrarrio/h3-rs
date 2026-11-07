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
use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, LaunchConfig, PushKernelArg};
use h3_core::backend::{
    Act, Backend, Conv1dShape, Conv3dShape, ConvEpilogue, ConvShape, ConvT1dShape, FlowStep, Gate, HeadNorm, Modulation, Pad,
    Rope, Snake,
};
use h3_core::quant::{Nvfp4, Q4, Q8};
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
    maxsel: usize,
    te: usize,
    heads: usize,
    d: usize,
    seq: usize,
    scale: f32,
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
    Nvfp4 { alpha: f32, first_high: bool },
}

pub struct Cuda {
    #[allow(dead_code)]
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
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
    q8_scratch: std::sync::Mutex<Option<CuBuf>>,
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
    rms: CudaFunction,
    layer_norm: CudaFunction,
    head_split: CudaFunction,
    head_split_warp: CudaFunction,
    patch: CudaFunction,
    flow_step: CudaFunction,
    axpby: CudaFunction,
    channel_affine: CudaFunction,
    fill: CudaFunction,
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
    vsa_fine3: [[CudaFunction; 4]; 2],
    dequant_q8: CudaFunction,
    dequant_q4: CudaFunction,
    dequant_nvfp4: CudaFunction,
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
        cudarc::nvrtc::CompileOptions { arch: Some(arch), options: vec!["--std=c++17".into()], ..Default::default() },
    )
    .map_err(err)
}

/// One cached cuBLASLt plan per (n, k, rows, epilogue).
struct LtPlan {
    desc: lt_sys::cublasLtMatmulDesc_t,
    a: lt_sys::cublasLtMatrixLayout_t,
    b: lt_sys::cublasLtMatrixLayout_t,
    c: lt_sys::cublasLtMatrixLayout_t,
    algo: lt_sys::cublasLtMatmulAlgo_t,
}

/// (n, k, rows, epilogue) → plan, `None` when cuBLASLt declined the shape.
type PlanCache = std::collections::HashMap<(usize, usize, usize, u8), Option<LtPlan>>;

struct Lt {
    handle: lt_sys::cublasLtHandle_t,
    workspace: CudaSlice<u8>,
    plans: std::sync::Mutex<PlanCache>,
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
        Ok(Self { handle, workspace, plans: Default::default() })
    }

    /// Column-major yᵀ[n, rows] = Wᵀ…: A = W (op T, stored k × n), B = x
    /// (k × rows), C = D = y (n × rows), all bf16, f32 accumulate. `epi` 0 =
    /// bias, 1 = tanh-GELU(·+bias). `None` = not supported.
    fn plan(&self, n: usize, k: usize, rows: usize, epi: u8) -> Option<LtPlan> {
        use lt_sys::cudaDataType_t::{CUDA_R_16BF, CUDA_R_32F};
        let a = lt_result::create_matrix_layout(CUDA_R_16BF, k as u64, n as u64, k as i64).ok()?;
        let b = lt_result::create_matrix_layout(CUDA_R_16BF, k as u64, rows as u64, k as i64).ok()?;
        let c = lt_result::create_matrix_layout(CUDA_R_16BF, n as u64, rows as u64, n as i64).ok()?;
        let desc = lt_result::create_matmul_desc(lt_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, CUDA_R_32F).ok()?;
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
            Some(LtPlan { desc, a, b, c, algo: h.ok()?.algo })
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
        let ctx = CudaContext::new(ordinal).map_err(err)?;
        let (maj, min) = ctx.compute_capability().map_err(err)?;
        let arch: &'static str = Box::leak(format!("compute_{maj}{min}").into_boxed_str());
        let ptx = compile_kernels(arch)?;
        let module = ctx.load_module(ptx).map_err(err)?;
        let get = |n: &str| module.load_function(n).map_err(err);
        let f = Kernels {
            bias_act: get("fw_bias_act")?,
            swiglu: get("fw_swiglu")?,
            gate_residual: get("fw_gate_residual")?,
            rms: get("fw_rms")?,
            layer_norm: get("fw_layer_norm")?,
            head_split: get("fw_head_split")?,
            head_split_warp: get("fw_head_split_warp")?,
            patch: get("fw_patch")?,
            flow_step: get("fw_flow_step")?,
            axpby: get("fw_axpby")?,
            channel_affine: get("fw_channel_affine")?,
            fill: get("fw_fill")?,
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
                [get("fw_vsa_fine3a_mma64")?, get("fw_vsa_fine3b_mma64")?, get("fw_vsa_fine3c_mma64")?, get("fw_vsa_fine3d_mma64")?],
                [get("fw_vsa_fine3a_mma128")?, get("fw_vsa_fine3b_mma128")?, get("fw_vsa_fine3c_mma128")?, get("fw_vsa_fine3d_mma128")?],
            ],
            dequant_q8: get("fw_dequant_q8")?,
            dequant_q4: get("fw_dequant_q4")?,
            dequant_nvfp4: get("fw_dequant_nvfp4")?,
            conv2d: get("fw_conv2d")?,
            conv2d_mma: get("fw_conv2d_mma")?,
            conv3d: get("fw_conv3d")?,
            conv3d_mma: get("fw_conv3d_mma")?,
        };
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone()).map_err(err)?;
        let lt = if env1("H3_CUDA_NO_LT") { None } else { Lt::new(&stream).ok() };
        let name = format!("cuda {} (sm_{maj}{min})", ctx.name().unwrap_or_default());
        Ok(Self {
            ctx,
            stream,
            blas,
            lt,
            module,
            f,
            name,
            ampere: maj >= 8,
            q8_scratch: std::sync::Mutex::new(None),
            scalar_conv: env1("H3_CUDA_SCALAR_CONV"),
            scalar_attn: env1("H3_CUDA_SCALAR_ATTN"),
            attn_v1: env1("H3_CUDA_ATTN_V1"),
            vsa_v2: env1("H3_CUDA_VSA_V2"),
            vsa_pair: env1("H3_CUDA_VSA_PAIR"),
            vsa_coarse1: env1("H3_CUDA_VSA_COARSE1"),
            vsa_dbg: std::env::var("H3_CUDA_VSA_DBG").ok().and_then(|v| v.parse().ok()).unwrap_or(0),
            vsa_fine3: match std::env::var("H3_CUDA_VSA_FINE").as_deref() {
                Ok("a") => Some(0),
                Ok("c") => Some(2),
                Ok("b") => Some(1),
                Ok("v1") => None,
                _ => Some(3),
            },
            prof: env1("H3_PROFILE").then(Default::default),
        })
    }

    fn ptr(&self, b: &CuBuf) -> u64 {
        let (p, _sync) = b.s.device_ptr(&self.stream);
        p
    }

    fn opt(&self, b: Option<&CuBuf>) -> u64 {
        b.map_or(0, |b| self.ptr(b))
    }

    /// Element offset into a bf16 buffer, in bytes.
    fn ptr_at(&self, b: &CuBuf, elems: usize) -> u64 {
        self.ptr(b) + (elems * 2) as u64
    }

    fn launch(&self, f: &CudaFunction, grid: (u32, u32, u32), block: (u32, u32, u32), args: &[Arg]) {
        let mut lb = self.stream.launch_builder(f);
        for a in args {
            match a {
                Arg::P(v) => lb.arg(v),
                Arg::I(v) => lb.arg(v),
                Arg::L(v) => lb.arg(v),
                Arg::F(v) => lb.arg(v),
            };
        }
        let cfg = LaunchConfig { grid_dim: grid, block_dim: block, shared_mem_bytes: 0 };
        // SAFETY: every kernel's parameter list is matched by the `args`
        // built next to its launch, and every pointer is a live allocation of
        // at least the extent the kernel indexes (lengths asserted by callers).
        self.timed(|| self.kernel_name(f).to_string(), || unsafe { lb.launch(cfg) }.expect("kernel launch"));
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
        let all: [(&CudaFunction, &'static str); 56] = [
            (&k.bias_act, "bias_act"),
            (&k.swiglu, "swiglu"),
            (&k.gate_residual, "gate_residual"),
            (&k.rms, "rms_norm (+residual, modulation)"),
            (&k.layer_norm, "layer_norm"),
            (&k.head_split, "head_split (qk-norm + rope)"),
            (&k.head_split_warp, "head_split (qk-norm + rope)"),
            (&k.patch, "patchify / unpatchify"),
            (&k.flow_step, "flow_step"),
            (&k.axpby, "axpby"),
            (&k.channel_affine, "channel_affine"),
            (&k.fill, "fill"),
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
            (&k.dequant_q8, "dequant int8"),
            (&k.dequant_q4, "dequant int4"),
            (&k.dequant_nvfp4, "dequant nvfp4"),
            (&k.conv2d, "conv2d (scalar)"),
            (&k.conv2d_mma, "conv2d (mma)"),
            (&k.conv3d, "conv3d (scalar)"),
            (&k.conv3d_mma, "conv3d (mma)"),
        ];
        all.iter().find(|(g, _)| std::ptr::eq(*g, f)).map_or("kernel", |(_, n)| n)
    }

    fn elems(n: usize) -> (u32, u32, u32) {
        ((n.div_ceil(256)) as u32, 1, 1)
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
        let mut rows: Vec<(&String, &(f64, usize))> = m.iter().filter(|(k, _)| k.as_str() != "gemm FLOP (total)").collect();
        rows.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
        let total: f64 = rows.iter().map(|r| r.1.0).sum();
        let gemm: f64 = rows.iter().filter(|r| r.0.starts_with("gemm")).map(|r| r.1.0).sum();
        let mut out = format!(
            "profile (CUDA, synced per op): {:.1} ms in ops; GEMMs {:.1} ms at {:.1} TFLOP/s\n",
            total * 1e3,
            gemm * 1e3,
            if gemm > 0.0 { flops / gemm / 1e12 } else { 0.0 }
        );
        for (k, (t, n)) in rows {
            out += &format!("  {k:40} {:9.1} ms {:5.1}%  {n:6} calls  {:8.3} ms/call\n", t * 1e3, 100.0 * t / total.max(1e-12), t * 1e3 / *n as f64);
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
            Gate::None => [Arg::I(0), Arg::P(0), Arg::P(0), Arg::P(0), Arg::I(0), Arg::I(0)],
            Gate::Vector(v) => [Arg::I(1), Arg::P(self.ptr(v)), Arg::P(0), Arg::P(0), Arg::I(0), Arg::I(0)],
            Gate::Table(m) => {
                assert!(m.table.table.f32 && m.table.idx.f32, "modulation tables come from upload_table / upload_u32");
                [Arg::I(2), Arg::P(0), Arg::P(self.ptr(m.table.table)), Arg::P(self.ptr(m.table.idx)), Arg::I(m.table.slots as i32), Arg::I(m.slot as i32)]
            }
        }
    }

    /// Modulation arguments: has_mod, table, idx, slots, scale, shift.
    fn mod_args(&self, m: &Modulation<'_, CuBuf>) -> [Arg; 6] {
        match m {
            Modulation::None => [Arg::I(0), Arg::P(0), Arg::P(0), Arg::I(0), Arg::I(0), Arg::I(0)],
            Modulation::Table { m, scale, shift } => {
                assert!(m.table.f32 && m.idx.f32, "modulation tables come from upload_table / upload_u32");
                [Arg::I(1), Arg::P(self.ptr(m.table)), Arg::P(self.ptr(m.idx)), Arg::I(m.slots as i32), Arg::I(*scale as i32), Arg::I(*shift as i32)]
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
        let mut args = vec![Arg::P(self.ptr(h)), Arg::P(self.opt(a)), Arg::P(self.ptr(out)), Arg::I(rows as i32), Arg::I(dim as i32), Arg::I(a.is_some() as i32)];
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
pub fn conformance(ordinal: usize, tol: f32) -> Result<Vec<h3_model::check::CheckResult>, CudaError> {
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
        CuBuf { s: Arc::new(self.stream.alloc_zeros::<u16>(len.max(1)).expect("cuda alloc")), len, f32: false }
    }

    fn upload(&self, data: &[f32]) -> CuBuf {
        let h: Vec<u16> = data.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
        let h = if h.is_empty() { vec![0] } else { h };
        CuBuf { s: Arc::new(self.stream.clone_htod(&h).expect("upload")), len: data.len(), f32: false }
    }

    fn upload_table(&self, data: &[f32]) -> CuBuf {
        let h: Vec<u16> = data
            .iter()
            .flat_map(|v| {
                let b = v.to_bits();
                [(b & 0xffff) as u16, (b >> 16) as u16]
            })
            .collect();
        let h = if h.is_empty() { vec![0, 0] } else { h };
        CuBuf { s: Arc::new(self.stream.clone_htod(&h).expect("upload")), len: data.len(), f32: true }
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
            unsafe { cudarc::driver::result::memcpy_htod_async(self.ptr(buf), &h, self.stream.cu_stream()).expect("write") };
        } else {
            let h: Vec<u16> = data.iter().map(|&v| bf16::from_f32(v).to_bits()).collect();
            // SAFETY: `h` is a host slice of `data.len()` u16s and the
            // destination holds at least that many; the copy is ordered on this
            // stream and `h` is kept alive until the stream is synchronised.
            unsafe { cudarc::driver::result::memcpy_htod_async(self.ptr(buf), &h, self.stream.cu_stream()).expect("write") };
        }
        self.stream.synchronize().expect("sync");
    }

    fn download(&self, buf: &CuBuf) -> Vec<f32> {
        let h = self.stream.clone_dtoh(&*buf.s).expect("download");
        if buf.f32 {
            h[..2 * buf.len].par_chunks_exact(2).map(|p| f32::from_bits(p[0] as u32 | (p[1] as u32) << 16)).collect()
        } else {
            h[..buf.len].par_iter().map(|&b| bf16::from_bits(b).to_f32()).collect()
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

    fn fill(&self, buf: &CuBuf, off: usize, len: usize, value: f32) {
        if len == 0 {
            return;
        }
        assert!(!buf.f32 && off + len <= buf.len);
        self.launch(&self.f.fill, Self::elems(len), (256, 1, 1), &[Arg::P(self.ptr(buf)), Arg::L(off as i64), Arg::L(len as i64), Arg::F(value)]);
    }

    fn weight(&self, data: &[f32], rows: usize, cols: usize) -> CuWeight {
        assert_eq!(data.len(), rows * cols);
        CuWeight { buf: self.upload(data), rows, cols, quant: None }
    }

    fn weight_q8(&self, q: &Q8) -> CuWeight {
        let mut words: Vec<u16> = q.q.chunks(2).map(|c| c[0] as u8 as u16 | (c.get(1).map_or(0, |&v| v as u8 as u16) << 8)).collect();
        if words.is_empty() {
            words.push(0);
        }
        CuWeight {
            buf: CuBuf { s: Arc::new(self.stream.clone_htod(&words).expect("upload")), len: words.len(), f32: false },
            rows: q.rows,
            cols: q.cols,
            quant: Some((Quant::Q8, Backend::upload_table(self, &q.scales), q.group)),
        }
    }

    fn weight_q4(&self, q: &Q4) -> CuWeight {
        let bytes = q.packed();
        let mut words: Vec<u16> = bytes.chunks(2).map(|c| c[0] as u16 | (c.get(1).map_or(0, |&v| v as u16) << 8)).collect();
        if words.is_empty() {
            words.push(0);
        }
        let sz: Vec<f32> = q.scales.iter().zip(&q.zeros).flat_map(|(&s, &z)| [s, z]).collect();
        CuWeight {
            buf: CuBuf { s: Arc::new(self.stream.clone_htod(&words).expect("upload")), len: words.len(), f32: false },
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
        let mut codes: Vec<u16> = q.packed.chunks(2).map(|c| c[0] as u16 | (c.get(1).map_or(0, |&v| v as u16) << 8)).collect();
        if codes.is_empty() {
            codes.push(0);
        }
        let mut scales: Vec<u16> = q.scales.chunks(2).map(|c| c[0] as u16 | (c.get(1).map_or(0, |&v| v as u16) << 8)).collect();
        if scales.is_empty() {
            scales.push(0);
        }
        let gc = CuBuf { s: Arc::new(self.stream.clone_htod(&codes).expect("upload")), len: codes.len(), f32: false };
        let gs = CuBuf { s: Arc::new(self.stream.clone_htod(&scales).expect("upload")), len: scales.len(), f32: false };
        CuWeight { buf: gc, rows: q.rows, cols: q.cols, quant: Some((Quant::Nvfp4 { alpha: q.alpha, first_high: q.first_high }, gs, 16)) }
    }

    fn weight_bytes(&self, w: &CuWeight) -> usize {
        w.buf.len * 2 + w.quant.as_ref().map_or(0, |(_, s, _)| s.len * if s.f32 { 4 } else { 2 })
    }

    fn sync(&self) {
        self.stream.synchronize().expect("sync");
    }

    fn profile_report(&self) -> Option<String> {
        self.profile_table()
    }

    fn profile_reset(&self) {
        if let Some(p) = &self.prof {
            self.stream.synchronize().expect("sync");
            p.lock().unwrap().clear();
        }
    }

    fn linear(&self, x: &CuBuf, x_row: usize, rows: usize, w: &CuWeight, bias: Option<&CuBuf>, act: Act, y: &CuBuf, y_row: usize) {
        let (n, k) = (w.rows, w.cols);
        assert!((x_row + rows) * k <= x.len && (y_row + rows) * n <= y.len);
        if rows == 0 {
            return;
        }
        let mut scratch = self.q8_scratch.lock().unwrap();
        let wp = match &w.quant {
            None => self.ptr(&w.buf),
            Some((kind, scales, group)) => {
                if scratch.as_ref().is_none_or(|b| b.len < n * k) {
                    *scratch = Some(self.alloc(n * k));
                }
                let dst = scratch.as_ref().unwrap();
                match kind {
                    Quant::Q8 | Quant::Q4 => self.launch(
                        if *kind == Quant::Q8 { &self.f.dequant_q8 } else { &self.f.dequant_q4 },
                        Self::elems(n * k),
                        (256, 1, 1),
                        &[Arg::P(self.ptr(dst)), Arg::P(self.ptr(&w.buf)), Arg::P(self.ptr(scales)), Arg::L((n * k) as i64), Arg::I(*group as i32)],
                    ),
                    Quant::Nvfp4 { alpha, first_high } => self.launch(
                        &self.f.dequant_nvfp4,
                        Self::elems(n * k),
                        (256, 1, 1),
                        &[Arg::P(self.ptr(dst)), Arg::P(self.ptr(&w.buf)), Arg::P(self.ptr(scales)), Arg::L((n * k) as i64), Arg::I(k as i32), Arg::F(*alpha), Arg::I(*first_high as i32)],
                    ),
                }
                self.ptr(dst)
            }
        };
        let act_code = match act {
            Act::None => 0,
            Act::GeluTanh => 1,
            Act::GeluErf => 2,
            Act::Silu => 3,
        };
        let (alpha, beta) = (1.0f32, 0.0f32);
        let yp = self.ptr_at(y, y_row * n);
        let xp = self.ptr_at(x, x_row * k);
        // cuBLASLt with the bias (+ tanh-GELU) in the epilogue: no second pass.
        if let (Some(lt), Some(b)) = (&self.lt, bias) {
            let epi = (act == Act::GeluTanh) as u8;
            let mut plans = lt.plans.lock().unwrap();
            let plan = plans.entry((n, k, rows, epi)).or_insert_with(|| lt.plan(n, k, rows, epi));
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
                        &[Arg::P(yp), Arg::P(0), Arg::L(total as i64), Arg::I(n as i32), Arg::I(0), Arg::I(act_code)],
                    );
                }
                return;
            }
        }
        // Row-major y[m,n] = x[m,k]·W[n,k]ᵀ is column-major yᵀ = W·xᵀ:
        // A = W (transposed), B = x, C = y, all with leading dims k, k, n.
        // SAFETY: pointers and extents checked above; alpha/beta outlive the call.
        self.timed(
            || format!("gemm n={n} k={k}{}", if rows < 1024 { " (small)" } else { "" }),
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
                &[Arg::P(yp), Arg::P(self.opt(bias)), Arg::L(total as i64), Arg::I(n as i32), Arg::I(bias.is_some() as i32), Arg::I(act_code)],
            );
        }
    }

    fn swiglu(&self, x: &CuBuf, rows: usize, dim: usize, out: &CuBuf) {
        let n = rows * dim;
        assert!(2 * n <= x.len && n <= out.len);
        self.launch(&self.f.swiglu, Self::elems(n), (256, 1, 1), &[Arg::P(self.ptr(x)), Arg::P(self.ptr(out)), Arg::L(n as i64), Arg::I(dim as i32)]);
    }

    fn rms_norm(&self, x: &CuBuf, rows: usize, dim: usize, w: Option<&CuBuf>, modulation: Modulation<'_, CuBuf>, eps: f32, out: &CuBuf) {
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
        let mut args = vec![Arg::P(self.ptr(h)), Arg::P(self.ptr(a)), Arg::L(n as i64), Arg::I(dim as i32)];
        args.extend(self.gate_args(&gate));
        self.launch(&self.f.gate_residual, Self::elems(n), (256, 1, 1), &args);
    }

    fn layer_norm(&self, x: &CuBuf, rows: usize, dim: usize, w: Option<&CuBuf>, b: Option<&CuBuf>, eps: f32, out: &CuBuf) {
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
        assert!(heads <= 256, "head_split keeps per-head statistics for up to 256 heads");
        assert!(heads * rows * hd <= out.len);
        if let Some(r) = &rope {
            assert!(r.cos.f32 && r.sin.f32, "RoPE tables must come from upload_table");
            assert!(r.rot.is_multiple_of(2) && r.rot <= hd);
        }
        let w = norm.as_ref().and_then(|n| n.w);
        let warp_kernel = hd.is_multiple_of(32) && hd <= 128;
        self.launch(
            if warp_kernel { &self.f.head_split_warp } else { &self.f.head_split },
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

    fn attention(&self, q: &CuBuf, k: &CuBuf, v: &CuBuf, heads: usize, kv_heads: usize, sq: usize, sk: usize, d: usize, causal: bool, out: &CuBuf) {
        assert!(d <= 128 && heads.is_multiple_of(kv_heads));
        assert!(heads * sq * d <= q.len && kv_heads * sk * d <= k.len && kv_heads * sk * d <= v.len && sq * heads * d <= out.len);
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

    fn vsa_prepare(&self, spec: &VsaSpec) -> CuVsa {
        let plan = &spec.plan;
        let (nb, h, d) = (plan.num_tiles(), spec.heads, spec.head_dim);
        assert!(nb <= 4096, "VSA selection holds at most 4096 tiles per block");
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
            maxsel,
            te: plan.tile_elems,
            heads: h,
            d,
            seq: plan.seq,
            scale: spec.scale,
        }
    }

    fn vsa_attention(&self, q: &CuBuf, k: &CuBuf, v: &CuBuf, gate: Option<&CuBuf>, s: &CuVsa, out: &CuBuf) {
        let n = s.heads * s.seq * s.d;
        assert!(n <= q.len && n <= k.len && n <= v.len && n <= out.len);
        if let Some(g) = gate {
            assert!(n <= g.len);
        }
        let grid = (s.nb as u32, s.heads as u32, 1);
        let vec_pool = s.d.is_multiple_of(4) && s.d <= 128;
        self.launch(
            if vec_pool { &self.f.vsa_pool } else { &self.f.vsa_pool_scalar },
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
            if coarse4 { &self.f.vsa_coarse4 } else { &self.f.vsa_coarse },
            if coarse4 { (s.nb.div_ceil(4) as u32, s.heads as u32, 1) } else { grid },
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
        if let (Some(f), true, false, true, true) = (pair, self.ampere, self.scalar_attn, self.vsa_pair, s.te == 64) {
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
            args.extend([Arg::P(self.ptr(&s.usel)), Arg::P(self.ptr(&s.nusel)), Arg::I(s.npairs as i32), Arg::I(s.ustride as i32), Arg::I(self.vsa_dbg)]);
            self.launch(f, (s.npairs as u32, s.heads as u32, 1), (256, 1, 1), &args);
            return;
        }
        if let (Some(var), true, false, true) = (self.vsa_fine3, self.ampere, self.scalar_attn, s.d == 64 || s.d == 128) {
            args.push(Arg::F(s.scale));
            tail(&mut args);
            let f = &self.f.vsa_fine3[if s.d == 128 { 1 } else { 0 }][var];
            self.launch(f, (s.nb as u32 * (s.te / 64) as u32, s.heads as u32, 1), (128, 1, 1), &args);
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
            self.launch(f, (s.nb as u32 * sub, s.heads as u32, 1), (threads, 1, 1), &args);
        } else {
            args.push(Arg::I(s.d as i32));
            args.push(Arg::F(s.scale));
            tail(&mut args);
            self.launch(&self.f.vsa_fine, grid, (64, 1, 1), &args);
        }
    }

    fn patchify(&self, latent: &CuBuf, c: usize, t: usize, h: usize, w: usize, patch: (usize, usize, usize), out: &CuBuf, out_row: usize) {
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

    fn unpatchify(&self, x: &CuBuf, x_row: usize, x_stride: usize, col: usize, c: usize, t: usize, h: usize, w: usize, patch: (usize, usize, usize), out: &CuBuf) {
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

    fn flow_step(&self, x: &CuBuf, off: usize, v: &CuBuf, v_off: usize, noise: Option<&CuBuf>, n: usize, step: FlowStep) {
        assert!(off + n <= x.len && v_off + n <= v.len);
        let (mode, sigma, next) = match step {
            FlowStep::Euler { sigma, next } => (0, sigma, next),
            FlowStep::Renoise { sigma, next } => {
                assert!(noise.is_some_and(|z| off + n <= z.len), "renoise needs noise");
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
            &[Arg::P(self.ptr(y)), Arg::L(off as i64), Arg::F(a), Arg::P(self.ptr(x)), Arg::L(x_off as i64), Arg::F(b), Arg::L(n as i64)],
        );
    }

    fn channel_affine(&self, x: &CuBuf, c: usize, inner: usize, scale: &CuBuf, shift: Option<&CuBuf>, out: &CuBuf) {
        let n = c * inner;
        assert!(n <= x.len && n <= out.len && c <= scale.len);
        self.launch(
            &self.f.channel_affine,
            Self::elems(n),
            (256, 1, 1),
            &[Arg::P(self.ptr(x)), Arg::P(self.ptr(out)), Arg::L(n as i64), Arg::I(inner as i32), Arg::P(self.ptr(scale)), Arg::I(shift.is_some() as i32), Arg::P(self.opt(shift))],
        );
    }

    fn conv2d(&self, a: &CuBuf, b: Option<&CuBuf>, s: &ConvShape, w: &CuWeight, bias: Option<&CuBuf>, ep: ConvEpilogue<'_, CuBuf>, out: &CuBuf) {
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
            if mma { &self.f.conv2d_mma } else { &self.f.conv2d },
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

    fn conv3d(&self, x: &CuBuf, s: &Conv3dShape, w: &CuWeight, bias: Option<&CuBuf>, ep: ConvEpilogue<'_, CuBuf>, out: &CuBuf) {
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
            self.launch(&self.f.conv3d_mma, (s.cout.div_ceil(64) as u32, gy, gz), (128, 1, 1), &args);
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

    fn conv1d(&self, x: &CuBuf, s: &Conv1dShape, w: &CuWeight, bias: Option<&CuBuf>, ep: ConvEpilogue<'_, CuBuf>, out: &CuBuf) {
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

    fn replicate_pad1d(&self, x: &CuBuf, c: usize, len: usize, left: usize, right: usize, out: &CuBuf) {
        let n = c * (len + left + right);
        assert!(c * len <= x.len && n <= out.len && len > 0);
        self.launch(
            &self.f.replicate_pad1d,
            Self::elems(n),
            (256, 1, 1),
            &[Arg::P(self.ptr(x)), Arg::P(self.ptr(out)), Arg::L(n as i64), Arg::I(len as i32), Arg::I(left as i32), Arg::I(right as i32)],
        );
    }

    fn conv_transpose1d(&self, x: &CuBuf, s: &ConvT1dShape, w: &CuWeight, bias: Option<&CuBuf>, out: &CuBuf) {
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

    fn group_norm(&self, x: &CuBuf, c: usize, t: usize, hw: usize, groups: usize, per_frame: bool, w: &CuBuf, b: &CuBuf, eps: f32, act: Act, out: &CuBuf) {
        let n = c * t * hw;
        assert!(n <= x.len && n <= out.len && c.is_multiple_of(groups));
        let fg = if per_frame { t * groups } else { groups };
        let stats = self.words(&vec![0; 2 * fg]);
        self.launch(
            &self.f.gn_stats,
            (fg as u32, 1, 1),
            (256, 1, 1),
            &[Arg::P(self.ptr(x)), Arg::P(self.ptr(&stats)), Arg::I(c as i32), Arg::I(t as i32), Arg::I(hw as i32), Arg::I(groups as i32), Arg::I(per_frame as i32), Arg::F(eps)],
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
        self.launch(&self.f.snake, Self::elems(n), (256, 1, 1), &[Arg::P(self.ptr(x)), Arg::L(n as i64), Arg::I(len as i32), Arg::P(alpha), Arg::P(beta), Arg::I(kind)]);
    }

    fn trilinear_resize(&self, x: &CuBuf, c: usize, t: usize, h: usize, w: usize, t2: usize, h2: usize, w2: usize, out: &CuBuf) {
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

    fn pixel_shuffle(&self, x: &CuBuf, frames: usize, c: usize, h: usize, w: usize, r: usize, out: &CuBuf) {
        let n = frames * c * h * w * r * r;
        assert!(n <= x.len && n <= out.len);
        self.launch(
            &self.f.pixel_shuffle,
            Self::elems(n),
            (256, 1, 1),
            &[Arg::P(self.ptr(x)), Arg::P(self.ptr(out)), Arg::L(n as i64), Arg::I(c as i32), Arg::I(h as i32), Arg::I(w as i32), Arg::I(r as i32)],
        );
    }

    fn tanh_clamp(&self, x: &CuBuf, n: usize, s: f32) {
        assert!(n <= x.len);
        self.launch(&self.f.tanh_clamp, Self::elems(n), (256, 1, 1), &[Arg::P(self.ptr(x)), Arg::L(n as i64), Arg::F(s)]);
    }

    fn clamp(&self, x: &CuBuf, n: usize, lo: f32, hi: f32) {
        assert!(n <= x.len);
        self.launch(&self.f.clamp, Self::elems(n), (256, 1, 1), &[Arg::P(self.ptr(x)), Arg::L(n as i64), Arg::F(lo), Arg::F(hi)]);
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
        let Ok(path) = std::env::var("H3_PTX_OUT") else { return };
        match std::panic::catch_unwind(|| super::compile_kernels("compute_89")) {
            Ok(Ok(ptx)) => std::fs::write(&path, ptx.to_src()).unwrap(),
            Ok(Err(e)) => panic!("{e}"),
            Err(_) => eprintln!("skipping: libnvrtc not loadable"),
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
