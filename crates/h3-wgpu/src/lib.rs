//! WebGPU backend (wgpu): one WGSL kernel per [`Backend`] op (`fw_<op>`, the
//! CUDA twins share the names).
//!
//! Runs natively on Vulkan / Metal / DX12 and in the browser on `wasm32`.
//! Activations are f32; weights are f16 pairs packed in `u32` (core
//! `unpack2x16float`, no `shader-f16` needed), int8 / 4-bit codes with their
//! scales, or f32 when the values do not fit f16. Dispatches are recorded
//! into one command encoder and submitted in batches; a host write or read
//! flushes first, so queue order always matches program order.
//!
//! Environment: `H3_PROFILE=1` per-kernel GPU time (timestamp queries),
//! `H3_WGPU_F16=1` f16-math GEMM where the adapter has `shader-f16`,
//! `H3_ALLOW_SOFTWARE=1` accepts a CPU rasterizer (lavapipe) adapter.

use h3_core::backend::{
    Act, Backend, BoxRows, Conv1dShape, Conv3dShape, ConvEpilogue, ConvShape, ConvT1dShape,
    FlowStep, Gate, HeadNorm, Modulation, Pad, Rope, Snake,
};
use h3_core::quant::{Mxfp8, Nvfp4, Q4, Q4Packed, Q8};
use h3_core::vsa::VsaSpec;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use wgpu::util::DeviceExt;

const COMMON: &str = include_str!("shaders/common.wgsl");
const LINEAR3: &str = include_str!("shaders/linear3.wgsl");
const LINEAR4: &str = include_str!("shaders/linear4.wgsl");
const LINEAR3H: &str = include_str!("shaders/linear3h.wgsl");
const NORM: &str = include_str!("shaders/norm.wgsl");
const HEADS: &str = include_str!("shaders/head_split.wgsl");
const ATTN: &str = include_str!("shaders/attention.wgsl");
const FLASH: &str = include_str!("shaders/flash.wgsl");
const FLASH3: &str = include_str!("shaders/flash3.wgsl");
const FLASH3H: &str = include_str!("shaders/flash3h.wgsl");
const VSA_POOL: &str = include_str!("shaders/vsa_pool.wgsl");
const VSA_COARSE: &str = include_str!("shaders/vsa_coarse.wgsl");
const ELEM: &str = include_str!("shaders/elementwise.wgsl");
const MX_ACT: &str = include_str!("shaders/mx_act.wgsl");
const NV_AMAX: &str = include_str!("shaders/nv_amax.wgsl");
const NV_ACT: &str = include_str!("shaders/nv_act.wgsl");
const VSA_QK8: &str = include_str!("shaders/vsa_qk8.wgsl");
const ATTN_QK8: &str = include_str!("shaders/attn_qk8.wgsl");
const ATTN_PV8: &str = include_str!("shaders/attn_pv8.wgsl");
const GROUP_NORM: &str = include_str!("shaders/group_norm.wgsl");
const CONV: &str = include_str!("shaders/conv.wgsl");
const CONV3D: &str = include_str!("shaders/conv3d.wgsl");
const CONV3D_DIRECT: &str = include_str!("shaders/conv3d_direct.wgsl");
const CONV1D: &str = include_str!("shaders/conv1d.wgsl");
const CONVT1D: &str = include_str!("shaders/convt1d.wgsl");
const BOX_ROWS: &str = include_str!("shaders/box_rows.wgsl");

/// Token-sized GEMMs only take the f16-math path (activations of small
/// projections can exceed f16's range).
const F16_MIN_ROWS: usize = 1024;
/// Shared memory the flash kernel declares.
const FLASH_SHARED: u32 = 2048 * 4 + 4160 * 4 + 5 * 64 * 4 + 4;
/// Shared memory flash3 declares (vec4 panels, S tile, partial max / sum,
/// row state, row tokens, the any-key flag).
const FLASH3_SHARED: u32 = 512 * 16 + 4160 * 4 + 2 * 256 * 4 + 5 * 64 * 4 + 4;
/// Shared memory the VSA coarse kernel declares (scores + keep flags + reduction).
const COARSE_SHARED: u32 = 4096 * 4 + 128 * 4 + 256 * 4 + 4;
const MAX_TILES: usize = 4096;
/// Submit after this many recorded dispatches.
const BATCH: usize = 64;
/// Submissions the host may run ahead of the GPU (`H3_WGPU_INFLIGHT`). wgpu
/// frees a dropped buffer only once its submissions are done and the device is
/// polled; without a bound the host queued a whole denoise step of per-op
/// temporaries and streamed block weights before the GPU finished any of it
/// (a 480p int4 draft peaked at 47 GB resident, 54 GB streamed, on an RTX PRO
/// 6000). Native only: the browser frees through its own event loop.
const INFLIGHT: usize = 4;
/// Bytes of uploads (`create_buffer_init`, `write`) between submissions of
/// the queue's pending writes. wgpu stages each upload in a host-visible copy
/// that it frees only once a submission carrying it has completed and the
/// device is polled: a `generate` with `--tables` submits nothing between
/// load and the first step, so the staged copy of every weight stayed alive
/// until then (480p int4 on an RTX PRO 6000: 40.7 GB at the end of load, a
/// 47.3 GB peak at the first step, 27.1 GB once it was freed).
const STAGE_FLUSH: u64 = 256 << 20;
/// Max workgroups per dispatch dimension.
const MAX_DIM: u32 = 65535;

#[derive(Clone)]
pub struct GBuf {
    buf: Arc<wgpu::Buffer>,
    len: usize,
}

/// VSA state for one layout.
pub struct GVsa {
    slot: GBuf,
    bsz: GBuf,
    regions: GBuf,
    pooled: GBuf,
    coarse: GBuf,
    sel: GBuf,
    nb: usize,
    nprefix: usize,
    nregions: usize,
    heads: usize,
    d: usize,
    seq: usize,
    scale: f32,
    tile: usize,
    /// INT8 Q·Kᵀ emulation (`vsa_qk8.wgsl`): per-head K mean, rounded Q / K copies.
    qk8: Option<(GBuf, GBuf, GBuf)>,
    /// FP8 P·V emulation (`attn_qk8.wgsl` modes 0 / 3 / 4 on V): per-head V mean, E4M3 codes, per-head scale.
    pv8: Option<(GBuf, GBuf, GBuf)>,
    sub: usize,
    /// Every tile kept: the dense attention path plus the gated pooled branch.
    dense: bool,
    /// `slot_of_row` (tile = slot / tile), for the dense path's gate add.
    rowslot: GBuf,
}

struct Prof {
    qs: wgpu::QuerySet,
    cap: u32,
    slots: Vec<String>,
    totals: HashMap<String, (f64, u64)>,
    period_ns: f64,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct WgpuOptions {
    /// f16 math where the adapter has `shader-f16` (`H3_WGPU_F16=1`: the
    /// f16 GEMM and `flash3h`; `=attn`: `flash3h` alone). The f16 GEMM is
    /// 64×64 and only replaces linear3 (`H3_WGPU_LINEAR=3`, or under 128 rows).
    pub f16: bool,
    /// With `f16`: the f16-math GEMM too (false for `H3_WGPU_F16=attn`).
    pub f16_gemm: bool,
    /// Accept a CPU rasterizer adapter (`H3_ALLOW_SOFTWARE=1`).
    pub allow_software: bool,
    /// Cap one storage binding at this many bytes, below the device's limit,
    /// forcing the row-chunked GEMM / head_split paths (`H3_WGPU_MAX_BINDING`;
    /// the conformance test sets 64 KiB).
    pub max_binding: Option<u64>,
    /// GEMM kernel (`H3_WGPU_LINEAR`): 4 = 128×128 tiles, 8×8 per thread
    /// (`linear4.wgsl`, the default; GEMMs under 128 rows run 3), 3 = 64×64,
    /// 4×4 per thread (`linear3.wgsl`). The two agree bit for bit.
    pub linear: u32,
    /// Attention kernel (`H3_WGPU_FLASH`): 3 = `flash3.wgsl` (the default),
    /// 1 = `flash.wgsl`, 0 = neither (the unfused attention path).
    pub flash: u32,
}

impl WgpuOptions {
    pub fn from_env() -> Self {
        Self {
            f16: std::env::var("H3_WGPU_F16").is_ok_and(|v| v == "1" || v == "attn"),
            f16_gemm: std::env::var("H3_WGPU_F16").is_ok_and(|v| v == "1"),
            allow_software: std::env::var("H3_ALLOW_SOFTWARE").is_ok_and(|v| v == "1"),
            max_binding: std::env::var("H3_WGPU_MAX_BINDING")
                .ok()
                .and_then(|v| v.parse().ok()),
            linear: match std::env::var("H3_WGPU_LINEAR").as_deref() {
                Ok("3") => 3,
                _ => 4,
            },
            flash: match std::env::var("H3_WGPU_FLASH").as_deref() {
                Ok("0") => 0,
                Ok("1") => 1,
                _ => 3,
            },
        }
    }
}

pub struct GWeight {
    buf: GBuf,
    rows: usize,
    cols: usize,
    f16: bool,
    /// Weight-only quantized: `buf` holds the codes, this buffer the
    /// per-group scales (Q8) or interleaved scale / zero pairs (Q4), then the
    /// group size.
    quant: Option<(Quant, GBuf, usize)>,
    /// MXFP8 W8A8: the weight is stored dequantized (no FP8 tensor cores
    /// here) and `linear` rounds its activations through E4M3 / E8M0 blocks
    /// first (`mx_act.wgsl`), so the result equals the CPU reference.
    mx: bool,
    /// NVFP4 W4A4: stored dequantized like `mx`, and `linear` rounds its
    /// activations through NVFP4 first (`nv_amax.wgsl` + `nv_act.wgsl`).
    a4: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Quant {
    Q8,
    Q4,
}

struct Pending {
    enc: Option<wgpu::CommandEncoder>,
    n: usize,
}

pub struct Wgpu {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    attn_bk: usize,
    flash_ok: bool,
    /// `flash3.wgsl` rather than `flash.wgsl` where flash runs.
    flash3: bool,
    /// `linear4.wgsl` for GEMMs of ≥ 128 rows.
    linear4: bool,
    coarse_ok: bool,
    f16: bool,
    /// With `f16`: the f16-math GEMM (`linear3h`) as well as `flash3h`.
    f16_gemm: bool,
    max_binding: Option<u64>,
    pipelines: Mutex<HashMap<String, Arc<wgpu::ComputePipeline>>>,
    names: Mutex<HashMap<usize, String>>,
    tag: Mutex<Option<String>>,
    prof: Option<Mutex<Prof>>,
    pending: Mutex<Pending>,
    /// Submissions not yet known to be done, oldest first ([`INFLIGHT`]).
    inflight: Mutex<std::collections::VecDeque<wgpu::SubmissionIndex>>,
    max_inflight: usize,
    /// Upload bytes since the pending writes were last submitted ([`STAGE_FLUSH`]).
    staged: std::sync::atomic::AtomicU64,
    /// [`STAGE_FLUSH`], or `H3_WGPU_STAGE_FLUSH_MB` (0 = never, the old behaviour).
    stage_flush: u64,
    dummy: GBuf,
    pub allow_f16_weights: bool,
}

#[derive(Debug)]
pub struct InitError(pub String);

impl std::fmt::Display for InitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for InitError {}

impl Wgpu {
    pub async fn new_async() -> Result<Self, InitError> {
        Self::with_options_async(WgpuOptions::from_env()).await
    }

    pub async fn with_options_async(opts: WgpuOptions) -> Result<Self, InitError> {
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
                ..Default::default()
            })
            .await
            .map_err(|e| InitError(format!("no WebGPU adapter: {e}")))?;
        let info = adapter.get_info();
        if info.device_type == wgpu::DeviceType::Cpu && !opts.allow_software {
            return Err(InitError(format!(
                "the only WebGPU adapter is a software rasterizer ({}); set H3_ALLOW_SOFTWARE=1 to use it",
                info.name
            )));
        }
        let limits = adapter.limits();
        let profile = std::env::var("H3_PROFILE").is_ok_and(|v| v == "1")
            && adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY);
        let f16 = opts.f16 && adapter.features().contains(wgpu::Features::SHADER_F16);
        let mut features = wgpu::Features::empty();
        if profile {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }
        if f16 {
            features |= wgpu::Features::SHADER_F16;
        }
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("h3"),
                required_features: features,
                required_limits: limits.clone(),
                ..Default::default()
            })
            .await
            .map_err(|e| InitError(format!("request_device: {e}")))?;
        let wgs = device.limits().max_compute_workgroup_storage_size as usize;
        let attn_bk = if wgs >= 2 * 32 * 128 * 4 { 32 } else { 16 };
        let flash_ok = wgs >= FLASH_SHARED as usize && opts.flash != 0;
        let flash3 = opts.flash == 3 && wgs >= FLASH3_SHARED as usize;
        let linear4 = opts.linear == 4;
        let coarse_ok = wgs >= COARSE_SHARED as usize;
        let dummy = GBuf {
            buf: Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("dummy"),
                size: 16,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })),
            len: 4,
        };
        let prof = profile.then(|| {
            let cap = 2048;
            Mutex::new(Prof {
                qs: device.create_query_set(&wgpu::QuerySetDescriptor {
                    label: Some("profile"),
                    ty: wgpu::QueryType::Timestamp,
                    count: 2 * cap,
                }),
                cap,
                slots: Vec::new(),
                totals: HashMap::new(),
                period_ns: queue.get_timestamp_period() as f64,
            })
        });
        Ok(Self {
            info,
            device,
            queue,
            attn_bk,
            flash_ok,
            flash3,
            linear4,
            coarse_ok,
            f16,
            f16_gemm: f16 && opts.f16_gemm,
            max_binding: opts.max_binding,
            pipelines: Mutex::new(HashMap::new()),
            names: Mutex::new(HashMap::new()),
            tag: Mutex::new(None),
            prof,
            pending: Mutex::new(Pending { enc: None, n: 0 }),
            inflight: Mutex::new(Default::default()),
            staged: Default::default(),
            stage_flush: std::env::var("H3_WGPU_STAGE_FLUSH_MB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .map_or(STAGE_FLUSH, |mb| if mb == 0 { u64::MAX } else { mb << 20 }),
            max_inflight: std::env::var("H3_WGPU_INFLIGHT")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(INFLIGHT)
                .max(1),
            dummy,
            allow_f16_weights: true,
        })
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn new() -> Result<Self, InitError> {
        pollster::block_on(Self::new_async())
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub fn with_options(opts: WgpuOptions) -> Result<Self, InitError> {
        pollster::block_on(Self::with_options_async(opts))
    }

    pub fn f16(&self) -> bool {
        self.f16
    }

    /// A CPU rasterizer (llvmpipe / lavapipe / SwiftShader / WARP): fine for
    /// conformance checks, meaningless for timing.
    pub fn is_software(&self) -> bool {
        self.info.device_type == wgpu::DeviceType::Cpu
    }

    pub fn adapter_info(&self) -> &wgpu::AdapterInfo {
        &self.info
    }

    pub fn limits(&self) -> wgpu::Limits {
        self.device.limits()
    }

    fn pipeline(&self, key: &str, src: impl FnOnce() -> String) -> Arc<wgpu::ComputePipeline> {
        let mut map = self.pipelines.lock().unwrap();
        if let Some(p) = map.get(key) {
            return p.clone();
        }
        let body = src();
        let code = if body.contains("vec4<f16>") {
            format!("enable f16;\n{COMMON}\n{body}")
        } else {
            format!("{COMMON}\n{body}")
        };
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(key),
                source: wgpu::ShaderSource::Wgsl(code.into()),
            });
        let p = Arc::new(
            self.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(key),
                    layout: None,
                    module: &module,
                    entry_point: Some("main"),
                    compilation_options: Default::default(),
                    cache: None,
                }),
        );
        map.insert(key.to_string(), p.clone());
        self.names
            .lock()
            .unwrap()
            .insert(Arc::as_ptr(&p) as usize, key.to_string());
        p
    }

    /// Per-kernel GPU time since the last reset (`H3_PROFILE=1`).
    pub fn profile_table(&self) -> Option<String> {
        let prof = self.prof.as_ref()?;
        self.flush();
        let p = prof.lock().unwrap();
        let total: f64 = p.totals.values().map(|v| v.0).sum();
        let mut rows: Vec<_> = p.totals.iter().collect();
        rows.sort_by(|a, b| b.1.0.total_cmp(&a.1.0));
        let mut out = format!(
            "GPU time by kernel ({:.1} ms total, timestamp queries):\n",
            total / 1e6
        );
        for (name, (ns, n)) in rows {
            out += &format!(
                "  {:>9.1} ms {:>5.1}%  {:>6} calls {:>9.3} ms/call  {name}\n",
                ns / 1e6,
                100.0 * ns / total.max(1.0),
                n,
                ns / 1e6 / *n as f64
            );
        }
        Some(out)
    }

    pub fn profile_reset(&self) {
        if let Some(prof) = &self.prof {
            self.flush();
            prof.lock().unwrap().totals.clear();
        }
    }

    fn weight_src(src: &str, f16: bool) -> String {
        if f16 {
            src.replace("{W_TYPE}", "u32").replace(
                "{W_LOAD}",
                "let v = unpack2x16float(w[i >> 1u]); return select(v.x, v.y, (i & 1u) == 1u);",
            )
        } else {
            src.replace("{W_TYPE}", "f32")
                .replace("{W_LOAD}", "return w[i];")
        }
    }

    /// GEMM reading int8 codes packed 4 per word, scaled per group (`p._p`
    /// columns per scale, binding 5).
    fn q8_src(src: &str) -> String {
        src.replace("{W_TYPE}", "u32")
            .replace("fn load_w(", "@group(0) @binding(5) var<storage, read> wscale: array<f32>;\nfn load_w(")
            .replace(
                "{W_LOAD}",
                "let b = (w[i >> 2u] >> ((i & 3u) * 8u)) << 24u; return f32(bitcast<i32>(b) >> 24u) * wscale[i / p._p];",
            )
    }

    /// The 4-bit twin: 8 codes per word, `(q − zero) · scale` per group with
    /// (scale, zero) pairs at binding 5.
    fn q4_src(src: &str) -> String {
        src.replace("{W_TYPE}", "u32")
            .replace("fn load_w(", "@group(0) @binding(5) var<storage, read> wsz: array<vec2<f32>>;\nfn load_w(")
            .replace(
                "{W_LOAD}",
                "let c = f32((w[i >> 3u] >> ((i & 7u) * 4u)) & 15u); let g = wsz[i / p._p]; return (c - g.y) * g.x;",
            )
    }

    fn dispatch(
        &self,
        pipe: &wgpu::ComputePipeline,
        params: &[u32],
        bufs: &[&GBuf],
        groups: (u32, u32, u32),
    ) {
        let full: Vec<(&GBuf, Option<(u64, u64)>)> = bufs.iter().map(|b| (*b, None)).collect();
        self.dispatch_ranges(pipe, params, &full, groups);
    }

    /// Bytes one storage binding may cover (2 GiB on NVIDIA Vulkan); larger
    /// activations are bound as row sub-ranges by the ops that touch them.
    fn max_binding_bytes(&self) -> u64 {
        let l = self.device.limits();
        let dev = (l.max_storage_buffer_binding_size as u64).min(l.max_buffer_size);
        match self.max_binding {
            Some(v) => v.min(dev),
            None => dev,
        }
    }

    /// Rows per chunk so that `rows × width × 4` bytes (plus the alignment
    /// remainder) fit one binding, for the widest of the given widths.
    fn row_chunk(&self, widths: &[usize]) -> usize {
        let lim = self.max_binding_bytes() as usize;
        let wmax = widths.iter().copied().max().unwrap_or(1).max(1);
        (lim.saturating_sub(256) / (wmax * 4)).max(1)
    }

    /// A binding covering elements `[start, start + len)`: the byte range,
    /// aligned down to the device's offset alignment, and the element offset
    /// of `start` inside it.
    fn sub_range(&self, start: usize, len: usize) -> ((u64, u64), usize) {
        let align = (self.device.limits().min_storage_buffer_offset_alignment as usize).max(4);
        let bytes = start * 4;
        let off = bytes - bytes % align;
        let rem = (bytes - off) / 4;
        ((off as u64, ((rem + len) * 4) as u64), rem)
    }

    /// `dispatch` where a buffer may be bound as a byte sub-range `(offset, size)`.
    fn dispatch_ranges(
        &self,
        pipe: &wgpu::ComputePipeline,
        params: &[u32],
        bufs: &[(&GBuf, Option<(u64, u64)>)],
        groups: (u32, u32, u32),
    ) {
        let tag = self.tag.lock().unwrap().take();
        let slot = self.prof.as_ref().map(|prof| {
            let full = {
                let p = prof.lock().unwrap();
                p.slots.len() as u32 >= p.cap
            };
            if full {
                self.flush();
            }
            let mut p = prof.lock().unwrap();
            let base = self
                .names
                .lock()
                .unwrap()
                .get(&(pipe as *const _ as usize))
                .cloned()
                .unwrap_or_else(|| "?".into());
            p.slots.push(match tag {
                Some(t) => format!("{base} {t}"),
                None => base,
            });
            (p.slots.len() - 1) as u32
        });
        let mut words = params.to_vec();
        while words.len() % 4 != 0 {
            words.push(0);
        }
        let ubuf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&words),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: ubuf.as_entire_binding(),
        }];
        for (i, (b, range)) in bufs.iter().enumerate() {
            let resource = match range {
                None => b.buf.as_entire_binding(),
                Some((offset, size)) => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &b.buf,
                    offset: *offset,
                    size: std::num::NonZeroU64::new(*size),
                }),
            };
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32 + 1,
                resource,
            });
        }
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &pipe.get_bind_group_layout(0),
            entries: &entries,
        });
        let mut pend = self.pending.lock().unwrap();
        let enc = pend
            .enc
            .get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
        {
            let guard = self.prof.as_ref().map(|p| p.lock().unwrap());
            let tw = match (&guard, slot) {
                (Some(p), Some(i)) => Some(wgpu::ComputePassTimestampWrites {
                    query_set: &p.qs,
                    beginning_of_pass_write_index: Some(2 * i),
                    end_of_pass_write_index: Some(2 * i + 1),
                }),
                _ => None,
            };
            let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: tw,
            });
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(groups.0, groups.1, groups.2);
        }
        pend.n += 1;
        if pend.n >= BATCH {
            let enc = pend.enc.take().unwrap();
            pend.n = 0;
            drop(pend);
            self.submit(enc);
        }
    }

    /// Submit one batch, then wait for the oldest submissions until at most
    /// `max_inflight` are outstanding; the wait also frees what they dropped.
    fn submit(&self, enc: wgpu::CommandEncoder) {
        let idx = self.queue.submit([enc.finish()]);
        self.track(idx);
    }

    /// A buffer filled at creation; every [`STAGE_FLUSH`] bytes the pending
    /// writes go out in a submission of their own, bounded like the others.
    fn init_buffer(&self, contents: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
        let buf = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents,
                usage,
            });
        self.staged_bytes(contents.len());
        buf
    }

    fn staged_bytes(&self, n: usize) {
        use std::sync::atomic::Ordering;
        let total = self.staged.fetch_add(n as u64, Ordering::Relaxed) + n as u64;
        if total >= self.stage_flush {
            self.staged.store(0, Ordering::Relaxed);
            // Recorded dispatches first, then the pending writes alone.
            self.flush();
            let idx = self.queue.submit([]);
            self.track(idx);
        }
    }

    fn track(&self, idx: wgpu::SubmissionIndex) {
        #[cfg(target_arch = "wasm32")]
        let _ = idx;
        #[cfg(not(target_arch = "wasm32"))]
        {
            let mut q = self.inflight.lock().unwrap();
            q.push_back(idx);
            while q.len() > self.max_inflight {
                let oldest = q.pop_front().unwrap();
                let _ = self.device.poll(wgpu::PollType::Wait {
                    submission_index: Some(oldest),
                    timeout: None,
                });
            }
        }
    }

    fn flush(&self) {
        let mut pend = self.pending.lock().unwrap();
        let mut enc = pend.enc.take();
        pend.n = 0;
        drop(pend);
        let mut readback = None;
        if let Some(prof) = &self.prof {
            let p = prof.lock().unwrap();
            let n = p.slots.len() as u32;
            if n > 0 {
                let bytes = (2 * n as u64) * 8;
                let resolve = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: bytes,
                    usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                });
                let read = self.device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                let e = enc
                    .get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
                e.resolve_query_set(&p.qs, 0..2 * n, &resolve, 0);
                e.copy_buffer_to_buffer(&resolve, 0, &read, 0, bytes);
                readback = Some(read);
            }
        }
        if let Some(enc) = enc {
            self.submit(enc);
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let (Some(read), Some(prof)) = (readback, &self.prof) {
            let slice = read.slice(..);
            slice.map_async(wgpu::MapMode::Read, |_| {});
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            let ticks: Vec<u64> =
                bytemuck::cast_slice::<u8, u64>(&slice.get_mapped_range().expect("timestamps"))
                    .to_vec();
            let mut p = prof.lock().unwrap();
            let period = p.period_ns;
            let slots = std::mem::take(&mut p.slots);
            for (i, name) in slots.into_iter().enumerate() {
                let ns = ticks[2 * i + 1].saturating_sub(ticks[2 * i]) as f64 * period;
                let e = p.totals.entry(name).or_insert((0.0, 0));
                e.0 += ns;
                e.1 += 1;
            }
        }
    }

    /// `n` workgroups as a (x, y) grid within the per-dimension limit.
    fn grid(n: usize) -> (u32, u32, u32) {
        let n = n.max(1) as u32;
        if n <= MAX_DIM {
            (n, 1, 1)
        } else {
            let y = n.div_ceil(MAX_DIM);
            (n.div_ceil(y), y, 1)
        }
    }

    fn upload_u32_buf(&self, data: &[u32]) -> GBuf {
        let data = if data.is_empty() { &[0u32][..] } else { data };
        GBuf {
            buf: Arc::new(self.init_buffer(
                bytemuck::cast_slice(data),
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            )),
            len: data.len(),
        }
    }
}

/// Parameters of the elementwise kernel (see `elementwise.wgsl`).
#[derive(Default, Clone, Copy)]
struct Ep {
    n: usize,
    op: u32,
    dim: usize,
    slots: usize,
    slot: usize,
    off: usize,
    xoff: usize,
    c: usize,
    t: usize,
    h: usize,
    w: usize,
    pt: usize,
    ph: usize,
    pw: usize,
    t2: usize,
    h2: usize,
    w2: usize,
    left: usize,
    len: usize,
    r: usize,
    frames: usize,
    stride: usize,
    col: usize,
    has_z: bool,
    f: [f32; 4],
}

impl Ep {
    fn words(&self) -> Vec<u32> {
        let mut v: Vec<u32> = [
            self.n,
            self.op as usize,
            self.dim,
            self.slots,
            self.slot,
            self.off,
            self.xoff,
            self.c,
            self.t,
            self.h,
            self.w,
            self.pt,
            self.ph,
            self.pw,
            self.t2,
            self.h2,
            self.w2,
            self.left,
            self.len,
            self.r,
            self.frames,
            self.stride,
            self.col,
            self.has_z as usize,
        ]
        .iter()
        .map(|&x| x as u32)
        .collect();
        v.extend(self.f.iter().map(|f| f.to_bits()));
        v
    }
}

impl Wgpu {
    /// [`Backend::linear`] without the MXFP8 activation rounding.
    #[allow(clippy::too_many_arguments)]
    fn linear_plain(
        &self,
        x: &GBuf,
        x_row: usize,
        rows: usize,
        w: &GWeight,
        bias: Option<&GBuf>,
        act: Act,
        y: &GBuf,
        y_row: usize,
    ) {
        if rows == 0 {
            return;
        }
        if self.prof.is_some() {
            *self.tag.lock().unwrap() = Some(format!("{}x{} {act:?}", w.rows, w.cols));
        }
        let act_code = match act {
            Act::None => 0,
            Act::GeluTanh => 1,
            Act::GeluErf => 2,
            Act::Silu => 3,
        };
        // 128×128 tiles for GEMMs of 128+ rows (RTX PRO 6000 profile: the
        // linear3 GEMMs were 62 % of a 480p int4 step). The f16-math GEMM
        // (linear3h, 64×64) only ever replaces linear3: in place of linear4
        // it made a 480p native forward 24.5 → 41.5 s (FFN up 113 → 249 ms;
        // docs/runs/2026-11-25-vast-pro6000ws-f16-staging).
        let big = self.linear4 && rows >= 128;
        let half = self.f16 && self.f16_gemm && w.f16 && rows >= F16_MIN_ROWS && !big;
        let (src, tile, name) = if big {
            (LINEAR4, 128, "linear4")
        } else {
            (LINEAR3, 64, "linear3")
        };
        let pipe = if let Some((kind, ..)) = &w.quant {
            match kind {
                Quant::Q8 => self.pipeline(&format!("{name}_q8"), || Self::q8_src(src)),
                Quant::Q4 => self.pipeline(&format!("{name}_q4"), || Self::q4_src(src)),
            }
        } else if half {
            self.pipeline("linear3h", || LINEAR3H.to_string())
        } else {
            let key = format!("{name}_{}", if w.f16 { "f16" } else { "f32" });
            self.pipeline(&key, || Self::weight_src(src, w.f16))
        };
        let p = [
            rows as u32,
            w.rows as u32,
            w.cols as u32,
            (x_row * w.cols) as u32,
            (y_row * w.rows) as u32,
            bias.is_some() as u32,
            act_code,
            w.quant.as_ref().map_or(0, |q| q.2) as u32,
        ];
        let b = bias.unwrap_or(&self.dummy);
        let lim = self.max_binding_bytes();
        if (x.len as u64) * 4 <= lim && (y.len as u64) * 4 <= lim {
            let groups = (w.rows.div_ceil(tile) as u32, rows.div_ceil(tile) as u32, 1);
            return match &w.quant {
                Some((_, scales, _)) => {
                    self.dispatch(&pipe, &p, &[x, &w.buf, b, y, scales], groups)
                }
                None => self.dispatch(&pipe, &p, &[x, &w.buf, b, y], groups),
            };
        }
        // An activation larger than one binding (the DiT's fused QKV at
        // 38k tokens is 3.3 GB): run the GEMM over row chunks, binding each
        // chunk's rows of x and y as sub-ranges; the binding offset is
        // aligned down and the remainder rides in the element offsets.
        let chunk = self.row_chunk(&[w.cols, w.rows]);
        let mut r0 = 0;
        while r0 < rows {
            let n = chunk.min(rows - r0);
            let mut pc = p;
            pc[0] = n as u32;
            let (xr, xo) = self.sub_range((x_row + r0) * w.cols, n * w.cols);
            let (yr, yo) = self.sub_range((y_row + r0) * w.rows, n * w.rows);
            pc[3] = xo as u32;
            pc[4] = yo as u32;
            let (xr, yr) = (Some(xr), Some(yr));
            let groups = (w.rows.div_ceil(tile) as u32, n.div_ceil(tile) as u32, 1);
            match &w.quant {
                Some((_, scales, _)) => self.dispatch_ranges(
                    &pipe,
                    &pc,
                    &[(x, xr), (&w.buf, None), (b, None), (y, yr), (scales, None)],
                    groups,
                ),
                None => self.dispatch_ranges(
                    &pipe,
                    &pc,
                    &[(x, xr), (&w.buf, None), (b, None), (y, yr)],
                    groups,
                ),
            }
            r0 += n;
        }
    }

    /// `Mxfp8::round_rows` of `rows` rows of `x` from `x_row` into a fresh
    /// `[rows, k]` buffer (`mx_act.wgsl`); the slice must fit one binding.
    fn mx_round(&self, x: &GBuf, x_row: usize, rows: usize, k: usize) -> GBuf {
        assert!(k.is_multiple_of(32), "MXFP8 needs k % 32 == 0");
        let out = self.alloc(rows * k);
        let blocks = rows * k / 32;
        let (xr, xo) = self.sub_range(x_row * k, rows * k);
        let pipe = self.pipeline("mx_act", || MX_ACT.to_string());
        let p = [blocks as u32, xo as u32, 0, 0];
        self.dispatch_ranges(
            &pipe,
            &p,
            &[(x, Some(xr)), (&out, None)],
            Self::grid(blocks.div_ceil(256)),
        );
        out
    }

    /// max |bf16(x)| over elements `[start, start + len)` into a one-element
    /// buffer (`nv_amax.wgsl`, two levels of workgroup trees).
    fn nv_amax(&self, x: &GBuf, start: usize, len: usize) -> GBuf {
        const CHUNK: usize = 4096;
        let pipe = self.pipeline("nv_amax", || NV_AMAX.to_string());
        // First level over binding-sized slices of x, partials side by side.
        let slice = (self.row_chunk(&[1]) / CHUNK).max(1) * CHUNK;
        let groups: usize = (0..len)
            .step_by(slice)
            .map(|o| (len - o).min(slice).div_ceil(CHUNK))
            .sum();
        let mut part = self.alloc(groups.max(1));
        let mut g0 = 0;
        for o in (0..len).step_by(slice) {
            let n = (len - o).min(slice);
            let (xr, xo) = self.sub_range(start + o, n);
            let g = n.div_ceil(CHUNK);
            let p = [n, xo, CHUNK, 1, g0, 0, 0, 0].map(|v| v as u32);
            self.dispatch_ranges(&pipe, &p, &[(x, Some(xr)), (&part, None)], Self::grid(g));
            g0 += g;
        }
        let mut n = groups;
        while n > 1 {
            let g = n.div_ceil(CHUNK);
            let next = self.alloc(g);
            let p = [n, 0, CHUNK, 0, 0, 0, 0, 0].map(|v| v as u32);
            self.dispatch(&pipe, &p, &[&part, &next], Self::grid(g));
            part = next;
            n = g;
        }
        part
    }

    /// `Nvfp4::round_rows` of `rows` rows of `x` from `x_row` with the call's
    /// maximum `amax` (from [`Self::nv_amax`]) into a fresh `[rows, k]` buffer.
    fn nv_round(&self, x: &GBuf, x_row: usize, rows: usize, k: usize, amax: &GBuf) -> GBuf {
        assert!(k.is_multiple_of(16), "NVFP4 needs k % 16 == 0");
        let out = self.alloc(rows * k);
        let blocks = rows * k / 16;
        let (xr, xo) = self.sub_range(x_row * k, rows * k);
        let pipe = self.pipeline("nv_act", || NV_ACT.to_string());
        let p = [blocks as u32, xo as u32, 0, 0];
        self.dispatch_ranges(
            &pipe,
            &p,
            &[(x, Some(xr)), (amax, None), (&out, None)],
            Self::grid(blocks.div_ceil(256)),
        );
        out
    }

    fn elementwise(
        &self,
        p: &Ep,
        x: &GBuf,
        y: Option<&GBuf>,
        aux: Option<&GBuf>,
        z: Option<&GBuf>,
        ui: Option<&GBuf>,
    ) {
        let pipe = self.pipeline("elementwise", || ELEM.to_string());
        let d = &self.dummy;
        self.dispatch(
            &pipe,
            &p.words(),
            &[
                x,
                y.unwrap_or(d),
                aux.unwrap_or(d),
                z.unwrap_or(d),
                ui.unwrap_or(d),
            ],
            Self::grid(p.n.div_ceil(256)),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn norm(
        &self,
        h: &GBuf,
        a: Option<&GBuf>,
        rows: usize,
        dim: usize,
        mode: u32,
        gate: Gate<'_, GBuf>,
        w: Option<&GBuf>,
        modulation: Modulation<'_, GBuf>,
        eps: f32,
        out: &GBuf,
    ) {
        let pipe = self.pipeline("norm", || NORM.to_string());
        let d = &self.dummy;
        let (gkind, gslots, gslot, gbuf, gidx) = match gate {
            Gate::None => (0u32, 0usize, 0usize, d, d),
            Gate::Vector(v) => (1, 0, 0, v, d),
            Gate::Table(m) => (2, m.table.slots, m.slot, m.table.table, m.table.idx),
        };
        let (has_mod, mslots, mscale, mshift, mt, midx) = match modulation {
            Modulation::None => (0u32, 0usize, 0usize, 0usize, d, d),
            Modulation::Table { m, scale, shift } => (1, m.slots, scale, shift, m.table, m.idx),
        };
        let p = [
            rows as u32,
            dim as u32,
            mode,
            gkind,
            gslots as u32,
            gslot as u32,
            w.is_some() as u32,
            has_mod,
            mslots as u32,
            mscale as u32,
            mshift as u32,
            (mode == 2 && a.is_some()) as u32,
            eps.to_bits(),
            0,
            0,
            0,
        ];
        self.dispatch(
            &pipe,
            &p,
            &[h, a.unwrap_or(d), gbuf, gidx, w.unwrap_or(d), mt, midx, out],
            Self::grid(rows),
        );
    }

    /// Flash kernel dispatch: mode 0 dense, mode 1 VSA fine stage.
    #[allow(clippy::too_many_arguments)]
    fn flash(
        &self,
        mode: u32,
        q: &GBuf,
        k: &GBuf,
        v: &GBuf,
        out: &GBuf,
        heads: usize,
        kv_heads: usize,
        sq: usize,
        sk: usize,
        d: usize,
        scale: f32,
        causal: bool,
        vsa: Option<(&GVsa, Option<&GBuf>)>,
        pv8: Option<(&GBuf, &GBuf)>,
    ) {
        // flash3h (f16 math) where f16 is on, except pv8: its E4M3 V codes
        // times P overflow an f16 partial sum.
        let pipe = if self.flash3 && self.f16 && pv8.is_none() {
            self.pipeline("flash3h", || FLASH3H.to_string())
        } else if self.flash3 {
            self.pipeline("flash3", || FLASH3.to_string())
        } else {
            self.pipeline("flash", || FLASH.to_string())
        };
        let (nb, tile, sub, has_gate) = vsa.map_or((0, 64, 1, 0), |(s, g)| {
            (s.nb, s.tile, s.sub, g.is_some() as u32)
        });
        let p = [
            heads as u32,
            (heads / kv_heads) as u32,
            sq as u32,
            sk as u32,
            d as u32,
            (scale * std::f32::consts::LOG2_E).to_bits(),
            mode,
            causal as u32,
            nb as u32,
            tile as u32,
            sub as u32,
            has_gate,
            pv8.is_some() as u32,
            0,
            0,
            0,
        ];
        let blocks = if mode == 0 { sq.div_ceil(64) } else { nb * sub };
        let dm = &self.dummy;
        let (slot, sel, coarse, gate) = match vsa {
            Some((s, g)) => (&s.slot, &s.sel, &s.coarse, g.unwrap_or(dm)),
            None => (dm, dm, dm, dm),
        };
        let (vmean, vscale) = pv8.unwrap_or((dm, dm));
        assert!(blocks as u32 <= MAX_DIM && heads as u32 <= MAX_DIM);
        self.dispatch(
            &pipe,
            &p,
            &[q, k, v, out, slot, sel, coarse, gate, vmean, vscale],
            (blocks as u32, heads as u32, 1),
        );
    }

    fn conv_pipe(&self, name: &str, src: &str, w: &GWeight) -> Arc<wgpu::ComputePipeline> {
        assert!(w.quant.is_none(), "conv weights are not quantized");
        self.pipeline(
            &format!("{name}_{}", if w.f16 { "f16" } else { "f32" }),
            || Self::weight_src(src, w.f16),
        )
    }

    fn epilogue<'a>(&'a self, ep: &ConvEpilogue<'a, GBuf>) -> (u32, &'a GBuf) {
        match ep {
            ConvEpilogue::None => (0, &self.dummy),
            ConvEpilogue::Relu => (1, &self.dummy),
            ConvEpilogue::AddRelu(r) => (2, r),
            ConvEpilogue::Add(r) => (3, r),
            ConvEpilogue::Prelu(slope) => (4, slope),
        }
    }

    /// (y, z) workgroup counts for `m` 64-row tiles.
    fn mgrid(m: usize) -> (u32, u32) {
        let mt = m.div_ceil(64) as u32;
        if mt <= MAX_DIM {
            (mt.max(1), 1)
        } else {
            let z = mt.div_ceil(MAX_DIM);
            (mt.div_ceil(z), z)
        }
    }
}

impl Backend for Wgpu {
    type Buf = GBuf;
    type Vsa = GVsa;
    type Weight = GWeight;

    fn name(&self) -> String {
        format!(
            "wgpu {:?}: {} ({}{}){}",
            self.info.backend,
            self.info.name,
            self.info.driver,
            if self.info.driver_info.is_empty() {
                String::new()
            } else {
                format!(" {}", self.info.driver_info)
            },
            if self.f16 { " +f16" } else { "" }
        )
    }

    fn max_buffer_elems(&self) -> usize {
        (self.max_binding_bytes() / 4) as usize
    }

    fn alloc(&self, len: usize) -> GBuf {
        GBuf {
            buf: Arc::new(self.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: (len.max(1) * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })),
            len,
        }
    }

    fn upload(&self, data: &[f32]) -> GBuf {
        let padded;
        let bytes: &[u8] = if data.is_empty() {
            padded = [0f32];
            bytemuck::cast_slice(&padded)
        } else {
            bytemuck::cast_slice(data)
        };
        GBuf {
            buf: Arc::new(self.init_buffer(
                bytes,
                wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_SRC
                    | wgpu::BufferUsages::COPY_DST,
            )),
            len: data.len(),
        }
    }

    fn upload_u32(&self, data: &[u32]) -> GBuf {
        self.upload_u32_buf(data)
    }

    fn write(&self, buf: &GBuf, data: &[f32]) {
        self.flush();
        self.queue
            .write_buffer(&buf.buf, 0, bytemuck::cast_slice(data));
        self.staged_bytes(data.len() * 4);
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn download(&self, buf: &GBuf) -> Vec<f32> {
        let size = (buf.len.max(1) * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        {
            let mut pend = self.pending.lock().unwrap();
            let enc = pend
                .enc
                .get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
            enc.copy_buffer_to_buffer(&buf.buf, 0, &staging, 0, size);
        }
        self.flush();
        let slice = staging.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .expect("device lost while reading back");
        rx.recv().unwrap().expect("map failed");
        let data = slice.get_mapped_range().expect("mapped range");
        let out = bytemuck::cast_slice::<u8, f32>(&data)[..buf.len].to_vec();
        drop(data);
        staging.unmap();
        out
    }

    #[cfg(target_arch = "wasm32")]
    fn download(&self, _buf: &GBuf) -> Vec<f32> {
        panic!("blocking readback is unavailable in the browser; use Wgpu::download_async")
    }

    fn buf_len(&self, buf: &GBuf) -> usize {
        buf.len
    }

    fn box_rows(&self, grid: &GBuf, packed: &GBuf, b: &BoxRows, scatter: bool) {
        let n = b.len();
        if n == 0 {
            return;
        }
        assert!(
            u32::try_from(n).is_ok(),
            "box_rows: {n} elements exceed u32 indexing"
        );
        let pipe = self.pipeline("box_rows", || BOX_ROWS.to_string());
        let p = [
            n,
            b.width,
            b.rows(),
            b.heads,
            b.count.1,
            b.grid.1,
            b.grid.2,
            b.t.0,
            b.y.0,
            b.x.0,
            b.y.1 - b.y.0,
            b.x.1 - b.x.0,
            b.step.0,
            b.step.1,
            b.grid_off,
            b.grid_head,
            b.grid_row,
            b.packed_off,
            b.packed_win,
            b.packed_head,
            b.packed_row,
            scatter as usize,
            0,
            0,
        ]
        .map(|v| v as u32);
        self.dispatch(&pipe, &p, &[grid, packed], Self::grid(n.div_ceil(256)));
    }

    fn copy(&self, src: &GBuf, so: usize, dst: &GBuf, d: usize, len: usize) {
        if len == 0 {
            return;
        }
        if Arc::ptr_eq(&src.buf, &dst.buf) {
            // A buffer cannot be both source and destination of one copy.
            let tmp = self.alloc(len);
            self.copy(src, so, &tmp, 0, len);
            self.copy(&tmp, 0, dst, d, len);
            return;
        }
        let mut pend = self.pending.lock().unwrap();
        let enc = pend
            .enc
            .get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
        enc.copy_buffer_to_buffer(
            &src.buf,
            (so * 4) as u64,
            &dst.buf,
            (d * 4) as u64,
            (len * 4) as u64,
        );
    }

    fn fill(&self, buf: &GBuf, off: usize, len: usize, value: f32) {
        if len == 0 {
            return;
        }
        let p = Ep {
            n: len,
            op: 14,
            off,
            f: [value, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        self.elementwise(&p, buf, None, None, None, None);
    }

    fn weight(&self, data: &[f32], rows: usize, cols: usize) -> GWeight {
        assert_eq!(data.len(), rows * cols);
        let fits = data.iter().all(|v| v.abs() <= 65000.0);
        if self.allow_f16_weights && cols % 2 == 0 && fits {
            let packed: Vec<u32> = data
                .chunks_exact(2)
                .map(|p| {
                    let lo = half::f16::from_f32(p[0]).to_bits() as u32;
                    let hi = half::f16::from_f32(p[1]).to_bits() as u32;
                    lo | (hi << 16)
                })
                .collect();
            let buf = GBuf {
                buf: Arc::new(
                    self.init_buffer(bytemuck::cast_slice(&packed), wgpu::BufferUsages::STORAGE),
                ),
                len: packed.len(),
            };
            GWeight {
                buf,
                rows,
                cols,
                f16: true,
                quant: None,
                mx: false,
                a4: false,
            }
        } else {
            GWeight {
                buf: self.upload(data),
                rows,
                cols,
                f16: false,
                quant: None,
                mx: false,
                a4: false,
            }
        }
    }

    fn weight_nvfp4(&self, q: &Nvfp4) -> GWeight {
        GWeight {
            a4: q.act4,
            ..self.weight(&q.dequantize(), q.rows, q.cols)
        }
    }

    fn weight_mxfp8(&self, q: &Mxfp8) -> GWeight {
        GWeight {
            mx: true,
            a4: false,
            ..self.weight(&q.dequantize(), q.rows, q.cols)
        }
    }

    fn weight_q8(&self, q: &Q8) -> GWeight {
        let packed: Vec<u32> =
            q.q.chunks(4)
                .map(|c| {
                    c.iter()
                        .enumerate()
                        .fold(0u32, |a, (j, &v)| a | ((v as u8 as u32) << (8 * j)))
                })
                .collect();
        GWeight {
            buf: self.upload_u32_buf(&packed),
            rows: q.rows,
            cols: q.cols,
            f16: false,
            quant: Some((Quant::Q8, self.upload(&q.scales), q.group)),
            mx: false,
            a4: false,
        }
    }

    fn weight_q4(&self, q: &Q4) -> GWeight {
        self.weight_q4_packed(&q.to_packed())
    }

    fn weight_q4_packed(&self, q: &Q4Packed) -> GWeight {
        // `packed` is a multiple of 4 bytes; a Vec<u8> need not be 4-aligned.
        let copy: Vec<u32>;
        let words: &[u32] = match bytemuck::try_cast_slice(&q.packed) {
            Ok(w) => w,
            Err(_) => {
                copy = bytemuck::pod_collect_to_vec(&q.packed);
                &copy
            }
        };
        GWeight {
            buf: self.upload_u32_buf(words),
            rows: q.rows,
            cols: q.cols,
            f16: false,
            quant: Some((Quant::Q4, self.upload(&q.sz), q.group)),
            mx: false,
            a4: false,
        }
    }

    fn weight_bytes(&self, w: &GWeight) -> usize {
        w.buf.len * 4 + w.quant.as_ref().map_or(0, |q| q.1.len * 4)
    }

    fn sync(&self) {
        self.flush();
        #[cfg(not(target_arch = "wasm32"))]
        {
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            });
            self.inflight.lock().unwrap().clear();
        }
    }

    fn profile_report(&self) -> Option<String> {
        self.profile_table()
    }
    fn profile_reset(&self) {
        Wgpu::profile_reset(self)
    }

    fn linear(
        &self,
        x: &GBuf,
        x_row: usize,
        rows: usize,
        w: &GWeight,
        bias: Option<&GBuf>,
        act: Act,
        y: &GBuf,
        y_row: usize,
    ) {
        if rows == 0 {
            return;
        }
        if w.a4 {
            // NVFP4 W4A4: the whole call's maximum first, then each row chunk
            // rounded with it (`Nvfp4::round_rows` is per call, not per chunk).
            let k = w.cols;
            let amax = self.nv_amax(x, x_row * k, rows * k);
            let chunk = self.row_chunk(&[w.cols, w.rows]);
            let mut r0 = 0;
            while r0 < rows {
                let n = chunk.min(rows - r0);
                let xq = self.nv_round(x, x_row + r0, n, k, &amax);
                self.linear_plain(&xq, 0, n, w, bias, act, y, y_row + r0);
                r0 += n;
            }
            return;
        }
        if w.mx {
            // MXFP8 W8A8: round the activations per 32-column block into a
            // scratch copy first, in row chunks that fit one binding.
            let chunk = self.row_chunk(&[w.cols, w.rows]);
            let mut r0 = 0;
            while r0 < rows {
                let n = chunk.min(rows - r0);
                let xq = self.mx_round(x, x_row + r0, n, w.cols);
                self.linear_plain(&xq, 0, n, w, bias, act, y, y_row + r0);
                r0 += n;
            }
            return;
        }
        self.linear_plain(x, x_row, rows, w, bias, act, y, y_row);
    }

    fn swiglu(&self, x: &GBuf, rows: usize, dim: usize, out: &GBuf) {
        let p = Ep {
            n: rows * dim,
            op: 13,
            dim,
            ..Default::default()
        };
        self.elementwise(&p, out, Some(x), None, None, None);
    }

    fn rms_norm(
        &self,
        x: &GBuf,
        rows: usize,
        dim: usize,
        w: Option<&GBuf>,
        modulation: Modulation<'_, GBuf>,
        eps: f32,
        out: &GBuf,
    ) {
        self.norm(x, None, rows, dim, 0, Gate::None, w, modulation, eps, out);
    }

    #[allow(clippy::too_many_arguments)]
    fn residual_rms_norm(
        &self,
        h: &GBuf,
        a: &GBuf,
        rows: usize,
        dim: usize,
        gate: Gate<'_, GBuf>,
        w: Option<&GBuf>,
        modulation: Modulation<'_, GBuf>,
        eps: f32,
        normed: &GBuf,
    ) {
        self.norm(h, Some(a), rows, dim, 1, gate, w, modulation, eps, normed);
    }

    fn gate_residual(&self, h: &GBuf, a: &GBuf, rows: usize, dim: usize, gate: Gate<'_, GBuf>) {
        let n = rows * dim;
        match gate {
            Gate::None => self.elementwise(
                &Ep {
                    n,
                    op: 0,
                    dim,
                    ..Default::default()
                },
                h,
                Some(a),
                None,
                None,
                None,
            ),
            Gate::Vector(g) => self.elementwise(
                &Ep {
                    n,
                    op: 1,
                    dim,
                    ..Default::default()
                },
                h,
                Some(a),
                Some(g),
                None,
                None,
            ),
            Gate::Table(m) => self.elementwise(
                &Ep {
                    n,
                    op: 2,
                    dim,
                    slots: m.table.slots,
                    slot: m.slot,
                    ..Default::default()
                },
                h,
                Some(a),
                Some(m.table.table),
                None,
                Some(m.table.idx),
            ),
        }
    }

    fn layer_norm(
        &self,
        x: &GBuf,
        rows: usize,
        dim: usize,
        w: Option<&GBuf>,
        b: Option<&GBuf>,
        eps: f32,
        out: &GBuf,
    ) {
        self.norm(
            x,
            b,
            rows,
            dim,
            2,
            Gate::None,
            w,
            Modulation::None,
            eps,
            out,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn head_split(
        &self,
        src: &GBuf,
        rows: usize,
        stride: usize,
        col_off: usize,
        heads: usize,
        hd: usize,
        norm: Option<HeadNorm<'_, GBuf>>,
        rope: Option<Rope<'_, GBuf>>,
        out: &GBuf,
    ) {
        assert!(heads <= 256, "head_split handles at most 256 heads");
        let pipe = self.pipeline("head_split", || HEADS.to_string());
        let d = &self.dummy;
        let (has_norm, w, eps) = match norm {
            Some(n) => (1u32, n.w.unwrap_or(d), n.eps),
            None => (0, d, 0.0),
        };
        let (has_rope, cs, sn, rot) = match rope {
            Some(r) => (1u32, r.cos, r.sin, r.rot),
            None => (0, d, d, 0),
        };
        let has_w = norm.and_then(|n| n.w).is_some() as u32;
        let chunk = if (src.len as u64) * 4 <= self.max_binding_bytes() {
            rows
        } else {
            self.row_chunk(&[stride])
        };
        let mut r0 = 0;
        while r0 < rows {
            let n = chunk.min(rows - r0);
            let (sr, rem) = if chunk == rows {
                (None, 0)
            } else {
                let (r, rem) = self.sub_range(r0 * stride, n * stride);
                (Some(r), rem)
            };
            let p = [
                n as u32,
                stride as u32,
                (col_off + rem) as u32,
                heads as u32,
                hd as u32,
                has_norm,
                has_w,
                has_rope,
                rot as u32,
                eps.to_bits(),
                r0 as u32,
                rows as u32,
            ];
            self.dispatch_ranges(
                &pipe,
                &p,
                &[(src, sr), (w, None), (cs, None), (sn, None), (out, None)],
                Self::grid(n),
            );
            r0 += n;
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn attention(
        &self,
        q: &GBuf,
        k: &GBuf,
        v: &GBuf,
        heads: usize,
        kv_heads: usize,
        sq: usize,
        sk: usize,
        d: usize,
        causal: bool,
        out: &GBuf,
    ) {
        assert!(d <= 128, "attention kernel supports head_dim <= 128");
        assert!(heads % kv_heads == 0);
        if self.flash_ok && d % 16 == 0 {
            self.flash(
                0,
                q,
                k,
                v,
                out,
                heads,
                kv_heads,
                sq,
                sk,
                d,
                1.0 / (d as f32).sqrt(),
                causal,
                None,
                None,
            );
            return;
        }
        let bk = self.attn_bk;
        let pipe = self.pipeline(&format!("attention_bk{bk}"), || {
            ATTN.replace("{BK_D}", &(bk * 128).to_string())
                .replace("{BK}", &bk.to_string())
        });
        let p = [
            heads as u32,
            (heads / kv_heads) as u32,
            sq as u32,
            sk as u32,
            d as u32,
            (1.0 / (d as f32).sqrt()).to_bits(),
            causal as u32,
            0,
        ];
        assert!(heads as u32 <= MAX_DIM);
        self.dispatch(
            &pipe,
            &p,
            &[q, k, v, out],
            (sq.div_ceil(64) as u32, heads as u32, 1),
        );
    }

    fn attention_qk8(
        &self,
        q: &GBuf,
        k: &GBuf,
        v: &GBuf,
        heads: usize,
        kv_heads: usize,
        sq: usize,
        sk: usize,
        d: usize,
        causal: bool,
        pv8: bool,
        out: &GBuf,
    ) {
        // Emulation: the rounded f32 copies through the exact kernel (or, with
        // FP8 P·V, through the oracle's two-pass kernel).
        let (kmean, q8, k8) = (
            self.alloc(kv_heads * d),
            self.alloc(heads * sq * d),
            self.alloc(kv_heads * sk * d),
        );
        let pipe = self.pipeline("attn_qk8", || ATTN_QK8.to_string());
        let mut p = [
            heads as u32,
            kv_heads as u32,
            sq as u32,
            sk as u32,
            d as u32,
            0,
            0,
            0,
        ];
        self.dispatch(
            &pipe,
            &p,
            &[q, k, &kmean, &q8, &k8],
            (kv_heads as u32, 1, 1),
        );
        p[5] = 1;
        self.dispatch(
            &pipe,
            &p,
            &[q, k, &kmean, &q8, &k8],
            (sq.div_ceil(64) as u32, heads as u32, 1),
        );
        p[5] = 2;
        self.dispatch(
            &pipe,
            &p,
            &[q, k, &kmean, &q8, &k8],
            (sk.div_ceil(64) as u32, kv_heads as u32, 1),
        );
        if !pv8 {
            self.attention(&q8, &k8, v, heads, kv_heads, sq, sk, d, causal, out);
            return;
        }
        // V: mean (mode 0 on V), scale (mode 3), codes (mode 4).
        let (vmean, v8, vscale) = (
            self.alloc(kv_heads * d),
            self.alloc(kv_heads * sk * d),
            self.alloc(kv_heads.max(1)),
        );
        p[5] = 0;
        self.dispatch(
            &pipe,
            &p,
            &[q, v, &vmean, &v8, &vscale],
            (kv_heads as u32, 1, 1),
        );
        p[5] = 3;
        self.dispatch(
            &pipe,
            &p,
            &[q, v, &vmean, &v8, &vscale],
            (kv_heads as u32, 1, 1),
        );
        p[5] = 4;
        self.dispatch(
            &pipe,
            &p,
            &[q, v, &vmean, &v8, &vscale],
            (sk.div_ceil(64) as u32, kv_heads as u32, 1),
        );
        let pipe8 = self.pipeline("attn_pv8", || ATTN_PV8.to_string());
        let p8 = [
            heads as u32,
            kv_heads as u32,
            sq as u32,
            sk as u32,
            d as u32,
            causal as u32,
            (1.0 / (d as f32).sqrt()).to_bits(),
            0,
        ];
        self.dispatch(
            &pipe8,
            &p8,
            &[&q8, &k8, &v8, &vmean, &vscale, out],
            (sq.div_ceil(64) as u32, heads as u32, 1),
        );
    }

    fn vsa_supported(&self, head_dim: usize) -> bool {
        self.flash_ok && self.coarse_ok && head_dim % 16 == 0 && head_dim <= 128
    }

    fn vsa_prepare(&self, spec: &VsaSpec) -> GVsa {
        let plan = &spec.plan;
        let (nb, h, d) = (plan.num_tiles(), spec.heads, spec.head_dim);
        assert!(
            nb <= MAX_TILES,
            "VSA coarse stage holds at most {MAX_TILES} tiles ({nb} tiles)"
        );
        assert!(plan.tile_elems % 64 == 0);
        let mut regions = Vec::new();
        for r in &plan.regions {
            regions.extend([r.start as u32, r.end as u32, r.topk as u32, 0]);
        }
        GVsa {
            slot: self.upload_u32_buf(&plan.slot_src),
            bsz: self.upload_u32_buf(&plan.block_sizes),
            regions: self.upload_u32_buf(&regions),
            pooled: self.alloc(3 * h * nb * d),
            coarse: self.alloc(h * nb * d),
            sel: self.upload_u32_buf(&vec![0u32; h * nb * (nb + 1)]),
            nb,
            nprefix: plan.num_prefix_tiles,
            nregions: plan.regions.len(),
            heads: h,
            d,
            seq: plan.seq,
            scale: spec.scale,
            tile: plan.tile_elems,
            sub: plan.tile_elems / 64,
            qk8: spec.qk8.then(|| {
                (
                    self.alloc(h * d),
                    self.alloc(h * plan.seq * d),
                    self.alloc(h * plan.seq * d),
                )
            }),
            pv8: spec.pv8.then(|| {
                (
                    self.alloc(h * d),
                    self.alloc(h * plan.seq * d),
                    self.alloc(h),
                )
            }),
            dense: plan.is_dense(),
            rowslot: self.upload_u32_buf(&plan.slot_of_row()),
        }
    }

    fn vsa_attention(
        &self,
        q: &GBuf,
        k: &GBuf,
        v: &GBuf,
        gate: Option<&GBuf>,
        s: &GVsa,
        out: &GBuf,
    ) {
        assert!(
            self.vsa_supported(s.d),
            "VSA needs the flash kernel (head_dim multiple of 16, <= 128)"
        );
        let pool = self.pipeline("vsa_pool", || VSA_POOL.to_string());
        self.dispatch(
            &pool,
            &[
                s.heads as u32,
                s.seq as u32,
                s.d as u32,
                s.nb as u32,
                s.tile as u32,
                0,
                0,
                0,
            ],
            &[q, k, v, &s.slot, &s.bsz, &s.pooled],
            (s.nb as u32, s.heads as u32, 1),
        );
        let coarse = self.pipeline("vsa_coarse", || VSA_COARSE.to_string());
        let p = [
            s.heads as u32,
            s.d as u32,
            s.nb as u32,
            s.nprefix as u32,
            s.nregions as u32,
            s.scale.to_bits(),
            0,
            0,
        ];
        self.dispatch(
            &coarse,
            &p,
            &[&s.pooled, &s.coarse, &s.sel, &s.regions],
            (s.nb as u32, s.heads as u32, 1),
        );
        if s.dense {
            // A dense plan: the dense attention (its INT8 / FP8 recipe when the
            // spec asks) plus the gated pooled branch; see `vsa_reference`.
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
                let n = s.seq * s.heads * s.d;
                self.elementwise(
                    &Ep {
                        n,
                        op: 20,
                        dim: s.d,
                        slots: s.heads,
                        slot: s.nb,
                        off: s.tile,
                        len: s.seq,
                        ..Default::default()
                    },
                    out,
                    Some(g),
                    Some(&s.coarse),
                    None,
                    Some(&s.rowslot),
                );
            }
            return;
        }
        // INT8 Q·Kᵀ emulation: the fine stage reads the rounded copies; the
        // tile means and selection above used the exact values.
        if let Some((kmean, q8, k8)) = &s.qk8 {
            let pipe = self.pipeline("vsa_qk8", || VSA_QK8.to_string());
            let base = [
                s.heads as u32,
                s.seq as u32,
                s.d as u32,
                s.nb as u32,
                s.tile as u32,
                0,
                0,
                0,
            ];
            self.dispatch(
                &pipe,
                &base,
                &[q, k, &s.slot, &s.bsz, kmean, q8, k8],
                (s.heads as u32, 1, 1),
            );
            let mut p1 = base;
            p1[5] = 1;
            self.dispatch(
                &pipe,
                &p1,
                &[q, k, &s.slot, &s.bsz, kmean, q8, k8],
                (s.nb as u32, s.heads as u32, 1),
            );
            if let Some((vmean, v8, vscale)) = &s.pv8 {
                // V: mean (mode 0 on V), per-head scale (mode 3), codes (mode 4).
                let pipe = self.pipeline("attn_qk8", || ATTN_QK8.to_string());
                let mut pv = [
                    s.heads as u32,
                    s.heads as u32,
                    s.seq as u32,
                    s.seq as u32,
                    s.d as u32,
                    0,
                    0,
                    0,
                ];
                self.dispatch(
                    &pipe,
                    &pv,
                    &[q, v, vmean, v8, vscale],
                    (s.heads as u32, 1, 1),
                );
                pv[5] = 3;
                self.dispatch(
                    &pipe,
                    &pv,
                    &[q, v, vmean, v8, vscale],
                    (s.heads as u32, 1, 1),
                );
                pv[5] = 4;
                self.dispatch(
                    &pipe,
                    &pv,
                    &[q, v, vmean, v8, vscale],
                    (s.seq.div_ceil(64) as u32, s.heads as u32, 1),
                );
                self.flash(
                    1,
                    q8,
                    k8,
                    v8,
                    out,
                    s.heads,
                    s.heads,
                    s.seq,
                    s.seq,
                    s.d,
                    s.scale,
                    false,
                    Some((s, gate)),
                    Some((vmean, vscale)),
                );
                return;
            }
            self.flash(
                1,
                q8,
                k8,
                v,
                out,
                s.heads,
                s.heads,
                s.seq,
                s.seq,
                s.d,
                s.scale,
                false,
                Some((s, gate)),
                None,
            );
            return;
        }
        self.flash(
            1,
            q,
            k,
            v,
            out,
            s.heads,
            s.heads,
            s.seq,
            s.seq,
            s.d,
            s.scale,
            false,
            Some((s, gate)),
            None,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn patchify(
        &self,
        latent: &GBuf,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        patch: (usize, usize, usize),
        out: &GBuf,
        out_row: usize,
    ) {
        let n = c * t * h * w;
        let p = Ep {
            n,
            op: 11,
            off: out_row,
            c,
            t,
            h,
            w,
            pt: patch.0,
            ph: patch.1,
            pw: patch.2,
            ..Default::default()
        };
        self.elementwise(&p, out, Some(latent), None, None, None);
    }

    #[allow(clippy::too_many_arguments)]
    fn unpatchify(
        &self,
        x: &GBuf,
        x_row: usize,
        x_stride: usize,
        col: usize,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        patch: (usize, usize, usize),
        out: &GBuf,
    ) {
        let n = c * t * h * w;
        let p = Ep {
            n,
            op: 12,
            off: x_row,
            stride: x_stride,
            col,
            c,
            t,
            h,
            w,
            pt: patch.0,
            ph: patch.1,
            pw: patch.2,
            ..Default::default()
        };
        self.elementwise(&p, out, Some(x), None, None, None);
    }

    fn flow_step(
        &self,
        x: &GBuf,
        off: usize,
        v: &GBuf,
        v_off: usize,
        noise: Option<&GBuf>,
        n: usize,
        step: FlowStep,
    ) {
        let (op, f) = match step {
            FlowStep::Euler { sigma, next } => (3, [sigma, next, 0.0, 0.0]),
            FlowStep::Renoise { sigma, next } => (4, [sigma, next, 0.0, 0.0]),
            FlowStep::Terminal { sigma } => (5, [sigma, 0.0, 0.0, 0.0]),
        };
        let z = if op == 4 {
            Some(noise.expect("renoise needs noise"))
        } else {
            None
        };
        let p = Ep {
            n,
            op,
            off,
            xoff: v_off,
            f,
            ..Default::default()
        };
        self.elementwise(&p, x, Some(v), z, None, None);
    }

    fn axpby(&self, y: &GBuf, off: usize, a: f32, x: &GBuf, x_off: usize, b: f32, n: usize) {
        if Arc::ptr_eq(&y.buf, &x.buf) {
            let p = Ep {
                n,
                op: 9,
                off,
                xoff: x_off,
                f: [a, b, 0.0, 0.0],
                ..Default::default()
            };
            self.elementwise(&p, y, None, None, None, None);
        } else {
            let p = Ep {
                n,
                op: 8,
                off,
                xoff: x_off,
                f: [a, b, 0.0, 0.0],
                ..Default::default()
            };
            self.elementwise(&p, y, Some(x), None, None, None);
        }
    }

    fn channel_affine(
        &self,
        x: &GBuf,
        c: usize,
        inner: usize,
        scale: &GBuf,
        shift: Option<&GBuf>,
        out: &GBuf,
    ) {
        let p = Ep {
            n: c * inner,
            op: 10,
            dim: inner,
            has_z: shift.is_some(),
            ..Default::default()
        };
        self.elementwise(&p, out, Some(x), Some(scale), shift, None);
    }

    fn conv2d(
        &self,
        a: &GBuf,
        b: Option<&GBuf>,
        s: &ConvShape,
        w: &GWeight,
        bias: Option<&GBuf>,
        ep: ConvEpilogue<'_, GBuf>,
        out: &GBuf,
    ) {
        assert_eq!((w.rows, w.cols), (s.cout, s.k_len()), "conv weight shape");
        if self.prof.is_some() {
            *self.tag.lock().unwrap() = Some(format!("{}->{} k{}", s.cin(), s.cout, s.k));
        }
        let pipe = self.conv_pipe("conv2d", CONV, w);
        let (epc, res) = self.epilogue(&ep);
        let p = [
            s.frames as u32,
            s.cin_a as u32,
            s.cin_b as u32,
            s.cout as u32,
            s.h as u32,
            s.w as u32,
            if s.upsample { 2 } else { 1 },
            s.k as u32,
            b.is_some() as u32,
            bias.is_some() as u32,
            epc,
            0,
        ];
        let (oh, ow) = s.out_hw();
        let (gy, gz) = Self::mgrid(s.frames * oh * ow);
        let groups = (s.cout.div_ceil(64) as u32, gy, gz);
        self.dispatch(
            &pipe,
            &p,
            &[
                a,
                b.unwrap_or(&self.dummy),
                &w.buf,
                bias.unwrap_or(&self.dummy),
                res,
                out,
            ],
            groups,
        );
    }

    fn conv3d(
        &self,
        x: &GBuf,
        s: &Conv3dShape,
        w: &GWeight,
        bias: Option<&GBuf>,
        ep: ConvEpilogue<'_, GBuf>,
        out: &GBuf,
    ) {
        assert!(
            !matches!(ep, ConvEpilogue::Prelu(_)),
            "PReLU is a conv2d epilogue"
        );
        assert_eq!((w.rows, w.cols), (s.cout, s.k_len()), "conv3d weight shape");
        if self.prof.is_some() {
            *self.tag.lock().unwrap() =
                Some(format!("{}->{} k{:?} g{}", s.cin, s.cout, s.k, s.groups));
        }
        let (epc, res) = self.epilogue(&ep);
        let (ot, oh, ow) = s.out_dims();
        let mut p = vec![
            s.cin as u32,
            s.cout as u32,
            s.t as u32,
            s.h as u32,
            s.w as u32,
            s.k.0 as u32,
            s.k.1 as u32,
            s.k.2 as u32,
            s.stride.0 as u32,
            s.stride.1 as u32,
            s.stride.2 as u32,
            s.pad_t.0 as u32,
            s.pad_h.0 as u32,
            s.pad_w.0 as u32,
            (s.spatial_pad == Pad::Reflect) as u32,
            bias.is_some() as u32,
            epc,
            ot as u32,
            oh as u32,
            ow as u32,
        ];
        let bufs = [x, &w.buf, bias.unwrap_or(&self.dummy), res, out];
        if s.groups == 1 {
            let pipe = self.conv_pipe("conv3d", CONV3D, w);
            let (gy, gz) = Self::mgrid(ot * oh * ow);
            self.dispatch(&pipe, &p, &bufs, (s.cout.div_ceil(64) as u32, gy, gz));
        } else {
            let pipe = self.conv_pipe("conv3d_direct", CONV3D_DIRECT, w);
            let n = s.out_len();
            p.extend([s.groups as u32, n as u32, 0, 0]);
            self.dispatch(&pipe, &p, &bufs, Self::grid(n.div_ceil(256)));
        }
    }

    fn conv1d(
        &self,
        x: &GBuf,
        s: &Conv1dShape,
        w: &GWeight,
        bias: Option<&GBuf>,
        ep: ConvEpilogue<'_, GBuf>,
        out: &GBuf,
    ) {
        assert!(
            !matches!(ep, ConvEpilogue::Prelu(_)),
            "PReLU is a conv2d epilogue"
        );
        assert_eq!(
            (w.rows, w.cols),
            (s.cout, (s.cin / s.groups) * s.k),
            "conv1d weight shape"
        );
        let pipe = self.conv_pipe("conv1d", CONV1D, w);
        let (epc, res) = self.epilogue(&ep);
        let n = s.out_len();
        let p = [
            s.cin as u32,
            s.cout as u32,
            s.groups as u32,
            s.len as u32,
            s.k as u32,
            s.stride as u32,
            s.dilation as u32,
            s.pad.0 as u32,
            s.out_len_1d() as u32,
            bias.is_some() as u32,
            epc,
            n as u32,
        ];
        self.dispatch(
            &pipe,
            &p,
            &[x, &w.buf, bias.unwrap_or(&self.dummy), res, out],
            Self::grid(n.div_ceil(256)),
        );
    }

    fn replicate_pad1d(
        &self,
        x: &GBuf,
        c: usize,
        len: usize,
        left: usize,
        right: usize,
        out: &GBuf,
    ) {
        let p = Ep {
            n: c * (len + left + right),
            op: 15,
            left,
            len,
            r: right,
            ..Default::default()
        };
        self.elementwise(&p, out, Some(x), None, None, None);
    }

    fn conv_transpose1d(
        &self,
        x: &GBuf,
        s: &ConvT1dShape,
        w: &GWeight,
        bias: Option<&GBuf>,
        out: &GBuf,
    ) {
        assert_eq!(
            (w.rows, w.cols),
            (s.cin, (s.cout / s.groups) * s.k),
            "conv_transpose1d weight shape"
        );
        let pipe = self.conv_pipe("convt1d", CONVT1D, w);
        let n = s.out_len();
        let p = [
            s.cin as u32,
            s.cout as u32,
            s.groups as u32,
            s.len as u32,
            s.k as u32,
            s.stride as u32,
            s.pad as u32,
            s.out_len_1d() as u32,
            bias.is_some() as u32,
            n as u32,
            0,
            0,
        ];
        self.dispatch(
            &pipe,
            &p,
            &[x, &w.buf, bias.unwrap_or(&self.dummy), out],
            Self::grid(n.div_ceil(256)),
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn group_norm(
        &self,
        x: &GBuf,
        c: usize,
        t: usize,
        hw: usize,
        groups: usize,
        per_frame: bool,
        w: &GBuf,
        b: &GBuf,
        eps: f32,
        act: Act,
        out: &GBuf,
    ) {
        let pipe = self.pipeline("group_norm", || GROUP_NORM.to_string());
        let act_code = match act {
            Act::None => 0,
            Act::GeluTanh => 1,
            Act::GeluErf => 2,
            Act::Silu => 3,
        };
        let p = [
            c as u32,
            t as u32,
            hw as u32,
            groups as u32,
            per_frame as u32,
            act_code,
            eps.to_bits(),
            (c / groups) as u32,
        ];
        let n = if per_frame { t * groups } else { groups };
        assert!(n as u32 <= MAX_DIM);
        self.dispatch(&pipe, &p, &[x, w, b, out], (n as u32, 1, 1));
    }

    fn snake(&self, x: &GBuf, c: usize, len: usize, snake: Snake<'_, GBuf>) {
        match snake {
            Snake::Alpha(a) => self.elementwise(
                &Ep {
                    n: c * len,
                    op: 17,
                    len,
                    ..Default::default()
                },
                x,
                None,
                Some(a),
                None,
                None,
            ),
            Snake::AlphaBeta { alpha, beta } => self.elementwise(
                &Ep {
                    n: c * len,
                    op: 18,
                    len,
                    ..Default::default()
                },
                x,
                None,
                Some(alpha),
                Some(beta),
                None,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn trilinear_resize(
        &self,
        x: &GBuf,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        t2: usize,
        h2: usize,
        w2: usize,
        out: &GBuf,
    ) {
        let p = Ep {
            n: c * t2 * h2 * w2,
            op: 19,
            c,
            t,
            h,
            w,
            t2,
            h2,
            w2,
            ..Default::default()
        };
        self.elementwise(&p, out, Some(x), None, None, None);
    }

    fn pixel_shuffle(
        &self,
        x: &GBuf,
        frames: usize,
        c: usize,
        h: usize,
        w: usize,
        r: usize,
        out: &GBuf,
    ) {
        let p = Ep {
            n: frames * c * h * w * r * r,
            op: 16,
            c,
            h,
            w,
            r,
            frames,
            ..Default::default()
        };
        self.elementwise(&p, out, Some(x), None, None, None);
    }

    fn tanh_clamp(&self, x: &GBuf, n: usize, s: f32) {
        self.elementwise(
            &Ep {
                n,
                op: 6,
                f: [s, 0.0, 0.0, 0.0],
                ..Default::default()
            },
            x,
            None,
            None,
            None,
            None,
        );
    }

    fn clamp(&self, x: &GBuf, n: usize, lo: f32, hi: f32) {
        self.elementwise(
            &Ep {
                n,
                op: 7,
                f: [lo, hi, 0.0, 0.0],
                ..Default::default()
            },
            x,
            None,
            None,
            None,
            None,
        );
    }
}

impl Wgpu {
    /// Non-blocking readback for the browser (and usable natively).
    pub async fn download_async(&self, buf: &GBuf) -> Vec<f32> {
        let size = (buf.len.max(1) * 4) as u64;
        let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        {
            let mut pend = self.pending.lock().unwrap();
            let enc = pend
                .enc
                .get_or_insert_with(|| self.device.create_command_encoder(&Default::default()));
            enc.copy_buffer_to_buffer(&buf.buf, 0, &staging, 0, size);
        }
        self.flush();
        let (tx, rx) = futures_oneshot();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |r| tx.send(r));
        #[cfg(not(target_arch = "wasm32"))]
        let _ = self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        });
        rx.await.expect("map failed");
        let data = staging.slice(..).get_mapped_range().expect("mapped range");
        let out = bytemuck::cast_slice::<u8, f32>(&data)[..buf.len].to_vec();
        drop(data);
        staging.unmap();
        out
    }
}

fn futures_oneshot<T>() -> (OneTx<T>, OneRx<T>) {
    let s = Arc::new(Mutex::new((None, None::<std::task::Waker>)));
    (OneTx(s.clone()), OneRx(s))
}
struct OneTx<T>(Arc<Mutex<(Option<T>, Option<std::task::Waker>)>>);
struct OneRx<T>(Arc<Mutex<(Option<T>, Option<std::task::Waker>)>>);
impl<T> OneTx<T> {
    fn send(self, v: T) {
        let mut g = self.0.lock().unwrap();
        g.0 = Some(v);
        if let Some(w) = g.1.take() {
            w.wake();
        }
    }
}
impl<T> std::future::Future for OneRx<T> {
    type Output = T;
    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> std::task::Poll<T> {
        let mut g = self.0.lock().unwrap();
        match g.0.take() {
            Some(v) => std::task::Poll::Ready(v),
            None => {
                g.1 = Some(cx.waker().clone());
                std::task::Poll::Pending
            }
        }
    }
}
