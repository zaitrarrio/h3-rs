//! Backend conformance: every op, then the tiny models, against the CPU
//! reference on the same random inputs. `h3 check --backend …` runs this on
//! real hardware; the WebGPU backend runs it in CI on lavapipe.
//!
//! Shapes hit tile edges: rows and columns that are not multiples of 64, odd
//! head counts, sequence lengths below one VSA tile, partial tiles.

use crate::{
    audio_vae::{AudioLatents, AudioVae},
    dit::{Attention, H3Dit, LoadOptions},
    taeh3::TaeH3,
    upscaler::Upscaler,
    vae::{VideoVae, Volume},
};
use h3_core::{
    backend::{Act, Backend, Conv1dShape, Conv3dShape, ConvEpilogue, ConvShape, ConvT1dShape, FlowStep, Gate, HeadNorm, ModRef, ModTable, Modulation, Pad, Rope, Snake},
    config::{AudioVaeConfig, H3Config, TaeH3Config, UpscalerConfig, VideoVaeConfig},
    packing::{self, TEXT_TAG},
    quant::{Mxfp8, Nvfp4, Q4, Q8, WeightQuant},
    rng::Rng,
    vsa::{Segment, VsaSpec},
    weights::Synthetic,
};
use h3_cpu::Cpu;

#[derive(Debug, Clone)]
pub struct CheckResult {
    pub name: String,
    /// `max |a − b| / (max |b| + 1e-6)` over the output.
    pub rel_err: f32,
    pub ok: bool,
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "output lengths differ");
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())) + 1e-6;
    let worst = a.iter().zip(b).fold(0f32, |m, (x, y)| {
        let d = (x - y).abs();
        if d.is_nan() { f32::INFINITY } else { m.max(d) }
    });
    worst / scale
}

struct Pair<'a, B: Backend> {
    be: &'a B,
    tol: f32,
    out: Vec<CheckResult>,
}

impl<B: Backend> Pair<'_, B> {
    fn record(&mut self, name: impl Into<String>, got: Vec<f32>, want: Vec<f32>) {
        let e = rel(&got, &want);
        self.out.push(CheckResult { name: name.into(), rel_err: e, ok: e <= self.tol });
    }
}

/// Type-erased access so one closure can drive two backends.
trait Runner {
    fn run(&self, f: &mut dyn FnMut(&dyn ErasedBackend) -> Vec<f32>) -> Vec<f32>;
}

/// The subset of ops exercised by the checks, over erased buffers.
pub trait ErasedBackend {
    fn linear(&self, x: &[f32], x_row: usize, rows: usize, w: &Mat, bias: Option<&[f32]>, act: Act, y_rows: usize, y_row: usize) -> Vec<f32>;
    fn swiglu(&self, x: &[f32], rows: usize, dim: usize) -> Vec<f32>;
    fn rms_norm(&self, x: &[f32], rows: usize, dim: usize, w: Option<&[f32]>, m: Option<(&[f32], &[u32], usize, usize, usize)>, eps: f32) -> Vec<f32>;
    fn residual_rms_norm(&self, h: &[f32], a: &[f32], rows: usize, dim: usize, gate: GateSpec<'_>, w: Option<&[f32]>, m: Option<(&[f32], &[u32], usize, usize, usize)>, eps: f32) -> (Vec<f32>, Vec<f32>);
    fn gate_residual(&self, h: &[f32], a: &[f32], rows: usize, dim: usize, gate: GateSpec<'_>) -> Vec<f32>;
    fn layer_norm(&self, x: &[f32], rows: usize, dim: usize, w: Option<&[f32]>, b: Option<&[f32]>, eps: f32) -> Vec<f32>;
    fn head_split(&self, src: &[f32], rows: usize, stride: usize, col: usize, heads: usize, hd: usize, norm: Option<(Option<&[f32]>, f32)>, rope: Option<(&[f32], &[f32], usize)>) -> Vec<f32>;
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], heads: usize, kv: usize, sq: usize, sk: usize, d: usize, causal: bool) -> Vec<f32>;
    fn attention_qk8(&self, q: &[f32], k: &[f32], v: &[f32], heads: usize, kv: usize, sq: usize, sk: usize, d: usize, causal: bool, pv8: bool) -> Vec<f32>;
    fn vsa(&self, q: &[f32], k: &[f32], v: &[f32], gate: Option<&[f32]>, spec: &VsaSpec) -> Vec<f32>;
    fn patchify(&self, latent: &[f32], c: usize, t: usize, h: usize, w: usize, patch: (usize, usize, usize), out_rows: usize, out_row: usize) -> Vec<f32>;
    fn unpatchify(&self, x: &[f32], x_row: usize, stride: usize, col: usize, c: usize, t: usize, h: usize, w: usize, patch: (usize, usize, usize)) -> Vec<f32>;
    fn flow_step(&self, x: &[f32], off: usize, v: &[f32], v_off: usize, noise: Option<&[f32]>, n: usize, step: FlowStep) -> Vec<f32>;
    fn axpby(&self, y: &[f32], off: usize, a: f32, x: &[f32], x_off: usize, b: f32, n: usize) -> Vec<f32>;
    fn channel_affine(&self, x: &[f32], c: usize, inner: usize, scale: &[f32], shift: Option<&[f32]>) -> Vec<f32>;
    fn conv2d(&self, a: &[f32], b: Option<&[f32]>, shape: &ConvShape, w: &[f32], bias: Option<&[f32]>, ep: Ep<'_>) -> Vec<f32>;
    fn conv3d(&self, x: &[f32], shape: &Conv3dShape, w: &[f32], bias: Option<&[f32]>, ep: Ep<'_>) -> Vec<f32>;
    fn conv1d(&self, x: &[f32], shape: &Conv1dShape, w: &[f32], bias: Option<&[f32]>, ep: Ep<'_>) -> Vec<f32>;
    fn replicate_pad1d(&self, x: &[f32], c: usize, len: usize, left: usize, right: usize) -> Vec<f32>;
    fn conv_transpose1d(&self, x: &[f32], shape: &ConvT1dShape, w: &[f32], bias: Option<&[f32]>) -> Vec<f32>;
    fn group_norm(&self, x: &[f32], c: usize, t: usize, hw: usize, groups: usize, per_frame: bool, w: &[f32], b: &[f32], eps: f32, act: Act) -> Vec<f32>;
    fn snake(&self, x: &[f32], c: usize, len: usize, alpha: &[f32], beta: Option<&[f32]>) -> Vec<f32>;
    fn trilinear(&self, x: &[f32], c: usize, t: usize, h: usize, w: usize, t2: usize, h2: usize, w2: usize) -> Vec<f32>;
    fn pixel_shuffle(&self, x: &[f32], frames: usize, c: usize, h: usize, w: usize, r: usize) -> Vec<f32>;
    fn tanh_clamp(&self, x: &[f32], s: f32) -> Vec<f32>;
    fn clamp(&self, x: &[f32], lo: f32, hi: f32) -> Vec<f32>;
}

/// A weight in one of the formats the backends accept.
pub enum Mat<'a> {
    F32(&'a [f32], usize, usize),
    Q8(&'a Q8),
    Q4(&'a Q4),
    Nvfp4(&'a Nvfp4),
    Mxfp8(&'a Mxfp8),
}

#[derive(Clone, Copy)]
pub enum GateSpec<'a> {
    None,
    Vector(&'a [f32]),
    Table(&'a [f32], &'a [u32], usize, usize),
}

#[derive(Clone, Copy)]
pub enum Ep<'a> {
    None,
    Relu,
    AddRelu(&'a [f32]),
    Add(&'a [f32]),
}

struct BackendRunner<'a, B: Backend>(&'a B);

impl<B: Backend> Runner for BackendRunner<'_, B> {
    fn run(&self, f: &mut dyn FnMut(&dyn ErasedBackend) -> Vec<f32>) -> Vec<f32> {
        f(self)
    }
}

impl<B: Backend> BackendRunner<'_, B> {
    fn mat(&self, m: &Mat<'_>) -> B::Weight {
        match m {
            Mat::F32(w, r, c) => self.0.weight(w, *r, *c),
            Mat::Q8(q) => self.0.weight_q8(q),
            Mat::Q4(q) => self.0.weight_q4(q),
            Mat::Nvfp4(q) => self.0.weight_nvfp4(q),
            Mat::Mxfp8(q) => self.0.weight_mxfp8(q),
        }
    }
}

impl<B: Backend> ErasedBackend for BackendRunner<'_, B> {
    fn linear(&self, x: &[f32], x_row: usize, rows: usize, w: &Mat, bias: Option<&[f32]>, act: Act, y_rows: usize, y_row: usize) -> Vec<f32> {
        let be = self.0;
        let (gx, gw) = (be.upload(x), self.mat(w));
        let gb = bias.map(|b| be.upload(b));
        let n = match w {
            Mat::F32(_, r, _) => *r,
            Mat::Q8(q) => q.rows,
            Mat::Q4(q) => q.rows,
            Mat::Nvfp4(q) => q.rows,
            Mat::Mxfp8(q) => q.rows,
        };
        let gy = be.alloc(y_rows * n);
        be.linear(&gx, x_row, rows, &gw, gb.as_ref(), act, &gy, y_row);
        be.download(&gy)
    }
    fn swiglu(&self, x: &[f32], rows: usize, dim: usize) -> Vec<f32> {
        let be = self.0;
        let o = be.alloc(rows * dim);
        be.swiglu(&be.upload(x), rows, dim, &o);
        be.download(&o)
    }
    fn rms_norm(&self, x: &[f32], rows: usize, dim: usize, w: Option<&[f32]>, m: Option<(&[f32], &[u32], usize, usize, usize)>, eps: f32) -> Vec<f32> {
        let be = self.0;
        let gx = be.upload(x);
        let gw = w.map(|w| be.upload(w));
        let gm = m.map(|(t, i, _, _, _)| (be.upload_table(t), be.upload_u32(i)));
        let o = be.alloc(rows * dim);
        let modulation = match (&gm, m) {
            (Some((t, i)), Some((_, _, slots, sc, sh))) => Modulation::Table { m: ModTable { table: t, idx: i, slots }, scale: sc, shift: sh },
            _ => Modulation::None,
        };
        be.rms_norm(&gx, rows, dim, gw.as_ref(), modulation, eps, &o);
        be.download(&o)
    }
    fn residual_rms_norm(&self, h: &[f32], a: &[f32], rows: usize, dim: usize, gate: GateSpec<'_>, w: Option<&[f32]>, m: Option<(&[f32], &[u32], usize, usize, usize)>, eps: f32) -> (Vec<f32>, Vec<f32>) {
        let be = self.0;
        let (gh, ga) = (be.upload(h), be.upload(a));
        let gw = w.map(|w| be.upload(w));
        let gm = m.map(|(t, i, _, _, _)| (be.upload_table(t), be.upload_u32(i)));
        let gg = match gate {
            GateSpec::Vector(v) => Some((be.upload(v), None)),
            GateSpec::Table(t, i, _, _) => Some((be.upload_table(t), Some(be.upload_u32(i)))),
            GateSpec::None => None,
        };
        let o = be.alloc(rows * dim);
        let modulation = match (&gm, m) {
            (Some((t, i)), Some((_, _, slots, sc, sh))) => Modulation::Table { m: ModTable { table: t, idx: i, slots }, scale: sc, shift: sh },
            _ => Modulation::None,
        };
        let g = match (&gg, gate) {
            (Some((v, None)), GateSpec::Vector(_)) => Gate::Vector(v),
            (Some((t, Some(i))), GateSpec::Table(_, _, slots, slot)) => Gate::Table(ModRef { table: ModTable { table: t, idx: i, slots }, slot }),
            _ => Gate::None,
        };
        be.residual_rms_norm(&gh, &ga, rows, dim, g, gw.as_ref(), modulation, eps, &o);
        (be.download(&gh), be.download(&o))
    }
    fn gate_residual(&self, h: &[f32], a: &[f32], rows: usize, dim: usize, gate: GateSpec<'_>) -> Vec<f32> {
        let be = self.0;
        let (gh, ga) = (be.upload(h), be.upload(a));
        let gg = match gate {
            GateSpec::Vector(v) => Some((be.upload(v), None)),
            GateSpec::Table(t, i, _, _) => Some((be.upload_table(t), Some(be.upload_u32(i)))),
            GateSpec::None => None,
        };
        let g = match (&gg, gate) {
            (Some((v, None)), GateSpec::Vector(_)) => Gate::Vector(v),
            (Some((t, Some(i))), GateSpec::Table(_, _, slots, slot)) => Gate::Table(ModRef { table: ModTable { table: t, idx: i, slots }, slot }),
            _ => Gate::None,
        };
        be.gate_residual(&gh, &ga, rows, dim, g);
        be.download(&gh)
    }
    fn layer_norm(&self, x: &[f32], rows: usize, dim: usize, w: Option<&[f32]>, b: Option<&[f32]>, eps: f32) -> Vec<f32> {
        let be = self.0;
        let gw = w.map(|w| be.upload(w));
        let gb = b.map(|b| be.upload(b));
        let o = be.alloc(rows * dim);
        be.layer_norm(&be.upload(x), rows, dim, gw.as_ref(), gb.as_ref(), eps, &o);
        be.download(&o)
    }
    fn head_split(&self, src: &[f32], rows: usize, stride: usize, col: usize, heads: usize, hd: usize, norm: Option<(Option<&[f32]>, f32)>, rope: Option<(&[f32], &[f32], usize)>) -> Vec<f32> {
        let be = self.0;
        let gs = be.upload(src);
        let nw = norm.and_then(|(w, _)| w.map(|w| be.upload(w)));
        let rp = rope.map(|(c, s, _)| (be.upload_table(c), be.upload_table(s)));
        let o = be.alloc(heads * rows * hd);
        let hn = norm.map(|(_, eps)| HeadNorm { w: nw.as_ref(), eps });
        let rr = match (&rp, rope) {
            (Some((c, s)), Some((_, _, rot))) => Some(Rope { cos: c, sin: s, rot }),
            _ => None,
        };
        be.head_split(&gs, rows, stride, col, heads, hd, hn, rr, &o);
        be.download(&o)
    }
    fn attention(&self, q: &[f32], k: &[f32], v: &[f32], heads: usize, kv: usize, sq: usize, sk: usize, d: usize, causal: bool) -> Vec<f32> {
        let be = self.0;
        let o = be.alloc(sq * heads * d);
        be.attention(&be.upload(q), &be.upload(k), &be.upload(v), heads, kv, sq, sk, d, causal, &o);
        be.download(&o)
    }
    fn attention_qk8(&self, q: &[f32], k: &[f32], v: &[f32], heads: usize, kv: usize, sq: usize, sk: usize, d: usize, causal: bool, pv8: bool) -> Vec<f32> {
        let be = self.0;
        let o = be.alloc(sq * heads * d);
        be.attention_qk8(&be.upload(q), &be.upload(k), &be.upload(v), heads, kv, sq, sk, d, causal, pv8, &o);
        be.download(&o)
    }
    fn vsa(&self, q: &[f32], k: &[f32], v: &[f32], gate: Option<&[f32]>, spec: &VsaSpec) -> Vec<f32> {
        let be = self.0;
        let st = be.vsa_prepare(spec);
        let o = be.alloc(spec.plan.seq * spec.heads * spec.head_dim);
        let g = gate.map(|g| be.upload(g));
        be.vsa_attention(&be.upload(q), &be.upload(k), &be.upload(v), g.as_ref(), &st, &o);
        be.download(&o)
    }
    fn patchify(&self, latent: &[f32], c: usize, t: usize, h: usize, w: usize, patch: (usize, usize, usize), out_rows: usize, out_row: usize) -> Vec<f32> {
        let be = self.0;
        let f = c * patch.0 * patch.1 * patch.2;
        let o = be.alloc(out_rows * f);
        be.patchify(&be.upload(latent), c, t, h, w, patch, &o, out_row);
        be.download(&o)
    }
    fn unpatchify(&self, x: &[f32], x_row: usize, stride: usize, col: usize, c: usize, t: usize, h: usize, w: usize, patch: (usize, usize, usize)) -> Vec<f32> {
        let be = self.0;
        let o = be.alloc(c * t * h * w);
        be.unpatchify(&be.upload(x), x_row, stride, col, c, t, h, w, patch, &o);
        be.download(&o)
    }
    fn flow_step(&self, x: &[f32], off: usize, v: &[f32], v_off: usize, noise: Option<&[f32]>, n: usize, step: FlowStep) -> Vec<f32> {
        let be = self.0;
        let gx = be.upload(x);
        let gn = noise.map(|z| be.upload(z));
        be.flow_step(&gx, off, &be.upload(v), v_off, gn.as_ref(), n, step);
        be.download(&gx)
    }
    fn axpby(&self, y: &[f32], off: usize, a: f32, x: &[f32], x_off: usize, b: f32, n: usize) -> Vec<f32> {
        let be = self.0;
        let gy = be.upload(y);
        be.axpby(&gy, off, a, &be.upload(x), x_off, b, n);
        be.download(&gy)
    }
    fn channel_affine(&self, x: &[f32], c: usize, inner: usize, scale: &[f32], shift: Option<&[f32]>) -> Vec<f32> {
        let be = self.0;
        let o = be.alloc(c * inner);
        let sh = shift.map(|s| be.upload(s));
        be.channel_affine(&be.upload(x), c, inner, &be.upload(scale), sh.as_ref(), &o);
        be.download(&o)
    }
    fn conv2d(&self, a: &[f32], b: Option<&[f32]>, shape: &ConvShape, w: &[f32], bias: Option<&[f32]>, ep: Ep<'_>) -> Vec<f32> {
        let be = self.0;
        let ga = be.upload(a);
        let gb = b.map(|b| be.upload(b));
        let gw = be.weight(w, shape.cout, shape.k_len());
        let gbias = bias.map(|b| be.upload(b));
        let res = match ep {
            Ep::AddRelu(r) | Ep::Add(r) => Some(be.upload(r)),
            _ => None,
        };
        let o = be.alloc(shape.out_len());
        let e = match (ep, &res) {
            (Ep::None, _) => ConvEpilogue::None,
            (Ep::Relu, _) => ConvEpilogue::Relu,
            (Ep::AddRelu(_), Some(r)) => ConvEpilogue::AddRelu(r),
            (Ep::Add(_), Some(r)) => ConvEpilogue::Add(r),
            _ => unreachable!(),
        };
        be.conv2d(&ga, gb.as_ref(), shape, &gw, gbias.as_ref(), e, &o);
        be.download(&o)
    }
    fn conv3d(&self, x: &[f32], shape: &Conv3dShape, w: &[f32], bias: Option<&[f32]>, ep: Ep<'_>) -> Vec<f32> {
        let be = self.0;
        let gw = be.weight(w, shape.cout, shape.k_len());
        let gbias = bias.map(|b| be.upload(b));
        let res = match ep {
            Ep::AddRelu(r) | Ep::Add(r) => Some(be.upload(r)),
            _ => None,
        };
        let o = be.alloc(shape.out_len());
        let e = match (ep, &res) {
            (Ep::None, _) => ConvEpilogue::None,
            (Ep::Relu, _) => ConvEpilogue::Relu,
            (Ep::AddRelu(_), Some(r)) => ConvEpilogue::AddRelu(r),
            (Ep::Add(_), Some(r)) => ConvEpilogue::Add(r),
            _ => unreachable!(),
        };
        be.conv3d(&be.upload(x), shape, &gw, gbias.as_ref(), e, &o);
        be.download(&o)
    }
    fn conv1d(&self, x: &[f32], shape: &Conv1dShape, w: &[f32], bias: Option<&[f32]>, ep: Ep<'_>) -> Vec<f32> {
        let be = self.0;
        let gw = be.weight(w, shape.cout, (shape.cin / shape.groups) * shape.k);
        let gbias = bias.map(|b| be.upload(b));
        let res = match ep {
            Ep::AddRelu(r) | Ep::Add(r) => Some(be.upload(r)),
            _ => None,
        };
        let o = be.alloc(shape.out_len());
        let e = match (ep, &res) {
            (Ep::None, _) => ConvEpilogue::None,
            (Ep::Relu, _) => ConvEpilogue::Relu,
            (Ep::AddRelu(_), Some(r)) => ConvEpilogue::AddRelu(r),
            (Ep::Add(_), Some(r)) => ConvEpilogue::Add(r),
            _ => unreachable!(),
        };
        be.conv1d(&be.upload(x), shape, &gw, gbias.as_ref(), e, &o);
        be.download(&o)
    }
    fn replicate_pad1d(&self, x: &[f32], c: usize, len: usize, left: usize, right: usize) -> Vec<f32> {
        let be = self.0;
        let o = be.alloc(c * (len + left + right));
        be.replicate_pad1d(&be.upload(x), c, len, left, right, &o);
        be.download(&o)
    }
    fn conv_transpose1d(&self, x: &[f32], shape: &ConvT1dShape, w: &[f32], bias: Option<&[f32]>) -> Vec<f32> {
        let be = self.0;
        let gw = be.weight(w, shape.cin, (shape.cout / shape.groups) * shape.k);
        let gbias = bias.map(|b| be.upload(b));
        let o = be.alloc(shape.out_len());
        be.conv_transpose1d(&be.upload(x), shape, &gw, gbias.as_ref(), &o);
        be.download(&o)
    }
    fn group_norm(&self, x: &[f32], c: usize, t: usize, hw: usize, groups: usize, per_frame: bool, w: &[f32], b: &[f32], eps: f32, act: Act) -> Vec<f32> {
        let be = self.0;
        let o = be.alloc(c * t * hw);
        be.group_norm(&be.upload(x), c, t, hw, groups, per_frame, &be.upload(w), &be.upload(b), eps, act, &o);
        be.download(&o)
    }
    fn snake(&self, x: &[f32], c: usize, len: usize, alpha: &[f32], beta: Option<&[f32]>) -> Vec<f32> {
        let be = self.0;
        let gx = be.upload(x);
        let ga = be.upload(alpha);
        match beta {
            None => be.snake(&gx, c, len, Snake::Alpha(&ga)),
            Some(b) => {
                let gb = be.upload(b);
                be.snake(&gx, c, len, Snake::AlphaBeta { alpha: &ga, beta: &gb });
            }
        }
        be.download(&gx)
    }
    fn trilinear(&self, x: &[f32], c: usize, t: usize, h: usize, w: usize, t2: usize, h2: usize, w2: usize) -> Vec<f32> {
        let be = self.0;
        let o = be.alloc(c * t2 * h2 * w2);
        be.trilinear_resize(&be.upload(x), c, t, h, w, t2, h2, w2, &o);
        be.download(&o)
    }
    fn pixel_shuffle(&self, x: &[f32], frames: usize, c: usize, h: usize, w: usize, r: usize) -> Vec<f32> {
        let be = self.0;
        let o = be.alloc(frames * c * h * w * r * r);
        be.pixel_shuffle(&be.upload(x), frames, c, h, w, r, &o);
        be.download(&o)
    }
    fn tanh_clamp(&self, x: &[f32], s: f32) -> Vec<f32> {
        let be = self.0;
        let gx = be.upload(x);
        be.tanh_clamp(&gx, x.len(), s);
        be.download(&gx)
    }
    fn clamp(&self, x: &[f32], lo: f32, hi: f32) -> Vec<f32> {
        let be = self.0;
        let gx = be.upload(x);
        be.clamp(&gx, x.len(), lo, hi);
        be.download(&gx)
    }
}

impl<B: Backend> Pair<'_, B> {
    fn op(&mut self, name: impl Into<String>, f: impl Fn(&dyn ErasedBackend) -> Vec<f32>) {
        let mut g = |r: &dyn ErasedBackend| f(r);
        let got = BackendRunner(self.be).run(&mut g);
        let want = BackendRunner(&Cpu).run(&mut g);
        self.record(name, got, want);
    }
}

pub fn run_all<B: Backend>(be: &B, tol: f32) -> Vec<CheckResult> {
    let mut p = Pair { be, tol, out: Vec::new() };
    let mut rng = Rng::new(42);
    ops(&mut p, &mut rng);
    models(&mut p);
    p.out
}

/// Just the op checks (fast).
pub fn run_ops<B: Backend>(be: &B, tol: f32) -> Vec<CheckResult> {
    let mut p = Pair { be, tol, out: Vec::new() };
    let mut rng = Rng::new(42);
    ops(&mut p, &mut rng);
    p.out
}

fn ops<B: Backend>(p: &mut Pair<'_, B>, rng: &mut Rng) {
    // linear: odd M/N/K, row offsets, every activation and weight format.
    for &(m, k, n, act) in &[
        (1, 8, 4, Act::None),
        (37, 70, 33, Act::GeluTanh),
        (130, 96, 200, Act::GeluErf),
        (65, 256, 64, Act::Silu),
        (300, 96, 260, Act::None),
        (129, 1536, 130, Act::None),
        (1100, 1536, 200, Act::GeluTanh),
    ] {
        let x = rng.normal((m + 3) * k);
        let w = rng.uniform(n * k, 0.2);
        let b = rng.normal(n);
        p.op(format!("linear {m}x{k}x{n} {act:?}"), |r| r.linear(&x, 3, m, &Mat::F32(&w, n, k), Some(&b), act, m + 2, 2));
        if k % 2 == 0 && m > 1 {
            let q8 = Q8::quantize(&w, n, k);
            p.op(format!("linear int8/{} {m}x{k}x{n}", q8.group), |r| r.linear(&x, 3, m, &Mat::Q8(&q8), Some(&b), act, m + 2, 2));
            let q4 = Q4::quantize(&w, n, k, 64, k % 64 != 0);
            p.op(format!("linear int4/{} {m}x{k}x{n}", q4.group), |r| r.linear(&x, 3, m, &Mat::Q4(&q4), Some(&b), act, m + 2, 2));
            if k % 16 == 0 {
                let nv = synth_nvfp4(&w, n, k);
                p.op(format!("linear nvfp4 {m}x{k}x{n}"), |r| r.linear(&x, 3, m, &Mat::Nvfp4(&nv), None, act, m + 2, 2));
                // The same matrix in ComfyUI's nibble order (first code high).
                let nvh = Nvfp4 {
                    packed: nv.packed.iter().map(|b| (b << 4) | (b >> 4)).collect(),
                    first_high: true,
                    ..nv.clone()
                };
                p.op(format!("linear nvfp4 first-high {m}x{k}x{n}"), |r| r.linear(&x, 3, m, &Mat::Nvfp4(&nvh), None, act, m + 2, 2));
            }
            if k % 32 == 0 {
                // MXFP8 W8A8: the reference rounds the activations too, so the
                // FP8 tensor-core path and the emulation must both match it.
                let mx = Mxfp8::quantize(&w, n, k);
                p.op(format!("linear mxfp8 {m}x{k}x{n} {act:?}"), |r| r.linear(&x, 3, m, &Mat::Mxfp8(&mx), Some(&b), act, m + 2, 2));
                p.op(format!("linear mxfp8 no-bias {m}x{k}x{n}"), |r| r.linear(&x, 0, m, &Mat::Mxfp8(&mx), None, Act::None, m, 0));
            }
        }
    }
    // MXFP8 with n a multiple of 256: the CUDA backend splits such GEMMs along N
    // when `H3_CUDA_MX_SPLIT_MIN` allows (the default threshold is 24576).
    {
        let (m, k, n) = (130, 64, 512);
        let x = rng.normal((m + 3) * k);
        let w = rng.uniform(n * k, 0.2);
        let b = rng.normal(n);
        let mx = Mxfp8::quantize(&w, n, k);
        p.op(format!("linear mxfp8 {m}x{k}x{n} GeluTanh (split-able)"), |r| r.linear(&x, 3, m, &Mat::Mxfp8(&mx), Some(&b), Act::GeluTanh, m + 2, 2));
        p.op(format!("linear mxfp8 no-bias {m}x{k}x{n} (split-able)"), |r| r.linear(&x, 1, m, &Mat::Mxfp8(&mx), None, Act::Silu, m + 1, 1));
    }
    // swiglu
    {
        let x = rng.normal(19 * 2 * 48);
        p.op("swiglu 19x48", |r| r.swiglu(&x, 19, 48));
    }
    // Norms with row-indexed tables.
    let (rows, dim) = (19, 96);
    let h = rng.normal(rows * dim);
    let a = rng.normal(rows * dim);
    let table = rng.uniform(3 * 6 * dim, 0.5);
    let idx: Vec<u32> = (0..rows as u32).map(|i| i % 3).collect();
    let nw: Vec<f32> = rng.uniform(dim, 0.5).iter().map(|v| v + 1.0).collect();
    let nb = rng.uniform(dim, 0.5);
    p.op("rms_norm plain", |r| r.rms_norm(&h, rows, dim, Some(&nw), None, 1e-5));
    p.op("rms_norm mod", |r| r.rms_norm(&h, rows, dim, Some(&nw), Some((&table, &idx, 6, 1, 0)), 1e-5));
    p.op("residual_rms_norm table-gate", |r| {
        let (hh, nn) = r.residual_rms_norm(&h, &a, rows, dim, GateSpec::Table(&table, &idx, 6, 2), Some(&nw), Some((&table, &idx, 6, 4, 3)), 1e-5);
        [hh, nn].concat()
    });
    p.op("residual_rms_norm vector-gate", |r| {
        let (hh, nn) = r.residual_rms_norm(&h, &a, rows, dim, GateSpec::Vector(&nb), Some(&nw), None, 1e-5);
        [hh, nn].concat()
    });
    p.op("residual_rms_norm plain", |r| {
        let (hh, nn) = r.residual_rms_norm(&h, &a, rows, dim, GateSpec::None, None, None, 1e-5);
        [hh, nn].concat()
    });
    p.op("gate_residual table", |r| r.gate_residual(&h, &a, rows, dim, GateSpec::Table(&table, &idx, 6, 5)));
    p.op("gate_residual vector", |r| r.gate_residual(&h, &a, rows, dim, GateSpec::Vector(&nb)));
    p.op("layer_norm", |r| r.layer_norm(&h, rows, dim, Some(&nw), Some(&nb), 1e-5));
    p.op("layer_norm no-affine", |r| r.layer_norm(&h, rows, dim, None, None, 1e-6));
    // head_split: 3 heads of 32 out of a 3·96-wide row, per-head norm, partial rope (24 of 32).
    {
        let (rows, heads, hd) = (23, 3, 32);
        let src = rng.normal(rows * 3 * heads * hd);
        let w = rng.uniform(hd, 0.5).iter().map(|v| v + 1.0).collect::<Vec<_>>();
        let rot = 24;
        let cos = rng.uniform(rows * rot / 2, 1.0);
        let sin = rng.uniform(rows * rot / 2, 1.0);
        p.op("head_split norm+rope", |r| r.head_split(&src, rows, 3 * heads * hd, heads * hd, heads, hd, Some((Some(&w), 1e-5)), Some((&cos, &sin, rot))));
        let cos2 = rng.uniform(rows * hd / 2, 1.0);
        let sin2 = rng.uniform(rows * hd / 2, 1.0);
        p.op("head_split no-affine norm full rope", |r| r.head_split(&src, rows, 3 * heads * hd, 0, heads, hd, Some((None, 1e-5)), Some((&cos2, &sin2, hd))));
        p.op("head_split plain", |r| r.head_split(&src, rows, 3 * heads * hd, 2 * heads * hd, heads, hd, None, None));
    }
    // attention: odd lengths, GQA, causal, head_dim 16/32/64/128.
    for &(heads, kv, sq, sk, d, causal) in &[(3, 3, 37, 37, 16, false), (4, 2, 70, 70, 32, true), (2, 2, 130, 65, 64, false), (2, 1, 33, 33, 128, true)] {
        let q = rng.normal(heads * sq * d);
        let k = rng.normal(kv * sk * d);
        let v = rng.normal(kv * sk * d);
        p.op(format!("attention h{heads} kv{kv} {sq}x{sk} d{d} causal={causal}"), |r| r.attention(&q, &k, &v, heads, kv, sq, sk, d, causal));
        // INT8 Q·Kᵀ: K shifted on one channel so the mean subtraction matters;
        // the int8 kernels exist for d 64 / 128, the others emulate.
        let kshift: Vec<f32> = k.iter().enumerate().map(|(i, x)| if i % d == 1 { x + 3.0 } else { *x }).collect();
        p.op(format!("attention qk8 h{heads} kv{kv} {sq}x{sk} d{d} causal={causal}"), |r| r.attention_qk8(&q, &kshift, &v, heads, kv, sq, sk, d, causal, false));
        // FP8 P·V on top: V shifted on one channel so its mean matters.
        let vshift: Vec<f32> = v.iter().enumerate().map(|(i, x)| if i % d == 2 { x + 2.0 } else { *x }).collect();
        p.op(format!("attention qk8+pv8 h{heads} kv{kv} {sq}x{sk} d{d} causal={causal}"), |r| r.attention_qk8(&q, &kshift, &vshift, heads, kv, sq, sk, d, causal, true));
    }
    // VSA: a dense prefix, two regions (one sparse, one dense), partial tiles, with and without gate.
    for &(tile, d, odd) in &[(64usize, 32usize, false), (128, 64, false), (64, 128, true)] {
        // `odd`: the H3 shape (64-token tiles, 128-wide heads) with an odd
        // number of tiles and a single prefix tile, so the paired-tile fine
        // stage sees a prefix + video pair and a partnerless last tile.
        let segs = if odd {
            [Segment::Dense(40), Segment::Video { grid: (5, 6, 9), sparsity: 0.7 }, Segment::Dense(3), Segment::Video { grid: (3, 4, 4), sparsity: 0.5 }]
        } else {
            [Segment::Dense(70), Segment::Video { grid: (3, 5, 9), sparsity: 0.6 }, Segment::Dense(5), Segment::Video { grid: (2, 4, 4), sparsity: 0.0 }]
        };
        let spec = VsaSpec::new(&segs, tile, 2, d);
        let n = spec.plan.seq * 2 * d;
        let (q, k, v, g) = (rng.normal(n), rng.normal(n), rng.normal(n), rng.normal(n));
        p.op(format!("vsa tile{tile} d{d} gate"), |r| r.vsa(&q, &k, &v, Some(&g), &spec));
        p.op(format!("vsa tile{tile} d{d} no-gate"), |r| r.vsa(&q, &k, &v, None, &spec));
        // INT8 Q·Kᵀ in the fine stage: K shifted by its mean so the rounding
        // matters (a mean of 3 on one channel).
        let spec8 = spec.clone().with_qk8(true);
        let kshift: Vec<f32> = k.iter().enumerate().map(|(i, x)| if i % d == 1 { x + 3.0 } else { *x }).collect();
        p.op(format!("vsa tile{tile} d{d} qk8 gate"), |r| r.vsa(&q, &kshift, &v, Some(&g), &spec8));
        p.op(format!("vsa tile{tile} d{d} qk8 no-gate"), |r| r.vsa(&q, &kshift, &v, None, &spec8));
        // FP8 P·V on top: V shifted on one channel so its mean matters.
        let spec88 = spec.clone().with_pv8(true);
        let vshift: Vec<f32> = v.iter().enumerate().map(|(i, x)| if i % d == 2 { x + 2.0 } else { *x }).collect();
        p.op(format!("vsa tile{tile} d{d} qk8+pv8 gate"), |r| r.vsa(&q, &kshift, &vshift, Some(&g), &spec88));
        p.op(format!("vsa tile{tile} d{d} qk8+pv8 no-gate"), |r| r.vsa(&q, &kshift, &vshift, None, &spec88));
    }
    // patchify / unpatchify with offsets.
    {
        let (c, t, h, w) = (6, 3, 6, 10);
        let lat = rng.normal(c * t * h * w);
        p.op("patchify 1x2x2", |r| r.patchify(&lat, c, t, h, w, (1, 2, 2), 3 * 3 * 5 + 4, 4));
        p.op("patchify 1x1x1", |r| r.patchify(&lat, c, t, h, w, (1, 1, 1), t * h * w, 0));
        let rows = rng.normal((45 + 2) * (24 + 5));
        p.op("unpatchify", |r| r.unpatchify(&rows, 2, 29, 5, c, t, h, w, (1, 2, 2)));
        let tok = rng.normal(2 * 4 * 3 * 3 * 4 * 16 * 16);
        p.op("unpatchify 4x16x16", |r| r.unpatchify(&tok, 0, 3 * 4 * 16 * 16, 0, 3, 8, 48, 64, (4, 16, 16)));
    }
    // flow steps, axpby, channel_affine.
    {
        let n = 1000;
        let x = rng.normal(n + 7);
        let v = rng.normal(n + 7);
        let z = rng.normal(n + 7);
        p.op("flow euler", |r| r.flow_step(&x, 7, &v, 3, None, n, FlowStep::Euler { sigma: 0.9, next: 0.6 }));
        p.op("flow renoise", |r| r.flow_step(&x, 7, &v, 3, Some(&z), n, FlowStep::Renoise { sigma: 0.9, next: 0.6 }));
        p.op("flow terminal", |r| r.flow_step(&x, 0, &v, 0, None, n, FlowStep::Terminal { sigma: 0.3 }));
        p.op("axpby", |r| r.axpby(&x, 7, 0.5, &v, 3, -1.5, n));
        let (c, inner) = (7, 33);
        let s = rng.uniform(c, 1.0);
        let sh = rng.uniform(c, 1.0);
        p.op("channel_affine", |r| r.channel_affine(&x[..c * inner], c, inner, &s, Some(&sh)));
    }
    // conv2d (TAEH3 shapes): concat, upsample, epilogues.
    for &(frames, cin_a, cin_b, cout, h, w, up, k, ep) in &[
        (2usize, 5usize, 5usize, 7usize, 6usize, 9usize, false, 3usize, 1u8),
        (3, 6, 0, 8, 5, 7, true, 3, 0),
        (2, 4, 0, 12, 6, 6, false, 1, 0),
        (1, 3, 0, 3, 17, 33, false, 3, 2),
    ] {
        let s = ConvShape { frames, cin_a, cin_b, cout, h, w, upsample: up, k };
        let a = rng.normal(frames * cin_a * h * w);
        let b = rng.normal(frames * cin_b.max(1) * h * w);
        let wt = rng.uniform(cout * s.k_len(), 0.3);
        let bias = rng.normal(cout);
        let res = rng.normal(s.out_len());
        p.op(format!("conv2d {frames}x{cin_a}+{cin_b}->{cout} {h}x{w} up={up} k{k} ep{ep}"), |r| {
            let e = match ep {
                0 => Ep::None,
                1 => Ep::Relu,
                _ => Ep::AddRelu(&res),
            };
            r.conv2d(&a, (cin_b > 0).then_some(&b[..]), &s, &wt, Some(&bias), e)
        });
    }
    // conv3d: causal reflect, zero same, strided downsampler, depthwise temporal, 1×1×1 shortcut, Add epilogue.
    {
        let (c, t, h, w) = (5, 4, 6, 7);
        let x = rng.normal(c * t * h * w);
        let mk = |cout: usize, s: Conv3dShape, rng: &mut Rng| -> (Conv3dShape, Vec<f32>, Vec<f32>) {
            let wt = rng.uniform(cout * s.k_len(), 0.3);
            let b = rng.normal(cout);
            (s, wt, b)
        };
        let (s1, w1, b1) = mk(6, Conv3dShape::causal(c, 6, t, h, w, 3), rng);
        p.op("conv3d causal reflect", |r| r.conv3d(&x, &s1, &w1, Some(&b1), Ep::None));
        let (s2, w2, b2) = mk(c, Conv3dShape::same(c, c, t, h, w, 3), rng);
        p.op("conv3d same add", |r| r.conv3d(&x, &s2, &w2, Some(&b2), Ep::Add(&x)));
        let s3 = Conv3dShape { stride: (2, 2, 2), pad_h: (0, 1), pad_w: (0, 1), ..Conv3dShape::causal(c, 8, t, h, w, 3) };
        let (s3, w3, b3) = mk(8, s3, rng);
        p.op("conv3d strided downsampler", |r| r.conv3d(&x, &s3, &w3, Some(&b3), Ep::None));
        let s4 = Conv3dShape { groups: c, k: (5, 1, 1), pad_t: (2, 2), pad_h: (0, 0), pad_w: (0, 0), ..Conv3dShape::same(c, c, t, h, w, 1) };
        let (s4, w4, b4) = mk(c, s4, rng);
        p.op("conv3d depthwise temporal", |r| r.conv3d(&x, &s4, &w4, Some(&b4), Ep::None));
        let s5 = Conv3dShape { pad_t: (0, 0), pad_h: (0, 0), pad_w: (0, 0), ..Conv3dShape::causal(c, 9, t, h, w, 1) };
        let (s5, w5, b5) = mk(9, s5, rng);
        p.op("conv3d 1x1x1", |r| r.conv3d(&x, &s5, &w5, Some(&b5), Ep::None));
        let s6 = Conv3dShape { spatial_pad: Pad::Zero, ..Conv3dShape::causal(c, 4, t, h, w, 3) };
        let (s6, w6, b6) = mk(4, s6, rng);
        p.op("conv3d causal zero relu", |r| r.conv3d(&x, &s6, &w6, Some(&b6), Ep::Relu));
    }
    // conv1d / transposed / replicate pad / snake.
    {
        let (c, len) = (6, 53);
        let x = rng.normal(c * len);
        let s1 = Conv1dShape { cin: c, cout: 8, groups: 1, len, k: 7, stride: 1, dilation: 3, pad: (9, 9) };
        let w1 = rng.uniform(8 * c * 7, 0.3);
        let b1 = rng.normal(8);
        p.op("conv1d dilated", |r| r.conv1d(&x, &s1, &w1, Some(&b1), Ep::None));
        let s2 = Conv1dShape { cin: c, cout: 12, groups: 1, len, k: 8, stride: 4, dilation: 1, pad: (2, 2) };
        let w2 = rng.uniform(12 * c * 8, 0.3);
        p.op("conv1d strided", |r| r.conv1d(&x, &s2, &w2, None, Ep::None));
        let s3 = Conv1dShape { cin: c, cout: c, groups: c, len, k: 12, stride: 2, dilation: 1, pad: (0, 0) };
        let w3 = rng.uniform(c * 12, 0.3);
        p.op("conv1d depthwise stride2", |r| r.conv1d(&x, &s3, &w3, None, Ep::None));
        let s4 = Conv1dShape { cin: c, cout: c, groups: 1, len, k: 1, stride: 1, dilation: 1, pad: (0, 0) };
        let w4 = rng.uniform(c * c, 0.3);
        p.op("conv1d 1x1 add", |r| r.conv1d(&x, &s4, &w4, None, Ep::Add(&x)));
        p.op("replicate_pad1d", |r| r.replicate_pad1d(&x, c, len, 5, 6));
        let ts = ConvT1dShape { cin: c, cout: 4, groups: 1, len, k: 9, stride: 5, pad: 2 };
        let wt = rng.uniform(c * 4 * 9, 0.3);
        let bt = rng.normal(4);
        p.op("conv_transpose1d", |r| r.conv_transpose1d(&x, &ts, &wt, Some(&bt)));
        let td = ConvT1dShape { cin: c, cout: c, groups: c, len, k: 12, stride: 2, pad: 0 };
        let wd = rng.uniform(c * 12, 0.3);
        p.op("conv_transpose1d depthwise", |r| r.conv_transpose1d(&x, &td, &wd, None));
        // The snake parameters enter sin(e^α·x): a bf16 backend's rounding of α
        // is a phase error, so every backend gets bf16-representable values.
        let r16 = |v: &f32| half::bf16::from_f32(*v).to_f32();
        let alpha = rng.uniform(c, 0.5).iter().map(|v| r16(&(v + 1.0))).collect::<Vec<_>>();
        let beta = rng.uniform(c, 0.5).iter().map(r16).collect::<Vec<_>>();
        p.op("snake alpha", |r| r.snake(&x, c, len, &alpha, None));
        p.op("snake beta", |r| r.snake(&x, c, len, &alpha, Some(&beta)));
    }
    // group_norm, trilinear, pixel_shuffle, clamps.
    {
        let (c, t, hw) = (8, 3, 15);
        let x = rng.normal(c * t * hw);
        let w = rng.uniform(c, 0.5).iter().map(|v| v + 1.0).collect::<Vec<_>>();
        let b = rng.uniform(c, 0.5);
        p.op("group_norm per-frame silu", |r| r.group_norm(&x, c, t, hw, 4, true, &w, &b, 1e-6, Act::Silu));
        p.op("group_norm volume", |r| r.group_norm(&x, c, t, hw, 2, false, &w, &b, 1e-5, Act::None));
        let v = rng.normal(3 * 2 * 4 * 5);
        p.op("trilinear up", |r| r.trilinear(&v, 3, 2, 4, 5, 2, 7, 11));
        p.op("trilinear same-t", |r| r.trilinear(&v, 3, 2, 4, 5, 2, 8, 10));
        let ps = rng.normal(2 * 12 * 3 * 4);
        p.op("pixel_shuffle", |r| r.pixel_shuffle(&ps, 2, 3, 3, 4, 2));
        let big = rng.uniform(777, 6.0);
        p.op("tanh_clamp", |r| r.tanh_clamp(&big, 3.0));
        p.op("clamp", |r| r.clamp(&big, -1.0, 1.0));
    }
}

/// A deterministic NVFP4 matrix: quantize `w` to E2M1 with per-16 E4M3 scales.
pub fn synth_nvfp4(w: &[f32], rows: usize, cols: usize) -> Nvfp4 {
    use h3_core::quant::{E2M1, NVFP4_GROUP};
    let groups = cols / NVFP4_GROUP;
    let alpha = 1.0f32;
    let mut scales = vec![0u8; rows * groups];
    let mut packed = vec![0u8; rows * cols / 2];
    for r in 0..rows {
        for g in 0..groups {
            let blk = &w[r * cols + g * NVFP4_GROUP..r * cols + (g + 1) * NVFP4_GROUP];
            let amax = blk.iter().fold(0f32, |m, v| m.max(v.abs()));
            // E4M3 scale ≈ amax / 6, encoded as the nearest representable.
            let s = (amax / 6.0).max(2f32.powi(-6));
            let e = s.log2().floor().clamp(-6.0, 8.0);
            let man = ((s / 2f32.powf(e) - 1.0) * 8.0).round().clamp(0.0, 7.0);
            let byte = (((e as i32 + 7) as u8) << 3) | man as u8;
            let sv = h3_core::quant::e4m3_to_f32(byte) * alpha;
            scales[r * groups + g] = byte;
            for (i, &v) in blk.iter().enumerate() {
                let target = v / sv;
                let code = (0..16u8).min_by(|&a, &b| (E2M1[a as usize] - target).abs().total_cmp(&(E2M1[b as usize] - target).abs())).unwrap();
                let c = g * NVFP4_GROUP + i;
                let idx = (r * cols + c) / 2;
                if c % 2 == 0 {
                    packed[idx] |= code;
                } else {
                    packed[idx] |= code << 4;
                }
            }
        }
    }
    Nvfp4 { packed, scales, alpha, rows, cols, first_high: false }
}

fn models<B: Backend>(p: &mut Pair<'_, B>) {
    for quant in [WeightQuant::Native, WeightQuant::Int8, WeightQuant::Int4 { group: 32, hqq: true, ffn_only: false }] {
        dit_check(p, Attention::Dense { qk8: false, pv8: false }, quant, false);
        dit_check(p, Attention::Vsa { sparsity: 0.5, tile: 64, ref_sparsity: 0.5, qk8: false, pv8: false }, quant, true);
    }
    dit_check(p, Attention::Dense { qk8: true, pv8: false }, WeightQuant::Native, false);
    dit_check(p, Attention::Dense { qk8: true, pv8: true }, WeightQuant::Native, false);
    dit_check(p, Attention::Vsa { sparsity: 0.5, tile: 64, ref_sparsity: 0.5, qk8: true, pv8: false }, WeightQuant::Native, false);
    dit_check(p, Attention::Vsa { sparsity: 0.5, tile: 64, ref_sparsity: 0.5, qk8: true, pv8: true }, WeightQuant::Native, false);
    vae_check(p);
    audio_check(p);
    tae_check(p);
    upscaler_check(p);
}

fn dit_forward<B: Backend>(be: &B, attention: Attention, quant: WeightQuant, gate: bool, stream: bool) -> Vec<f32> {
    let cfg = H3Config::tiny_vsa();
    let src = Synthetic { seed: 11 };
    // Gate presence: the synthetic source answers every name, so the loader
    // sees gates; a gate-free source hides them.
    struct NoGate(Synthetic);
    impl h3_core::weights::WeightSource for NoGate {
        fn get(&self, n: &str, s: &[usize]) -> h3_core::Result<Vec<f32>> {
            self.0.get(n, s)
        }
        fn contains(&self, n: &str) -> bool {
            !n.contains("to_gate_compress") && self.0.contains(n)
        }
    }
    let src_dyn: Box<dyn h3_core::weights::WeightSource> = if gate { Box::new(src) } else { Box::new(NoGate(src)) };
    let dit = H3Dit::load(be, cfg.clone(), src_dyn.as_ref(), &LoadOptions { quant, stream_blocks: stream }, &[]).unwrap();
    // A small FL2VA layout: 6 text rows (one vision tag), one first keyframe,
    // 3×4×6 latent, 5 audio latents.
    let text = vec![TEXT_TAG, TEXT_TAG, 0, TEXT_TAG, TEXT_TAG, TEXT_TAG];
    let layout = packing::build_layout(&text, 3, 4, 6, 5, cfg.patch, &[packing::Anchor::First]).unwrap();
    let mut rng = Rng::new(5);
    let embeds = rng.normal(text.len() * cfg.text_dim);
    let refined = dit.refine_text(be, &embeds);
    let ws = dit.workspace(be, &layout, 7, attention).unwrap();
    let tables = dit.adaln_tables(be, src_dyn.as_ref(), &[], &[0.25, 0.7, 0.999, 1.0], None).unwrap();
    let (unique, inverse) = packing::row_timesteps(&layout, 0.25, 0.7, 0.999, 1.0);
    let mods = dit.step_mods(be, &tables, &unique, &inverse, &layout.token_tags, false).unwrap();
    let video = rng.normal(layout.video_indices.len() * cfg.video_patch_dim());
    let audio = rng.normal(layout.audio_indices.len() * cfg.audio_in_channels);
    let (gv, ga) = (be.upload(&video), be.upload(&audio));
    let vo = be.alloc(video.len());
    let ao = be.alloc(audio.len());
    dit.forward(be, &ws, &gv, &ga, &refined, &mods, None, &vo, &ao);
    let mut out = be.download(&vo);
    out.extend(be.download(&ao));
    out
}

fn dit_check<B: Backend>(p: &mut Pair<'_, B>, attention: Attention, quant: WeightQuant, gate: bool) {
    let got = dit_forward(p.be, attention, quant, gate, false);
    let want = dit_forward(&Cpu, attention, quant, gate, false);
    p.record(format!("dit tiny {attention:?} {quant}"), got, want.clone());
    // Streamed blocks (`--stream-blocks`): the same forward with every block's
    // weights uploaded per step through the backend's prefetch path.
    let streamed = dit_forward(p.be, attention, quant, gate, true);
    p.record(format!("dit tiny {attention:?} {quant} streamed"), streamed, want);
}

fn vae_check<B: Backend>(p: &mut Pair<'_, B>) {
    let run = |be: &dyn ErasedVae| be.run();
    trait ErasedVae {
        fn run(&self) -> Vec<f32>;
    }
    struct V<'a, B: Backend>(&'a B);
    impl<B: Backend> ErasedVae for V<'_, B> {
        fn run(&self) -> Vec<f32> {
            let be = self.0;
            let cfg = VideoVaeConfig::tiny();
            let vae = VideoVae::load(be, cfg, &Synthetic { seed: 2 }, true, true).unwrap();
            let mut r = Rng::new(3);
            let px = Volume::from_data(3, 17, 32, 48, r.uniform(3 * 17 * 32 * 48, 0.5).iter().map(|v| v + 0.5).collect());
            let m = vae.encode_pixels(be, &px, None).unwrap();
            let z = Volume::from_data(4, 2, 2, 3, r.normal(4 * 2 * 2 * 3));
            let mut frames = Vec::new();
            vae.decode(be, &z, None, &mut |v| frames.push(v.clone())).unwrap();
            let mut out = m.data;
            out.extend(Volume::cat_t(&frames).data);
            out
        }
    }
    let got = run(&V(p.be));
    let want = run(&V(&Cpu));
    p.record("video vae tiny encode+decode", got, want);
}

fn audio_check<B: Backend>(p: &mut Pair<'_, B>) {
    fn run<B: Backend>(be: &B) -> Vec<f32> {
        let cfg = AudioVaeConfig::tiny();
        let vae = AudioVae::load(be, cfg.clone(), &Synthetic { seed: 9 }, true, true).unwrap();
        let wave: Vec<f32> = (0..95).map(|i| ((i as f32) * 0.3).sin() * 0.5).collect();
        let post = vae.encode(be, &wave).unwrap();
        let mut z = post.mode();
        let mut out = z.data.clone();
        vae.normalize_latents(&mut z);
        let mut r = Rng::new(4);
        let z2 = AudioLatents { channels: cfg.latent_channels, frames: 6, data: r.normal(cfg.latent_channels * 6) };
        out.extend(vae.decode(be, &z2).unwrap());
        out
    }
    p.record("audio vae tiny encode+decode", run(p.be), run(&Cpu));
}

fn tae_check<B: Backend>(p: &mut Pair<'_, B>) {
    fn run<B: Backend>(be: &B) -> Vec<f32> {
        let cfg = TaeH3Config::tiny();
        let tae = TaeH3::load(be, cfg.clone(), &Synthetic { seed: 3 }).unwrap();
        let (h, w) = (2, 3);
        let z: Vec<f32> = (0..7 * cfg.latent_channels * h * w).map(|i| ((i * 31) % 17) as f32 * 0.1 - 0.8).collect();
        let mut st = tae.state();
        let per = cfg.latent_channels * h * w;
        let a = tae.decode_chunk(be, &mut st, &z[..5 * per], 5, h, w);
        let b = tae.decode_chunk(be, &mut st, &z[5 * per..], 2, h, w);
        [a.data, b.data].concat()
    }
    p.record("taeh3 tiny", run(p.be), run(&Cpu));
}

fn upscaler_check<B: Backend>(p: &mut Pair<'_, B>) {
    fn run<B: Backend>(be: &B) -> Vec<f32> {
        let up = Upscaler::load(be, UpscalerConfig::tiny(), &Synthetic { seed: 7 }).unwrap();
        let x = Volume::from_data(4, 2, 4, 4, (0..128).map(|i| (i % 7) as f32 * 0.1).collect());
        up.upscale(be, &x, (96, 128)).unwrap().data
    }
    p.record("upscaler tiny", run(p.be), run(&Cpu));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_agrees_with_itself() {
        let res = run_all(&Cpu, 1e-5);
        for r in &res {
            assert!(r.ok, "{} rel_err {}", r.name, r.rel_err);
        }
        assert!(res.len() > 60);
    }
}
