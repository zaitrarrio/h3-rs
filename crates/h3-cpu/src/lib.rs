//! The CPU reference backend: plain f32, one obvious loop per op, rayon over
//! rows. It is the ground truth every GPU kernel is compared against
//! (`h3 check`), so clarity beats speed here — but it runs the tiny graphs in
//! unit tests and a short low-resolution clip end to end.

use h3_core::backend::{
    Act, Backend, BoxRows, Conv1dShape, Conv3dShape, ConvEpilogue, ConvShape, ConvT1dShape,
    FlowStep, Gate, HeadNorm, Modulation, Pad, Rope, Snake,
};
use h3_core::quant::{Mxfp8, Nvfp4, Q4, Q8};
use h3_core::vsa::{VsaSpec, select};
use rayon::prelude::*;
use std::sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard};

#[derive(Clone, Default)]
pub struct CpuBuf(Arc<RwLock<Vec<f32>>>);

impl CpuBuf {
    fn r(&self) -> RwLockReadGuard<'_, Vec<f32>> {
        self.0
            .try_read()
            .expect("buffer aliased with a written output")
    }
    fn w(&self) -> RwLockWriteGuard<'_, Vec<f32>> {
        self.0.try_write().expect("output buffer aliases an input")
    }
}

pub struct CpuWeight {
    pub data: Vec<f32>,
    pub rows: usize,
    pub cols: usize,
    /// MXFP8 W8A8: `linear` rounds its activations through E4M3 with E8M0
    /// block scales ([`Mxfp8::round_rows`]) before the f32 GEMM.
    pub mx: bool,
}

#[derive(Default)]
pub struct Cpu;

pub fn gelu_tanh(x: f32) -> f32 {
    const C: f32 = 0.797_884_6; // sqrt(2/pi)
    0.5 * x * (1.0 + (C * (x + 0.044715 * x * x * x)).tanh())
}

/// Abramowitz–Stegun 7.1.26 is not enough for a reference; use the erf of
/// libm through `f64`.
pub fn gelu_erf(x: f32) -> f32 {
    let z = x as f64 / std::f64::consts::SQRT_2;
    (0.5 * x as f64 * (1.0 + erf(z))) as f32
}

/// erf with ~1e-12 accuracy (W. J. Cody's rational approximations as used by
/// libm; sufficient for an f32 reference).
pub fn erf(x: f64) -> f64 {
    // Series for small |x|, continued fraction via erfc for large |x|.
    let ax = x.abs();
    if ax < 2.0 {
        // Taylor series: erf(x) = 2/√π Σ (-1)^n x^(2n+1) / (n! (2n+1))
        let mut term = ax;
        let mut sum = ax;
        let x2 = ax * ax;
        let mut n = 0.0;
        while term.abs() > 1e-17 * sum.abs() {
            n += 1.0;
            term *= -x2 / n;
            sum += term / (2.0 * n + 1.0);
            if n > 200.0 {
                break;
            }
        }
        (2.0 / std::f64::consts::PI.sqrt() * sum).copysign(x)
    } else {
        // erfc continued fraction (Lentz), accurate for x ≥ 2.
        let mut f = ax;
        let mut c = ax;
        let mut d = 0.0;
        let tiny = 1e-300;
        for k in 1..200 {
            let a = k as f64 / 2.0;
            d = ax + a * d;
            d = if d.abs() < tiny { tiny } else { 1.0 / d };
            c = ax + a / c;
            if c.abs() < tiny {
                c = tiny;
            }
            let delta = c * d;
            f *= delta;
            if (delta - 1.0).abs() < 1e-16 {
                break;
            }
        }
        let erfc = (-ax * ax).exp() / (f * std::f64::consts::PI.sqrt());
        (1.0 - erfc).copysign(x)
    }
}

pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

pub fn act(a: Act, x: f32) -> f32 {
    match a {
        Act::None => x,
        Act::GeluTanh => gelu_tanh(x),
        Act::GeluErf => gelu_erf(x),
        Act::Silu => silu(x),
    }
}

fn rms_row(
    src: &[f32],
    dst: &mut [f32],
    w: Option<&[f32]>,
    m: Option<(&[f32], usize, usize, usize)>,
    eps: f32,
) {
    let dim = src.len();
    let ms = src.iter().map(|&v| v as f64 * v as f64).sum::<f64>() / dim as f64;
    let inv = (1.0 / (ms + eps as f64).sqrt()) as f32;
    for j in 0..dim {
        let mut y = src[j] * inv;
        if let Some(w) = w {
            y *= w[j];
        }
        dst[j] = y;
    }
    if let Some((table, base, scale, shift)) = m {
        let sc = &table[base + scale * dim..][..dim];
        let sh = &table[base + shift * dim..][..dim];
        for j in 0..dim {
            dst[j] = dst[j] * (1.0 + sc[j]) + sh[j];
        }
    }
}

/// Resolved modulation: table, idx, slots, scale, shift.
struct ModView<'a> {
    table: RwLockReadGuard<'a, Vec<f32>>,
    idx: RwLockReadGuard<'a, Vec<f32>>,
    slots: usize,
    scale: usize,
    shift: usize,
}

impl ModView<'_> {
    fn of<'a>(m: &Modulation<'a, CpuBuf>) -> Option<ModView<'a>> {
        match m {
            Modulation::None => None,
            Modulation::Table { m, scale, shift } => Some(ModView {
                table: m.table.r(),
                idx: m.idx.r(),
                slots: m.slots,
                scale: *scale,
                shift: *shift,
            }),
        }
    }
    fn row(&self, r: usize, dim: usize) -> (&[f32], usize, usize, usize) {
        let e = self.idx[r].to_bits() as usize;
        (&self.table, e * self.slots * dim, self.scale, self.shift)
    }
}

/// u32 side tables are stored bit-for-bit in f32 buffers.
fn u32_to_f32_bits(v: &[u32]) -> Vec<f32> {
    v.iter().map(|&x| f32::from_bits(x)).collect()
}

struct GateView<'a> {
    kind: GateKind<'a>,
}
enum GateKind<'a> {
    None,
    Vector(RwLockReadGuard<'a, Vec<f32>>),
    Table {
        table: RwLockReadGuard<'a, Vec<f32>>,
        idx: RwLockReadGuard<'a, Vec<f32>>,
        slots: usize,
        slot: usize,
    },
}

impl GateView<'_> {
    fn of<'a>(g: &Gate<'a, CpuBuf>) -> GateView<'a> {
        GateView {
            kind: match g {
                Gate::None => GateKind::None,
                Gate::Vector(v) => GateKind::Vector(v.r()),
                Gate::Table(m) => GateKind::Table {
                    table: m.table.table.r(),
                    idx: m.table.idx.r(),
                    slots: m.table.slots,
                    slot: m.slot,
                },
            },
        }
    }
    fn apply(&self, r: usize, dim: usize, h: &mut [f32], a: &[f32]) {
        match &self.kind {
            GateKind::None => h.iter_mut().zip(a).for_each(|(x, y)| *x += y),
            GateKind::Vector(g) => h
                .iter_mut()
                .zip(a)
                .zip(g.iter())
                .for_each(|((x, y), g)| *x += y * g),
            GateKind::Table {
                table,
                idx,
                slots,
                slot,
            } => {
                let e = idx[r].to_bits() as usize;
                let g = &table[(e * slots + slot) * dim..][..dim];
                h.iter_mut()
                    .zip(a)
                    .zip(g)
                    .for_each(|((x, y), g)| *x += y * g);
            }
        }
    }
}

/// VSA-H3 reference: pooled means in f64, per-region top-k by score then
/// lower tile index, fine attention with an online softmax, the gated pooled
/// branch added last.
/// The dense attention reference: softmax(q·kᵀ/√d)·v per head over `[heads][sq][d]`
/// / `[kv_heads][sk][d]` inputs, output `[sq][heads][d]`; GQA by `heads / kv_heads`.
#[allow(clippy::too_many_arguments)]
pub fn attention_reference_into(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    heads: usize,
    kv_heads: usize,
    sq: usize,
    sk: usize,
    d: usize,
    causal: bool,
    out: &mut [f32],
) {
    let scale = 1.0 / (d as f32).sqrt();
    let group = heads / kv_heads;
    out[..sq * heads * d]
        .par_chunks_mut(heads * d)
        .enumerate()
        .for_each(|(i, orow)| {
            let mut p = vec![0.0f32; sk];
            let limit = if causal {
                (i + 1 + sk - sq).min(sk)
            } else {
                sk
            };
            for h in 0..heads {
                let kh = h / group;
                let qi = &q[(h * sq + i) * d..][..d];
                let mut mx = f32::NEG_INFINITY;
                for j in 0..limit {
                    let kj = &k[(kh * sk + j) * d..][..d];
                    let s = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale;
                    p[j] = s;
                    mx = mx.max(s);
                }
                let mut den = 0.0f64;
                for pj in p[..limit].iter_mut() {
                    *pj = (*pj - mx).exp();
                    den += *pj as f64;
                }
                let o = &mut orow[h * d..][..d];
                let mut acc = vec![0.0f64; d];
                for (j, &pj) in p[..limit].iter().enumerate() {
                    let vj = &v[(kh * sk + j) * d..][..d];
                    for c in 0..d {
                        acc[c] += pj as f64 * vj[c] as f64;
                    }
                }
                for c in 0..d {
                    o[c] = (acc[c] / den) as f32;
                }
            }
        });
}

pub fn vsa_reference(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: Option<&[f32]>,
    spec: &VsaSpec,
    out: &mut [f32],
) {
    vsa_reference_impl(q, k, v, gate, spec, out, None);
}

/// [`vsa_reference`] that also fills `slack` (same shape as `out`) with the
/// FP8 P·V rounding slack of `h3_core::vsa::attention_pv8_reference_slack`
/// over the selected keys; zero without `spec.pv8`.
pub fn vsa_reference_slack(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: Option<&[f32]>,
    spec: &VsaSpec,
    out: &mut [f32],
    slack: &mut [f32],
) {
    vsa_reference_impl(q, k, v, gate, spec, out, Some(slack));
}

fn vsa_reference_impl(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    gate: Option<&[f32]>,
    spec: &VsaSpec,
    out: &mut [f32],
    mut slack: Option<&mut [f32]>,
) {
    let (plan, heads, d) = (&spec.plan, spec.heads, spec.head_dim);
    let want_slack = slack.is_some();
    let (seq, nb, te) = (plan.seq, plan.num_tiles(), plan.tile_elems);
    let scale = spec.scale as f64;
    // INT8 Q·Kᵀ: the fine stage sees the rounded Q / shifted-and-rounded K;
    // the tile means, selection and gated branch stay on the exact values.
    let rounded = spec.qk8.then(|| h3_core::vsa::qk8_round(q, k, spec));
    let (qf, kf): (&[f32], &[f32]) = match &rounded {
        Some((qr, kr)) => (qr, kr),
        None => (q, k),
    };
    // FP8 P·V: V − mean(V) as E4M3 codes with one scale per head; the fine
    // stage rounds P to E4M3 at 64× with the row maximum rounded up to an
    // integer (h3_core::vsa::attention_pv8_reference's recipe, over the
    // selected tiles).
    let v8 = spec
        .pv8
        .then(|| h3_core::vsa::attn_v8_round(v, heads, seq, d));
    // A dense plan (every tile kept: dense layers, the dense first steps of a
    // run) is dense attention over the sequence in its natural order plus the
    // gated pooled branch; the 8-bit recipes are the dense ones
    // (`attn_qk8_round`, `attention_pv8_reference`), not the tile-grouped
    // scales of the fine stage, so a backend may run its dense kernel here.
    let dense_out: Option<(Vec<f32>, Option<Vec<f32>>)> = plan.is_dense().then(|| {
        if spec.pv8 {
            if want_slack {
                let (o, sl) = h3_core::vsa::attention_pv8_reference_slack(
                    q, k, v, heads, heads, seq, seq, d, false,
                );
                (o, Some(sl))
            } else {
                (
                    h3_core::vsa::attention_pv8_reference(
                        q, k, v, heads, heads, seq, seq, d, false,
                    ),
                    None,
                )
            }
        } else {
            let mut o = vec![0f32; seq * heads * d];
            if spec.qk8 {
                let (q8, k8) = h3_core::vsa::attn_qk8_round(q, k, heads, heads, seq, seq, d);
                attention_reference_into(&q8, &k8, v, heads, heads, seq, seq, d, false, &mut o);
            } else {
                attention_reference_into(q, k, v, heads, heads, seq, seq, d, false, &mut o);
            }
            (o, None)
        }
    });
    let per_head: Vec<Vec<(usize, Vec<f32>, Option<Vec<f32>>)>> = (0..heads)
        .into_par_iter()
        .map(|h| {
            fn slice<'b>(b: &'b [f32], off: usize, d: usize) -> &'b [f32] {
                &b[off..off + d]
            }
            let row =
                |b: &[f32], tok: usize| -> Vec<f32> { slice(b, (h * seq + tok) * d, d).to_vec() };
            // 1. Tile means over real tokens.
            let mut mean = vec![vec![0f64; nb * d]; 3];
            for t in 0..nb {
                let n = plan.block_sizes[t] as f64;
                for s in 0..plan.block_sizes[t] as usize {
                    let tok = plan.slot_src[t * te + s] as usize;
                    for (m, src) in mean.iter_mut().zip([q, k, v]) {
                        for (c, x) in row(src, tok).iter().enumerate() {
                            m[t * d + c] += *x as f64;
                        }
                    }
                }
                for m in mean.iter_mut() {
                    for c in 0..d {
                        m[t * d + c] /= n;
                    }
                }
            }
            let (qc, kc, vc) = (&mean[0], &mean[1], &mean[2]);
            // 2. Scores, 3. selection, gated pooled branch.
            let mut pooled = vec![0f32; nb * d];
            let mut keep = vec![Vec::new(); nb];
            for qt in 0..nb {
                let s: Vec<f32> = (0..nb)
                    .map(|kt| {
                        ((0..d).map(|c| qc[qt * d + c] * kc[kt * d + c]).sum::<f64>() * scale)
                            as f32
                    })
                    .collect();
                if gate.is_some() {
                    let m = s.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let e: Vec<f64> = s.iter().map(|x| ((x - m) as f64).exp()).collect();
                    let z: f64 = e.iter().sum();
                    for (kt, w) in e.iter().enumerate() {
                        for c in 0..d {
                            pooled[qt * d + c] += (w / z * vc[kt * d + c]) as f32;
                        }
                    }
                }
                let sel = select(plan, qt, &s);
                keep[qt] = sel
                    .iter()
                    .enumerate()
                    .filter(|&(_, &k)| k)
                    .map(|(i, _)| i)
                    .collect();
            }
            // 4. Fine attention, 5. combine.
            let mut rows: Vec<(usize, Vec<f32>, Option<Vec<f32>>)> = Vec::with_capacity(seq);
            if let Some((dense, dslack)) = &dense_out {
                for qt in 0..nb {
                    for qs in 0..plan.block_sizes[qt] as usize {
                        let qtok = plan.slot_src[qt * te + qs] as usize;
                        let base = (qtok * heads + h) * d;
                        let mut o = dense[base..base + d].to_vec();
                        if let Some(g) = gate {
                            let gr = row(g, qtok);
                            for c in 0..d {
                                o[c] += pooled[qt * d + c] * gr[c];
                            }
                        }
                        rows.push((
                            qtok,
                            o,
                            dslack.as_ref().map(|sl| sl[base..base + d].to_vec()),
                        ));
                    }
                }
                return rows;
            }
            for qt in 0..nb {
                for qs in 0..plan.block_sizes[qt] as usize {
                    let qtok = plan.slot_src[qt * te + qs] as usize;
                    let qr = row(qf, qtok);
                    let (mut m, mut z) = (f32::NEG_INFINITY, 0f64);
                    let mut acc = vec![0f64; d];
                    if let Some((codes, vmean, vs)) = &v8 {
                        let sl2 = scale as f32 * std::f32::consts::LOG2_E;
                        let mut s2 = Vec::new();
                        let mut mx = f32::NEG_INFINITY;
                        for &kt in &keep[qt] {
                            for ks in 0..plan.block_sizes[kt] as usize {
                                let ktok = plan.slot_src[kt * te + ks] as usize;
                                let kr = row(kf, ktok);
                                let s = (0..d).map(|c| qr[c] * kr[c]).sum::<f32>() * sl2;
                                mx = mx.max(s);
                                s2.push((ktok, s));
                            }
                        }
                        let mi = mx.ceil();
                        let mut l = 0f32;
                        let mut a = vec![0f32; d];
                        let mut sl = want_slack.then(|| vec![0f32; d]);
                        for &(ktok, s) in &s2 {
                            let pr = (s - mi + h3_core::vsa::PV8_SHIFT).exp2();
                            l += pr;
                            let p8 =
                                h3_core::quant::e4m3_decode(h3_core::quant::e4m3_satfinite(pr));
                            let vr = &codes[(h * seq + ktok) * d..][..d];
                            for c in 0..d {
                                a[c] += p8 * vr[c];
                            }
                            if let Some(sl) = sl.as_mut() {
                                let w =
                                    h3_core::vsa::e4m3_flip_slack(pr, h3_core::vsa::PV8_FLIP_EPS);
                                if w > 0.0 {
                                    for c in 0..d {
                                        sl[c] += w * vr[c].abs();
                                    }
                                }
                            }
                        }
                        let inv = if l > 0.0 { vs[h] / l } else { 0.0 };
                        let mut o: Vec<f32> =
                            (0..d).map(|c| a[c] * inv + vmean[h * d + c]).collect();
                        if let Some(g) = gate {
                            let gr = row(g, qtok);
                            for c in 0..d {
                                o[c] += pooled[qt * d + c] * gr[c];
                            }
                        }
                        if let Some(sl) = sl.as_mut() {
                            sl.iter_mut().for_each(|x| *x *= inv);
                        }
                        rows.push((qtok, o, sl));
                        continue;
                    }
                    for &kt in &keep[qt] {
                        for ks in 0..plan.block_sizes[kt] as usize {
                            let ktok = plan.slot_src[kt * te + ks] as usize;
                            let kr = row(kf, ktok);
                            let s = ((0..d).map(|c| qr[c] as f64 * kr[c] as f64).sum::<f64>()
                                * scale) as f32;
                            if s > m {
                                let shrink = if m.is_finite() {
                                    ((m - s) as f64).exp()
                                } else {
                                    0.0
                                };
                                z *= shrink;
                                acc.iter_mut().for_each(|a| *a *= shrink);
                                m = s;
                            }
                            let e = ((s - m) as f64).exp();
                            z += e;
                            let vr = row(v, ktok);
                            for c in 0..d {
                                acc[c] += e * vr[c] as f64;
                            }
                        }
                    }
                    let mut o: Vec<f32> = (0..d).map(|c| (acc[c] / z) as f32).collect();
                    if let Some(g) = gate {
                        let gr = row(g, qtok);
                        for c in 0..d {
                            o[c] += pooled[qt * d + c] * gr[c];
                        }
                    }
                    rows.push((qtok, o, None));
                }
            }
            rows
        })
        .collect();
    for (h, rows) in per_head.into_iter().enumerate() {
        for (tok, o, sl) in rows {
            out[(tok * heads + h) * d..][..d].copy_from_slice(&o);
            if let (Some(sl), Some(slack)) = (sl, slack.as_deref_mut()) {
                slack[(tok * heads + h) * d..][..d].copy_from_slice(&sl);
            }
        }
    }
}

/// Reflect / zero padding index: `None` for a zero.
#[inline]
fn pad_index(p: isize, n: usize, mode: Pad) -> Option<usize> {
    if p >= 0 && (p as usize) < n {
        return Some(p as usize);
    }
    match mode {
        Pad::Zero => None,
        Pad::Reflect => {
            let n = n as isize;
            let mut q = p;
            // PyTorch reflect (no edge repeat); pads smaller than the dim.
            while q < 0 || q >= n {
                if q < 0 {
                    q = -q;
                }
                if q >= n {
                    q = 2 * (n - 1) - q;
                }
            }
            Some(q as usize)
        }
    }
}

/// `i` indexes the output (the residual), `ch` is its output channel (the
/// PReLU slope; `conv2d` only).
fn epilogue(
    ep: &ConvEpilogue<'_, CpuBuf>,
    res: Option<&RwLockReadGuard<'_, Vec<f32>>>,
    i: usize,
    ch: usize,
    acc: f32,
) -> f32 {
    match ep {
        ConvEpilogue::None => acc,
        ConvEpilogue::Relu => acc.max(0.0),
        ConvEpilogue::AddRelu(_) => (acc + res.unwrap()[i]).max(0.0),
        ConvEpilogue::Add(_) => acc + res.unwrap()[i],
        ConvEpilogue::Prelu(_) => {
            if acc >= 0.0 {
                acc
            } else {
                res.unwrap()[ch] * acc
            }
        }
    }
}

fn residual_of<'a>(ep: &ConvEpilogue<'a, CpuBuf>) -> Option<RwLockReadGuard<'a, Vec<f32>>> {
    match ep {
        ConvEpilogue::AddRelu(r) | ConvEpilogue::Add(r) | ConvEpilogue::Prelu(r) => Some(r.r()),
        _ => None,
    }
}

impl Backend for Cpu {
    type Buf = CpuBuf;
    type Vsa = VsaSpec;
    type Weight = CpuWeight;

    fn name(&self) -> String {
        format!("cpu (rayon, {} threads)", rayon::current_num_threads())
    }

    fn alloc(&self, len: usize) -> CpuBuf {
        CpuBuf(Arc::new(RwLock::new(vec![0.0; len])))
    }
    fn upload(&self, data: &[f32]) -> CpuBuf {
        CpuBuf(Arc::new(RwLock::new(data.to_vec())))
    }
    fn upload_u32(&self, data: &[u32]) -> CpuBuf {
        self.upload(&u32_to_f32_bits(data))
    }
    fn write(&self, buf: &CpuBuf, data: &[f32]) {
        buf.w()[..data.len()].copy_from_slice(data);
    }
    fn download(&self, buf: &CpuBuf) -> Vec<f32> {
        buf.r().clone()
    }
    fn buf_len(&self, buf: &CpuBuf) -> usize {
        buf.r().len()
    }
    fn box_rows(&self, grid: &CpuBuf, packed: &CpuBuf, b: &BoxRows, scatter: bool) {
        let n = b.heads * b.rows() * b.width;
        if scatter {
            let src = packed.r();
            let mut dst = grid.w();
            for i in 0..n {
                let (g, p) = b.index(i);
                dst[g] = src[p];
            }
        } else {
            let src = grid.r();
            let mut dst = packed.w();
            for i in 0..n {
                let (g, p) = b.index(i);
                dst[p] = src[g];
            }
        }
    }

    fn copy(&self, src: &CpuBuf, so: usize, dst: &CpuBuf, d: usize, len: usize) {
        if Arc::ptr_eq(&src.0, &dst.0) {
            let mut b = dst.w();
            let tmp = b[so..so + len].to_vec();
            b[d..d + len].copy_from_slice(&tmp);
            return;
        }
        let s = src.r();
        dst.w()[d..d + len].copy_from_slice(&s[so..so + len]);
    }
    fn fill(&self, buf: &CpuBuf, off: usize, len: usize, value: f32) {
        buf.w()[off..off + len].fill(value);
    }
    fn weight(&self, data: &[f32], rows: usize, cols: usize) -> CpuWeight {
        assert_eq!(data.len(), rows * cols);
        CpuWeight {
            data: data.to_vec(),
            rows,
            cols,
            mx: false,
        }
    }
    fn weight_q8(&self, q: &Q8) -> CpuWeight {
        self.weight(&q.dequantize(), q.rows, q.cols)
    }
    fn weight_q4(&self, q: &Q4) -> CpuWeight {
        self.weight(&q.dequantize(), q.rows, q.cols)
    }
    fn weight_nvfp4(&self, q: &Nvfp4) -> CpuWeight {
        self.weight(&q.dequantize(), q.rows, q.cols)
    }
    fn weight_mxfp8(&self, q: &Mxfp8) -> CpuWeight {
        CpuWeight {
            data: q.dequantize(),
            rows: q.rows,
            cols: q.cols,
            mx: true,
        }
    }
    fn weight_bytes(&self, w: &CpuWeight) -> usize {
        w.data.len() * 4
    }
    fn sync(&self) {}

    fn linear(
        &self,
        x: &CpuBuf,
        x_row: usize,
        rows: usize,
        w: &CpuWeight,
        bias: Option<&CpuBuf>,
        a: Act,
        y: &CpuBuf,
        y_row: usize,
    ) {
        let (k, n) = (w.cols, w.rows);
        let x = x.r();
        let b = bias.map(|b| b.r());
        let mut y = y.w();
        let x = &x[x_row * k..(x_row + rows) * k];
        let xq;
        let x = if w.mx {
            xq = Mxfp8::round_rows(x, k);
            &xq[..]
        } else {
            x
        };
        // Rows × 4 output columns per task keeps the weight row in cache.
        y[y_row * n..(y_row + rows) * n]
            .par_chunks_mut(n)
            .zip(x.par_chunks(k))
            .for_each(|(yr, xr)| {
                for (o, out) in yr.iter_mut().enumerate() {
                    let wr = &w.data[o * k..(o + 1) * k];
                    let mut acc = [0.0f32; 8];
                    let mut i = 0;
                    while i + 8 <= k {
                        for j in 0..8 {
                            acc[j] += xr[i + j] * wr[i + j];
                        }
                        i += 8;
                    }
                    let mut s: f32 = acc.iter().sum();
                    while i < k {
                        s += xr[i] * wr[i];
                        i += 1;
                    }
                    if let Some(b) = &b {
                        s += b[o];
                    }
                    *out = act(a, s);
                }
            });
    }

    fn swiglu(&self, x: &CpuBuf, rows: usize, dim: usize, out: &CpuBuf) {
        let x = x.r();
        let mut o = out.w();
        o[..rows * dim]
            .par_chunks_mut(dim)
            .zip(x[..rows * 2 * dim].par_chunks(2 * dim))
            .for_each(|(o, xr)| {
                for j in 0..dim {
                    o[j] = xr[j] * silu(xr[dim + j]);
                }
            });
    }

    fn rms_norm(
        &self,
        x: &CpuBuf,
        rows: usize,
        dim: usize,
        w: Option<&CpuBuf>,
        modulation: Modulation<'_, CpuBuf>,
        eps: f32,
        out: &CpuBuf,
    ) {
        let x = x.r();
        let w = w.map(|w| w.r());
        let m = ModView::of(&modulation);
        let mut o = out.w();
        o[..rows * dim]
            .par_chunks_mut(dim)
            .zip(x[..rows * dim].par_chunks(dim))
            .enumerate()
            .for_each(|(r, (o, xr))| {
                rms_row(
                    xr,
                    o,
                    w.as_deref().map(|v| &v[..]),
                    m.as_ref().map(|m| m.row(r, dim)),
                    eps,
                );
            });
    }

    fn residual_rms_norm(
        &self,
        h: &CpuBuf,
        a: &CpuBuf,
        rows: usize,
        dim: usize,
        gate: Gate<'_, CpuBuf>,
        w: Option<&CpuBuf>,
        modulation: Modulation<'_, CpuBuf>,
        eps: f32,
        normed: &CpuBuf,
    ) {
        let a = a.r();
        let g = GateView::of(&gate);
        let w = w.map(|w| w.r());
        let m = ModView::of(&modulation);
        let mut hh = h.w();
        let mut o = normed.w();
        hh[..rows * dim]
            .par_chunks_mut(dim)
            .zip(a[..rows * dim].par_chunks(dim))
            .zip(o[..rows * dim].par_chunks_mut(dim))
            .enumerate()
            .for_each(|(r, ((hr, ar), or))| {
                g.apply(r, dim, hr, ar);
                rms_row(
                    hr,
                    or,
                    w.as_deref().map(|v| &v[..]),
                    m.as_ref().map(|m| m.row(r, dim)),
                    eps,
                );
            });
    }

    fn gate_residual(
        &self,
        h: &CpuBuf,
        a: &CpuBuf,
        rows: usize,
        dim: usize,
        gate: Gate<'_, CpuBuf>,
    ) {
        let a = a.r();
        let g = GateView::of(&gate);
        let mut hh = h.w();
        hh[..rows * dim]
            .par_chunks_mut(dim)
            .zip(a[..rows * dim].par_chunks(dim))
            .enumerate()
            .for_each(|(r, (hr, ar))| {
                g.apply(r, dim, hr, ar);
            });
    }

    fn layer_norm(
        &self,
        x: &CpuBuf,
        rows: usize,
        dim: usize,
        w: Option<&CpuBuf>,
        b: Option<&CpuBuf>,
        eps: f32,
        out: &CpuBuf,
    ) {
        let x = x.r();
        let w = w.map(|w| w.r());
        let b = b.map(|b| b.r());
        let mut o = out.w();
        o[..rows * dim]
            .par_chunks_mut(dim)
            .zip(x[..rows * dim].par_chunks(dim))
            .for_each(|(o, xr)| {
                let n = dim as f64;
                let mean = xr.iter().map(|&v| v as f64).sum::<f64>() / n;
                let var = xr
                    .iter()
                    .map(|&v| (v as f64 - mean) * (v as f64 - mean))
                    .sum::<f64>()
                    / n;
                let inv = 1.0 / (var + eps as f64).sqrt();
                for j in 0..dim {
                    let mut y = ((xr[j] as f64 - mean) * inv) as f32;
                    if let Some(w) = &w {
                        y *= w[j];
                    }
                    if let Some(b) = &b {
                        y += b[j];
                    }
                    o[j] = y;
                }
            });
    }

    fn head_split(
        &self,
        src: &CpuBuf,
        rows: usize,
        stride: usize,
        col_off: usize,
        heads: usize,
        hd: usize,
        norm: Option<HeadNorm<'_, CpuBuf>>,
        rope: Option<Rope<'_, CpuBuf>>,
        out: &CpuBuf,
    ) {
        let src = src.r();
        let nw = norm.and_then(|n| n.w.map(|w| w.r()));
        let eps = norm.map(|n| n.eps);
        let rp = rope.map(|r| (r.cos.r(), r.sin.r(), r.rot));
        let width = heads * hd;
        let mut out = out.w();
        let rows_out: Vec<Vec<f32>> = (0..rows)
            .into_par_iter()
            .map(|r| {
                let x = &src[r * stride + col_off..][..width];
                let mut v = x.to_vec();
                if let Some(eps) = eps {
                    for h in 0..heads {
                        let xh = &x[h * hd..][..hd];
                        let ms = xh.iter().map(|&a| a as f64 * a as f64).sum::<f64>() / hd as f64;
                        let inv = (1.0 / (ms + eps as f64).sqrt()) as f32;
                        for j in 0..hd {
                            v[h * hd + j] = xh[j] * inv * nw.as_ref().map_or(1.0, |w| w[j]);
                        }
                    }
                }
                if let Some((c, s, rot)) = &rp {
                    let half = rot / 2;
                    for h in 0..heads {
                        for i in 0..half {
                            let (cv, sv) = (c[r * half + i], s[r * half + i]);
                            let (x1, x2) = (v[h * hd + i], v[h * hd + half + i]);
                            v[h * hd + i] = x1 * cv - x2 * sv;
                            v[h * hd + half + i] = x2 * cv + x1 * sv;
                        }
                    }
                }
                v
            })
            .collect();
        for (r, v) in rows_out.iter().enumerate() {
            for h in 0..heads {
                out[(h * rows + r) * hd..][..hd].copy_from_slice(&v[h * hd..][..hd]);
            }
        }
    }

    fn attention(
        &self,
        q: &CpuBuf,
        k: &CpuBuf,
        v: &CpuBuf,
        heads: usize,
        kv_heads: usize,
        sq: usize,
        sk: usize,
        d: usize,
        causal: bool,
        out: &CpuBuf,
    ) {
        let (q, k, v) = (q.r(), k.r(), v.r());
        let mut out = out.w();
        attention_reference_into(
            &q,
            &k,
            &v,
            heads,
            kv_heads,
            sq,
            sk,
            d,
            causal,
            &mut out[..sq * heads * d],
        );
    }

    fn attention_qk8(
        &self,
        q: &CpuBuf,
        k: &CpuBuf,
        v: &CpuBuf,
        heads: usize,
        kv_heads: usize,
        sq: usize,
        sk: usize,
        d: usize,
        causal: bool,
        pv8: bool,
        out: &CpuBuf,
    ) {
        if pv8 {
            let o = {
                let (q, k, v) = (q.r(), k.r(), v.r());
                h3_core::vsa::attention_pv8_reference(
                    &q[..heads * sq * d],
                    &k[..kv_heads * sk * d],
                    &v[..kv_heads * sk * d],
                    heads,
                    kv_heads,
                    sq,
                    sk,
                    d,
                    causal,
                )
            };
            out.w()[..sq * heads * d].copy_from_slice(&o);
            return;
        }
        let (q8, k8) = {
            let (q, k) = (q.r(), k.r());
            h3_core::vsa::attn_qk8_round(
                &q[..heads * sq * d],
                &k[..kv_heads * sk * d],
                heads,
                kv_heads,
                sq,
                sk,
                d,
            )
        };
        self.attention(
            &self.upload(&q8),
            &self.upload(&k8),
            v,
            heads,
            kv_heads,
            sq,
            sk,
            d,
            causal,
            out,
        );
    }

    fn vsa_prepare(&self, spec: &VsaSpec) -> VsaSpec {
        spec.clone()
    }

    fn vsa_attention(
        &self,
        q: &CpuBuf,
        k: &CpuBuf,
        v: &CpuBuf,
        gate: Option<&CpuBuf>,
        spec: &VsaSpec,
        out: &CpuBuf,
    ) {
        let (q, k, v) = (q.r(), k.r(), v.r());
        let g = gate.map(|g| g.r());
        vsa_reference(&q, &k, &v, g.as_deref().map(|g| &g[..]), spec, &mut out.w());
    }

    fn patchify(
        &self,
        latent: &CpuBuf,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        (pt, ph, pw): (usize, usize, usize),
        out: &CpuBuf,
        out_row: usize,
    ) {
        let l = latent.r();
        let (tt, hh, ww) = (t / pt, h / ph, w / pw);
        let f = c * pt * ph * pw;
        let mut o = out.w();
        o[out_row * f..(out_row + tt * hh * ww) * f]
            .par_chunks_mut(f)
            .enumerate()
            .for_each(|(row, dst)| {
                let (ti, rem) = (row / (hh * ww), row % (hh * ww));
                let (hi, wi) = (rem / ww, rem % ww);
                for ci in 0..c {
                    for kt in 0..pt {
                        for kh in 0..ph {
                            for kw in 0..pw {
                                let src =
                                    ((ci * t + ti * pt + kt) * h + hi * ph + kh) * w + wi * pw + kw;
                                dst[((ci * pt + kt) * ph + kh) * pw + kw] = l[src];
                            }
                        }
                    }
                }
            });
    }

    fn unpatchify(
        &self,
        x: &CpuBuf,
        x_row: usize,
        x_stride: usize,
        col: usize,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        (pt, ph, pw): (usize, usize, usize),
        out: &CpuBuf,
    ) {
        let xr = x.r();
        let (tt, hh, ww) = (t / pt, h / ph, w / pw);
        let mut o = out.w();
        o[..c * t * h * w]
            .par_chunks_mut(t * h * w)
            .enumerate()
            .for_each(|(ci, plane)| {
                for ti in 0..tt {
                    for hi in 0..hh {
                        for wi in 0..ww {
                            let row = (ti * hh + hi) * ww + wi;
                            let base = (x_row + row) * x_stride + col;
                            for kt in 0..pt {
                                for kh in 0..ph {
                                    for kw in 0..pw {
                                        let feat = ((ci * pt + kt) * ph + kh) * pw + kw;
                                        plane[((ti * pt + kt) * h + hi * ph + kh) * w
                                            + wi * pw
                                            + kw] = xr[base + feat];
                                    }
                                }
                            }
                        }
                    }
                }
            });
    }

    fn flow_step(
        &self,
        x: &CpuBuf,
        off: usize,
        v: &CpuBuf,
        v_off: usize,
        noise: Option<&CpuBuf>,
        n: usize,
        step: FlowStep,
    ) {
        let v = v.r();
        let nz = noise.map(|z| z.r());
        let mut x = x.w();
        let xs = &mut x[off..off + n];
        let vs = &v[v_off..v_off + n];
        match step {
            FlowStep::Euler { sigma, next } => {
                let dt = sigma - next;
                xs.par_iter_mut().zip(vs).for_each(|(x, v)| *x += dt * v);
            }
            FlowStep::Renoise { sigma, next } => {
                let z = nz.expect("renoise needs noise");
                let z = &z[off..off + n];
                xs.par_iter_mut().zip(vs).zip(z).for_each(|((x, v), z)| {
                    let x0 = *x + sigma * v;
                    *x = (1.0 - next) * x0 + next * z;
                });
            }
            FlowStep::Terminal { sigma } => {
                xs.par_iter_mut().zip(vs).for_each(|(x, v)| *x += sigma * v)
            }
        }
    }

    fn axpby(&self, y: &CpuBuf, off: usize, a: f32, x: &CpuBuf, x_off: usize, b: f32, n: usize) {
        if Arc::ptr_eq(&y.0, &x.0) {
            let mut yy = y.w();
            let xs = yy[x_off..x_off + n].to_vec();
            yy[off..off + n]
                .iter_mut()
                .zip(xs)
                .for_each(|(y, x)| *y = a * *y + b * x);
            return;
        }
        let x = x.r();
        let mut yy = y.w();
        yy[off..off + n]
            .par_iter_mut()
            .zip(&x[x_off..x_off + n])
            .for_each(|(y, x)| *y = a * *y + b * x);
    }

    fn channel_affine(
        &self,
        x: &CpuBuf,
        c: usize,
        inner: usize,
        scale: &CpuBuf,
        shift: Option<&CpuBuf>,
        out: &CpuBuf,
    ) {
        let x = x.r();
        let s = scale.r();
        let sh = shift.map(|s| s.r());
        let mut o = out.w();
        o[..c * inner]
            .par_chunks_mut(inner)
            .zip(x[..c * inner].par_chunks(inner))
            .enumerate()
            .for_each(|(ci, (o, xr))| {
                let b = sh.as_ref().map_or(0.0, |v| v[ci]);
                for j in 0..inner {
                    o[j] = xr[j] * s[ci] + b;
                }
            });
    }

    fn conv2d(
        &self,
        a: &CpuBuf,
        b: Option<&CpuBuf>,
        s: &ConvShape,
        w: &CpuWeight,
        bias: Option<&CpuBuf>,
        ep: ConvEpilogue<'_, CpuBuf>,
        out: &CpuBuf,
    ) {
        assert_eq!(w.cols, s.k_len());
        assert_eq!(w.rows, s.cout);
        let a = a.r();
        let b = b.map(|b| b.r());
        let bias = bias.map(|b| b.r());
        let res = residual_of(&ep);
        let (oh, ow) = s.out_hw();
        let up = if s.upsample { 2 } else { 1 };
        let pad = (s.k / 2) as isize;
        let mut o = out.w();
        o[..s.out_len()]
            .par_chunks_mut(oh * ow)
            .enumerate()
            .for_each(|(fc, plane)| {
                let (f, co) = (fc / s.cout, fc % s.cout);
                let wr = &w.data[co * s.k_len()..][..s.k_len()];
                for y in 0..oh {
                    for xo in 0..ow {
                        let mut acc = bias.as_ref().map_or(0.0, |b| b[co]);
                        for ci in 0..s.cin() {
                            let (src, c, cn): (&[f32], usize, usize) = if ci < s.cin_a {
                                (&a, ci, s.cin_a)
                            } else {
                                (b.as_ref().unwrap(), ci - s.cin_a, s.cin_b)
                            };
                            for ky in 0..s.k {
                                let yy = y as isize + ky as isize - pad;
                                if yy < 0 || yy >= oh as isize {
                                    continue;
                                }
                                for kx in 0..s.k {
                                    let xx = xo as isize + kx as isize - pad;
                                    if xx < 0 || xx >= ow as isize {
                                        continue;
                                    }
                                    let (sy, sx) = (yy as usize / up, xx as usize / up);
                                    acc += wr[(ci * s.k + ky) * s.k + kx]
                                        * src[((f * cn + c) * s.h + sy) * s.w + sx];
                                }
                            }
                        }
                        let i = y * ow + xo;
                        plane[i] = epilogue(&ep, res.as_ref(), fc * oh * ow + i, co, acc);
                    }
                }
            });
    }

    fn conv3d(
        &self,
        x: &CpuBuf,
        s: &Conv3dShape,
        w: &CpuWeight,
        bias: Option<&CpuBuf>,
        ep: ConvEpilogue<'_, CpuBuf>,
        out: &CpuBuf,
    ) {
        assert!(
            !matches!(ep, ConvEpilogue::Prelu(_)),
            "PReLU is a conv2d epilogue"
        );
        assert_eq!(w.cols, s.k_len(), "conv3d weight columns");
        assert_eq!(w.rows, s.cout);
        assert_eq!(s.cin % s.groups, 0);
        assert_eq!(s.cout % s.groups, 0);
        let x = x.r();
        let bias = bias.map(|b| b.r());
        let res = residual_of(&ep);
        let (ot, oh, ow) = s.out_dims();
        let cg = s.cin / s.groups;
        let og = s.cout / s.groups;
        let (kt, kh, kw) = s.k;
        let mut o = out.w();
        o[..s.out_len()]
            .par_chunks_mut(oh * ow)
            .enumerate()
            .for_each(|(plane_i, plane)| {
                let (co, to) = (plane_i / ot, plane_i % ot);
                let g = co / og;
                let wr = &w.data[co * s.k_len()..][..s.k_len()];
                for yo in 0..oh {
                    for xo in 0..ow {
                        let mut acc = bias.as_ref().map_or(0.0, |b| b[co]);
                        for dt in 0..kt {
                            let ti = (to * s.stride.0 + dt) as isize - s.pad_t.0 as isize;
                            if ti < 0 || ti >= s.t as isize {
                                continue; // temporal padding is always zero
                            }
                            for dy in 0..kh {
                                let Some(yi) = pad_index(
                                    (yo * s.stride.1 + dy) as isize - s.pad_h.0 as isize,
                                    s.h,
                                    s.spatial_pad,
                                ) else {
                                    continue;
                                };
                                for dx in 0..kw {
                                    let Some(xi) = pad_index(
                                        (xo * s.stride.2 + dx) as isize - s.pad_w.0 as isize,
                                        s.w,
                                        s.spatial_pad,
                                    ) else {
                                        continue;
                                    };
                                    for cl in 0..cg {
                                        let ci = g * cg + cl;
                                        acc += wr[((cl * kt + dt) * kh + dy) * kw + dx]
                                            * x[((ci * s.t + ti as usize) * s.h + yi) * s.w + xi];
                                    }
                                }
                            }
                        }
                        let i = yo * ow + xo;
                        plane[i] =
                            epilogue(&ep, res.as_ref(), plane_i * oh * ow + i, usize::MAX, acc);
                    }
                }
            });
    }

    fn conv1d(
        &self,
        x: &CpuBuf,
        s: &Conv1dShape,
        w: &CpuWeight,
        bias: Option<&CpuBuf>,
        ep: ConvEpilogue<'_, CpuBuf>,
        out: &CpuBuf,
    ) {
        assert!(
            !matches!(ep, ConvEpilogue::Prelu(_)),
            "PReLU is a conv2d epilogue"
        );
        assert_eq!(w.cols, (s.cin / s.groups) * s.k);
        assert_eq!(w.rows, s.cout);
        let x = x.r();
        let bias = bias.map(|b| b.r());
        let res = residual_of(&ep);
        let ol = s.out_len_1d();
        let cg = s.cin / s.groups;
        let og = s.cout / s.groups;
        let mut o = out.w();
        o[..s.cout * ol]
            .par_chunks_mut(ol)
            .enumerate()
            .for_each(|(co, row)| {
                let g = co / og;
                let wr = &w.data[co * cg * s.k..][..cg * s.k];
                for t in 0..ol {
                    let mut acc = bias.as_ref().map_or(0.0, |b| b[co]);
                    for kk in 0..s.k {
                        let ti = (t * s.stride + kk * s.dilation) as isize - s.pad.0 as isize;
                        if ti < 0 || ti >= s.len as isize {
                            continue;
                        }
                        for cl in 0..cg {
                            acc += wr[cl * s.k + kk] * x[(g * cg + cl) * s.len + ti as usize];
                        }
                    }
                    row[t] = epilogue(&ep, res.as_ref(), co * ol + t, usize::MAX, acc);
                }
            });
    }

    fn replicate_pad1d(
        &self,
        x: &CpuBuf,
        c: usize,
        len: usize,
        left: usize,
        right: usize,
        out: &CpuBuf,
    ) {
        let x = x.r();
        let ol = len + left + right;
        let mut o = out.w();
        o[..c * ol]
            .par_chunks_mut(ol)
            .enumerate()
            .for_each(|(ci, row)| {
                for i in 0..ol {
                    let src = (i as isize - left as isize).clamp(0, len as isize - 1) as usize;
                    row[i] = x[ci * len + src];
                }
            });
    }

    fn conv_transpose1d(
        &self,
        x: &CpuBuf,
        s: &ConvT1dShape,
        w: &CpuWeight,
        bias: Option<&CpuBuf>,
        out: &CpuBuf,
    ) {
        assert_eq!(w.rows, s.cin);
        assert_eq!(w.cols, (s.cout / s.groups) * s.k);
        let x = x.r();
        let bias = bias.map(|b| b.r());
        let ol = s.out_len_1d();
        let cg = s.cin / s.groups;
        let og = s.cout / s.groups;
        let mut o = out.w();
        o[..s.cout * ol]
            .par_chunks_mut(ol)
            .enumerate()
            .for_each(|(co, row)| {
                let g = co / og;
                let col = co % og;
                let b = bias.as_ref().map_or(0.0, |b| b[co]);
                row.fill(b);
                for cl in 0..cg {
                    let ci = g * cg + cl;
                    let wr = &w.data[ci * og * s.k + col * s.k..][..s.k];
                    for i in 0..s.len {
                        let xv = x[ci * s.len + i];
                        for kk in 0..s.k {
                            let t = (i * s.stride + kk) as isize - s.pad as isize;
                            if t >= 0 && (t as usize) < ol {
                                row[t as usize] += xv * wr[kk];
                            }
                        }
                    }
                }
            });
    }

    fn group_norm(
        &self,
        x: &CpuBuf,
        c: usize,
        t: usize,
        hw: usize,
        groups: usize,
        per_frame: bool,
        w: &CpuBuf,
        b: &CpuBuf,
        eps: f32,
        a: Act,
        out: &CpuBuf,
    ) {
        let x = x.r();
        let (w, b) = (w.r(), b.r());
        let cg = c / groups;
        let frames = if per_frame { t } else { 1 };
        let tlen = if per_frame { 1 } else { t };
        // Statistics per (frame, group).
        let stats: Vec<(f64, f64)> = (0..frames * groups)
            .into_par_iter()
            .map(|fg| {
                let (f, g) = (fg / groups, fg % groups);
                let mut sum = 0.0f64;
                let mut sq = 0.0f64;
                let mut n = 0usize;
                for cl in 0..cg {
                    let ci = g * cg + cl;
                    for tl in 0..tlen {
                        let ti = if per_frame { f } else { tl };
                        let base = (ci * t + ti) * hw;
                        for &v in &x[base..base + hw] {
                            sum += v as f64;
                            sq += v as f64 * v as f64;
                            n += 1;
                        }
                    }
                }
                let mean = sum / n as f64;
                let var = (sq / n as f64 - mean * mean).max(0.0);
                (mean, 1.0 / (var + eps as f64).sqrt())
            })
            .collect();
        let mut o = out.w();
        o[..c * t * hw]
            .par_chunks_mut(hw)
            .enumerate()
            .for_each(|(ct, row)| {
                let (ci, ti) = (ct / t, ct % t);
                let g = ci / cg;
                let (mean, inv) = stats[if per_frame { ti * groups + g } else { g }];
                let base = ct * hw;
                for j in 0..hw {
                    let y = ((x[base + j] as f64 - mean) * inv) as f32 * w[ci] + b[ci];
                    row[j] = act(a, y);
                }
            });
    }

    fn snake(&self, x: &CpuBuf, c: usize, len: usize, snake: Snake<'_, CpuBuf>) {
        let mut xx = x.w();
        match snake {
            Snake::Alpha(a) => {
                let a = a.r();
                xx[..c * len]
                    .par_chunks_mut(len)
                    .enumerate()
                    .for_each(|(ci, row)| {
                        let al = a[ci];
                        let inv = 1.0 / (al + 1e-9);
                        for v in row.iter_mut() {
                            let s = (al * *v).sin();
                            *v += inv * s * s;
                        }
                    });
            }
            Snake::AlphaBeta { alpha, beta } => {
                let (a, b) = (alpha.r(), beta.r());
                xx[..c * len]
                    .par_chunks_mut(len)
                    .enumerate()
                    .for_each(|(ci, row)| {
                        let al = a[ci].exp();
                        let inv = 1.0 / (b[ci].exp() + 1e-9);
                        for v in row.iter_mut() {
                            let s = (al * *v).sin();
                            *v += inv * s * s;
                        }
                    });
            }
        }
    }

    fn trilinear_resize(
        &self,
        x: &CpuBuf,
        c: usize,
        t: usize,
        h: usize,
        w: usize,
        t2: usize,
        h2: usize,
        w2: usize,
        out: &CpuBuf,
    ) {
        let x = x.r();
        let coord = |o: usize, n_in: usize, n_out: usize| -> (usize, usize, f32) {
            if n_in == n_out {
                return (o, o, 0.0);
            }
            let scale = n_in as f32 / n_out as f32;
            let src = (scale * (o as f32 + 0.5) - 0.5).max(0.0);
            let i0 = (src.floor() as usize).min(n_in - 1);
            let i1 = (i0 + 1).min(n_in - 1);
            (i0, i1, src - i0 as f32)
        };
        let mut o = out.w();
        o[..c * t2 * h2 * w2]
            .par_chunks_mut(h2 * w2)
            .enumerate()
            .for_each(|(ct, plane)| {
                let (ci, to) = (ct / t2, ct % t2);
                let (t0, t1, lt) = coord(to, t, t2);
                for yo in 0..h2 {
                    let (y0, y1, ly) = coord(yo, h, h2);
                    for xo in 0..w2 {
                        let (x0, x1, lx) = coord(xo, w, w2);
                        let at =
                            |ti: usize, yi: usize, xi: usize| x[((ci * t + ti) * h + yi) * w + xi];
                        let lerp = |a: f32, b: f32, l: f32| a + (b - a) * l;
                        let c00 = lerp(at(t0, y0, x0), at(t0, y0, x1), lx);
                        let c01 = lerp(at(t0, y1, x0), at(t0, y1, x1), lx);
                        let c10 = lerp(at(t1, y0, x0), at(t1, y0, x1), lx);
                        let c11 = lerp(at(t1, y1, x0), at(t1, y1, x1), lx);
                        plane[yo * w2 + xo] = lerp(lerp(c00, c01, ly), lerp(c10, c11, ly), lt);
                    }
                }
            });
    }

    fn pixel_shuffle(
        &self,
        x: &CpuBuf,
        frames: usize,
        c: usize,
        h: usize,
        w: usize,
        r: usize,
        out: &CpuBuf,
    ) {
        let x = x.r();
        let mut o = out.w();
        let (oh, ow) = (h * r, w * r);
        o[..frames * c * oh * ow]
            .par_chunks_mut(oh * ow)
            .enumerate()
            .for_each(|(fc, plane)| {
                let (f, co) = (fc / c, fc % c);
                for y in 0..oh {
                    for xx in 0..ow {
                        let ci = co * r * r + (y % r) * r + (xx % r);
                        plane[y * ow + xx] = x[((f * c * r * r + ci) * h + y / r) * w + xx / r];
                    }
                }
            });
    }

    fn tanh_clamp(&self, x: &CpuBuf, n: usize, s: f32) {
        for v in x.w()[..n].iter_mut() {
            *v = (*v / s).tanh() * s;
        }
    }

    fn clamp(&self, x: &CpuBuf, n: usize, lo: f32, hi: f32) {
        for v in x.w()[..n].iter_mut() {
            *v = v.clamp(lo, hi);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3_core::backend::ModTable;
    use h3_core::rng::Rng;
    use h3_core::vsa::Segment;

    fn max_err(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs()))
    }

    /// All tiles selected and no gate: VSA is dense attention.
    #[test]
    fn vsa_with_every_tile_is_dense_attention() {
        let (heads, d) = (2, 16);
        let segs = [
            Segment::Dense(70),
            Segment::Video {
                grid: (5, 6, 9),
                sparsity: 0.0,
            },
        ];
        let spec = VsaSpec::new(&segs, 64, heads, d);
        assert!(spec.plan.is_dense());
        let s = spec.plan.seq;
        let mut r = Rng::new(1);
        let (q, k, v) = (
            r.normal(heads * s * d),
            r.normal(heads * s * d),
            r.normal(heads * s * d),
        );
        let be = Cpu;
        let (gq, gk, gv) = (be.upload(&q), be.upload(&k), be.upload(&v));
        let dense = be.alloc(s * heads * d);
        be.attention(&gq, &gk, &gv, heads, heads, s, s, d, false, &dense);
        let sparse = be.alloc(s * heads * d);
        let st = be.vsa_prepare(&spec);
        be.vsa_attention(&gq, &gk, &gv, None, &st, &sparse);
        assert!(max_err(&be.download(&dense), &be.download(&sparse)) < 1e-5);
        // A zero gate changes nothing; a non-zero gate does.
        let zero = be.alloc(heads * s * d);
        be.vsa_attention(&gq, &gk, &gv, Some(&zero), &st, &sparse);
        assert!(max_err(&be.download(&dense), &be.download(&sparse)) < 1e-5);
        let g = be.upload(&r.normal(heads * s * d));
        be.vsa_attention(&gq, &gk, &gv, Some(&g), &st, &sparse);
        assert!(max_err(&be.download(&dense), &be.download(&sparse)) > 1e-3);
    }

    #[test]
    fn gqa_equals_repeated_kv_and_causal_masks() {
        let (heads, kv, s, d) = (4, 2, 7, 8);
        let mut r = Rng::new(3);
        let q = r.normal(heads * s * d);
        let k = r.normal(kv * s * d);
        let v = r.normal(kv * s * d);
        let be = Cpu;
        let out = be.alloc(s * heads * d);
        be.attention(
            &be.upload(&q),
            &be.upload(&k),
            &be.upload(&v),
            heads,
            kv,
            s,
            s,
            d,
            false,
            &out,
        );
        // Repeat each KV head twice.
        let rep = |x: &[f32]| -> Vec<f32> {
            let mut o = Vec::new();
            for h in 0..heads {
                o.extend_from_slice(&x[(h / 2) * s * d..(h / 2 + 1) * s * d]);
            }
            o
        };
        let out2 = be.alloc(s * heads * d);
        be.attention(
            &be.upload(&q),
            &be.upload(&rep(&k)),
            &be.upload(&rep(&v)),
            heads,
            heads,
            s,
            s,
            d,
            false,
            &out2,
        );
        assert!(max_err(&be.download(&out), &be.download(&out2)) < 1e-6);
        // Causal: row 0 attends only to key 0 → equals v[0].
        be.attention(
            &be.upload(&q),
            &be.upload(&k),
            &be.upload(&v),
            heads,
            kv,
            s,
            s,
            d,
            true,
            &out,
        );
        let o = be.download(&out);
        for h in 0..heads {
            assert!(max_err(&o[h * d..(h + 1) * d], &v[(h / 2) * s * d..][..d]) < 1e-6);
        }
    }

    #[test]
    fn patchify_round_trips_channel_major() {
        let (c, t, h, w) = (3, 2, 4, 6);
        let mut r = Rng::new(4);
        let lat = r.normal(c * t * h * w);
        let be = Cpu;
        let rows = t * (h / 2) * (w / 2);
        let f = c * 4;
        let tok = be.alloc(rows * f);
        be.patchify(&be.upload(&lat), c, t, h, w, (1, 2, 2), &tok, 0);
        let tk = be.download(&tok);
        // Feature order: ((c·1 + 0)·2 + kh)·2 + kw.
        assert_eq!(tk[0 * f + 1], lat[1]); // c0, kh0, kw1 → latent (0,0,0,1)
        assert_eq!(tk[0 * f + 2], lat[w]); // c0, kh1, kw0 → latent (0,0,1,0)
        assert_eq!(tk[0 * f + 4], lat[t * h * w]); // c1
        let back = be.alloc(c * t * h * w);
        be.unpatchify(&tok, 0, f, 0, c, t, h, w, (1, 2, 2), &back);
        assert_eq!(be.download(&back), lat);
    }

    #[test]
    fn rope_rotates_half_and_norm_is_per_head() {
        let (rows, heads, hd, rot) = (2, 2, 8, 4);
        let be = Cpu;
        let src: Vec<f32> = (0..rows * heads * hd).map(|i| i as f32 * 0.1).collect();
        let cos = vec![0.0f32; rows * rot / 2];
        let sin = vec![1.0f32; rows * rot / 2];
        let out = be.alloc(heads * rows * hd);
        be.head_split(
            &be.upload(&src),
            rows,
            heads * hd,
            0,
            heads,
            hd,
            None,
            Some(Rope {
                cos: &be.upload(&cos),
                sin: &be.upload(&sin),
                rot,
            }),
            &out,
        );
        let o = be.download(&out);
        // cos 0, sin 1: (x1, x2) → (−x2, x1) on channels 0..2 / 2..4; 4..8 untouched.
        assert_eq!(o[0], -src[2]);
        assert_eq!(o[2], src[0]);
        assert_eq!(o[5], src[5]);
        // head 1 row 0 lands at [heads, rows, hd] index (1·rows + 0)·hd.
        assert_eq!(o[rows * hd + 4], src[hd + 4]);
        let w = be.upload(&vec![2.0; hd]);
        be.head_split(
            &be.upload(&src),
            rows,
            heads * hd,
            0,
            heads,
            hd,
            Some(HeadNorm {
                w: Some(&w),
                eps: 0.0,
            }),
            None,
            &out,
        );
        let o = be.download(&out);
        let ms: f32 = (0..hd).map(|j| o[j] * o[j]).sum::<f32>() / hd as f32;
        assert!((ms - 4.0).abs() < 1e-4);
    }

    #[test]
    fn conv3d_reflect_and_causal_padding() {
        // One channel, t=1, 3×3 image, 3×3×3 kernel of ones over the last
        // temporal tap: with causal padding (2 in front) the current frame
        // sits under tap kt = 2.
        let be = Cpu;
        let x: Vec<f32> = (1..=9).map(|v| v as f32).collect();
        let mut w = vec![0.0f32; 27];
        for i in 18..27 {
            w[i] = 1.0;
        }
        let s = Conv3dShape::causal(1, 1, 1, 3, 3, 3);
        let out = be.alloc(s.out_len());
        be.conv3d(
            &be.upload(&x),
            &s,
            &be.weight(&w, 1, 27),
            None,
            ConvEpilogue::None,
            &out,
        );
        let o = be.download(&out);
        // Centre pixel: sum of all 9. Corner (0,0) with reflect: rows {1,0,1} × cols {1,0,1}.
        assert_eq!(o[4], 45.0);
        let corner = x[4] + x[3] + x[4] + x[1] + x[0] + x[1] + x[4] + x[3] + x[4];
        assert_eq!(o[0], corner);
        // The causal pad reads only t = 0 from the front taps: temporal taps 0 and 1 see zeros when t=1.
        let mut wt = vec![0.0f32; 27];
        wt[4] = 1.0; // kt = 0 centre
        be.conv3d(
            &be.upload(&x),
            &s,
            &be.weight(&wt, 1, 27),
            None,
            ConvEpilogue::None,
            &out,
        );
        assert!(be.download(&out).iter().all(|&v| v == 0.0));
        // Stride-2 downsampler with asymmetric zero padding (0,1): 3→2.
        let ds = Conv3dShape {
            h: 4,
            w: 4,
            stride: (1, 2, 2),
            pad_h: (0, 1),
            pad_w: (0, 1),
            spatial_pad: Pad::Zero,
            pad_t: (2, 0),
            ..s
        };
        assert_eq!(ds.out_dims(), (1, 2, 2));
    }

    #[test]
    fn conv1d_transpose_and_groups() {
        let be = Cpu;
        // Depthwise conv1d: 2 channels, k=3, pad 1, identity taps per channel.
        let x = vec![1.0f32, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0];
        let s = Conv1dShape {
            cin: 2,
            cout: 2,
            groups: 2,
            len: 4,
            k: 3,
            stride: 1,
            dilation: 1,
            pad: (1, 1),
        };
        let w = vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0]; // ch0 identity, ch1 shift left
        let out = be.alloc(s.out_len());
        be.conv1d(
            &be.upload(&x),
            &s,
            &be.weight(&w, 2, 3),
            None,
            ConvEpilogue::None,
            &out,
        );
        assert_eq!(
            be.download(&out),
            vec![1.0, 2.0, 3.0, 4.0, 20.0, 30.0, 40.0, 0.0]
        );
        // Transposed: cin 1 → cout 1, k 4, stride 2, pad 1: (len−1)·2 − 2 + 4.
        let ts = ConvT1dShape {
            cin: 1,
            cout: 1,
            groups: 1,
            len: 3,
            k: 4,
            stride: 2,
            pad: 1,
        };
        assert_eq!(ts.out_len_1d(), 6);
        let o = be.alloc(6);
        be.conv_transpose1d(
            &be.upload(&[1.0, 0.0, 0.0]),
            &ts,
            &be.weight(&[1.0, 2.0, 3.0, 4.0], 1, 4),
            None,
            &o,
        );
        // x[0] spreads to t = 0·2 − 1 + k = k − 1: taps k=1..3 land at 0..2.
        assert_eq!(be.download(&o), vec![2.0, 3.0, 4.0, 0.0, 0.0, 0.0]);
    }

    #[test]
    fn group_norm_per_frame_and_trilinear_identity_and_pixel_shuffle() {
        let be = Cpu;
        // c=4, t=2, hw=2, groups 2, per-frame: frame 1 scaled by 10 normalises the same.
        let base = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let mut x = vec![0.0f32; 4 * 2 * 2];
        for c in 0..4 {
            for j in 0..2 {
                x[(c * 2) * 2 + j] = base[c * 2 + j];
                x[(c * 2 + 1) * 2 + j] = 10.0 * base[c * 2 + j];
            }
        }
        let out = be.alloc(x.len());
        let w = be.upload(&[1.0; 4]);
        let b = be.upload(&[0.0; 4]);
        be.group_norm(
            &be.upload(&x),
            4,
            2,
            2,
            2,
            true,
            &w,
            &b,
            1e-6,
            Act::None,
            &out,
        );
        let o = be.download(&out);
        for c in 0..4 {
            for j in 0..2 {
                assert!((o[(c * 2) * 2 + j] - o[(c * 2 + 1) * 2 + j]).abs() < 1e-4);
            }
        }
        let same = be.alloc(8);
        be.trilinear_resize(&be.upload(&base), 1, 2, 2, 2, 2, 2, 2, &same);
        assert_eq!(be.download(&same), base.to_vec());
        let up = be.alloc(4);
        be.trilinear_resize(&be.upload(&[1.0, 3.0]), 1, 1, 1, 2, 1, 1, 4, &up);
        assert_eq!(be.download(&up), vec![1.0, 1.5, 2.5, 3.0]);
        let ps = be.alloc(4);
        be.pixel_shuffle(&be.upload(&[1.0, 2.0, 3.0, 4.0]), 1, 1, 1, 1, 2, &ps);
        assert_eq!(be.download(&ps), vec![1.0, 2.0, 3.0, 4.0]);
        let ps2 = be.alloc(8);
        be.pixel_shuffle(
            &be.upload(&[1.0, 5.0, 2.0, 6.0, 3.0, 7.0, 4.0, 8.0]),
            1,
            1,
            2,
            1,
            2,
            &ps2,
        );
        assert_eq!(
            be.download(&ps2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        );
    }

    #[test]
    fn modulated_norms_read_row_entries() {
        let be = Cpu;
        let (rows, dim) = (3, 4);
        let x: Vec<f32> = (0..rows * dim).map(|i| (i % 4) as f32 - 1.5).collect();
        // Two entries × 6 slots × dim; entry 1 has scale 1 / shift 2 at slots (1, 0).
        let mut table = vec![0.0f32; 2 * 6 * dim];
        for c in 0..dim {
            table[(6 + 1) * dim + c] = 1.0;
            table[(6 + 0) * dim + c] = 2.0;
            table[(6 + 2) * dim + c] = 0.5; // gate
        }
        let idx = be.upload_u32(&[0, 1, 0]);
        let tab = be.upload_table(&table);
        let out = be.alloc(rows * dim);
        let m = ModTable {
            table: &tab,
            idx: &idx,
            slots: 6,
        };
        be.rms_norm(
            &be.upload(&x),
            rows,
            dim,
            None,
            Modulation::Table {
                m,
                scale: 1,
                shift: 0,
            },
            0.0,
            &out,
        );
        let o = be.download(&out);
        let ms = (x[..dim].iter().map(|v| v * v).sum::<f32>() / dim as f32).sqrt();
        assert!((o[0] - x[0] / ms).abs() < 1e-6);
        assert!((o[dim] - (x[dim] / ms * 2.0 + 2.0)).abs() < 1e-6);
        let h = be.upload(&vec![1.0; rows * dim]);
        let a = be.upload(&vec![2.0; rows * dim]);
        be.gate_residual(
            &h,
            &a,
            rows,
            dim,
            Gate::Table(h3_core::backend::ModRef { table: m, slot: 2 }),
        );
        let hv = be.download(&h);
        assert_eq!(hv[0], 1.0);
        assert_eq!(hv[dim], 2.0);
        assert!((gelu_erf(1.0) - 0.841_345).abs() < 1e-5);
        assert!((erf(0.5) - 0.520_499_877_8).abs() < 1e-9);
        assert!((erf(2.5) - 0.999_593_047_7).abs() < 1e-9);
    }
}
