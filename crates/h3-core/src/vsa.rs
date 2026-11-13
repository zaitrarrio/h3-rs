//! Video Sparse Attention for H3's packed sequence (FastVideo
//! `video_sparse_attn_h3.py`).
//!
//! One joint attention runs over `[text | conditions | audio | video]`, so:
//!
//! 1. **Tile.** Dense segments (text, audio, image references) are cut into
//!    prefix tiles of `tile_elems` consecutive rows that never straddle a
//!    segment boundary; every video region is cut into 3-D tiles of
//!    `(4, 4, 4)` (64), `(4, 4, 8)` (128) or `(4, 8, 8)` (256) tokens, partial
//!    tiles padded. Rows are permuted to `[prefix tiles][region 0][region 1]…`.
//! 2. **Pool.** Mean of Q, K and V over the real tokens of each tile (f32).
//! 3. **Score.** `q̄·k̄ / √d` over all tiles.
//! 4. **Select.** Prefix-tile queries are dense. A video query keeps every
//!    prefix tile (`exempt`) and, in each video region, its top-k tiles with
//!    `k = max(1, min(⌈(1 − s)·n⌉, n))` for that region's sparsity `s`.
//!    Sparsity 0 keeps everything (dense layers / dense first steps).
//! 5. **Fine.** Full-resolution attention from each query tile to its
//!    selected tiles.
//! 6. **Gate.** With a trained `to_gate_compress`: `out += softmax(scores) ·
//!    V̄ · gate`, the pooled branch broadcast to every row of the query tile.
//!
//! This module is the geometry; backends consume [`VsaSpec`].

/// Slot value for padding in [`TilePlan::slot_src`].
pub const PAD: u32 = u32::MAX;

pub fn tile_shape(elems: usize) -> Option<(usize, usize, usize)> {
    match elems {
        64 => Some((4, 4, 4)),
        128 => Some((4, 4, 8)),
        256 => Some((4, 8, 8)),
        _ => None,
    }
}

/// Tiles to keep per query tile in a region of `n` tiles (`compute_topk`).
pub fn topk_for(sparsity: f64, n: usize) -> usize {
    if sparsity <= 0.0 {
        return n.max(1);
    }
    (((1.0 - sparsity) * n as f64).ceil() as usize).clamp(1, n.max(1))
}

/// One span of the packed sequence, in packed order.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Segment {
    /// `rows` consecutive non-video rows (text, audio, image references).
    Dense(usize),
    /// A `(t, h, w)` token grid, row-major in packed order, with its own
    /// keep rate.
    Video { grid: (usize, usize, usize), sparsity: f64 },
}

impl Segment {
    pub fn rows(&self) -> usize {
        match self {
            Segment::Dense(n) => *n,
            Segment::Video { grid: (t, h, w), .. } => t * h * w,
        }
    }
}

/// One video region's tiles: `[start, end)` in tile numbering (prefix tiles
/// first) and how many of them each video query keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Region {
    pub start: usize,
    pub end: usize,
    pub topk: usize,
}

impl Region {
    pub fn num_tiles(&self) -> usize {
        self.end - self.start
    }
    pub fn is_dense(&self) -> bool {
        self.topk >= self.num_tiles()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TilePlan {
    pub tile: (usize, usize, usize),
    pub tile_elems: usize,
    pub seq: usize,
    pub num_prefix_tiles: usize,
    pub regions: Vec<Region>,
    /// Packed row of each padded slot, [`PAD`] for padding: `num_tiles · tile_elems`.
    pub slot_src: Vec<u32>,
    /// Real rows per tile.
    pub block_sizes: Vec<u32>,
}

impl TilePlan {
    pub fn new(segments: &[Segment], tile_elems: usize) -> Self {
        let tile = tile_shape(tile_elems).expect("tile size 64, 128 or 256");
        let (ts, hs, ws) = tile;
        let mut slot_src = Vec::new();
        let mut block_sizes = Vec::new();
        let mut prefix_sizes = Vec::new();
        let mut prefix_rows: Vec<u32> = Vec::new();
        let mut videos = Vec::new();
        let mut cursor = 0usize;
        for s in segments {
            match *s {
                Segment::Dense(n) => {
                    let (full, rem) = (n / tile_elems, n % tile_elems);
                    prefix_sizes.extend(std::iter::repeat_n(tile_elems, full));
                    if rem > 0 {
                        prefix_sizes.push(rem);
                    }
                    prefix_rows.extend((cursor..cursor + n).map(|r| r as u32));
                    cursor += n;
                }
                Segment::Video { grid, sparsity } => {
                    videos.push((cursor, grid, sparsity));
                    cursor += grid.0 * grid.1 * grid.2;
                }
            }
        }
        // Prefix tiles: consecutive rows, each dense segment's remainder tile
        // padded, never crossing a segment.
        let mut it = prefix_rows.into_iter();
        for &n in &prefix_sizes {
            for _ in 0..n {
                slot_src.push(it.next().unwrap());
            }
            slot_src.extend(std::iter::repeat_n(PAD, tile_elems - n));
            block_sizes.push(n as u32);
        }
        let num_prefix_tiles = prefix_sizes.len();
        let mut regions = Vec::new();
        for (start, (t, h, w), sparsity) in videos {
            let tiles = (t.div_ceil(ts), h.div_ceil(hs), w.div_ceil(ws));
            let first = block_sizes.len();
            for tt in 0..tiles.0 {
                for hh in 0..tiles.1 {
                    for ww in 0..tiles.2 {
                        let mut n = 0;
                        for z in tt * ts..((tt + 1) * ts).min(t) {
                            for y in hh * hs..((hh + 1) * hs).min(h) {
                                for x in ww * ws..((ww + 1) * ws).min(w) {
                                    slot_src.push((start + (z * h + y) * w + x) as u32);
                                    n += 1;
                                }
                            }
                        }
                        slot_src.extend(std::iter::repeat_n(PAD, tile_elems - n));
                        block_sizes.push(n as u32);
                    }
                }
            }
            let end = block_sizes.len();
            regions.push(Region { start: first, end, topk: topk_for(sparsity, end - first) });
        }
        Self { tile, tile_elems, seq: cursor, num_prefix_tiles, regions, slot_src, block_sizes }
    }

    pub fn num_tiles(&self) -> usize {
        self.block_sizes.len()
    }
    pub fn num_video_tiles(&self) -> usize {
        self.num_tiles() - self.num_prefix_tiles
    }
    /// Whether every query reads every tile (nothing to select).
    pub fn is_dense(&self) -> bool {
        self.regions.iter().all(Region::is_dense)
    }
    /// The plan with every region kept whole (dense layers / dense steps).
    pub fn dense(&self) -> Self {
        let mut p = self.clone();
        for r in &mut p.regions {
            r.topk = r.num_tiles();
        }
        p
    }
    /// Inverse permutation: `tile_of_row[r]` is the padded slot of packed row `r`.
    pub fn slot_of_row(&self) -> Vec<u32> {
        let mut v = vec![PAD; self.seq];
        for (slot, &r) in self.slot_src.iter().enumerate() {
            if r != PAD {
                v[r as usize] = slot as u32;
            }
        }
        v
    }
    /// Key tiles the queries of tile `qt` may read: all tiles for a prefix
    /// query; prefix tiles plus `topk` of each region for a video query. The
    /// region selection itself needs scores, so this returns the maximum.
    pub fn keys_per_query_tile(&self, qt: usize) -> usize {
        if qt < self.num_prefix_tiles {
            self.num_tiles()
        } else {
            self.num_prefix_tiles + self.regions.iter().map(|r| r.topk).sum::<usize>()
        }
    }
}

/// What a backend needs to build its VSA state.
#[derive(Debug, Clone)]
pub struct VsaSpec {
    pub plan: TilePlan,
    pub heads: usize,
    pub head_dim: usize,
    pub scale: f32,
    /// INT8 Q·Kᵀ in the fine stage (SageAttention's recipe): per head, K is
    /// shifted by its mean over the sequence (softmax-invariant), then Q and
    /// K are rounded to int8 with one scale per (head, tile); P·V stays
    /// bf16 / f32 and the coarse stage keeps the exact tile means. The CPU
    /// reference and WebGPU round the same way ([`qk8_round`]); CUDA runs
    /// the int8 tensor cores on the codes.
    pub qk8: bool,
    /// FP8 P·V in the fine stage as well (implies `qk8`): V smoothed by its
    /// per-head mean over the sequence and rounded to E4M3 with one scale
    /// per head ([`attn_v8_round`]), P rounded to E4M3 at a 64× scale with
    /// the row maximum rounded up to an integer in log2 units
    /// ([`attention_pv8_reference`]'s recipe); the gated pooled branch
    /// stays exact.
    pub pv8: bool,
}

impl VsaSpec {
    pub fn new(segments: &[Segment], tile_elems: usize, heads: usize, head_dim: usize) -> Self {
        Self { plan: TilePlan::new(segments, tile_elems), heads, head_dim, scale: 1.0 / (head_dim as f32).sqrt(), qk8: false, pv8: false }
    }
    pub fn with_qk8(self, qk8: bool) -> Self {
        Self { qk8: qk8 || self.pv8, ..self }
    }
    pub fn with_pv8(self, pv8: bool) -> Self {
        Self { qk8: self.qk8 || pv8, pv8, ..self }
    }
    pub fn dense(&self) -> Self {
        Self { plan: self.plan.dense(), ..self.clone() }
    }
}

/// The host twin of the INT8 Q·Kᵀ rounding: `q`, `k` are `[heads][seq][d]`
/// f32 (the device's bf16 values); returns them rounded as the fine stage
/// sees them. Per head, `kmean[c]` is the mean of K over the sequence; per
/// (head, tile) the scale is `max |x| / 127` over the tile's real tokens
/// (`x = q`, or `x = k − kmean`), codes are `round_ties_even(x / scale)` in
/// `[−127, 127]`, and the value is `code · scale`. A tile whose max is 0
/// keeps zeros.
pub fn qk8_round(q: &[f32], k: &[f32], spec: &VsaSpec) -> (Vec<f32>, Vec<f32>) {
    let (plan, heads, d) = (&spec.plan, spec.heads, spec.head_dim);
    let (seq, nb, te) = (plan.seq, plan.num_tiles(), plan.tile_elems);
    let mut qo = q.to_vec();
    let mut ko = k.to_vec();
    for h in 0..heads {
        let base = h * seq * d;
        let mut kmean = vec![0f64; d];
        for tok in 0..seq {
            for c in 0..d {
                kmean[c] += k[base + tok * d + c] as f64;
            }
        }
        let kmean: Vec<f32> = kmean.iter().map(|m| (m / seq as f64) as f32).collect();
        for t in 0..nb {
            let toks: Vec<usize> = (0..plan.block_sizes[t] as usize).map(|s| plan.slot_src[t * te + s] as usize).collect();
            let (mut qmax, mut kmax) = (0f32, 0f32);
            for &tok in &toks {
                for c in 0..d {
                    qmax = qmax.max(q[base + tok * d + c].abs());
                    kmax = kmax.max((k[base + tok * d + c] - kmean[c]).abs());
                }
            }
            let (qs, ks) = (qmax / 127.0, kmax / 127.0);
            // `x * (1 / scale)`, the kernels' form (`__float2int_rn(v * inv)`):
            // `x / scale` differs in the last bit one time in a million, and
            // a code flip moves a score by ~3e-3, enough to re-round P.
            let (qi, ki) = (if qs > 0.0 { 1.0 / qs } else { 0.0 }, if ks > 0.0 { 1.0 / ks } else { 0.0 });
            for &tok in &toks {
                for c in 0..d {
                    let i = base + tok * d + c;
                    qo[i] = (q[i] * qi).round_ties_even().clamp(-127.0, 127.0) * qs;
                    ko[i] = ((k[i] - kmean[c]) * ki).round_ties_even().clamp(-127.0, 127.0) * ks;
                }
            }
        }
    }
    (qo, ko)
}

/// Rows per INT8 scale tile of dense attention (`attn_qk8_round`).
pub const QK8_TILE: usize = 64;

/// The INT8 Q·Kᵀ rounding of dense attention (SageAttention's recipe, the
/// dense twin of [`qk8_round`]): per head, Q rows in tiles of [`QK8_TILE`]
/// share one scale `max|q| / 127`; per KV head, K is shifted by its mean over
/// the keys and rounded per tile with `max|k − mean| / 127`. Both come back
/// as f32 copies (K without the mean: a per-query constant softmax ignores)
/// so that `attention(q', k', v)` is the oracle of the int8 kernels. Layouts
/// `[heads][sq][d]` and `[kv_heads][sk][d]`.
pub fn attn_qk8_round(q: &[f32], k: &[f32], heads: usize, kv_heads: usize, sq: usize, sk: usize, d: usize) -> (Vec<f32>, Vec<f32>) {
    let round = |x: &[f32], out: &mut [f32], rows: usize, mean: Option<&[f32]>| {
        for t0 in (0..rows).step_by(QK8_TILE) {
            let t1 = (t0 + QK8_TILE).min(rows);
            let mut mx = 0f32;
            for r in t0..t1 {
                for c in 0..d {
                    mx = mx.max((x[r * d + c] - mean.map_or(0.0, |m| m[c])).abs());
                }
            }
            let sc = mx / 127.0;
            let inv = if sc > 0.0 { 1.0 / sc } else { 0.0 }; // the kernels' `v * inv`
            for r in t0..t1 {
                for c in 0..d {
                    let v = x[r * d + c] - mean.map_or(0.0, |m| m[c]);
                    out[r * d + c] = (v * inv).round_ties_even().clamp(-127.0, 127.0) * sc;
                }
            }
        }
    };
    let mut qo = vec![0f32; heads * sq * d];
    for h in 0..heads {
        round(&q[h * sq * d..][..sq * d], &mut qo[h * sq * d..][..sq * d], sq, None);
    }
    let mut ko = vec![0f32; kv_heads * sk * d];
    for h in 0..kv_heads {
        let kh = &k[h * sk * d..][..sk * d];
        let mut mean = vec![0f64; d];
        for r in 0..sk {
            for c in 0..d {
                mean[c] += kh[r * d + c] as f64;
            }
        }
        let mean: Vec<f32> = mean.iter().map(|m| (m / sk.max(1) as f64) as f32).collect();
        round(kh, &mut ko[h * sk * d..][..sk * d], sk, Some(&mean));
    }
    (qo, ko)
}

/// FP8 P·V (SageAttention2's recipe) on the dense INT8 path: V is smoothed by
/// its per-channel mean over the keys (per KV head), rounded to E4M3 with one
/// scale `max|v − mean| / 448` per KV head, and stored as unitless codes.
/// Returns `(codes as f32, mean [kv_heads][d], scale [kv_heads])`.
pub fn attn_v8_round(v: &[f32], kv_heads: usize, sk: usize, d: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut codes = vec![0f32; kv_heads * sk * d];
    let mut mean = vec![0f32; kv_heads * d];
    let mut scale = vec![0f32; kv_heads];
    for h in 0..kv_heads {
        let vh = &v[h * sk * d..][..sk * d];
        let mut acc = vec![0f64; d];
        for r in 0..sk {
            for c in 0..d {
                acc[c] += vh[r * d + c] as f64;
            }
        }
        for c in 0..d {
            mean[h * d + c] = (acc[c] / sk.max(1) as f64) as f32;
        }
        let mut mx = 0f32;
        for r in 0..sk {
            for c in 0..d {
                mx = mx.max((vh[r * d + c] - mean[h * d + c]).abs());
            }
        }
        let sc = mx / 448.0;
        scale[h] = sc;
        let inv = if sc > 0.0 { 1.0 / sc } else { 0.0 };
        for r in 0..sk {
            for c in 0..d {
                let x = (vh[r * d + c] - mean[h * d + c]) * inv;
                codes[(h * sk + r) * d + c] = crate::quant::e4m3_decode(crate::quant::e4m3_satfinite(x));
            }
        }
    }
    (codes, mean, scale)
}

/// Rows per 64× P scale: P is rounded to E4M3 as `exp2(s − ⌈m⌉ + 6)`, so
/// every weight down to 2^-12 of the row's largest sits in E4M3's normal
/// range; the kernels' running maxima are integers too, so their rescales
/// are exact powers of two and round the same way.
pub const PV8_SHIFT: f32 = 6.0;

/// Relative distance within which the checker treats a P value as sitting
/// on an E4M3 rounding boundary (2⁻¹⁴): the kernels' scores and `ex2`
/// differ from the oracle's f32 arithmetic by a few ulps, ~1e-6 relative
/// in P, and a flip of the largest P of a row is 1/16 of that row.
pub const PV8_FLIP_EPS: f32 = 6.103_515_6e-5;

/// The width of the E4M3 rounding step at `x` when `x` lies within `eps`
/// (relative) of a boundary between two codes, else 0: the slack a kernel
/// that rounds `x·(1 ± eps)` may be given against the oracle's rounding.
pub fn e4m3_flip_slack(x: f32, eps: f32) -> f32 {
    let lo = crate::quant::e4m3_decode(crate::quant::e4m3_satfinite(x * (1.0 - eps)));
    let hi = crate::quant::e4m3_decode(crate::quant::e4m3_satfinite(x * (1.0 + eps)));
    (hi - lo).abs()
}

/// The 8-bit attention oracle: INT8 Q·Kᵀ ([`attn_qk8_round`]) and FP8 P·V
/// ([`attn_v8_round`]). Scores in log2 units; the row maximum rounded up to
/// an integer; `p = exp2(s − ⌈m⌉ + PV8_SHIFT)` rounded to E4M3 for the
/// product and kept exact for the denominator; `o = Σ p₈·v₈ · scale / Σ p
/// + mean`. Output `[sq][heads][d]`.
#[allow(clippy::too_many_arguments)]
pub fn attention_pv8_reference(q: &[f32], k: &[f32], v: &[f32], heads: usize, kv_heads: usize, sq: usize, sk: usize, d: usize, causal: bool) -> Vec<f32> {
    pv8_reference(q, k, v, heads, kv_heads, sq, sk, d, causal, None)
}

/// [`attention_pv8_reference`] plus, per output element, the slack of the
/// P roundings that sit within [`PV8_FLIP_EPS`] of an E4M3 boundary
/// (`Σ step(p_j)·|v₈ⱼ| · scale / Σ p`): what a kernel whose scores differ
/// from this f32 arithmetic in the last bits may legitimately differ by.
#[allow(clippy::too_many_arguments)]
pub fn attention_pv8_reference_slack(q: &[f32], k: &[f32], v: &[f32], heads: usize, kv_heads: usize, sq: usize, sk: usize, d: usize, causal: bool) -> (Vec<f32>, Vec<f32>) {
    let mut slack = vec![0f32; sq * heads * d];
    let out = pv8_reference(q, k, v, heads, kv_heads, sq, sk, d, causal, Some(&mut slack));
    (out, slack)
}

#[allow(clippy::too_many_arguments)]
fn pv8_reference(q: &[f32], k: &[f32], v: &[f32], heads: usize, kv_heads: usize, sq: usize, sk: usize, d: usize, causal: bool, mut slack: Option<&mut [f32]>) -> Vec<f32> {
    let (q8, k8) = attn_qk8_round(q, k, heads, kv_heads, sq, sk, d);
    let (v8, mean, vs) = attn_v8_round(v, kv_heads, sk, d);
    let group = heads / kv_heads;
    let sl2 = std::f32::consts::LOG2_E / (d as f32).sqrt();
    let mut out = vec![0f32; sq * heads * d];
    let mut s2 = vec![0f32; sk];
    for i in 0..sq {
        let limit = if causal { (i + 1 + sk - sq).min(sk) } else { sk };
        for h in 0..heads {
            let kh = h / group;
            let qi = &q8[(h * sq + i) * d..][..d];
            let mut m = f32::NEG_INFINITY;
            for (j, sj) in s2[..limit].iter_mut().enumerate() {
                let kj = &k8[(kh * sk + j) * d..][..d];
                *sj = qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * sl2;
                m = m.max(*sj);
            }
            let mi = m.ceil();
            let mut l = 0f32;
            let mut acc = vec![0f32; d];
            let mut sl = slack.as_ref().map(|_| vec![0f32; d]);
            for (j, &sj) in s2[..limit].iter().enumerate() {
                let p = (sj - mi + PV8_SHIFT).exp2();
                l += p;
                let p8 = crate::quant::e4m3_decode(crate::quant::e4m3_satfinite(p));
                let vj = &v8[(kh * sk + j) * d..][..d];
                for c in 0..d {
                    acc[c] += p8 * vj[c];
                }
                if let Some(sl) = sl.as_mut() {
                    let w = e4m3_flip_slack(p, PV8_FLIP_EPS);
                    if w > 0.0 {
                        for c in 0..d {
                            sl[c] += w * vj[c].abs();
                        }
                    }
                }
            }
            let inv = if l > 0.0 { vs[kh] / l } else { 0.0 };
            let o = &mut out[(i * heads + h) * d..][..d];
            for c in 0..d {
                o[c] = acc[c] * inv + mean[kh * d + c];
            }
            if let (Some(sl), Some(slack)) = (sl, slack.as_deref_mut()) {
                let so = &mut slack[(i * heads + h) * d..][..d];
                for c in 0..d {
                    so[c] = sl[c] * inv;
                }
            }
        }
    }
    out
}

/// Select the key tiles of query tile `qt` from a row of pooled scores
/// (`scores[kt]` for all tiles), the host oracle of the selection rule. Ties
/// go to the lower tile index (PyTorch `topk` on CUDA is not specified for
/// ties; the kernels mirror this choice).
pub fn select(plan: &TilePlan, qt: usize, scores: &[f32]) -> Vec<bool> {
    let n = plan.num_tiles();
    let mut keep = vec![false; n];
    if qt < plan.num_prefix_tiles || plan.is_dense() {
        keep.iter_mut().for_each(|k| *k = true);
        return keep;
    }
    for k in keep.iter_mut().take(plan.num_prefix_tiles) {
        *k = true;
    }
    for r in &plan.regions {
        if r.is_dense() {
            for k in &mut keep[r.start..r.end] {
                *k = true;
            }
            continue;
        }
        let mut idx: Vec<usize> = (r.start..r.end).collect();
        idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]).then(a.cmp(&b)));
        for &i in &idx[..r.topk] {
            keep[i] = true;
        }
    }
    keep
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e4m3_slack_marks_boundaries_only() {
        // 1.0625 is the midpoint of the codes 1.0 and 1.125.
        assert_eq!(e4m3_flip_slack(1.0625, PV8_FLIP_EPS), 0.125);
        assert_eq!(e4m3_flip_slack(1.05, PV8_FLIP_EPS), 0.0);
        assert_eq!(e4m3_flip_slack(40.0, PV8_FLIP_EPS), 0.0);
        // 34 is the midpoint of the codes 32 and 36 (step 4 in [32, 64)).
        assert_eq!(e4m3_flip_slack(34.0, PV8_FLIP_EPS), 4.0);
        assert_eq!(e4m3_flip_slack(36.0, PV8_FLIP_EPS), 0.0);
        assert_eq!(e4m3_flip_slack(0.0, PV8_FLIP_EPS), 0.0);
    }

    #[test]
    fn pv8_slack_bounds_a_boundary_flip() {
        // The slack output leaves the reference unchanged, is non-negative,
        // and bounds the effect of re-rounding the near-boundary P values
        // the other way (what a kernel's exp2 may do).
        let mut rng = crate::rng::Rng::new(7);
        let (heads, kv, sq, sk, d) = (2usize, 1usize, 40usize, 72usize, 32usize);
        let (q, k, v) = (rng.normal(heads * sq * d), rng.normal(kv * sk * d), rng.normal(kv * sk * d));
        let plain = attention_pv8_reference(&q, &k, &v, heads, kv, sq, sk, d, true);
        let (out, slack) = attention_pv8_reference_slack(&q, &k, &v, heads, kv, sq, sk, d, true);
        assert_eq!(plain, out);
        assert!(slack.iter().all(|&s| s >= 0.0 && s.is_finite()));
        // Re-derive with every P nudged by +eps/2 and by −eps/2: the outputs
        // move only where slack was granted, and by no more than it.
        let (q8, k8) = attn_qk8_round(&q, &k, heads, kv, sq, sk, d);
        let (v8, mean, vs) = attn_v8_round(&v, kv, sk, d);
        let sl2 = std::f32::consts::LOG2_E / (d as f32).sqrt();
        let mut flips = 0;
        for i in 0..sq {
            let limit = (i + 1 + sk - sq).min(sk);
            for h in 0..heads {
                let qi = &q8[(h * sq + i) * d..][..d];
                let s: Vec<f32> = (0..limit).map(|j| qi.iter().zip(&k8[j * d..][..d]).map(|(a, b)| a * b).sum::<f32>() * sl2).collect();
                let mi = s.iter().cloned().fold(f32::NEG_INFINITY, f32::max).ceil();
                for sign in [-1.0f32, 1.0] {
                    let mut l = 0f32;
                    let mut acc = vec![0f32; d];
                    for (j, &sj) in s.iter().enumerate() {
                        let p = (sj - mi + PV8_SHIFT).exp2();
                        l += p;
                        let p8 = crate::quant::e4m3_decode(crate::quant::e4m3_satfinite(p * (1.0 + sign * PV8_FLIP_EPS * 0.5)));
                        for c in 0..d {
                            acc[c] += p8 * v8[j * d + c];
                        }
                    }
                    for c in 0..d {
                        let o = acc[c] * vs[0] / l + mean[c];
                        let idx = (i * heads + h) * d + c;
                        let diff = (o - out[idx]).abs();
                        if diff > 0.0 {
                            flips += 1;
                            assert!(diff <= slack[idx] * (1.0 + 1e-5) + 1e-6, "row {i} head {h} ch {c}: moved {diff} with slack {}", slack[idx]);
                        }
                    }
                }
            }
        }
        // Boundaries are rare at this eps; the seed is chosen so a few are hit.
        assert!(flips > 0, "no near-boundary P in this draw; choose another seed");
    }

    #[test]
    fn t2va_plan_768p() {
        // 10 text + 414 audio rows dense, then a 37×24×42 video grid.
        let segs = [Segment::Dense(424), Segment::Video { grid: (37, 24, 42), sparsity: 0.8 }];
        let p = TilePlan::new(&segs, 64);
        assert_eq!(p.num_prefix_tiles, 7); // 6 full + one of 40
        assert_eq!(p.block_sizes[6], 40);
        let tiles = 10 * 6 * 11;
        assert_eq!(p.num_video_tiles(), tiles);
        assert_eq!(p.regions, vec![Region { start: 7, end: 7 + tiles, topk: topk_for(0.8, tiles) }]);
        assert_eq!(topk_for(0.8, tiles), 132);
        // Every row lands in exactly one slot.
        let mut seen = vec![false; p.seq];
        for &s in &p.slot_src {
            if s != PAD {
                assert!(!seen[s as usize]);
                seen[s as usize] = true;
            }
        }
        assert!(seen.iter().all(|&x| x));
        assert_eq!(p.block_sizes.iter().map(|&b| b as usize).sum::<usize>(), p.seq);
        // Last video tile: 1 frame (37 = 9·4 + 1), 4 rows, 2 cols (42 = 10·4 + 2).
        assert_eq!(*p.block_sizes.last().unwrap(), 1 * 4 * 2);
        let inv = p.slot_of_row();
        assert_eq!(inv[0], 0);
        assert_eq!(p.slot_src[inv[500] as usize], 500);
    }

    #[test]
    fn dense_segments_never_straddle_and_regions_select_independently() {
        let segs = [
            Segment::Dense(70),
            Segment::Video { grid: (2, 4, 8), sparsity: 0.5 },
            Segment::Dense(5),
            Segment::Video { grid: (4, 4, 4), sparsity: 0.0 },
        ];
        let p = TilePlan::new(&segs, 64);
        // 70 → tiles of 64 and 6; 5 → one tile of 5. All prefix tiles first.
        assert_eq!(p.num_prefix_tiles, 3);
        assert_eq!(&p.block_sizes[..3], &[64, 6, 5]);
        assert_eq!(p.regions.len(), 2);
        assert_eq!(p.regions[0], Region { start: 3, end: 5, topk: 1 });
        assert_eq!(p.regions[1], Region { start: 5, end: 6, topk: 1 });
        assert!(p.regions[1].is_dense());
        let scores = vec![0.0, 0.0, 0.0, 1.0, 2.0, 0.0];
        let keep = select(&p, 4, &scores);
        assert_eq!(keep, vec![true, true, true, false, true, true]);
        assert!(select(&p, 0, &scores).iter().all(|&k| k));
        assert!(TilePlan::new(&segs, 128).tile == (4, 4, 8));
        assert!(p.dense().is_dense());
    }
}

/// The segments of a packed [`crate::packing::Layout`]: text, condition and
/// audio rows dense, each reference video a region at `ref_sparsity`, the
/// target video the last region at `sparsity`. Continuation clips stay dense
/// (FastVideo `_h3_vsa_ref2va_segments`).
pub fn segments_for_layout(
    layout: &crate::packing::Layout,
    patch: (usize, usize, usize),
    sparsity: f64,
    ref_sparsity: f64,
) -> Vec<Segment> {
    use crate::packing::SegmentKind;
    let mut segs = Vec::new();
    let mut dense = layout.num_text_tokens();
    let flush = |segs: &mut Vec<Segment>, dense: &mut usize| {
        if *dense > 0 {
            segs.push(Segment::Dense(*dense));
            *dense = 0;
        }
    };
    if layout.reference_segments.is_empty() {
        dense += layout.num_condition_video_rows + layout.audio_indices.len();
    } else {
        for &(kind, rows, (t, h, w)) in &layout.reference_segments {
            if kind == SegmentKind::Video {
                flush(&mut segs, &mut dense);
                segs.push(Segment::Video { grid: (t / patch.0, h / patch.1, w / patch.2), sparsity: ref_sparsity });
            } else {
                dense += rows;
            }
        }
        dense += layout.num_target_audio_rows();
    }
    flush(&mut segs, &mut dense);
    segs.push(Segment::Video { grid: layout.video_grid(patch), sparsity });
    debug_assert_eq!(segs.iter().map(Segment::rows).sum::<usize>(), layout.sequence_length);
    segs
}
