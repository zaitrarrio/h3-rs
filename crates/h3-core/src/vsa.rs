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
}

impl VsaSpec {
    pub fn new(segments: &[Segment], tile_elems: usize, heads: usize, head_dim: usize) -> Self {
        Self { plan: TilePlan::new(segments, tile_elems), heads, head_dim, scale: 1.0 / (head_dim as f32).sqrt(), qk8: false }
    }
    pub fn with_qk8(self, qk8: bool) -> Self {
        Self { qk8, ..self }
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
            for &tok in &toks {
                for c in 0..d {
                    let i = base + tok * d + c;
                    qo[i] = if qs > 0.0 { (q[i] / qs).round_ties_even().clamp(-127.0, 127.0) * qs } else { 0.0 };
                    ko[i] = if ks > 0.0 { ((k[i] - kmean[c]) / ks).round_ties_even().clamp(-127.0, 127.0) * ks } else { 0.0 };
                }
            }
        }
    }
    (qo, ko)
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
