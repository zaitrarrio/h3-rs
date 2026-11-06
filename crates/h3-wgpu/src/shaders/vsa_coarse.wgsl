// fw_vsa_coarse: one workgroup per (query tile, head).
//   scores over all key tiles (mean q · mean k · scale); softmax →
//   coarse[h][qt] = Σ p · mean v (the gated pooled branch); then the key-tile
//   selection written as sel[(h·nb + qt)·(nb + 1)] = count, tile ids after:
//   a prefix query tile keeps every tile; a video query keeps every prefix
//   tile and, per region (start, end, topk), the top-k by score, ties to the
//   lower index — the set of `h3_core::vsa::select`.
struct P { heads: u32, d: u32, nb: u32, nprefix: u32, nregions: u32, scale: f32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> pooled: array<f32>;
@group(0) @binding(2) var<storage, read_write> coarse: array<f32>;
@group(0) @binding(3) var<storage, read_write> sel: array<u32>;
@group(0) @binding(4) var<storage, read> regions: array<u32>;   // [nregions][4]: start, end, topk, _

const MAXNB: u32 = 4096u;
var<workgroup> sc: array<f32, MAXNB>;
var<workgroup> keep: array<atomic<u32>, 128>;   // bitmask over tiles
var<workgroup> red: array<f32, 256>;
var<workgroup> cnt: atomic<u32>;

// Float → u32 with the same order.
fn okey(x: f32) -> u32 {
    let b = bitcast<u32>(x);
    return select(b | 0x80000000u, ~b, (b & 0x80000000u) != 0u);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let qt = wid.x;
    let h = wid.y;
    let plane = p.heads * p.nb * p.d;
    let qb = (h * p.nb + qt) * p.d;
    if (tid < 128u) { atomicStore(&keep[tid], 0u); }
    var mx = -3.0e38;
    for (var kt = tid; kt < p.nb; kt = kt + 256u) {
        let kb = plane + (h * p.nb + kt) * p.d;
        var dot = 0.0;
        for (var c = 0u; c < p.d; c = c + 1u) { dot = fma(pooled[qb + c], pooled[kb + c], dot); }
        let s = dot * p.scale;
        sc[kt] = s;
        mx = max(mx, s);
    }
    red[tid] = mx;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (tid < st) { red[tid] = max(red[tid], red[tid + st]); }
        workgroupBarrier();
    }
    let m = red[0];
    workgroupBarrier();
    var z = 0.0;
    for (var kt = tid; kt < p.nb; kt = kt + 256u) { z = z + exp(sc[kt] - m); }
    red[tid] = z;
    workgroupBarrier();
    for (var st = 128u; st > 0u; st = st >> 1u) {
        if (tid < st) { red[tid] = red[tid] + red[tid + st]; }
        workgroupBarrier();
    }
    let zt = red[0];
    if (tid < p.d) {
        var o = 0.0;
        for (var kt = 0u; kt < p.nb; kt = kt + 1u) {
            o = fma(exp(sc[kt] - m), pooled[2u * plane + (h * p.nb + kt) * p.d + tid], o);
        }
        coarse[qb + tid] = o / zt;
    }
    // Selection.
    if (qt < p.nprefix) {
        for (var kt = tid; kt < p.nb; kt = kt + 256u) { atomicOr(&keep[kt >> 5u], 1u << (kt & 31u)); }
    } else {
        for (var kt = tid; kt < p.nprefix; kt = kt + 256u) { atomicOr(&keep[kt >> 5u], 1u << (kt & 31u)); }
        for (var r = 0u; r < p.nregions; r = r + 1u) {
            let start = regions[r * 4u];
            let end = regions[r * 4u + 1u];
            let topk = regions[r * 4u + 2u];
            if (topk >= end - start) {
                for (var kt = start + tid; kt < end; kt = kt + 256u) { atomicOr(&keep[kt >> 5u], 1u << (kt & 31u)); }
                continue;
            }
            // Largest key T with count(key >= T) >= topk, bit by bit from the top.
            var t = 0u;
            for (var bit = 31i; bit >= 0i; bit = bit - 1i) {
                let cand = t | (1u << u32(bit));
                if (tid == 0u) { atomicStore(&cnt, 0u); }
                workgroupBarrier();
                var local = 0u;
                for (var kt = start + tid; kt < end; kt = kt + 256u) { if (okey(sc[kt]) >= cand) { local = local + 1u; } }
                atomicAdd(&cnt, local);
                workgroupBarrier();
                if (atomicLoad(&cnt) >= topk) { t = cand; }
                workgroupBarrier();
            }
            for (var kt = start + tid; kt < end; kt = kt + 256u) { if (okey(sc[kt]) > t) { atomicOr(&keep[kt >> 5u], 1u << (kt & 31u)); } }
            workgroupBarrier();
            if (tid == 0u) {
                var n = 0u;
                for (var kt = start; kt < end; kt = kt + 1u) { if (okey(sc[kt]) > t) { n = n + 1u; } }
                for (var kt = start; kt < end && n < topk; kt = kt + 1u) { if (okey(sc[kt]) == t) { atomicOr(&keep[kt >> 5u], 1u << (kt & 31u)); n = n + 1u; } }
            }
            workgroupBarrier();
        }
    }
    workgroupBarrier();
    if (tid == 0u) {
        let base = (h * p.nb + qt) * (p.nb + 1u);
        var n = 0u;
        for (var kt = 0u; kt < p.nb; kt = kt + 1u) { if ((atomicLoad(&keep[kt >> 5u]) & (1u << (kt & 31u))) != 0u) { sel[base + 1u + n] = kt; n = n + 1u; } }
        sel[base] = n;
    }
}
