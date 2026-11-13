// fw_attn_qk8: the INT8 Q·Kᵀ rounding of dense attention, emulated in f32
// (h3_core::vsa::attn_qk8_round). mode 0: one workgroup per KV head,
// kmean[h][c] = mean of K over the keys. mode 1: one workgroup per (64-row
// tile, head) of Q: qs = max|q| / 127 over the tile, q8 = round(q / qs)·qs.
// mode 2: the same for K per (tile, KV head) on k − kmean (the mean is not
// added back: softmax ignores a per-query constant). Rounding ties to even,
// clamped to ±127, as the int8 tensor cores see it.
// mode 3 / 4 (FP8 P·V, h3_core::vsa::attn_v8_round): with k = V, kmean its
// per-channel mean (mode 0 on V) and q8 unused: mode 3, one workgroup per KV
// head, writes vscale[h] = max|v − mean| / 448 into k8[h]; mode 4, one
// workgroup per (64-row tile, KV head), writes the unitless E4M3 codes of
// (v − mean) / vscale into q8 (reads the scale from k8[h]).
struct P { heads: u32, kvh: u32, sq: u32, sk: u32, d: u32, mode: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read_write> kmean: array<f32>;
@group(0) @binding(4) var<storage, read_write> q8: array<f32>;
@group(0) @binding(5) var<storage, read_write> k8: array<f32>;

var<workgroup> red: array<f32, 256>;

fn clamp_code(v: f32) -> f32 { return clamp(round(v), -127.0, 127.0); }

// E4M3 code of v (cvt.rn.satfinite), as mx_act.wgsl.
fn e4m3_sat(v: f32) -> u32 {
    let u = bitcast<u32>(v);
    let sign = (u >> 24u) & 0x80u;
    let a = u & 0x7FFFFFFFu;
    if (a > 0x7F800000u) { return sign | 0x7Fu; }
    if (a >= 0x43E00000u) { return sign | 0x7Eu; }
    if (a < 0x3C800000u) {
        let m = u32(round(bitcast<f32>(a) * 512.0));
        return sign | m;
    }
    var e = i32(a >> 23u) - 127;
    var m = (a >> 20u) & 7u;
    let rem = a & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (m & 1u) == 1u)) {
        m = m + 1u;
        if (m == 8u) { m = 0u; e = e + 1; }
    }
    return sign | (u32(e + 7) << 3u) | m;
}

fn e4m3_decode(b: u32) -> f32 {
    var sign = 1.0;
    if ((b & 0x80u) != 0u) { sign = -1.0; }
    let e = i32((b >> 3u) & 0xFu);
    let m = f32(b & 7u);
    if (e == 0) { return sign * m / 512.0; }
    return sign * (1.0 + m / 8.0) * exp2(f32(e - 7));
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    if (p.mode == 0u) {
        let h = wid.x;
        if (tid >= p.d) { return; }
        var acc = 0.0;
        for (var tok = 0u; tok < p.sk; tok = tok + 1u) {
            acc = acc + k[(h * p.sk + tok) * p.d + tid];
        }
        kmean[h * p.d + tid] = acc / f32(max(p.sk, 1u));
        return;
    }
    if (p.mode == 3u) {
        let h = wid.x;
        var mx = 0.0;
        for (var i = tid; i < p.sk * p.d; i = i + 256u) {
            mx = max(mx, abs(k[h * p.sk * p.d + i] - kmean[h * p.d + i % p.d]));
        }
        red[tid] = mx;
        workgroupBarrier();
        for (var s = 128u; s > 0u; s = s >> 1u) {
            if (tid < s) { red[tid] = max(red[tid], red[tid + s]); }
            workgroupBarrier();
        }
        if (tid == 0u) { k8[h] = red[0] / 448.0; }
        return;
    }
    if (p.mode == 4u) {
        let t = wid.x;
        let h = wid.y;
        let row0 = t * 64u;
        let rows = min(64u, p.sk - row0);
        let base = (h * p.sk + row0) * p.d;
        let sc = k8[h];
        var inv = 0.0;
        if (sc > 0.0) { inv = 1.0 / sc; }
        for (var i = tid; i < rows * p.d; i = i + 256u) {
            let c = i % p.d;
            q8[base + i] = e4m3_decode(e4m3_sat((k[base + i] - kmean[h * p.d + c]) * inv));
        }
        return;
    }
    let t = wid.x;
    let h = wid.y;
    let rows_total = select(p.sk, p.sq, p.mode == 1u);
    let row0 = t * 64u;
    let rows = min(64u, rows_total - row0);
    let base = (h * rows_total + row0) * p.d;
    let total = rows * p.d;
    var mx = 0.0;
    for (var i = tid; i < total; i = i + 256u) {
        let c = i % p.d;
        var v = 0.0;
        if (p.mode == 1u) { v = q[base + i]; } else { v = k[base + i] - kmean[h * p.d + c]; }
        mx = max(mx, abs(v));
    }
    red[tid] = mx;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) { red[tid] = max(red[tid], red[tid + s]); }
        workgroupBarrier();
    }
    let sc = red[0] / 127.0;
    for (var i = tid; i < total; i = i + 256u) {
        let c = i % p.d;
        var v = 0.0;
        if (p.mode == 1u) { v = q[base + i]; } else { v = k[base + i] - kmean[h * p.d + c]; }
        var r = 0.0;
        if (sc > 0.0) { r = clamp_code(v / sc) * sc; }
        if (p.mode == 1u) { q8[base + i] = r; } else { k8[base + i] = r; }
    }
}
