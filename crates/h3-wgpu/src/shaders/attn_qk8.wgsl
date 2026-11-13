// fw_attn_qk8: the INT8 Q·Kᵀ rounding of dense attention, emulated in f32
// (h3_core::vsa::attn_qk8_round). mode 0: one workgroup per KV head,
// kmean[h][c] = mean of K over the keys. mode 1: one workgroup per (64-row
// tile, head) of Q: qs = max|q| / 127 over the tile, q8 = round(q / qs)·qs.
// mode 2: the same for K per (tile, KV head) on k − kmean (the mean is not
// added back: softmax ignores a per-query constant). Rounding ties to even,
// clamped to ±127, as the int8 tensor cores see it.
struct P { heads: u32, kvh: u32, sq: u32, sk: u32, d: u32, mode: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read_write> kmean: array<f32>;
@group(0) @binding(4) var<storage, read_write> q8: array<f32>;
@group(0) @binding(5) var<storage, read_write> k8: array<f32>;

var<workgroup> red: array<f32, 256>;

fn clamp_code(v: f32) -> f32 { return clamp(round(v), -127.0, 127.0); }

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
