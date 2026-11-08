// fw_vsa_qk8: the INT8 Q·Kᵀ rounding of the VSA fine stage, emulated in f32
// (h3_core::vsa::qk8_round). mode 0: one workgroup per head, kmean[h][c] =
// mean of K over the sequence. mode 1: one workgroup per (tile, head): the
// scales max|q| / 127 and max|k − kmean| / 127 over the tile's real tokens,
// then q8 = round(q / qs)·qs and k8 = round((k − kmean) / ks)·ks (ties to
// even, clamped to ±127) for those tokens. CUDA runs int8 tensor cores on
// the same codes.
struct P { heads: u32, seq: u32, d: u32, nb: u32, te: u32, mode: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> slot: array<u32>;
@group(0) @binding(4) var<storage, read> bsz: array<u32>;
@group(0) @binding(5) var<storage, read_write> kmean: array<f32>;
@group(0) @binding(6) var<storage, read_write> q8: array<f32>;
@group(0) @binding(7) var<storage, read_write> k8: array<f32>;

var<workgroup> redq: array<f32, 256>;
var<workgroup> redk: array<f32, 256>;

fn clamp_code(v: f32) -> f32 { return clamp(round(v), -127.0, 127.0); }

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let tid = lid.x;
    if (p.mode == 0u) {
        let h = wid.x;
        if (tid >= p.d) { return; }
        var acc = 0.0;
        for (var tok = 0u; tok < p.seq; tok = tok + 1u) {
            acc = acc + k[(h * p.seq + tok) * p.d + tid];
        }
        kmean[h * p.d + tid] = acc / f32(p.seq);
        return;
    }
    let t = wid.x;
    let h = wid.y;
    let n = bsz[t];
    let total = n * p.d;
    var qm = 0.0;
    var km = 0.0;
    for (var i = tid; i < total; i = i + 256u) {
        let tok = slot[t * p.te + i / p.d];
        let c = i % p.d;
        let idx = (h * p.seq + tok) * p.d + c;
        qm = max(qm, abs(q[idx]));
        km = max(km, abs(k[idx] - kmean[h * p.d + c]));
    }
    redq[tid] = qm;
    redk[tid] = km;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (tid < s) {
            redq[tid] = max(redq[tid], redq[tid + s]);
            redk[tid] = max(redk[tid], redk[tid + s]);
        }
        workgroupBarrier();
    }
    let qs = redq[0] / 127.0;
    let ks = redk[0] / 127.0;
    for (var i = tid; i < total; i = i + 256u) {
        let tok = slot[t * p.te + i / p.d];
        let c = i % p.d;
        let idx = (h * p.seq + tok) * p.d + c;
        var qv = 0.0;
        var kv = 0.0;
        if (qs > 0.0) { qv = clamp_code(q[idx] / qs) * qs; }
        if (ks > 0.0) { kv = clamp_code((k[idx] - kmean[h * p.d + c]) / ks) * ks; }
        q8[idx] = qv;
        k8[idx] = kv;
    }
}
