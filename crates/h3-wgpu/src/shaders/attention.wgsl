// fw_attention (fallback for head dims that are not multiples of 16): one
// workgroup = 64 query rows of one head, one thread per row; K/V stream
// through shared memory {BK} keys at a time with an online softmax. GQA
// (query head h reads KV head h / group) and the causal mask
// (j > i + sk − sq). Output has heads merged ([sq, heads·d]).
struct P { heads: u32, group: u32, sq: u32, sk: u32, d: u32, scale: f32, causal: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

const BK: u32 = {BK}u;
var<workgroup> ks: array<f32, {BK_D}>;
var<workgroup> vs: array<f32, {BK_D}>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let h = wid.y;
    let kh = h / p.group;
    let i = wid.x * 64u + lid.x;
    let live = i < p.sq;
    let d = p.d;
    var qr: array<f32, 128>;
    var acc: array<f32, 128>;
    for (var c = 0u; c < 128u; c = c + 1u) { qr[c] = 0.0; acc[c] = 0.0; }
    if (live) {
        for (var c = 0u; c < d; c = c + 1u) { qr[c] = q[(h * p.sq + i) * d + c] * p.scale; }
    }
    var limit = p.sk;
    if (p.causal != 0u && live) { limit = min(p.sk, i + 1u + p.sk - p.sq); }
    var m = -3.0e38;
    var l = 0.0;
    var s: array<f32, {BK}>;
    for (var j0 = 0u; j0 < p.sk; j0 = j0 + BK) {
        let nk = min(BK, p.sk - j0);
        for (var e = lid.x; e < BK * d; e = e + 64u) {
            let jj = e / d;
            var kv = 0.0;
            var vv = 0.0;
            if (jj < nk) {
                let g = (kh * p.sk + j0 + jj) * d + (e % d);
                kv = k[g];
                vv = v[g];
            }
            ks[e] = kv;
            vs[e] = vv;
        }
        workgroupBarrier();
        if (live && j0 < limit) {
            let nkk = min(nk, limit - j0);
            var mt = m;
            for (var jj = 0u; jj < nkk; jj = jj + 1u) {
                var dot = 0.0;
                for (var c = 0u; c < d; c = c + 1u) { dot = fma(qr[c], ks[jj * d + c], dot); }
                s[jj] = dot;
                mt = max(mt, dot);
            }
            let corr = exp(m - mt);
            l = l * corr;
            for (var c = 0u; c < d; c = c + 1u) { acc[c] = acc[c] * corr; }
            for (var jj = 0u; jj < nkk; jj = jj + 1u) {
                let pj = exp(s[jj] - mt);
                l = l + pj;
                for (var c = 0u; c < d; c = c + 1u) { acc[c] = fma(pj, vs[jj * d + c], acc[c]); }
            }
            m = mt;
        }
        workgroupBarrier();
    }
    if (live) {
        let inv = 1.0 / l;
        for (var c = 0u; c < d; c = c + 1u) { out[(i * p.heads + h) * d + c] = acc[c] * inv; }
    }
}
