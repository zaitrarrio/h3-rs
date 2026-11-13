// The 8-bit attention oracle (h3_core::vsa::attention_pv8_reference) as a
// plain two-pass kernel: one invocation per (query row, head). q8 / k8 are
// the INT8-rounded copies (attn_qk8.wgsl modes 1 / 2), v8 the unitless E4M3
// codes of V − mean (modes 3 / 4) with mean [kvh][d] and vscale [kvh]. Scores
// in log2 units, the row maximum rounded up to an integer, p = exp2(s − ⌈m⌉
// + 6) rounded to E4M3 for the product and exact for the denominator,
// o = Σ p₈ v₈ · vscale / Σ p + mean. An emulation path: the CUDA kernel
// runs int8 / fp8 tensor cores on the same codes.
struct P { heads: u32, kvh: u32, sq: u32, sk: u32, d: u32, causal: u32, scale: f32, _b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q8: array<f32>;
@group(0) @binding(2) var<storage, read> k8: array<f32>;
@group(0) @binding(3) var<storage, read> v8: array<f32>;
@group(0) @binding(4) var<storage, read> vmean: array<f32>;
@group(0) @binding(5) var<storage, read> vscale: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

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

var<private> acc: array<f32, 128>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let h = gid.y;
    if (i >= p.sq) { return; }
    let kh = h / (p.heads / p.kvh);
    var limit = p.sk;
    if (p.causal != 0u) { limit = min(p.sk, i + 1u + p.sk - p.sq); }
    let sl2 = p.scale * 1.4426950408889634;
    let qb = (h * p.sq + i) * p.d;
    var m = -1e30;
    for (var j = 0u; j < limit; j = j + 1u) {
        var s = 0.0;
        let kb = (kh * p.sk + j) * p.d;
        for (var c = 0u; c < p.d; c = c + 1u) { s = s + q8[qb + c] * k8[kb + c]; }
        m = max(m, s * sl2);
    }
    let mi = ceil(m);
    for (var c = 0u; c < p.d; c = c + 1u) { acc[c] = 0.0; }
    var l = 0.0;
    for (var j = 0u; j < limit; j = j + 1u) {
        var s = 0.0;
        let kb = (kh * p.sk + j) * p.d;
        for (var c = 0u; c < p.d; c = c + 1u) { s = s + q8[qb + c] * k8[kb + c]; }
        let pr = exp2(s * sl2 - mi + 6.0);
        l = l + pr;
        let p8 = e4m3_decode(e4m3_sat(pr));
        for (var c = 0u; c < p.d; c = c + 1u) { acc[c] = acc[c] + p8 * v8[kb + c]; }
    }
    var inv = 0.0;
    if (l > 0.0) { inv = vscale[kh] / l; }
    let ob = (i * p.heads + h) * p.d;
    for (var c = 0u; c < p.d; c = c + 1u) { out[ob + c] = acc[c] * inv + vmean[kh * p.d + c]; }
}
