// fw_nv_act: the activation side of NVFP4 W4A4 (h3_core::quant::Nvfp4::
// round_rows). One thread per 16-column block: round to bf16, take the block
// maximum, its E4M3 scale `e4m3(amax_b / 6 · encode)` with encode = 2688 /
// amax (the tensor maximum in amax[0], from nv_amax.wgsl), each value's E2M1
// code `e2m1(v · (encode / scale))`, and write the decoded value
// `e2m1 · scale · decode`. Twin of the CPU reference and the CUDA fw_nv_act.
struct P { blocks: u32, x_off: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> amax: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

fn bf16_round(v: f32) -> f32 {
    var u = bitcast<u32>(v);
    let lsb = (u >> 16u) & 1u;
    u = u + 0x7FFFu + lsb;
    return bitcast<f32>(u & 0xFFFF0000u);
}

// a / b correctly rounded (round to nearest even), like Rust's `/` and CUDA's
// __fdiv_rn. WGSL's `/` may be off by 2.5 ulp (NVIDIA's Vulkan driver divides
// through a reciprocal), and one ulp flips an E2M1 code at a rounding
// boundary: the W4A4 checks failed at 1.2–3.2e-2 on an RTX PRO 6000. Long
// division of the 24-bit significands for positive normal operands with a
// normal quotient; anything else (zero, subnormal, inf, NaN) falls back to `/`.
fn div_rn(a: f32, b: f32) -> f32 {
    let ua = bitcast<u32>(a);
    let ub = bitcast<u32>(b);
    let ea = i32(ua >> 23u);
    let eb = i32(ub >> 23u);
    if (ea == 0 || eb == 0 || ea >= 255 || eb >= 255) { return a / b; }
    let mb = (ub & 0x7FFFFFu) | 0x800000u;
    var r = (ua & 0x7FFFFFu) | 0x800000u;
    var e = ea - eb + 127;
    if (r < mb) { r = r << 1u; e = e - 1; }
    // 25 quotient bits (24 + a guard bit), the remainder as the sticky bit.
    var q = 0u;
    for (var i = 0u; i < 25u; i = i + 1u) {
        q = q << 1u;
        if (r >= mb) { r = r - mb; q = q | 1u; }
        r = r << 1u;
    }
    let guard = q & 1u;
    q = q >> 1u;
    if (guard == 1u && (r != 0u || (q & 1u) == 1u)) { q = q + 1u; }
    if (q == 0x1000000u) { q = 0x800000u; e = e + 1; }
    if (e <= 0 || e >= 255) { return a / b; }
    return bitcast<f32>((u32(e) << 23u) | (q & 0x7FFFFFu));
}

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
    if (e == 0) { return sign * m * (1.0 / 512.0); }
    return sign * (1.0 + m / 8.0) * bitcast<f32>(u32(e - 7 + 127) << 23u);
}

// E2M1 value of v: round to nearest even, saturate at ±6 (cvt.rn.satfinite.e2m1x2).
fn e2m1_value(v: f32) -> f32 {
    let a = abs(v);
    var q = 6.0;
    if (a != a || a <= 0.25) { q = 0.0; }
    else if (a < 0.75) { q = 0.5; }
    else if (a <= 1.25) { q = 1.0; }
    else if (a < 1.75) { q = 1.5; }
    else if (a <= 2.5) { q = 2.0; }
    else if (a < 3.5) { q = 3.0; }
    else if (a <= 5.0) { q = 4.0; }
    if (v < 0.0) { q = -q; }
    return q;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let blk = wg_index(wid, nwg) * 256u + lid.x;
    if (blk >= p.blocks) { return; }
    let base = blk * 16u;
    let t_max = amax[0];
    var enc = 0.0;
    var dec = 0.0;
    if (t_max > 0.0 && t_max < 3.4e38) { enc = div_rn(2688.0, t_max); dec = div_rn(t_max, 2688.0); }
    var bmax = 0.0;
    for (var j = 0u; j < 16u; j = j + 1u) {
        let v = abs(bf16_round(x[p.x_off + base + j]));
        if (v > bmax || v != v) { bmax = v; }
    }
    let sf = e4m3_decode(e4m3_sat(div_rn(bmax, 6.0) * enc));
    var t = 0.0;
    if (sf > 0.0) { t = div_rn(enc, sf); }
    for (var j = 0u; j < 16u; j = j + 1u) {
        let v = bf16_round(x[p.x_off + base + j]);
        var q = 0.0;
        if (sf > 0.0) { q = e2m1_value(v * t); }
        out[base + j] = q * sf * dec;
    }
}
