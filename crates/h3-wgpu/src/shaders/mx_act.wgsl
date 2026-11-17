// fw_mx_act: the activation side of MXFP8 W8A8 (h3_core::quant::Mxfp8::round_rows).
// One thread per 32-column block: round the block to bf16 (what the device
// stores), take the block maximum, pick the E8M0 scale that keeps it within
// 448, quantize each value to E4M3 (round to nearest even, saturate) and
// write the decoded value. Bit-exact twin of the CPU reference and the CUDA
// `fw_mx_act_round` / `fw_mx_quant`.
struct P { blocks: u32, x_off: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;

fn bf16_round(v: f32) -> f32 {
    var u = bitcast<u32>(v);
    let lsb = (u >> 16u) & 1u;
    u = u + 0x7FFFu + lsb;
    return bitcast<f32>(u & 0xFFFF0000u);
}

// E4M3 code of v (cvt.rn.satfinite): returns the byte in the low 8 bits.
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
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let blk = wg_index(wid, nwg) * 256u + lid.x;
    if (blk >= p.blocks) { return; }
    let base = blk * 32u;
    var amax = 0.0;
    for (var j = 0u; j < 32u; j = j + 1u) {
        let v = abs(bf16_round(x[p.x_off + base + j]));
        if (v > amax || v != v) { amax = v; }
    }
    // E8M0 from the block max: the smallest power of two with amax · 2^-e ≤ 448.
    let bits = i32(bitcast<u32>(amax));
    let e0 = ((bits >> 23u) & 0xFF) - 135;
    let threshold = bitcast<f32>(u32((e0 + 135) << 23u) | 0x600000u);
    var e = e0;
    if (amax > threshold) { e = e + 1; }
    e = max(e, -127);
    let inv = bitcast<f32>(u32((127 - e) << 23u));
    let scale = bitcast<f32>(u32((e + 127) << 23u));
    for (var j = 0u; j < 32u; j = j + 1u) {
        let v = bf16_round(x[p.x_off + base + j]);
        out[base + j] = e4m3_decode(e4m3_sat(v * inv)) * scale;
    }
}
