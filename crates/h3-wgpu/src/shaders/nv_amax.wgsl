// fw_nv_amax: one pass of max |bf16(x)| (the NVFP4 activation recipe's tensor
// maximum, h3_core::quant::Nvfp4::round_rows). Workgroup g reduces elements
// [g·chunk, (g+1)·chunk) of x[x_off..x_off + n] into part[g]; run it again on
// the partials (`round` = 0 there: they are already bf16) until one is left.
// A tree in workgroup memory, no atomics (they miscounted on an AMD Mac).
struct P { n: u32, x_off: u32, chunk: u32, round: u32, out_off: u32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> part: array<f32>;
var<workgroup> red: array<f32, 256>;

fn bf16_round(v: f32) -> f32 {
    var u = bitcast<u32>(v);
    let lsb = (u >> 16u) & 1u;
    u = u + 0x7FFFu + lsb;
    return bitcast<f32>(u & 0xFFFF0000u);
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let g = wg_index(wid, nwg);
    let start = g * p.chunk;
    let end = min(start + p.chunk, p.n);
    var m = 0.0;
    for (var i = start + lid.x; i < end; i = i + 256u) {
        var v = x[p.x_off + i];
        if (p.round != 0u) { v = bf16_round(v); }
        v = abs(v);
        if (v > m || v != v) { m = v; }
    }
    red[lid.x] = m;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (lid.x < s) {
            let o = red[lid.x + s];
            if (o > red[lid.x] || o != o) { red[lid.x] = o; }
        }
        workgroupBarrier();
    }
    if (lid.x == 0u) { part[p.out_off + g] = red[0]; }
}
