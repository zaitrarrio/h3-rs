// fw_conv_transpose1d: PyTorch ConvTranspose1d, one thread per output sample.
// out[co, t] = bias[co] + Σ_ci Σ_k x[ci, i]·w[ci, co', k] with t = i·stride − pad + k.
// Weight [cin, (cout/groups)·k].
struct P { cin: u32, cout: u32, groups: u32, len: u32, k: u32, stride: u32, pad: u32, ol: u32,
           has_bias: u32, n: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<{W_TYPE}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

fn load_w(i: u32) -> f32 { {W_LOAD} }

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = wg_index(wid, nwg) * 256u + lid.x;
    if (i >= p.n) { return; }
    let co = i / p.ol;
    let t = i % p.ol;
    let cg = p.cin / p.groups;
    let og = p.cout / p.groups;
    let g = co / og;
    let col = co % og;
    var acc = 0.0;
    if (p.has_bias != 0u) { acc = bias[co]; }
    for (var kk = 0u; kk < p.k; kk = kk + 1u) {
        let num = i32(t + p.pad) - i32(kk);
        if (num < 0 || (u32(num) % p.stride) != 0u) { continue; }
        let src = u32(num) / p.stride;
        if (src >= p.len) { continue; }
        for (var cl = 0u; cl < cg; cl = cl + 1u) {
            let ci = g * cg + cl;
            acc = fma(load_w((ci * og + col) * p.k + kk), x[ci * p.len + src], acc);
        }
    }
    out[i] = acc;
    if (p.cin == 0xfffffffeu) { out[0] = bias[0]; }
}
