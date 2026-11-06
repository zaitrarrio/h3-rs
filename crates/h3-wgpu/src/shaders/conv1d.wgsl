// fw_conv1d: 1-D convolution over one [cin, len] signal, one thread per
// output sample, groups / stride / dilation / zero padding (left, right).
// Weight [cout, (cin/groups)·k]. ep: 0 none, 1 relu, 2 add+relu, 3 add.
struct P { cin: u32, cout: u32, groups: u32, len: u32, k: u32, stride: u32, dilation: u32, pl: u32,
           ol: u32, has_bias: u32, ep: u32, n: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<{W_TYPE}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> res: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

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
    var acc = 0.0;
    if (p.has_bias != 0u) { acc = bias[co]; }
    for (var kk = 0u; kk < p.k; kk = kk + 1u) {
        let ti = i32(t * p.stride + kk * p.dilation) - i32(p.pl);
        if (ti < 0 || ti >= i32(p.len)) { continue; }
        for (var cl = 0u; cl < cg; cl = cl + 1u) {
            acc = fma(load_w((co * cg + cl) * p.k + kk), x[(g * cg + cl) * p.len + u32(ti)], acc);
        }
    }
    if (p.ep == 1u) { acc = max(acc, 0.0); }
    if (p.ep == 2u) { acc = max(acc + res[i], 0.0); }
    if (p.ep == 3u) { acc = acc + res[i]; }
    out[i] = acc;
    if (p.cin == 0xfffffffeu) { out[0] = res[0] + bias[0]; }
}
