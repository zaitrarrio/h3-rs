// fw_conv3d_direct: grouped 3-D convolution, one thread per output voxel
// (depthwise temporal convs and other grouped shapes; the GEMM kernel takes
// groups = 1). Same padding and epilogue contract as fw_conv3d.
struct P { cin: u32, cout: u32, t: u32, h: u32, w: u32, kt: u32, kh: u32, kw: u32,
           st: u32, sh: u32, sw: u32, pt0: u32, ph0: u32, pw0: u32, reflect: u32, has_bias: u32,
           ep: u32, ot: u32, oh: u32, ow: u32, groups: u32, n: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<{W_TYPE}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> res: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

fn load_w(i: u32) -> f32 { {W_LOAD} }

fn pad_index(q: i32, n: u32) -> i32 {
    if (q >= 0 && q < i32(n)) { return q; }
    if (p.reflect == 0u) { return -1; }
    var r = q;
    let ni = i32(n);
    for (var it = 0u; it < 4u; it = it + 1u) {
        if (r < 0) { r = -r; }
        if (r >= ni) { r = 2 * (ni - 1) - r; }
    }
    return r;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = wg_index(wid, nwg) * 256u + lid.x;
    if (i >= p.n) { return; }
    let plane = p.ot * p.oh * p.ow;
    let co = i / plane;
    let m = i % plane;
    let to = m / (p.oh * p.ow);
    let yo = (m / p.ow) % p.oh;
    let xo = m % p.ow;
    let cg = p.cin / p.groups;
    let og = p.cout / p.groups;
    let g = co / og;
    let klen = cg * p.kt * p.kh * p.kw;
    var acc = 0.0;
    if (p.has_bias != 0u) { acc = bias[co]; }
    for (var cl = 0u; cl < cg; cl = cl + 1u) {
        let ci = g * cg + cl;
        for (var dt = 0u; dt < p.kt; dt = dt + 1u) {
            let ti = i32(to * p.st + dt) - i32(p.pt0);
            if (ti < 0 || ti >= i32(p.t)) { continue; }
            for (var dy = 0u; dy < p.kh; dy = dy + 1u) {
                let yi = pad_index(i32(yo * p.sh + dy) - i32(p.ph0), p.h);
                if (yi < 0) { continue; }
                for (var dx = 0u; dx < p.kw; dx = dx + 1u) {
                    let xi = pad_index(i32(xo * p.sw + dx) - i32(p.pw0), p.w);
                    if (xi < 0) { continue; }
                    let wi = co * klen + ((cl * p.kt + dt) * p.kh + dy) * p.kw + dx;
                    acc = fma(load_w(wi), x[((ci * p.t + u32(ti)) * p.h + u32(yi)) * p.w + u32(xi)], acc);
                }
            }
        }
    }
    if (p.ep == 1u) { acc = max(acc, 0.0); }
    if (p.ep == 2u) { acc = max(acc + res[i], 0.0); }
    if (p.ep == 3u) { acc = acc + res[i]; }
    out[i] = acc;
    if (p.cin == 0xfffffffeu) { out[0] = res[0] + bias[0]; }
}
