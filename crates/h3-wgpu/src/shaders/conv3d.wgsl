// fw_conv3d: implicit-GEMM 3-D convolution over one [cin, t, h, w] volume
// (groups = 1): M = ot·oh·ow output voxels, N = cout, K = cin·kt·kh·kw.
// Padding by address: temporal zero (front/back), spatial zero or reflect.
// ep: 0 none, 1 relu, 2 add+relu, 3 add.
struct P { cin: u32, cout: u32, t: u32, h: u32, w: u32, kt: u32, kh: u32, kw: u32,
           st: u32, sh: u32, sw: u32, pt0: u32, ph0: u32, pw0: u32, reflect: u32, has_bias: u32,
           ep: u32, ot: u32, oh: u32, ow: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<{W_TYPE}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read> res: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

fn load_w(i: u32) -> f32 { {W_LOAD} }

var<workgroup> xs: array<f32, 1024>;
var<workgroup> ws: array<f32, 1024>;

// Reflect / zero padding index, −1 for a zero.
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

fn load_x(m: u32, kidx: u32) -> f32 {
    let plane = p.oh * p.ow;
    let to = m / plane;
    let r = m % plane;
    let yo = r / p.ow;
    let xo = r % p.ow;
    let kvol = p.kt * p.kh * p.kw;
    let ci = kidx / kvol;
    let rr = kidx % kvol;
    let dt = rr / (p.kh * p.kw);
    let dy = (rr / p.kw) % p.kh;
    let dx = rr % p.kw;
    let ti = i32(to * p.st + dt) - i32(p.pt0);
    if (ti < 0 || ti >= i32(p.t)) { return 0.0; }
    let yi = pad_index(i32(yo * p.sh + dy) - i32(p.ph0), p.h);
    if (yi < 0) { return 0.0; }
    let xi = pad_index(i32(xo * p.sw + dx) - i32(p.pw0), p.w);
    if (xi < 0) { return 0.0; }
    return x[((ci * p.t + u32(ti)) * p.h + u32(yi)) * p.w + u32(xi)];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let mt = p.ot * p.oh * p.ow;
    let kt = p.cin * p.kt * p.kh * p.kw;
    let m0 = (wid.y + wid.z * nwg.y) * 64u;
    let n0 = wid.x * 64u;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    var c0 = vec4<f32>(0.0);
    var c1 = vec4<f32>(0.0);
    var c2 = vec4<f32>(0.0);
    var c3 = vec4<f32>(0.0);
    for (var k0 = 0u; k0 < kt; k0 = k0 + 16u) {
        for (var i = 0u; i < 4u; i = i + 1u) {
            let e = tid + i * 256u;
            let r = e / 16u;
            let kk = e % 16u;
            let gk = k0 + kk;
            var xv = 0.0;
            if (m0 + r < mt && gk < kt) { xv = load_x(m0 + r, gk); }
            xs[kk * 64u + r] = xv;
            var wv = 0.0;
            if (n0 + r < p.cout && gk < kt) { wv = load_w((n0 + r) * kt + gk); }
            ws[kk * 64u + r] = wv;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            let o = kk * 64u;
            let bb = vec4<f32>(ws[o + tx], ws[o + tx + 16u], ws[o + tx + 32u], ws[o + tx + 48u]);
            c0 = fma(vec4<f32>(xs[o + ty]), bb, c0);
            c1 = fma(vec4<f32>(xs[o + ty + 16u]), bb, c1);
            c2 = fma(vec4<f32>(xs[o + ty + 32u]), bb, c2);
            c3 = fma(vec4<f32>(xs[o + ty + 48u]), bb, c3);
        }
        workgroupBarrier();
    }
    let accs = array<vec4<f32>, 4>(c0, c1, c2, c3);
    for (var i = 0u; i < 4u; i = i + 1u) {
        let m = m0 + ty + 16u * i;
        if (m >= mt) { continue; }
        for (var j = 0u; j < 4u; j = j + 1u) {
            let n = n0 + tx + 16u * j;
            if (n >= p.cout) { continue; }
            var v = accs[i][j];
            if (p.has_bias != 0u) { v = v + bias[n]; }
            let o = n * mt + m;
            if (p.ep == 1u) { v = max(v, 0.0); }
            if (p.ep == 2u) { v = max(v + res[o], 0.0); }
            if (p.ep == 3u) { v = v + res[o]; }
            out[o] = v;
        }
    }
    if (p.cin == 0xfffffffeu) { out[0] = res[0] + bias[0]; }
}
