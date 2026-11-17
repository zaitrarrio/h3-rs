// fw_conv2d: implicit-GEMM conv2d, NCHW, stride 1, pad k/2: M = frames·OH·OW
// output pixels, N = cout, K = cin·k². The im2col happens in the tile loader,
// which also reads the concat (`cin_a` channels from `a`, then `cin_b` from
// `b`) and the nearest ×2 upsample (`up = 2`) by address.
// ep: 0 none, 1 relu, 2 add+relu, 3 add.
struct P { frames: u32, cin_a: u32, cin_b: u32, cout: u32, h: u32, w: u32, up: u32, k: u32,
           has_b: u32, has_bias: u32, ep: u32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read> w: array<{W_TYPE}>;
@group(0) @binding(4) var<storage, read> bias: array<f32>;
@group(0) @binding(5) var<storage, read> res: array<f32>;
@group(0) @binding(6) var<storage, read_write> out: array<f32>;

fn load_w(i: u32) -> f32 { {W_LOAD} }

var<workgroup> xs: array<f32, 1024>;
var<workgroup> ws: array<f32, 1024>;

fn load_x(m: u32, kidx: u32, oh: u32, ow: u32, kk2: u32) -> f32 {
    let plane = oh * ow;
    let f = m / plane;
    let r = m % plane;
    let oy = r / ow;
    let ox = r % ow;
    let ci = kidx / kk2;
    let rr = kidx % kk2;
    let ky = rr / p.k;
    let kx = rr % p.k;
    let pad = i32(p.k / 2u);
    let yy = i32(oy) + i32(ky) - pad;
    let xx = i32(ox) + i32(kx) - pad;
    if (yy < 0 || xx < 0 || yy >= i32(oh) || xx >= i32(ow)) { return 0.0; }
    let sy = u32(yy) / p.up;
    let sx = u32(xx) / p.up;
    if (ci < p.cin_a) {
        return a[((f * p.cin_a + ci) * p.h + sy) * p.w + sx];
    }
    return b[((f * p.cin_b + (ci - p.cin_a)) * p.h + sy) * p.w + sx];
}

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let oh = p.h * p.up;
    let ow = p.w * p.up;
    let mt = p.frames * oh * ow;
    let kk2 = p.k * p.k;
    let kt = (p.cin_a + p.cin_b) * kk2;
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
            if (m0 + r < mt && gk < kt) { xv = load_x(m0 + r, gk, oh, ow, kk2); }
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
    let plane = oh * ow;
    let accs = array<vec4<f32>, 4>(c0, c1, c2, c3);
    for (var i = 0u; i < 4u; i = i + 1u) {
        let m = m0 + ty + 16u * i;
        if (m >= mt) { continue; }
        let f = m / plane;
        let px = m % plane;
        for (var j = 0u; j < 4u; j = j + 1u) {
            let n = n0 + tx + 16u * j;
            if (n >= p.cout) { continue; }
            var v = accs[i][j];
            if (p.has_bias != 0u) { v = v + bias[n]; }
            let o = (f * p.cout + n) * plane + px;
            if (p.ep == 1u) { v = max(v, 0.0); }
            if (p.ep == 2u) { v = max(v + res[o], 0.0); }
            if (p.ep == 3u) { v = v + res[o]; }
            if (p.ep == 4u && v < 0.0) { v = res[n] * v; }
            out[o] = v;
        }
    }
    if (p.has_b == 7777u) { out[0] = b[0] + res[0] + bias[0]; }
}
