// fw_linear4: y[y_off + m·n + n'] = act(Σ_k x[x_off + m·k + k'] · W[n', k'] + b[n'])
// (element offsets), ported from fastwan-rs's linear4: 128×128 output tiles,
// 256 threads, 8×8 outputs per thread in sixteen named vec4 accumulators (no
// arrays: NVIDIA's Vulkan compiler spills them), BK = 16. Shared memory holds
// both operands transposed as vec4 ([k][row / 4]), so each k step is four vec4
// loads for 64 FMAs (linear3: eight scalar loads for 16). Each output sums
// over k in the same order with the same fma as linear3, so the two agree bit
// for bit. Same uniform block and bindings as linear3, so the f16 / int8 /
// int4 weight loaders (`load_w`, binding 5) apply unchanged. In fastwan-rs on
// an RTX 4090: 1.75× faster than linear3 per GEMM.
struct P { m: u32, n: u32, k: u32, x_off: u32, y_off: u32, has_bias: u32, act: u32, _p: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<{W_TYPE}>;
@group(0) @binding(3) var<storage, read> bias: array<f32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

fn load_w(i: u32) -> f32 { {W_LOAD} }

var<workgroup> xs: array<vec4<f32>, 512>;   // [16 k][128 rows / 4]
var<workgroup> ws: array<vec4<f32>, 512>;   // [16 k][128 cols / 4]

fn out1(m: u32, n: u32, v: f32) {
    if (n >= p.n) { return; }
    var o = v;
    if (p.has_bias != 0u) { o = o + bias[n]; }
    y[p.y_off + m * p.n + n] = act(o, p.act);
}

fn store4(m: u32, n: u32, v: vec4<f32>) {
    if (m >= p.m) { return; }
    out1(m, n, v.x);
    out1(m, n + 1u, v.y);
    out1(m, n + 2u, v.z);
    out1(m, n + 3u, v.w);
}

fn xload(m: u32, gk: u32) -> f32 {
    if (m >= p.m) { return 0.0; }
    return x[p.x_off + m * p.k + gk];
}

fn wload(n: u32, gk: u32) -> f32 {
    if (n >= p.n) { return 0.0; }
    return load_w(n * p.k + gk);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let m0 = wid.y * 128u;
    let n0 = wid.x * 128u;
    let tx = lid.x;
    let ty = lid.y;
    let tid = ty * 16u + tx;
    var c00 = vec4<f32>(0.0);
    var c01 = vec4<f32>(0.0);
    var c10 = vec4<f32>(0.0);
    var c11 = vec4<f32>(0.0);
    var c20 = vec4<f32>(0.0);
    var c21 = vec4<f32>(0.0);
    var c30 = vec4<f32>(0.0);
    var c31 = vec4<f32>(0.0);
    var c40 = vec4<f32>(0.0);
    var c41 = vec4<f32>(0.0);
    var c50 = vec4<f32>(0.0);
    var c51 = vec4<f32>(0.0);
    var c60 = vec4<f32>(0.0);
    var c61 = vec4<f32>(0.0);
    var c70 = vec4<f32>(0.0);
    var c71 = vec4<f32>(0.0);
    for (var k0 = 0u; k0 < p.k; k0 = k0 + 16u) {
        // Loads: each thread writes whole vec4s (a component store into a
        // shared vec4 may compile to read-modify-write and race): rows
        // 4q .. 4q+3 at k0 + kk, kk fastest across lanes for coalescing.
        for (var t = 0u; t < 2u; t = t + 1u) {
            let e = tid + t * 256u;
            let kk = e % 16u;
            let q4 = e / 16u;            // 0 .. 31: rows 4·q4 .. 4·q4 + 3
            let gk = k0 + kk;
            var xv = vec4<f32>(0.0);
            var wv = vec4<f32>(0.0);
            if (gk < p.k) {
                let mr = m0 + 4u * q4;
                let nr = n0 + 4u * q4;
                xv = vec4<f32>(xload(mr, gk), xload(mr + 1u, gk), xload(mr + 2u, gk), xload(mr + 3u, gk));
                wv = vec4<f32>(wload(nr, gk), wload(nr + 1u, gk), wload(nr + 2u, gk), wload(nr + 3u, gk));
            }
            xs[kk * 32u + q4] = xv;
            ws[kk * 32u + q4] = wv;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < 16u; kk = kk + 1u) {
            let o = kk * 32u;
            let a0 = xs[o + ty];
            let a1 = xs[o + 16u + ty];
            let b0 = ws[o + tx];
            let b1 = ws[o + 16u + tx];
            c00 = fma(vec4<f32>(a0.x), b0, c00);
            c01 = fma(vec4<f32>(a0.x), b1, c01);
            c10 = fma(vec4<f32>(a0.y), b0, c10);
            c11 = fma(vec4<f32>(a0.y), b1, c11);
            c20 = fma(vec4<f32>(a0.z), b0, c20);
            c21 = fma(vec4<f32>(a0.z), b1, c21);
            c30 = fma(vec4<f32>(a0.w), b0, c30);
            c31 = fma(vec4<f32>(a0.w), b1, c31);
            c40 = fma(vec4<f32>(a1.x), b0, c40);
            c41 = fma(vec4<f32>(a1.x), b1, c41);
            c50 = fma(vec4<f32>(a1.y), b0, c50);
            c51 = fma(vec4<f32>(a1.y), b1, c51);
            c60 = fma(vec4<f32>(a1.z), b0, c60);
            c61 = fma(vec4<f32>(a1.z), b1, c61);
            c70 = fma(vec4<f32>(a1.w), b0, c70);
            c71 = fma(vec4<f32>(a1.w), b1, c71);
        }
        workgroupBarrier();
    }
    store4(m0 + 4u * ty + 0u, n0 + 4u * tx, c00);
    store4(m0 + 4u * ty + 0u, n0 + 64u + 4u * tx, c01);
    store4(m0 + 4u * ty + 1u, n0 + 4u * tx, c10);
    store4(m0 + 4u * ty + 1u, n0 + 64u + 4u * tx, c11);
    store4(m0 + 4u * ty + 2u, n0 + 4u * tx, c20);
    store4(m0 + 4u * ty + 2u, n0 + 64u + 4u * tx, c21);
    store4(m0 + 4u * ty + 3u, n0 + 4u * tx, c30);
    store4(m0 + 4u * ty + 3u, n0 + 64u + 4u * tx, c31);
    store4(m0 + 64u + 4u * ty + 0u, n0 + 4u * tx, c40);
    store4(m0 + 64u + 4u * ty + 0u, n0 + 64u + 4u * tx, c41);
    store4(m0 + 64u + 4u * ty + 1u, n0 + 4u * tx, c50);
    store4(m0 + 64u + 4u * ty + 1u, n0 + 64u + 4u * tx, c51);
    store4(m0 + 64u + 4u * ty + 2u, n0 + 4u * tx, c60);
    store4(m0 + 64u + 4u * ty + 2u, n0 + 64u + 4u * tx, c61);
    store4(m0 + 64u + 4u * ty + 3u, n0 + 4u * tx, c70);
    store4(m0 + 64u + 4u * ty + 3u, n0 + 64u + 4u * tx, c71);
}
