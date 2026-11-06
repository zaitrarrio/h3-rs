// fw_norm: row kernels, one 256-thread workgroup per row.
//   mode 0 rms_norm:          out = rms(x)·w ⊙ mod
//   mode 1 residual_rms_norm: h += gate(a); out = rms(h)·w ⊙ mod   (in place on h)
//   mode 2 layer_norm:        out = LN(x)·w + b                     (b read from `a`)
// gate_kind: 0 none, 1 vector gbuf[c], 2 table gbuf[(gidx[row]·gslots + gslot)·dim + c].
// mod: has_mod → y·(1 + mt[(midx[row]·mslots + mscale)·dim + c]) + mt[(…+ mshift)·dim + c].
struct P { rows: u32, dim: u32, mode: u32, gate_kind: u32, gslots: u32, gslot: u32, has_w: u32, has_mod: u32,
           mslots: u32, mscale: u32, mshift: u32, has_b: u32, eps: f32, _a: u32, _b: u32, _c: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> h: array<f32>;
@group(0) @binding(2) var<storage, read> a: array<f32>;
@group(0) @binding(3) var<storage, read> gbuf: array<f32>;
@group(0) @binding(4) var<storage, read> gidx: array<u32>;
@group(0) @binding(5) var<storage, read> nw: array<f32>;
@group(0) @binding(6) var<storage, read> mt: array<f32>;
@group(0) @binding(7) var<storage, read> midx: array<u32>;
@group(0) @binding(8) var<storage, read_write> out: array<f32>;

var<workgroup> red: array<f32, 256>;

fn block_sum(v: f32, t: u32) -> f32 {
    red[t] = v;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) { red[t] = red[t] + red[t + s]; }
        workgroupBarrier();
    }
    let r = red[0];
    workgroupBarrier();
    return r;
}

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let row = wg_index(wid, nwg);
    if (row >= p.rows) { return; }
    let t = lid.x;
    let base = row * p.dim;
    var gbase = 0u;
    if (p.gate_kind == 2u) { gbase = (gidx[row] * p.gslots + p.gslot) * p.dim; }
    var s = 0.0;
    var q = 0.0;
    for (var j = t; j < p.dim; j = j + 256u) {
        var v = h[base + j];
        if (p.mode == 1u) {
            var g = 1.0;
            if (p.gate_kind == 1u) { g = gbuf[j]; }
            if (p.gate_kind == 2u) { g = gbuf[gbase + j]; }
            v = v + a[base + j] * g;
            h[base + j] = v;
        }
        s = s + v;
        q = q + v * v;
    }
    let mean = block_sum(s, t) / f32(p.dim);
    var inv: f32;
    var sub = 0.0;
    if (p.mode == 2u) {
        var vv = 0.0;
        for (var j = t; j < p.dim; j = j + 256u) {
            let dv = h[base + j] - mean;
            vv = vv + dv * dv;
        }
        inv = inverseSqrt(block_sum(vv, t) / f32(p.dim) + p.eps);
        sub = mean;
    } else {
        inv = inverseSqrt(block_sum(q, t) / f32(p.dim) + p.eps);
    }
    var sbase = 0u;
    var hbase = 0u;
    if (p.has_mod != 0u) {
        let e = midx[row] * p.mslots;
        sbase = (e + p.mscale) * p.dim;
        hbase = (e + p.mshift) * p.dim;
    }
    for (var j = t; j < p.dim; j = j + 256u) {
        var o = (h[base + j] - sub) * inv;
        if (p.has_w != 0u) { o = o * nw[j]; }
        if (p.mode == 2u && p.has_b != 0u) { o = o + a[j]; }
        if (p.has_mod != 0u) { o = o * (1.0 + mt[sbase + j]) + mt[hbase + j]; }
        out[base + j] = o;
    }
    if (p.rows == 0xfffffffeu) { out[0] = gbuf[0] + f32(gidx[0]) + mt[0] + f32(midx[0]) + a[0] + nw[0]; }
}
