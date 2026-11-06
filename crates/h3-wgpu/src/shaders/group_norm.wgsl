// fw_group_norm over [c, t, hw]: one workgroup per (frame, group) when
// per_frame, else per group over the whole volume. Two-pass statistics, then
// y = (x − mean)·inv·w[c] + b[c], then act.
struct P { c: u32, t: u32, hw: u32, groups: u32, per_frame: u32, act: u32, eps: f32, cg: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> w: array<f32>;
@group(0) @binding(3) var<storage, read> b: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;

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
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let g = wid.x % p.groups;
    let f = wid.x / p.groups;
    var tl = p.t;
    var t0 = 0u;
    if (p.per_frame != 0u) { tl = 1u; t0 = f; }
    let n = p.cg * tl * p.hw;
    var s = 0.0;
    for (var e = tid; e < n; e = e + 256u) {
        let cl = e / (tl * p.hw);
        let rem = e % (tl * p.hw);
        let ti = t0 + rem / p.hw;
        let i = rem % p.hw;
        s = s + x[((g * p.cg + cl) * p.t + ti) * p.hw + i];
    }
    let mean = block_sum(s, tid) / f32(n);
    var q = 0.0;
    for (var e = tid; e < n; e = e + 256u) {
        let cl = e / (tl * p.hw);
        let rem = e % (tl * p.hw);
        let ti = t0 + rem / p.hw;
        let i = rem % p.hw;
        let d = x[((g * p.cg + cl) * p.t + ti) * p.hw + i] - mean;
        q = q + d * d;
    }
    let inv = inverseSqrt(block_sum(q, tid) / f32(n) + p.eps);
    for (var e = tid; e < n; e = e + 256u) {
        let cl = e / (tl * p.hw);
        let rem = e % (tl * p.hw);
        let ti = t0 + rem / p.hw;
        let i = rem % p.hw;
        let ci = g * p.cg + cl;
        let idx = (ci * p.t + ti) * p.hw + i;
        out[idx] = act((x[idx] - mean) * inv * w[ci] + b[ci], p.act);
    }
}
