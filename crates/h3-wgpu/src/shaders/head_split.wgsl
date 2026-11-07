// fw_head_split: one 256-thread workgroup per token. Optional per-head RMS
// (over `hd` channels, optional weight[hd]), optional rotate-half RoPE over the
// first `rot` channels of every head (cos/sin [rows, rot/2]), scatter to
// [heads, rows_total, hd]. `rows` tokens of this dispatch start at global
// row `row0` (the source may be bound as a row sub-range: see
// Wgpu::head_split); cos/sin and the output are indexed by the global row.
struct P { rows: u32, stride: u32, col_off: u32, heads: u32, hd: u32, has_norm: u32, has_w: u32, has_rope: u32, rot: u32, eps: f32, row0: u32, rows_total: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read> nw: array<f32>;
@group(0) @binding(3) var<storage, read> cs: array<f32>;
@group(0) @binding(4) var<storage, read> sn: array<f32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;

var<workgroup> inv: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let r = wg_index(wid, nwg);
    if (r >= p.rows) { return; }
    let t = lid.x;
    let width = p.heads * p.hd;
    let base = r * p.stride + p.col_off;
    let g = r + p.row0;
    if (p.has_norm != 0u) {
        // One thread per head sums its channels (heads ≤ 256).
        for (var hh = t; hh < p.heads; hh = hh + 256u) {
            var s = 0.0;
            for (var c = 0u; c < p.hd; c = c + 1u) { let v = src[base + hh * p.hd + c]; s = s + v * v; }
            inv[hh] = inverseSqrt(s / f32(p.hd) + p.eps);
        }
    }
    workgroupBarrier();
    let half = p.rot / 2u;
    for (var j = t; j < width; j = j + 256u) {
        let hh = j / p.hd;
        let c = j % p.hd;
        var x = src[base + j];
        if (p.has_norm != 0u) {
            x = x * inv[hh];
            if (p.has_w != 0u) { x = x * nw[c]; }
        }
        if (p.has_rope != 0u && c < p.rot) {
            // rotate-half: channel c pairs with c + half (or c − half).
            var partner_c: u32;
            var ang: u32;
            if (c < half) { partner_c = c + half; ang = c; } else { partner_c = c - half; ang = c - half; }
            var y = src[base + hh * p.hd + partner_c];
            if (p.has_norm != 0u) {
                y = y * inv[hh];
                if (p.has_w != 0u) { y = y * nw[partner_c]; }
            }
            let co = cs[g * half + ang];
            let si = sn[g * half + ang];
            if (c < half) { x = x * co - y * si; } else { x = x * co + y * si; }
        }
        out[(hh * p.rows_total + g) * p.hd + c] = x;
    }
    if (p.rows == 0xfffffffeu) { out[0] = nw[0] + cs[0] + sn[0]; }
}
