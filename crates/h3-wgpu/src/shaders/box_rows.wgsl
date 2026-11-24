// fw_box_rows: Backend::box_rows. One thread per element of the packed
// side, i over [window, head, box row, width]; gather (scatter = 0:
// grid -> packed) or scatter. Index arithmetic as BoxRows::index.
struct P { n: u32, width: u32, rows: u32, heads: u32, cx: u32, gh: u32, gw: u32, t0: u32,
           y0: u32, x0: u32, ny: u32, nx: u32, sy: u32, sx: u32, goff: u32, ghead: u32,
           grow: u32, poff: u32, pwin: u32, phead: u32, prow: u32, scatter: u32, _a: u32, _b: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> g: array<f32>;
@group(0) @binding(2) var<storage, read_write> pk: array<f32>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = wg_index(wid, nwg) * 256u + lid.x;
    if (i >= p.n) { return; }
    let c = i % p.width;
    let r = (i / p.width) % p.rows;
    let h = (i / (p.width * p.rows)) % p.heads;
    let win = i / (p.width * p.rows * p.heads);
    let wy = win / p.cx;
    let wx = win % p.cx;
    let bx = r % p.nx;
    let by = (r / p.nx) % p.ny;
    let bt = r / (p.nx * p.ny);
    let gr = ((p.t0 + bt) * p.gh + p.y0 + wy * p.sy + by) * p.gw + p.x0 + wx * p.sx + bx;
    let gi = (p.goff + h * p.ghead + gr * p.grow) * p.width + c;
    let pi = (p.poff + win * p.pwin + h * p.phead + r * p.prow) * p.width + c;
    if (p.scatter != 0u) { g[gi] = pk[pi]; } else { pk[pi] = g[gi]; }
}
