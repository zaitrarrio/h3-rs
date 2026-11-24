// fw_elementwise: one thread per output element, 256 per group.
//   op 0  gate_residual none:   x += y
//   op 1  gate_residual vector: x += y · aux[i % dim]
//   op 2  gate_residual table:  x += y · aux[(ui[row]·slots + slot)·dim + c]
//   op 3  flow euler:           x[off+i] += (f0 − f1) · y[xoff+i]
//   op 4  flow renoise:         x0 = x + f0·y; x = (1 − f1)·x0 + f1·aux[off+i]
//   op 5  flow terminal:        x[off+i] += f0 · y[xoff+i]
//   op 6  tanh_clamp:           x = tanh(x / f0) · f0
//   op 7  clamp:                x = clamp(x, f0, f1)
//   op 8  axpby:                x[off+i] = f0·x[off+i] + f1·y[xoff+i]
//   op 9  axpby (self):         x[off+i] = f0·x[off+i] + f1·x[xoff+i]
//   op 10 channel_affine:       x[c, i] = y[c, i]·aux[c] + (has_z ? z[c] : 0)
//   op 11 patchify:             x[row + off, f] = y[latent]
//   op 12 unpatchify:           x[latent] = y[(off + row)·stride + col + f]
//   op 13 swiglu:               x[r, j] = y[r, j] · silu(y[r, dim + j])
//   op 14 fill:                 x[off+i] = f0
//   op 15 replicate_pad1d:      x[c, i] = y[c, clamp(i − left, 0, len − 1)]
//   op 16 pixel_shuffle:        [frames, c·r², h, w] → [frames, c, h·r, w·r]
//   op 17 snake alpha:          x += sin²(α·x) / (α + 1e-9)        (α = aux[c])
//   op 18 snake alpha-beta:     x += sin²(e^α·x) / (e^β + 1e-9)    (α = aux[c], β = z[c])
//   op 19 trilinear resize:     x[c, t2, h2, w2] from y[c, t, h, w], align_corners=false
//   op 20 vsa gate add:         x[tok, h, c] += aux[(h·slot + ui[tok] / off)·dim + c] · y[(h·len + tok)·dim + c]
//                               (dense-plan VSA: dim = d, slots = heads, slot = nb, off = te, len = seq)
//   op 21 box gather:           x[packed] = y[grid]   (Backend::box_rows; i over heads·rows·dim,
//   op 22 box scatter:          x[grid] = y[packed]    grid t·h·w, box start pt,ph,pw, size t2,h2,w2,
//                               rows = slots, grid off / head = off / stride, packed off / head = xoff / len)
struct P { n: u32, op: u32, dim: u32, slots: u32, slot: u32, off: u32, xoff: u32, c: u32,
           t: u32, h: u32, w: u32, pt: u32, ph: u32, pw: u32, t2: u32, h2: u32,
           w2: u32, left: u32, len: u32, r: u32, frames: u32, stride: u32, col: u32, has_z: u32,
           f0: f32, f1: f32, f2: f32, f3: f32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> x: array<f32>;
@group(0) @binding(2) var<storage, read> y: array<f32>;
@group(0) @binding(3) var<storage, read> aux: array<f32>;
@group(0) @binding(4) var<storage, read> z: array<f32>;
@group(0) @binding(5) var<storage, read> ui: array<u32>;

fn lerp(a: f32, b: f32, l: f32) -> f32 { return a + (b - a) * l; }

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
    let i = wg_index(wid, nwg) * 256u + lid.x;
    if (i >= p.n) { return; }
    switch p.op {
        case 0u: { x[i] = x[i] + y[i]; }
        case 1u: { x[i] = x[i] + y[i] * aux[i % p.dim]; }
        case 2u: {
            let row = i / p.dim;
            x[i] = x[i] + y[i] * aux[(ui[row] * p.slots + p.slot) * p.dim + i % p.dim];
        }
        case 3u: { x[p.off + i] = x[p.off + i] + (p.f0 - p.f1) * y[p.xoff + i]; }
        case 4u: {
            let x0 = x[p.off + i] + p.f0 * y[p.xoff + i];
            x[p.off + i] = (1.0 - p.f1) * x0 + p.f1 * aux[p.off + i];
        }
        case 5u: { x[p.off + i] = x[p.off + i] + p.f0 * y[p.xoff + i]; }
        case 6u: { x[i] = tanh_safe(x[i] / p.f0) * p.f0; }
        case 7u: { x[i] = clamp(x[i], p.f0, p.f1); }
        case 8u: { x[p.off + i] = p.f0 * x[p.off + i] + p.f1 * y[p.xoff + i]; }
        case 9u: { x[p.off + i] = p.f0 * x[p.off + i] + p.f1 * x[p.xoff + i]; }
        case 10u: {
            let cc = i / p.dim;
            var b = 0.0;
            if (p.has_z != 0u) { b = z[cc]; }
            x[i] = y[i] * aux[cc] + b;
        }
        case 11u: {
            // i = row·F + f, f = ((c'·pt + kt)·ph + kh)·pw + kw
            let f = p.c * p.pt * p.ph * p.pw;
            let row = i / f;
            let fi = i % f;
            let kw = fi % p.pw;
            let kh = (fi / p.pw) % p.ph;
            let kt = (fi / (p.pw * p.ph)) % p.pt;
            let cc = fi / (p.pw * p.ph * p.pt);
            let gw = p.w / p.pw;
            let gh = p.h / p.ph;
            let wi = row % gw;
            let hi = (row / gw) % gh;
            let ti = row / (gw * gh);
            x[(p.off + row) * f + fi] = y[((cc * p.t + ti * p.pt + kt) * p.h + hi * p.ph + kh) * p.w + wi * p.pw + kw];
        }
        case 12u: {
            // i = ((c'·t + tt)·h + yy)·w + xx
            let xx = i % p.w;
            let yy = (i / p.w) % p.h;
            let tt = (i / (p.w * p.h)) % p.t;
            let cc = i / (p.w * p.h * p.t);
            let gw = p.w / p.pw;
            let gh = p.h / p.ph;
            let row = ((tt / p.pt) * gh + yy / p.ph) * gw + xx / p.pw;
            let f = ((cc * p.pt + tt % p.pt) * p.ph + yy % p.ph) * p.pw + xx % p.pw;
            x[i] = y[(p.off + row) * p.stride + p.col + f];
        }
        case 13u: {
            let r = i / p.dim;
            let j = i % p.dim;
            x[i] = y[r * 2u * p.dim + j] * silu(y[r * 2u * p.dim + p.dim + j]);
        }
        case 14u: { x[p.off + i] = p.f0; }
        case 15u: {
            let ol = p.len + p.left + p.r;
            let cc = i / ol;
            let j = i % ol;
            let src = clamp(i32(j) - i32(p.left), 0, i32(p.len) - 1);
            x[i] = y[cc * p.len + u32(src)];
        }
        case 16u: {
            let oh = p.h * p.r;
            let ow = p.w * p.r;
            let xx = i % ow;
            let yy = (i / ow) % oh;
            let co = (i / (ow * oh)) % p.c;
            let f = i / (ow * oh * p.c);
            let ci = co * p.r * p.r + (yy % p.r) * p.r + (xx % p.r);
            x[i] = y[((f * p.c * p.r * p.r + ci) * p.h + yy / p.r) * p.w + xx / p.r];
        }
        case 17u: {
            let cc = i / p.len;
            let al = aux[cc];
            let s = sin(al * x[i]);
            x[i] = x[i] + s * s / (al + 1e-9);
        }
        case 18u: {
            let cc = i / p.len;
            let al = exp(aux[cc]);
            let be = exp(z[cc]);
            let s = sin(al * x[i]);
            x[i] = x[i] + s * s / (be + 1e-9);
        }
        case 21u, 22u: {
            let cc = i % p.dim;
            let r = (i / p.dim) % p.slots;
            let hh = i / (p.dim * p.slots);
            let bx = r % p.w2;
            let by = (r / p.w2) % p.h2;
            let bt = r / (p.w2 * p.h2);
            let grow = ((p.pt + bt) * p.h + p.ph + by) * p.w + p.pw + bx;
            let gi = (p.off + hh * p.stride + grow) * p.dim + cc;
            let pi = (p.xoff + hh * p.len + r) * p.dim + cc;
            if (p.op == 21u) { x[pi] = y[gi]; } else { x[gi] = y[pi]; }
        }
        case 20u: {
            let c = i % p.dim;
            let row = i / p.dim;
            let h = row % p.slots;
            let tok = row / p.slots;
            let sl = ui[tok];
            if (sl != 0xffffffffu) {
                let tile = sl / p.off;
                x[i] = x[i] + aux[(h * p.slot + tile) * p.dim + c] * y[(h * p.len + tok) * p.dim + c];
            }
        }
        default: {
            // trilinear: i over [c, t2, h2, w2]
            let xo = i % p.w2;
            let yo = (i / p.w2) % p.h2;
            let to = (i / (p.w2 * p.h2)) % p.t2;
            let cc = i / (p.w2 * p.h2 * p.t2);
            var t0 = to; var t1 = to; var lt = 0.0;
            if (p.t != p.t2) {
                let src = max(f32(p.t) / f32(p.t2) * (f32(to) + 0.5) - 0.5, 0.0);
                t0 = min(u32(floor(src)), p.t - 1u); t1 = min(t0 + 1u, p.t - 1u); lt = src - f32(t0);
            }
            var y0 = yo; var y1 = yo; var ly = 0.0;
            if (p.h != p.h2) {
                let src = max(f32(p.h) / f32(p.h2) * (f32(yo) + 0.5) - 0.5, 0.0);
                y0 = min(u32(floor(src)), p.h - 1u); y1 = min(y0 + 1u, p.h - 1u); ly = src - f32(y0);
            }
            var x0 = xo; var x1 = xo; var lx = 0.0;
            if (p.w != p.w2) {
                let src = max(f32(p.w) / f32(p.w2) * (f32(xo) + 0.5) - 0.5, 0.0);
                x0 = min(u32(floor(src)), p.w - 1u); x1 = min(x0 + 1u, p.w - 1u); lx = src - f32(x0);
            }
            let b = cc * p.t;
            let c00 = lerp(y[((b + t0) * p.h + y0) * p.w + x0], y[((b + t0) * p.h + y0) * p.w + x1], lx);
            let c01 = lerp(y[((b + t0) * p.h + y1) * p.w + x0], y[((b + t0) * p.h + y1) * p.w + x1], lx);
            let c10 = lerp(y[((b + t1) * p.h + y0) * p.w + x0], y[((b + t1) * p.h + y0) * p.w + x1], lx);
            let c11 = lerp(y[((b + t1) * p.h + y1) * p.w + x0], y[((b + t1) * p.h + y1) * p.w + x1], lx);
            x[i] = lerp(lerp(c00, c01, ly), lerp(c10, c11, ly), lt);
        }
    }
    if (p.op == 9999u) { x[0] = aux[0] + z[0] + f32(ui[0]) + y[0]; }
}
