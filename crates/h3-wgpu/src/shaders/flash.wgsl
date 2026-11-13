// fw_flash: register-tiled flash attention, for dense attention and the
// VSA-H3 fine stage.
//
// One workgroup = 64 query rows of one head, 256 threads as 16×16. Each key
// chunk of 64:
//   S = Q·Kᵀ    d in steps of 16 through shared memory, each thread a 4×4
//               block of S in four named vec4 (rows ty+16i, cols tx+16j);
//   softmax     online, base 2, one thread per row over the 64×65 tile;
//   O += P·V    each thread owns 4 rows × d/16 columns of O in registers.
//
// mode 0 dense: query rows wid.x·64 + r, key chunks of consecutive rows,
//               GQA (KV head h / group), optional causal mask j > i + sk − sq.
// mode 1 VSA:   query tile wid.x / sub, sub-chunk wid.x % sub of a tile of
//               `tile` slots (via `slot`); key chunks walk the tile's selected
//               list sel[(h·nb + qt)·(nb + 1)] = count, then tile ids, each
//               tile as `sub` chunks of 64 slots. With has_gate:
//               out += coarse[h][qt] · gate[h][token].
// pv8 (h3_core::vsa::attention_pv8_reference's P·V half): v holds the
// unitless E4M3 codes of V − vmean, the running row maximum is rounded up
// to an integer, P = exp2(s − ⌈m⌉ + 6) is rounded to E4M3 for the product
// and summed exactly, and the epilogue applies vscale[kv head] / l and adds
// vmean back before the gate term.
struct P { heads: u32, group: u32, sq: u32, sk: u32, d: u32, scale_log2: f32, mode: u32, causal: u32,
           nb: u32, tile: u32, sub: u32, has_gate: u32, pv8: u32, _p1: u32, _p2: u32, _p3: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> q: array<f32>;
@group(0) @binding(2) var<storage, read> k: array<f32>;
@group(0) @binding(3) var<storage, read> v: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<storage, read> slot: array<u32>;
@group(0) @binding(6) var<storage, read> sel: array<u32>;
@group(0) @binding(7) var<storage, read> coarse: array<f32>;
@group(0) @binding(8) var<storage, read> gate: array<f32>;
@group(0) @binding(9) var<storage, read> vmean: array<f32>;
@group(0) @binding(10) var<storage, read> vscale: array<f32>;

// E4M3 code of v (cvt.rn.satfinite) and back, as mx_act.wgsl.
fn e4m3_sat(v: f32) -> u32 {
    let u = bitcast<u32>(v);
    let sign = (u >> 24u) & 0x80u;
    let a = u & 0x7FFFFFFFu;
    if (a > 0x7F800000u) { return sign | 0x7Fu; }
    if (a >= 0x43E00000u) { return sign | 0x7Eu; }
    if (a < 0x3C800000u) {
        let m = u32(round(bitcast<f32>(a) * 512.0));
        return sign | m;
    }
    var e = i32(a >> 23u) - 127;
    var m = (a >> 20u) & 7u;
    let rem = a & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (m & 1u) == 1u)) {
        m = m + 1u;
        if (m == 8u) { m = 0u; e = e + 1; }
    }
    return sign | (u32(e + 7) << 3u) | m;
}
fn e4m3_decode(b: u32) -> f32 {
    var sign = 1.0;
    if ((b & 0x80u) != 0u) { sign = -1.0; }
    let e = i32((b >> 3u) & 0xFu);
    let m = f32(b & 7u);
    if (e == 0) { return sign * m / 512.0; }
    return sign * (1.0 + m / 8.0) * exp2(f32(e - 7));
}

const PAD: u32 = 0xffffffffu;
const SS: u32 = 65u;   // padded row stride of the S/P tile

var<workgroup> buf: array<f32, 2048>;     // Q|K panels (2×16×64) or a V panel (16×128)
var<workgroup> ss: array<f32, 4160>;      // 64 × 65
var<workgroup> rowm: array<f32, 64>;
var<workgroup> rowl: array<f32, 64>;
var<workgroup> rowc: array<f32, 64>;
var<workgroup> qrows: array<u32, 64>;     // token of each query row (PAD if none)
var<workgroup> krows: array<u32, 64>;     // token of each key row in this chunk
var<workgroup> kany: u32;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_index) tid: u32, @builtin(workgroup_id) wid: vec3<u32>) {
    let tx = tid % 16u;
    let ty = tid / 16u;
    let h = wid.y;
    let kh = h / p.group;
    let d = p.d;
    let nd = d / 16u;
    var qt = wid.x;
    var qsub = 0u;
    if (p.mode == 1u) { qt = wid.x / p.sub; qsub = wid.x % p.sub; }
    if (tid < 64u) {
        var tok = PAD;
        if (p.mode == 0u) {
            let r = qt * 64u + tid;
            if (r < p.sq) { tok = r; }
        } else {
            tok = slot[qt * p.tile + qsub * 64u + tid];
        }
        qrows[tid] = tok;
        rowm[tid] = -1e30;
        rowl[tid] = 0.0;
    }
    var o0a = vec4<f32>(0.0); var o0b = vec4<f32>(0.0);
    var o1a = vec4<f32>(0.0); var o1b = vec4<f32>(0.0);
    var o2a = vec4<f32>(0.0); var o2b = vec4<f32>(0.0);
    var o3a = vec4<f32>(0.0); var o3b = vec4<f32>(0.0);
    var nchunks = (p.sk + 63u) / 64u;
    var selbase = 0u;
    if (p.mode == 1u) {
        selbase = (h * p.nb + qt) * (p.nb + 1u);
        nchunks = sel[selbase] * p.sub;
    }
    let shift = p.sk - p.sq;
    workgroupBarrier();

    for (var c = 0u; c < nchunks; c = c + 1u) {
        if (tid < 64u) {
            var tok = PAD;
            if (p.mode == 0u) {
                let r = c * 64u + tid;
                if (r < p.sk) { tok = r; }
            } else {
                let tile = sel[selbase + 1u + c / p.sub];
                tok = slot[tile * p.tile + (c % p.sub) * 64u + tid];
            }
            krows[tid] = tok;
        }
        if (tid == 0u) { kany = 0u; }
        workgroupBarrier();
        if (tid < 64u && krows[tid] != PAD) { kany = 1u; }
        workgroupBarrier();
        if (kany == 0u) { continue; }
        var s0 = vec4<f32>(0.0);
        var s1 = vec4<f32>(0.0);
        var s2 = vec4<f32>(0.0);
        var s3 = vec4<f32>(0.0);
        for (var d0 = 0u; d0 < d; d0 = d0 + 16u) {
            for (var e = tid; e < 1024u; e = e + 256u) {
                let r = e / 16u;
                let kk = e % 16u;
                let qtok = qrows[r];
                var qv = 0.0;
                if (qtok != PAD) { qv = q[(h * p.sq + qtok) * d + d0 + kk]; }
                buf[kk * 64u + r] = qv;
                let ktok = krows[r];
                var kv = 0.0;
                if (ktok != PAD) { kv = k[(kh * p.sk + ktok) * d + d0 + kk]; }
                buf[1024u + kk * 64u + r] = kv;
            }
            workgroupBarrier();
            for (var kk = 0u; kk < 16u; kk = kk + 1u) {
                let o = kk * 64u;
                let b = vec4<f32>(buf[1024u + o + tx], buf[1024u + o + tx + 16u], buf[1024u + o + tx + 32u], buf[1024u + o + tx + 48u]);
                s0 = fma(vec4<f32>(buf[o + ty]), b, s0);
                s1 = fma(vec4<f32>(buf[o + ty + 16u]), b, s1);
                s2 = fma(vec4<f32>(buf[o + ty + 32u]), b, s2);
                s3 = fma(vec4<f32>(buf[o + ty + 48u]), b, s3);
            }
            workgroupBarrier();
        }
        let sv = array<vec4<f32>, 4>(s0, s1, s2, s3);
        for (var i = 0u; i < 4u; i = i + 1u) {
            let row = sv[i];
            let qtok = qrows[ty + 16u * i];
            for (var j = 0u; j < 4u; j = j + 1u) {
                let col = tx + 16u * j;
                let ktok = krows[col];
                var val = -1e30;
                if (ktok != PAD) {
                    val = row[j] * p.scale_log2;
                    if (p.causal != 0u && qtok != PAD && ktok > qtok + shift) { val = -1e30; }
                }
                ss[(ty + 16u * i) * SS + col] = val;
            }
        }
        workgroupBarrier();
        if (tid < 64u) {
            let base = tid * SS;
            var mx = rowm[tid];
            for (var j = 0u; j < 64u; j = j + 1u) { mx = max(mx, ss[base + j]); }
            if (p.pv8 != 0u && mx > -1e29) { mx = ceil(mx); }
            let corr = exp2(rowm[tid] - mx);
            var sum = 0.0;
            for (var j = 0u; j < 64u; j = j + 1u) {
                var e = 0.0;
                if (ss[base + j] > -1e29) {
                    if (p.pv8 != 0u) {
                        let pr = exp2(ss[base + j] - mx + 6.0);
                        sum = sum + pr;
                        e = e4m3_decode(e4m3_sat(pr));
                    } else {
                        e = exp2(ss[base + j] - mx);
                        sum = sum + e;
                    }
                }
                ss[base + j] = e;
            }
            rowl[tid] = rowl[tid] * corr + sum;
            rowm[tid] = mx;
            rowc[tid] = corr;
        }
        workgroupBarrier();
        let c0 = rowc[ty]; let c1 = rowc[ty + 16u]; let c2 = rowc[ty + 32u]; let c3 = rowc[ty + 48u];
        o0a = o0a * c0; o0b = o0b * c0;
        o1a = o1a * c1; o1b = o1b * c1;
        o2a = o2a * c2; o2b = o2b * c2;
        o3a = o3a * c3; o3b = o3b * c3;
        for (var k0 = 0u; k0 < 64u; k0 = k0 + 16u) {
            for (var e = tid; e < 16u * d; e = e + 256u) {
                let kk = e / d;
                let col = e % d;
                let ktok = krows[k0 + kk];
                var vv = 0.0;
                if (ktok != PAD) { vv = v[(kh * p.sk + ktok) * d + col]; }
                buf[kk * 128u + col] = vv;
            }
            workgroupBarrier();
            for (var kk = 0u; kk < 16u; kk = kk + 1u) {
                let vo = kk * 128u + tx;
                let va = vec4<f32>(buf[vo], buf[vo + 16u], buf[vo + 32u], buf[vo + 48u]);
                let vb = vec4<f32>(buf[vo + 64u], buf[vo + 80u], buf[vo + 96u], buf[vo + 112u]);
                let pc = k0 + kk;
                let p0 = vec4<f32>(ss[ty * SS + pc]);
                let p1 = vec4<f32>(ss[(ty + 16u) * SS + pc]);
                let p2 = vec4<f32>(ss[(ty + 32u) * SS + pc]);
                let p3 = vec4<f32>(ss[(ty + 48u) * SS + pc]);
                o0a = fma(p0, va, o0a); o0b = fma(p0, vb, o0b);
                o1a = fma(p1, va, o1a); o1b = fma(p1, vb, o1b);
                o2a = fma(p2, va, o2a); o2b = fma(p2, vb, o2b);
                o3a = fma(p3, va, o3a); o3b = fma(p3, vb, o3b);
            }
            workgroupBarrier();
        }
    }
    let oa = array<vec4<f32>, 4>(o0a, o1a, o2a, o3a);
    let ob = array<vec4<f32>, 4>(o0b, o1b, o2b, o3b);
    for (var i = 0u; i < 4u; i = i + 1u) {
        let r = ty + 16u * i;
        let tok = qrows[r];
        if (tok == PAD) { continue; }
        var inv = 0.0;
        if (rowl[r] > 0.0) { inv = 1.0 / rowl[r]; }
        if (p.pv8 != 0u) { inv = inv * vscale[kh]; }
        for (var j = 0u; j < nd; j = j + 1u) {
            let col = tx + 16u * j;
            var o: f32;
            if (j < 4u) { o = oa[i][j]; } else { o = ob[i][j - 4u]; }
            o = o * inv;
            if (p.pv8 != 0u) { o = o + vmean[kh * d + col]; }
            if (p.mode == 1u && p.has_gate != 0u) {
                o = o + coarse[(h * p.nb + qt) * d + col] * gate[(h * p.sq + tok) * d + col];
            }
            out[(tok * p.heads + h) * d + col] = o;
        }
    }
    if (p.heads == 0xfffffffeu) { out[0] = coarse[0] + gate[0] + f32(sel[0]) + f32(slot[0]) + vmean[0] + vscale[0]; }
}
