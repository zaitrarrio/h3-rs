// h3 CUDA kernels (NVRTC, no headers). Activations and weights are bf16
// stored as `unsigned short`; all arithmetic is f32 (f64 for the statistics
// that the CPU reference accumulates in f64). Lookup tables (RoPE, modulation,
// VSA state) are f32 / u32. Every kernel is `fw_<op>` (fastwan-rs convention:
// `norm` collides with a CUDA builtin); semantics are those of h3-cpu.

typedef unsigned short bf16_t;
typedef unsigned long long u64;

__device__ __forceinline__ float ld(const bf16_t* p, long i) {
    return __uint_as_float(((unsigned)p[i]) << 16);
}
__device__ __forceinline__ bf16_t bf(float f) {
    unsigned u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return (bf16_t)0x7fc0;  // NaN
    u += 0x7fffu + ((u >> 16) & 1u);                               // round to nearest even
    return (bf16_t)(u >> 16);
}
__device__ __forceinline__ float gelu_tanh(float x) {
    return 0.5f * x * (1.0f + tanhf(0.7978845608f * (x + 0.044715f * x * x * x)));
}
__device__ __forceinline__ float gelu_erf(float x) {
    return 0.5f * x * (1.0f + erff(x * 0.70710678118654752f));
}
__device__ __forceinline__ float silu(float x) {
    return x / (1.0f + __expf(-x));
}
// 0 none, 1 gelu(tanh), 2 gelu(erf), 3 silu
__device__ __forceinline__ float apply_act(int act, float v) {
    if (act == 1) return gelu_tanh(v);
    if (act == 2) return gelu_erf(v);
    if (act == 3) return silu(v);
    return v;
}
__device__ __forceinline__ float block_sum256(float v, float* red) {
    const int t = threadIdx.x;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((t & 31) == 0) red[t >> 5] = v;
    __syncthreads();
    float r = 0.0f;
    if (t < 8) r = red[t];
    if (t < 32) for (int o = 4; o > 0; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
    if (t == 0) red[8] = r;
    __syncthreads();
    r = red[8];
    __syncthreads();
    return r;
}
__device__ __forceinline__ double block_dsum256(double v, double* red) {
    const int t = threadIdx.x;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    if ((t & 31) == 0) red[t >> 5] = v;
    __syncthreads();
    double r = 0.0;
    if (t < 8) r = red[t];
    if (t < 32) for (int o = 4; o > 0; o >>= 1) r += __shfl_xor_sync(0xffffffffu, r, o);
    if (t == 0) red[8] = r;
    __syncthreads();
    r = red[8];
    __syncthreads();
    return r;
}
// PyTorch reflect padding index (no edge repeat); -1 for zero padding.
__device__ __forceinline__ int pad_index(int p, int n, int reflect) {
    if (p >= 0 && p < n) return p;
    if (!reflect) return -1;
    while (p < 0 || p >= n) {
        if (p < 0) p = -p;
        if (p >= n) p = 2 * (n - 1) - p;
    }
    return p;
}

// ---------------------------------------------------------------- elementwise

// y[rows, n] = act(y + bias) in place (the epilogue of a plain cuBLAS GEMM).
extern "C" __global__ void fw_bias_act(u64 y_, u64 b_, long total, int n, int has_bias, int act) {
    bf16_t* y = (bf16_t*)y_;
    const bf16_t* b = (const bf16_t*)b_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = ld(y, i);
    if (has_bias) v += ld(b, i % n);
    y[i] = bf(apply_act(act, v));
}

// out[r, j] = x[r, j] · silu(x[r, dim + j]); x rows are 2·dim wide.
extern "C" __global__ void fw_swiglu(u64 x_, u64 out_, long total, int dim) {
    const bf16_t* x = (const bf16_t*)x_;
    bf16_t* out = (bf16_t*)out_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const long r = i / dim; const int j = (int)(i - r * dim);
    out[i] = bf(ld(x, r * 2 * dim + j) * silu(ld(x, r * 2 * dim + dim + j)));
}

// gate kinds: 0 none, 1 vector[dim], 2 table[idx[row]][slot][dim]
__device__ __forceinline__ float gate_at(int kind, const bf16_t* gvec, const float* gtab, const unsigned* gidx,
                                         int slots, int slot, long row, int dim, int j) {
    if (kind == 1) return ld(gvec, j);
    if (kind == 2) return gtab[((long)gidx[row] * slots + slot) * dim + j];
    return 1.0f;
}

// in place: h += gate(a)
extern "C" __global__ void fw_gate_residual(u64 h_, u64 a_, long n, int dim, int kind, u64 gvec_, u64 gtab_, u64 gidx_, int slots, int slot) {
    bf16_t* h = (bf16_t*)h_;
    const bf16_t* a = (const bf16_t*)a_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long row = i / dim; const int j = (int)(i - row * dim);
    const float g = gate_at(kind, (const bf16_t*)gvec_, (const float*)gtab_, (const unsigned*)gidx_, slots, slot, row, dim, j);
    h[i] = bf(ld(h, i) + ld(a, i) * g);
}

// mode 0: out = rms(x)·w [mod]; mode 1: h += gate(a) (in place), out = rms(h)·w [mod].
// The residual sum is kept in registers (unrounded) when dim <= 256·24.
#define RMS_REG 24
extern "C" __global__ void fw_rms(u64 h_, u64 a_, u64 out_, int rows, int dim, int mode,
                                  int gkind, u64 gvec_, u64 gtab_, u64 gidx_, int gslots, int gslot,
                                  int has_w, u64 w_, int has_mod, u64 mtab_, u64 midx_, int mslots, int mscale, int mshift,
                                  float eps) {
    __shared__ double red[9];
    bf16_t* h = (bf16_t*)h_;
    const bf16_t* a = (const bf16_t*)a_;
    const bf16_t* w = (const bf16_t*)w_;
    bf16_t* out = (bf16_t*)out_;
    const int row = blockIdx.x;
    if (row >= rows) return;
    const long base = (long)row * dim;
    float cache[RMS_REG];
    const bool reg = dim <= 256 * RMS_REG;
    double ss = 0.0;
    for (int j = threadIdx.x, i = 0; j < dim; j += 256, ++i) {
        float v = ld(h, base + j);
        if (mode == 1) {
            const float g = gate_at(gkind, (const bf16_t*)gvec_, (const float*)gtab_, (const unsigned*)gidx_, gslots, gslot, row, dim, j);
            v += ld(a, base + j) * g;
            h[base + j] = bf(v);
        }
        if (reg) cache[i] = v;
        ss += (double)v * (double)v;
    }
    const double ms = block_dsum256(ss, red) / dim;
    const float inv = (float)(1.0 / sqrt(ms + (double)eps));
    const float* mt = (const float*)mtab_;
    const unsigned* mi = (const unsigned*)midx_;
    long mbase = 0;
    if (has_mod) mbase = (long)mi[row] * mslots * dim;
    for (int j = threadIdx.x, i = 0; j < dim; j += 256, ++i) {
        float v = reg ? cache[i] : ld(h, base + j);
        float y = v * inv;
        if (has_w) y *= ld(w, j);
        if (has_mod) y = y * (1.0f + mt[mbase + (long)mscale * dim + j]) + mt[mbase + (long)mshift * dim + j];
        out[base + j] = bf(y);
    }
}

extern "C" __global__ void fw_layer_norm(u64 x_, u64 out_, int rows, int dim, int has_w, u64 w_, int has_b, u64 b_, float eps) {
    __shared__ double red[9];
    const bf16_t *x = (const bf16_t*)x_, *w = (const bf16_t*)w_, *b = (const bf16_t*)b_;
    bf16_t* out = (bf16_t*)out_;
    const int row = blockIdx.x;
    if (row >= rows) return;
    const long base = (long)row * dim;
    float cache[RMS_REG];
    const bool reg = dim <= 256 * RMS_REG;
    double s = 0.0;
    for (int j = threadIdx.x, i = 0; j < dim; j += 256, ++i) { const float v = ld(x, base + j); if (reg) cache[i] = v; s += v; }
    const double mean = block_dsum256(s, red) / dim;
    double q = 0.0;
    for (int j = threadIdx.x, i = 0; j < dim; j += 256, ++i) { const double d = (double)(reg ? cache[i] : ld(x, base + j)) - mean; q += d * d; }
    const double inv = 1.0 / sqrt(block_dsum256(q, red) / dim + (double)eps);
    for (int j = threadIdx.x, i = 0; j < dim; j += 256, ++i) {
        float y = (float)(((double)(reg ? cache[i] : ld(x, base + j)) - mean) * inv);
        if (has_w) y *= ld(w, j);
        if (has_b) y += ld(b, j);
        out[base + j] = bf(y);
    }
}

// Columns [col_off, col_off + heads·hd) of row r → out[head][r][hd]; per-head
// RMSNorm (optional weight [hd]) and rotate-half RoPE over the first `rot`
// channels (cos/sin f32 [rows, rot/2]).
extern "C" __global__ void fw_head_split(u64 src_, u64 out_, int rows, int stride, int col_off, int heads, int hd,
                                         int has_norm, int has_w, u64 w_, float eps, int has_rope, u64 cos_, u64 sin_, int rot) {
    __shared__ float hs[256];
    __shared__ float hinv[256];
    const bf16_t *src = (const bf16_t*)src_, *w = (const bf16_t*)w_;
    const float *cs = (const float*)cos_, *sn = (const float*)sin_;
    bf16_t* out = (bf16_t*)out_;
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int width = heads * hd;
    const long base = (long)r * stride + col_off;
    if (has_norm) {
        for (int i = threadIdx.x; i < heads; i += 256) hs[i] = 0.0f;
        __syncthreads();
        for (int j = threadIdx.x; j < width; j += 256) { const float v = ld(src, base + j); atomicAdd(&hs[j / hd], v * v); }
        __syncthreads();
        for (int i = threadIdx.x; i < heads; i += 256) hinv[i] = rsqrtf(hs[i] / hd + eps);
        __syncthreads();
    }
    const int half = rot / 2;
    for (int j = threadIdx.x; j < width; j += 256) {
        const int hh = j / hd, c = j - hh * hd;
        if (has_rope && c >= half && c < rot) continue;  // written by the partner channel
        float x1 = ld(src, base + j);
        if (has_norm) { x1 *= hinv[hh]; if (has_w) x1 *= ld(w, c); }
        const long o = ((long)hh * rows + r) * hd + c;
        if (has_rope && c < half) {
            float x2 = ld(src, base + j + half);
            if (has_norm) { x2 *= hinv[hh]; if (has_w) x2 *= ld(w, c + half); }
            const long ti = (long)r * half + c;
            const float co = cs[ti], si = sn[ti];
            out[o] = bf(x1 * co - x2 * si);
            out[o + half] = bf(x2 * co + x1 * si);
        } else {
            out[o] = bf(x1);
        }
    }
}

// dir 0: patchify latent[c,t,h,w] → rows at out_row (features channel-major);
// dir 1: unpatchify rows x[(x_row+row)·stride + col + feat] → latent.
extern "C" __global__ void fw_patch(u64 dst_, u64 src_, long n, int c, int t, int hh, int ww, int pt, int ph, int pw,
                                    int dir, long row0, int stride, int col) {
    bf16_t* dst = (bf16_t*)dst_;
    const bf16_t* src = (const bf16_t*)src_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int gt = t / pt, gh = hh / ph, gw = ww / pw;
    const int f = c * pt * ph * pw;
    // i enumerates latent elements (cc, tt, yy, xx)
    const int xx = (int)(i % ww), yy = (int)((i / ww) % hh), tt = (int)((i / ((long)ww * hh)) % t);
    const int cc = (int)(i / ((long)ww * hh * t));
    const long tok = ((long)(tt / pt) * gh + yy / ph) * gw + xx / pw;
    const int feat = ((cc * pt + tt % pt) * ph + yy % ph) * pw + xx % pw;
    (void)gt;
    if (dir == 0) {
        dst[(row0 + tok) * f + feat] = src[i];
    } else {
        dst[i] = src[(row0 + tok) * stride + col + feat];
    }
}

// mode 0 euler: x += (σ − σ')·v; 1 renoise: x = (1 − σ')(x + σ v) + σ' e; 2 terminal: x += σ v.
extern "C" __global__ void fw_flow_step(u64 x_, long off, u64 v_, long voff, u64 e_, long n, int mode, float sigma, float next) {
    bf16_t* x = (bf16_t*)x_;
    const bf16_t *v = (const bf16_t*)v_, *e = (const bf16_t*)e_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xv = ld(x, off + i), vv = ld(v, voff + i);
    float r;
    if (mode == 0) r = xv + (sigma - next) * vv;
    else if (mode == 1) { const float x0 = xv + sigma * vv; r = (1.0f - next) * x0 + next * ld(e, off + i); }
    else r = xv + sigma * vv;
    x[off + i] = bf(r);
}

extern "C" __global__ void fw_axpby(u64 y_, long off, float a, u64 x_, long xoff, float b, long n) {
    bf16_t* y = (bf16_t*)y_;
    const bf16_t* x = (const bf16_t*)x_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xv = ld(x, xoff + i), yv = ld(y, off + i);
    y[off + i] = bf(a * yv + b * xv);
}

extern "C" __global__ void fw_channel_affine(u64 x_, u64 out_, long n, int inner, u64 scale_, int has_shift, u64 shift_) {
    const bf16_t *x = (const bf16_t*)x_, *s = (const bf16_t*)scale_, *sh = (const bf16_t*)shift_;
    bf16_t* out = (bf16_t*)out_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long c = i / inner;
    out[i] = bf(ld(x, i) * ld(s, c) + (has_shift ? ld(sh, c) : 0.0f));
}

extern "C" __global__ void fw_fill(u64 x_, long off, long n, float v) {
    bf16_t* x = (bf16_t*)x_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[off + i] = bf(v);
}

extern "C" __global__ void fw_tanh_clamp(u64 x_, long n, float s) {
    bf16_t* x = (bf16_t*)x_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = bf(tanhf(ld(x, i) / s) * s);
}

extern "C" __global__ void fw_clamp(u64 x_, long n, float lo, float hi) {
    bf16_t* x = (bf16_t*)x_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = bf(fminf(fmaxf(ld(x, i), lo), hi));
}

// kind 0: x + sin²(α x)/(α + 1e-9); kind 1 (log scale): x + sin²(e^α x)/(e^β + 1e-9).
extern "C" __global__ void fw_snake(u64 x_, long n, int len, u64 alpha_, u64 beta_, int kind) {
    bf16_t* x = (bf16_t*)x_;
    const bf16_t *al = (const bf16_t*)alpha_, *be = (const bf16_t*)beta_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long c = i / len;
    float a = ld(al, c), inv;
    if (kind == 0) inv = 1.0f / (a + 1e-9f);
    else { a = __expf(a); inv = 1.0f / (__expf(ld(be, c)) + 1e-9f); }
    const float v = ld(x, i), s = sinf(a * v);
    x[i] = bf(v + inv * s * s);
}

// [frames, c·r², h, w] → [frames, c, h·r, w·r]
extern "C" __global__ void fw_pixel_shuffle(u64 x_, u64 out_, long n, int c, int h, int w, int r) {
    const bf16_t* x = (const bf16_t*)x_;
    bf16_t* out = (bf16_t*)out_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int oh = h * r, ow = w * r;
    const int xx = (int)(i % ow), yy = (int)((i / ow) % oh);
    const long fc = i / ((long)ow * oh);
    const int co = (int)(fc % c); const long f = fc / c;
    const int ci = co * r * r + (yy % r) * r + (xx % r);
    out[i] = x[((f * c * r * r + ci) * (long)h + yy / r) * w + xx / r];
}

// trilinear, align_corners = false
__device__ __forceinline__ void lin_coord(int o, int n_in, int n_out, int* i0, int* i1, float* l) {
    if (n_in == n_out) { *i0 = o; *i1 = o; *l = 0.0f; return; }
    const float scale = (float)n_in / (float)n_out;
    const float src = fmaxf(scale * ((float)o + 0.5f) - 0.5f, 0.0f);
    int a = (int)floorf(src); if (a > n_in - 1) a = n_in - 1;
    int b = a + 1; if (b > n_in - 1) b = n_in - 1;
    *i0 = a; *i1 = b; *l = src - (float)a;
}
extern "C" __global__ void fw_trilinear(u64 x_, u64 out_, long n, int c, int t, int h, int w, int t2, int h2, int w2) {
    const bf16_t* x = (const bf16_t*)x_;
    bf16_t* out = (bf16_t*)out_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int xo = (int)(i % w2), yo = (int)((i / w2) % h2), to = (int)((i / ((long)w2 * h2)) % t2);
    const long ci = i / ((long)w2 * h2 * t2);
    int t0, t1, y0, y1, x0, x1; float lt, ly, lx;
    lin_coord(to, t, t2, &t0, &t1, &lt);
    lin_coord(yo, h, h2, &y0, &y1, &ly);
    lin_coord(xo, w, w2, &x0, &x1, &lx);
    #define AT(ti, yi, xi) ld(x, ((ci * t + (ti)) * (long)h + (yi)) * w + (xi))
    const float c00 = AT(t0, y0, x0) + (AT(t0, y0, x1) - AT(t0, y0, x0)) * lx;
    const float c01 = AT(t0, y1, x0) + (AT(t0, y1, x1) - AT(t0, y1, x0)) * lx;
    const float c10 = AT(t1, y0, x0) + (AT(t1, y0, x1) - AT(t1, y0, x0)) * lx;
    const float c11 = AT(t1, y1, x0) + (AT(t1, y1, x1) - AT(t1, y1, x0)) * lx;
    #undef AT
    const float a = c00 + (c01 - c00) * ly, b = c10 + (c11 - c10) * ly;
    out[i] = bf(a + (b - a) * lt);
}

// ---------------------------------------------------------------- group norm

// Statistics per (frame, group) → stats[2·fg] = mean, stats[2·fg+1] = inv std (f32).
// per_frame: elements (c in group, frame f, hw); else (c in group, all t, hw).
extern "C" __global__ void fw_group_norm_stats(u64 x_, u64 stats_, int c, int t, int hw, int groups, int per_frame, float eps) {
    __shared__ double red[9];
    const bf16_t* x = (const bf16_t*)x_;
    float* stats = (float*)stats_;
    const int fg = blockIdx.x;
    const int f = per_frame ? fg / groups : 0, g = fg % groups;
    const int cg = c / groups, tlen = per_frame ? 1 : t;
    const long n = (long)cg * tlen * hw;
    double s = 0.0, q = 0.0;
    for (long e = threadIdx.x; e < n; e += 256) {
        const int cl = (int)(e / ((long)tlen * hw)); const long rem = e - (long)cl * tlen * hw;
        const int tl = (int)(rem / hw), j = (int)(rem - (long)tl * hw);
        const int ci = g * cg + cl, ti = per_frame ? f : tl;
        const double v = ld(x, ((long)ci * t + ti) * hw + j);
        s += v; q += v * v;
    }
    const double mean = block_dsum256(s, red) / (double)n;
    const double var = fmax(block_dsum256(q, red) / (double)n - mean * mean, 0.0);
    if (threadIdx.x == 0) { stats[2 * fg] = (float)mean; stats[2 * fg + 1] = (float)(1.0 / sqrt(var + (double)eps)); }
}

extern "C" __global__ void fw_group_norm_apply(u64 x_, u64 out_, u64 stats_, long n, int c, int t, int hw, int groups, int per_frame,
                                               u64 w_, u64 b_, int act) {
    const bf16_t *x = (const bf16_t*)x_, *w = (const bf16_t*)w_, *b = (const bf16_t*)b_;
    const float* stats = (const float*)stats_;
    bf16_t* out = (bf16_t*)out_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long ct = i / hw; const int ti = (int)(ct % t); const int ci = (int)(ct / t);
    const int g = ci / (c / groups);
    const int fg = per_frame ? ti * groups + g : g;
    const float y = (ld(x, i) - stats[2 * fg]) * stats[2 * fg + 1] * ld(w, ci) + ld(b, ci);
    out[i] = bf(apply_act(act, y));
}

// ---------------------------------------------------------------- 1-D convs

extern "C" __global__ void fw_replicate_pad1d(u64 x_, u64 out_, long n, int len, int left, int right) {
    const bf16_t* x = (const bf16_t*)x_;
    bf16_t* out = (bf16_t*)out_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int ol = len + left + right;
    const long c = i / ol; int p = (int)(i - c * ol) - left;
    p = p < 0 ? 0 : (p > len - 1 ? len - 1 : p);
    out[i] = x[c * len + p];
}

// out[co, t] = ep(bias + Σ_{cl,k} w[co, cl·K + k] · x[g·cg + cl, t·stride + k·dil − pad0]); ep 0/1/2/3 = none/relu/addrelu/add
extern "C" __global__ void fw_conv1d(u64 x_, u64 w_, u64 bias_, u64 res_, u64 out_, int cin, int cout, int groups, int len,
                                     int k, int stride, int dil, int pad0, int ol, int has_bias, int ep) {
    const bf16_t *x = (const bf16_t*)x_, *w = (const bf16_t*)w_, *bias = (const bf16_t*)bias_, *res = (const bf16_t*)res_;
    bf16_t* out = (bf16_t*)out_;
    const long n = (long)cout * ol;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int co = (int)(i / ol), tt = (int)(i - (long)co * ol);
    const int cg = cin / groups, og = cout / groups, g = co / og;
    float acc = has_bias ? ld(bias, co) : 0.0f;
    const bf16_t* wr = w + (long)co * cg * k;
    for (int kk = 0; kk < k; ++kk) {
        const int ti = tt * stride + kk * dil - pad0;
        if (ti < 0 || ti >= len) continue;
        for (int cl = 0; cl < cg; ++cl) acc = fmaf(ld(wr, cl * k + kk), ld(x, (long)(g * cg + cl) * len + ti), acc);
    }
    if (ep == 1) acc = fmaxf(acc, 0.0f);
    if (ep == 2) acc = fmaxf(acc + ld(res, i), 0.0f);
    if (ep == 3) acc += ld(res, i);
    out[i] = bf(acc);
}

// PyTorch ConvTranspose1d: weight [cin, (cout/groups)·k]; out[co, t] = bias + Σ x[ci, i] w[ci, col, kk], t = i·stride − pad + kk.
extern "C" __global__ void fw_conv_transpose1d(u64 x_, u64 w_, u64 bias_, u64 out_, int cin, int cout, int groups, int len,
                                               int k, int stride, int pad, int ol, int has_bias) {
    const bf16_t *x = (const bf16_t*)x_, *w = (const bf16_t*)w_, *bias = (const bf16_t*)bias_;
    bf16_t* out = (bf16_t*)out_;
    const long n = (long)cout * ol;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int co = (int)(i / ol), tt = (int)(i - (long)co * ol);
    const int cg = cin / groups, og = cout / groups, g = co / og, col = co - g * og;
    float acc = has_bias ? ld(bias, co) : 0.0f;
    for (int kk = 0; kk < k; ++kk) {
        const int num = tt + pad - kk;
        if (num < 0 || num % stride != 0) continue;
        const int ii = num / stride;
        if (ii >= len) continue;
        for (int cl = 0; cl < cg; ++cl) {
            const int ci = g * cg + cl;
            acc = fmaf(ld(x, (long)ci * len + ii), ld(w, ((long)ci * og + col) * k + kk), acc);
        }
    }
    out[i] = bf(acc);
}

// ---------------------------------------------------------------- attention

// Scalar flash attention: 64 query rows (one per thread) of one head per
// block; K/V tiles of BK keys in shared memory. GQA: query head h reads KV
// head h / (heads / kv). Causal: key j allowed when j <= i + (sk − sq).
#define BK 32
extern "C" __global__ void fw_attention(u64 q_, u64 k_, u64 v_, u64 out_, int heads, int kv_heads, int sq, int sk, int d, int causal, float scale) {
    __shared__ float ks[BK * 128];
    __shared__ float vs[BK * 128];
    const bf16_t *q = (const bf16_t*)q_, *k = (const bf16_t*)k_, *v = (const bf16_t*)v_;
    bf16_t* out = (bf16_t*)out_;
    const int h = blockIdx.y, kh = h / (heads / kv_heads), i = blockIdx.x * 64 + threadIdx.x;
    const bool live = i < sq;
    const int limit = causal ? min(sk, i + 1 + sk - sq) : sk;   // keys this row may read
    const int blimit = causal ? min(sk, min(sq - 1, blockIdx.x * 64 + 63) + 1 + sk - sq) : sk;
    float qr[128], acc[128], s[BK];
    for (int c = 0; c < d; ++c) { qr[c] = live ? ld(q, ((long)h * sq + i) * d + c) * scale : 0.0f; acc[c] = 0.0f; }
    float m = -3.0e38f, l = 0.0f;
    for (int j0 = 0; j0 < blimit; j0 += BK) {
        const int nk = min(BK, sk - j0);
        for (int e = threadIdx.x; e < BK * d; e += 64) {
            const int jj = e / d;
            float kv = 0.0f, vv = 0.0f;
            if (jj < nk) { const long g = ((long)kh * sk + j0 + jj) * d + (e % d); kv = ld(k, g); vv = ld(v, g); }
            ks[e] = kv; vs[e] = vv;
        }
        __syncthreads();
        if (live) {
            const int nr = min(nk, limit - j0);
            if (nr > 0) {
                float mt = m;
                for (int jj = 0; jj < nr; ++jj) {
                    float dot = 0.0f;
                    for (int c = 0; c < d; ++c) dot = fmaf(qr[c], ks[jj * d + c], dot);
                    s[jj] = dot; mt = fmaxf(mt, dot);
                }
                const float corr = __expf(m - mt);
                l *= corr;
                for (int c = 0; c < d; ++c) acc[c] *= corr;
                for (int jj = 0; jj < nr; ++jj) {
                    const float p = __expf(s[jj] - mt);
                    l += p;
                    for (int c = 0; c < d; ++c) acc[c] = fmaf(p, vs[jj * d + c], acc[c]);
                }
                m = mt;
            }
        }
        __syncthreads();
    }
    if (live) {
        const float invl = l > 0.0f ? 1.0f / l : 0.0f;
        for (int c = 0; c < d; ++c) out[((long)i * heads + h) * d + c] = bf(acc[c] * invl);
    }
}

// Tensor-core flash attention (sm_80+): FlashAttention-2 structure on bf16
// mma.sync.m16n8k16 with f32 accumulation, templated on the head dim D (64 or
// 128). One block = 4 warps = 64 query rows of one head (16 per warp); K and
// V tiles of 64 keys in shared memory; online softmax in base 2. The S
// fragments are reused as the A operand of P·V, so P never touches shared
// memory. V fragments come from ldmatrix.trans.
#define MMA_BK 64

__device__ __forceinline__ void mma_bf16(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
    return (unsigned)bf(lo) | ((unsigned)bf(hi) << 16);
}
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, bool valid) {
    const unsigned d = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(d), "l"(gmem), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
__device__ __forceinline__ void cp_async_wait1() { asm volatile("cp.async.wait_group 1;\n" ::); }

// Body shared by dense attention (mode 0) and the VSA fine stage (mode 1).
// Mode 0: query rows qt·64.., keys in 64-chunks, GQA head kh, causal mask.
// Mode 1: query rows from slot[qt·te + qsub·64 + r]; key chunk c is 64 keys
// of selected tile sel[h][qt][c / sub] (sub = te / 64 chunks per tile); the
// epilogue adds pooled[h][qt] · gate (when has_gate).
template <int D>
__device__ __forceinline__ void attn_mma_body(int mode, const bf16_t* q, const bf16_t* k, const bf16_t* v, bf16_t* out,
        int heads, int kv_heads, int sq, int sk, int causal, float scale,
        const unsigned* slot, const unsigned* sel, const unsigned* nsel, const float* pooled, const bf16_t* gate,
        int has_gate, int nb, int np, int dense_all, int te, int maxsel,
        bf16_t* ks, bf16_t* vs, unsigned char* kvalid) {
    constexpr int KS_STRIDE = D + 8;
    constexpr int KK = D / 16;   // k-steps of Q·Kᵀ
    constexpr int DT = D / 8;    // 8-wide dim tiles of the output
    constexpr int CH = D / 8;    // 16-byte chunks per key row
    constexpr int RP = 128 / CH; // key rows loaded per pass
    const int h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int sub = mode == 0 ? 1 : te / 64;
    const int qt = mode == 0 ? blockIdx.x : blockIdx.x / sub;
    const int qsub = mode == 0 ? 0 : blockIdx.x % sub;
    const int kh = mode == 0 ? h / (heads / kv_heads) : h;
    const int lr0 = warp * 16 + g, lr1 = lr0 + 8;
    int tok0, tok1;
    if (mode == 0) {
        tok0 = qt * 64 + lr0 < sq ? qt * 64 + lr0 : -1;
        tok1 = qt * 64 + lr1 < sq ? qt * 64 + lr1 : -1;
    } else {
        const unsigned a = slot[(long)qt * te + qsub * 64 + lr0], b = slot[(long)qt * te + qsub * 64 + lr1];
        tok0 = a == 0xffffffffu ? -1 : (int)a;
        tok1 = b == 0xffffffffu ? -1 : (int)b;
    }
    const float sl2 = scale * 1.4426950408889634f;

    unsigned qa[KK][4];
    const unsigned* q0 = (const unsigned*)(q + ((long)h * sq + (tok0 < 0 ? 0 : tok0)) * D);
    const unsigned* q1 = (const unsigned*)(q + ((long)h * sq + (tok1 < 0 ? 0 : tok1)) * D);
    #pragma unroll
    for (int kk = 0; kk < KK; ++kk) {
        const int c = (kk * 16 + t * 2) >> 1;
        qa[kk][0] = tok0 >= 0 ? q0[c] : 0u;
        qa[kk][1] = tok1 >= 0 ? q1[c] : 0u;
        qa[kk][2] = tok0 >= 0 ? q0[c + 4] : 0u;
        qa[kk][3] = tok1 >= 0 ? q1[c + 4] : 0u;
    }
    float o[DT][4];
    #pragma unroll
    for (int i = 0; i < DT; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.0f, l1 = 0.0f;

    int nchunks;
    if (mode == 0) {
        nchunks = (sk + MMA_BK - 1) / MMA_BK;
        if (causal) {
            const int maxrow = min(sq - 1, qt * 64 + 63);
            const int maxkey = maxrow + sk - sq;            // inclusive
            nchunks = min(nchunks, maxkey / MMA_BK + 1);
            if (maxkey < 0) nchunks = 0;
        }
    } else {
        nchunks = (int)nsel[(long)h * nb + qt] * sub;
    }
    const bool all_tiles = mode == 1 && (qt < np || dense_all);
    const bf16_t* kbase = k + (long)kh * sk * D;
    const bf16_t* vbase = v + (long)kh * sk * D;
    const int ch = tid % CH, krow = tid / CH;
    auto key_tok = [&](int c, int key) -> int {
        if (mode == 0) return c * MMA_BK + key < sk ? c * MMA_BK + key : -1;
        const unsigned tile = all_tiles ? (unsigned)(c / sub) : sel[((long)h * nb + qt) * maxsel + c / sub];
        const unsigned s2 = slot[(long)tile * te + (c % sub) * 64 + key];
        return s2 == 0xffffffffu ? -1 : (int)s2;
    };
    auto load = [&](bf16_t* dst, const bf16_t* src, int c, unsigned char* valid) {
        #pragma unroll
        for (int j = 0; j < MMA_BK / RP; ++j) {
            const int key = krow + RP * j, tk = key_tok(c, key);
            cp_async16(dst + key * KS_STRIDE + ch * 8, src + (long)(tk < 0 ? 0 : tk) * D + ch * 8, tk >= 0);
            if (valid && ch == 0) valid[key] = tk >= 0;
        }
    };
    if (nchunks > 0) {
        load(ks, kbase, 0, kvalid);
        cp_async_commit();
        load(vs, vbase, 0, 0);
        cp_async_commit();
    }

    for (int c = 0; c < nchunks; ++c) {
        const unsigned char* kv = kvalid + (c & 1) * MMA_BK;
        cp_async_wait1();          // K_c landed (V_c may still be in flight)
        __syncthreads();
        float s[8][4];
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
            const bf16_t* kr = ks + (nt * 8 + g) * KS_STRIDE;
            #pragma unroll
            for (int kk = 0; kk < KK; ++kk) {
                const unsigned b0 = *(const unsigned*)(kr + kk * 16 + t * 2);
                const unsigned b1 = *(const unsigned*)(kr + kk * 16 + 8 + t * 2);
                mma_bf16(s[nt], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b0, b1);
            }
        }
        float mx0 = m0, mx1 = m1;
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
                const int key = nt * 8 + t * 2 + cc;
                bool ok0 = kv[key], ok1 = ok0;
                if (mode == 0 && causal) {
                    const int j = c * MMA_BK + key;
                    ok0 = ok0 && tok0 >= 0 && j <= tok0 + sk - sq;
                    ok1 = ok1 && tok1 >= 0 && j <= tok1 + sk - sq;
                }
                s[nt][cc] = ok0 ? s[nt][cc] * sl2 : -1e30f;
                s[nt][2 + cc] = ok1 ? s[nt][2 + cc] * sl2 : -1e30f;
                mx0 = fmaxf(mx0, s[nt][cc]);
                mx1 = fmaxf(mx1, s[nt][2 + cc]);
            }
        }
        __syncthreads();           // every warp is done with ks
        if (c + 1 < nchunks) load(ks, kbase, c + 1, kvalid + ((c + 1) & 1) * MMA_BK);
        cp_async_commit();
        #pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
        }
        const float c0 = exp2f(m0 - mx0), c1 = exp2f(m1 - mx1);
        m0 = mx0; m1 = mx1;
        l0 *= c0; l1 *= c1;
        #pragma unroll
        for (int i = 0; i < DT; ++i) { o[i][0] *= c0; o[i][1] *= c0; o[i][2] *= c1; o[i][3] *= c1; }
        cp_async_wait1();          // V_c landed (K_{c+1} may still be in flight)
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            float p[8];
            #pragma unroll
            for (int cc = 0; cc < 4; ++cc) {
                p[cc] = exp2f(s[2 * kk][cc] - (cc < 2 ? m0 : m1));
                p[4 + cc] = exp2f(s[2 * kk + 1][cc] - (cc < 2 ? m0 : m1));
            }
            l0 += p[0] + p[1] + p[4] + p[5];
            l1 += p[2] + p[3] + p[6] + p[7];
            const unsigned a0 = pack_bf16(p[0], p[1]), a1 = pack_bf16(p[2], p[3]);
            const unsigned a2 = pack_bf16(p[4], p[5]), a3 = pack_bf16(p[6], p[7]);
            const int vkey = kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
            #pragma unroll
            for (int dt = 0; dt < DT; dt += 2) {
                const unsigned addr = (unsigned)__cvta_generic_to_shared(vs + vkey * KS_STRIDE + dt * 8 + (lane >> 4) * 8);
                unsigned b0, b1, b2, b3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));
                mma_bf16(o[dt], a0, a1, a2, a3, b0, b1);
                mma_bf16(o[dt + 1], a0, a1, a2, a3, b2, b3);
            }
        }
        __syncthreads();           // every warp is done with vs
        if (c + 1 < nchunks) load(vs, vbase, c + 1, 0);
        cp_async_commit();
    }
    #pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, off);
        l1 += __shfl_xor_sync(0xffffffffu, l1, off);
    }
    const float i0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, i1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    #pragma unroll
    for (int dt = 0; dt < DT; ++dt) {
        const int c = dt * 8 + t * 2;
        float a0 = o[dt][0] * i0, a1 = o[dt][1] * i0, b0 = o[dt][2] * i1, b1 = o[dt][3] * i1;
        if (mode == 1 && has_gate) {
            const float* cr = pooled + ((long)h * nb + qt) * D + c;
            if (tok0 >= 0) { const bf16_t* gr = gate + ((long)h * sq + tok0) * D + c; a0 += cr[0] * ld(gr, 0); a1 += cr[1] * ld(gr, 1); }
            if (tok1 >= 0) { const bf16_t* gr = gate + ((long)h * sq + tok1) * D + c; b0 += cr[0] * ld(gr, 0); b1 += cr[1] * ld(gr, 1); }
        }
        if (tok0 >= 0) *(unsigned*)(out + ((long)tok0 * heads + h) * D + c) = pack_bf16(a0, a1);
        if (tok1 >= 0) *(unsigned*)(out + ((long)tok1 * heads + h) * D + c) = pack_bf16(b0, b1);
    }
}

#define ATTN_MMA_KERNELS(D)                                                                                          \
extern "C" __global__ void __launch_bounds__(128) fw_attention_mma##D(u64 q_, u64 k_, u64 v_, u64 out_, int heads,  \
        int kv_heads, int sq, int sk, int causal, float scale) {                                                     \
    __shared__ __align__(16) bf16_t ks[MMA_BK * (D + 8)];                                                            \
    __shared__ __align__(16) bf16_t vs[MMA_BK * (D + 8)];                                                            \
    __shared__ unsigned char kvalid[2 * MMA_BK];                                                                     \
    attn_mma_body<D>(0, (const bf16_t*)q_, (const bf16_t*)k_, (const bf16_t*)v_, (bf16_t*)out_, heads, kv_heads,    \
                     sq, sk, causal, scale, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ks, vs, kvalid);                        \
}                                                                                                                    \
extern "C" __global__ void __launch_bounds__(128) fw_vsa_fine_mma##D(u64 q_, u64 k_, u64 v_, u64 out_, int heads,   \
        int seq, float scale, u64 slot_, u64 sel_, u64 nsel_, u64 pooled_, u64 gate_, int has_gate, int nb, int np,  \
        int dense_all, int te, int maxsel) {                                                                         \
    __shared__ __align__(16) bf16_t ks[MMA_BK * (D + 8)];                                                            \
    __shared__ __align__(16) bf16_t vs[MMA_BK * (D + 8)];                                                            \
    __shared__ unsigned char kvalid[2 * MMA_BK];                                                                     \
    attn_mma_body<D>(1, (const bf16_t*)q_, (const bf16_t*)k_, (const bf16_t*)v_, (bf16_t*)out_, heads, heads,       \
                     seq, seq, 0, scale, (const unsigned*)slot_, (const unsigned*)sel_, (const unsigned*)nsel_,      \
                     (const float*)pooled_, (const bf16_t*)gate_, has_gate, nb, np, dense_all, te, maxsel, ks, vs,   \
                     kvalid);                                                                                        \
}
ATTN_MMA_KERNELS(64)
ATTN_MMA_KERNELS(128)

// ---------------------------------------------------------------- VSA

// Tile means (f32): block (tile, head), thread c < d. pooled is [3][heads][nb][d].
extern "C" __global__ void fw_vsa_pool(u64 q_, u64 k_, u64 v_, u64 slot_, u64 bsz_, u64 pooled_, int heads, int seq, int d, int nb, int te) {
    const bf16_t *q = (const bf16_t*)q_, *k = (const bf16_t*)k_, *v = (const bf16_t*)v_;
    const unsigned *slot = (const unsigned*)slot_, *bsz = (const unsigned*)bsz_;
    float* pooled = (float*)pooled_;
    const int t = blockIdx.x, h = blockIdx.y, c = threadIdx.x;
    if (c >= d) return;
    const unsigned n = bsz[t];
    float sq = 0.f, sk = 0.f, sv = 0.f;
    for (unsigned i = 0; i < n; ++i) {
        const long idx = ((long)h * seq + slot[(long)t * te + i]) * d + c;
        sq += ld(q, idx); sk += ld(k, idx); sv += ld(v, idx);
    }
    const float inv = 1.0f / (float)(n ? n : 1);
    const long o = ((long)h * nb + t) * d + c, plane = (long)heads * nb * d;
    pooled[o] = sq * inv; pooled[plane + o] = sk * inv; pooled[2 * plane + o] = sv * inv;
}

__device__ __forceinline__ unsigned okey(float x) {
    const unsigned b = __float_as_uint(x);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

#define VSA_MAX_TILES 4096

// Mark the top-`need` tiles of sc[lo, hi) in keep[] (ties: lowest index
// first, as h3_core::vsa::select): radix select on order-preserving keys, 8
// bits per pass, then a stable scan. All 256 threads participate.
__device__ void vsa_mark_topk(const float* sc, unsigned char* keep, int lo, int hi, unsigned need,
                              unsigned* hist, unsigned* wsum, unsigned* sh_prefix, unsigned* sh_need) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    unsigned prefix = 0u, mask = 0u;
    for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        hist[tid] = 0u;
        __syncthreads();
        for (int kt = lo + tid; kt < hi; kt += 256) {
            const unsigned key = okey(sc[kt]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (warp == 0) {
            unsigned c[8], s = 0u;
            #pragma unroll
            for (int b = 0; b < 8; ++b) { c[b] = hist[255 - 8 * lane - b]; s += c[b]; }
            unsigned incl = s;
            #pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const unsigned y = __shfl_up_sync(0xffffffffu, incl, o);
                if (lane >= o) incl += y;
            }
            const unsigned excl = incl - s;
            const unsigned hit = __ballot_sync(0xffffffffu, incl >= need && excl < need);
            if (lane == __ffs(hit) - 1) {
                unsigned above = excl;
                int b = 0;
                for (; b < 7 && above + c[b] < need; ++b) above += c[b];
                const unsigned bin = 255u - 8u * lane - b;
                *sh_prefix = prefix | (bin << shift);
                *sh_need = need - above;
            }
        }
        __syncthreads();
        prefix = *sh_prefix; need = *sh_need; mask |= 255u << shift;
        __syncthreads();
    }
    // prefix = threshold key T; take every key > T and the first `need` keys == T (ascending index).
    const unsigned lt = (1u << lane) - 1u;
    unsigned base_eq = 0u;
    for (int c0 = lo; c0 < hi; c0 += 256) {
        const int kt = c0 + tid;
        const unsigned key = kt < hi ? okey(sc[kt]) : 0u;
        const bool eq = kt < hi && key == prefix, gt = kt < hi && key > prefix;
        const unsigned b = __ballot_sync(0xffffffffu, eq);
        if (lane == 0) wsum[warp] = __popc(b);
        __syncthreads();
        unsigned woff = 0u, tot = 0u;
        for (int w = 0; w < 8; ++w) { if (w < warp) woff += wsum[w]; tot += wsum[w]; }
        const unsigned tie_rank = base_eq + woff + __popc(b & lt);
        base_eq += tot;
        __syncthreads();
        if (kt < hi) keep[kt] = (gt || (eq && tie_rank < need)) ? 1 : 0;
    }
    __syncthreads();
}

// VSA selection + pooled branch: block (query tile, head), 256 threads;
// nb <= 4096, d <= 128, d % 4 == 0. Scores over all tiles (one warp per key
// tile); prefix-tile queries and dense plans keep every tile, video queries
// keep every prefix tile plus the top-k of each region (regions [start, end,
// topk] × nreg); sel[h][qt][0..nsel) ascending. With has_gate the softmax
// over all tile scores times the pooled V goes to pooled_out[h][qt][d].
extern "C" __global__ void __launch_bounds__(256) fw_vsa_coarse(u64 pooled_, u64 pooled_out_, u64 sel_, u64 nsel_, u64 reg_,
        int heads, int d, int nb, int np, int nreg, int dense_all, int maxsel, int has_gate, float scale) {
    __shared__ float sc[VSA_MAX_TILES];
    __shared__ unsigned char keep[VSA_MAX_TILES];
    __shared__ float qrow[128];
    __shared__ float part[128];
    __shared__ unsigned hist[256];
    __shared__ unsigned wsum[8];
    __shared__ float red[9];
    __shared__ unsigned sh_prefix, sh_need;
    const float* pooled = (const float*)pooled_;
    float* pooled_out = (float*)pooled_out_;
    unsigned* sel = (unsigned*)sel_;
    unsigned* nsel = (unsigned*)nsel_;
    const unsigned* reg = (const unsigned*)reg_;
    const int qt = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const long plane = (long)heads * nb * d, qb = ((long)h * nb + qt) * d;
    if (tid < d) qrow[tid] = pooled[qb + tid];
    __syncthreads();

    // 1. scores
    const int d4 = d >> 2;
    for (int kt = warp; kt < nb; kt += 8) {
        const float4* kr = (const float4*)(pooled + plane + ((long)h * nb + kt) * d);
        float dot = 0.f;
        for (int c = lane; c < d4; c += 32) {
            const float4 kv = kr[c];
            dot += qrow[4 * c] * kv.x + qrow[4 * c + 1] * kv.y + qrow[4 * c + 2] * kv.z + qrow[4 * c + 3] * kv.w;
        }
        for (int o = 16; o > 0; o >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, o);
        if (lane == 0) sc[kt] = dot * scale;
    }
    __syncthreads();

    // 2. selection marks
    if (qt < np || dense_all) {
        for (int kt = tid; kt < nb; kt += 256) keep[kt] = 1;
        __syncthreads();
    } else {
        for (int kt = tid; kt < nb; kt += 256) keep[kt] = kt < np ? 1 : 0;
        __syncthreads();
        for (int r = 0; r < nreg; ++r) {
            const int lo = (int)reg[3 * r], hi = (int)reg[3 * r + 1];
            const unsigned need = reg[3 * r + 2];
            if (need >= (unsigned)(hi - lo)) {
                for (int kt = lo + tid; kt < hi; kt += 256) keep[kt] = 1;
                __syncthreads();
            } else {
                vsa_mark_topk(sc, keep, lo, hi, need, hist, wsum, &sh_prefix, &sh_need);
            }
        }
    }

    // 3. stable compaction of keep[] into sel (ascending kt); all-tile queries
    //    (prefix rows, dense plans) are read directly by the fine stage.
    unsigned* o = sel + ((long)h * nb + qt) * maxsel;
    const unsigned lt = (1u << lane) - 1u;
    unsigned base_sel = (qt < np || dense_all) ? (unsigned)nb : 0u;
    for (int c0 = (qt < np || dense_all) ? nb : 0; c0 < nb; c0 += 256) {
        const int kt = c0 + tid;
        const bool take = kt < nb && keep[kt];
        const unsigned b = __ballot_sync(0xffffffffu, take);
        if (lane == 0) wsum[warp] = __popc(b);
        __syncthreads();
        unsigned woff = 0u, tot = 0u;
        for (int w = 0; w < 8; ++w) { if (w < warp) woff += wsum[w]; tot += wsum[w]; }
        if (take) o[base_sel + woff + __popc(b & lt)] = (unsigned)kt;
        base_sel += tot;
        __syncthreads();
    }
    if (tid == 0) nsel[(long)h * nb + qt] = base_sel;

    // 4. pooled branch: softmax over the tile scores times pooled V.
    if (!has_gate) return;
    float mx = -3.0e38f;
    for (int kt = tid; kt < nb; kt += 256) mx = fmaxf(mx, sc[kt]);
    for (int s2 = 16; s2 > 0; s2 >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, s2));
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    float m = red[0];
    for (int i = 1; i < 8; ++i) m = fmaxf(m, red[i]);
    __syncthreads();
    float z = 0.f;
    for (int kt = tid; kt < nb; kt += 256) { const float e = __expf(sc[kt] - m); sc[kt] = e; z += e; }
    const float zt = block_sum256(z, red);   // syncs, so sc[] holds the weights
    const int c = tid & 127, half = tid >> 7, mid = (nb + 1) >> 1;
    float acc = 0.f;
    if (c < d) {
        const float* vp = pooled + 2 * plane + (long)h * nb * d + c;
        for (int kt = half ? mid : 0; kt < (half ? nb : mid); ++kt) acc = fmaf(sc[kt], vp[(long)kt * d], acc);
        if (half) part[c] = acc;
    }
    __syncthreads();
    if (!half && c < d) pooled_out[qb + c] = (acc + part[c]) / zt;
}

// Scalar VSA fine stage (any d <= 128, any SM, any tile size): 64 threads
// take the query tile's slots in groups of 64; selected tiles streamed through
// shared memory in halves of 32 keys.
extern "C" __global__ void fw_vsa_fine(u64 q_, u64 k_, u64 v_, u64 out_, int heads, int seq, int d, float scale,
        u64 slot_, u64 sel_, u64 nsel_, u64 pooled_, u64 gate_, int has_gate, int nb, int np, int dense_all, int te, int maxsel) {
    __shared__ float kss[32 * 128];
    __shared__ float vss[32 * 128];
    __shared__ int ktok[32];
    const bf16_t *q = (const bf16_t*)q_, *k = (const bf16_t*)k_, *v = (const bf16_t*)v_, *gate = (const bf16_t*)gate_;
    const unsigned *slot = (const unsigned*)slot_, *sel = (const unsigned*)sel_, *nsel = (const unsigned*)nsel_;
    const float* pooled = (const float*)pooled_;
    bf16_t* out = (bf16_t*)out_;
    const int qt = blockIdx.x, h = blockIdx.y;
    const bool all_tiles = qt < np || dense_all;
    const unsigned ns = all_tiles ? (unsigned)nb : nsel[(long)h * nb + qt];
    const unsigned* mysel = sel + ((long)h * nb + qt) * maxsel;
    for (int base = 0; base < te; base += 64) {
        const unsigned qs = slot[(long)qt * te + base + threadIdx.x];
        const bool live = qs != 0xffffffffu;
        float qr[128], acc[128];
        for (int c = 0; c < d; ++c) { qr[c] = live ? ld(q, ((long)h * seq + qs) * d + c) * scale : 0.f; acc[c] = 0.f; }
        float m = -1e30f, l = 0.f;
        for (unsigned ci = 0; ci < ns; ++ci) {
            const unsigned tile = all_tiles ? ci : mysel[ci];
            for (int half = 0; half < te; half += 32) {
                __syncthreads();
                if (threadIdx.x < 32) { const unsigned s = slot[(long)tile * te + half + threadIdx.x]; ktok[threadIdx.x] = s == 0xffffffffu ? -1 : (int)s; }
                __syncthreads();
                for (int e = threadIdx.x; e < 32 * d; e += 64) {
                    const int j = e / d, c = e % d;
                    const int kt = ktok[j];
                    kss[e] = kt >= 0 ? ld(k, ((long)h * seq + kt) * d + c) : 0.f;
                    vss[e] = kt >= 0 ? ld(v, ((long)h * seq + kt) * d + c) : 0.f;
                }
                __syncthreads();
                if (live) {
                    for (int j = 0; j < 32; ++j) {
                        if (ktok[j] < 0) continue;
                        float dot = 0.f;
                        for (int c = 0; c < d; ++c) dot = fmaf(qr[c], kss[j * d + c], dot);
                        const float mn = fmaxf(m, dot), corr = __expf(m - mn), p = __expf(dot - mn);
                        l = l * corr + p;
                        for (int c = 0; c < d; ++c) acc[c] = acc[c] * corr + p * vss[j * d + c];
                        m = mn;
                    }
                }
            }
        }
        if (live) {
            const float invl = l > 0.f ? 1.0f / l : 0.f;
            const float* cr = pooled + ((long)h * nb + qt) * d;
            for (int c = 0; c < d; ++c) {
                float o = acc[c] * invl;
                if (has_gate) o += cr[c] * ld(gate, ((long)h * seq + qs) * d + c);
                out[((long)qs * heads + h) * d + c] = bf(o);
            }
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------- dequant

// int8 weight-only (h3_core::quant::Q8): one f32 scale per `group` columns.
extern "C" __global__ void fw_dequant_q8(u64 out_, u64 q_, u64 s_, long n, int group) {
    bf16_t* out = (bf16_t*)out_;
    const signed char* q = (const signed char*)q_;
    const float* s = (const float*)s_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = bf((float)q[i] * s[i / group]);
}

// 4-bit weight-only (h3_core::quant::Q4): codes 2 per byte (low nibble first), (scale, zero) pairs per group.
extern "C" __global__ void fw_dequant_q4(u64 out_, u64 q_, u64 sz_, long n, int group) {
    bf16_t* out = (bf16_t*)out_;
    const unsigned char* q = (const unsigned char*)q_;
    const float* sz = (const float*)sz_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float c = (float)((q[i >> 1] >> ((i & 1) * 4)) & 15);
    long g = i / group;
    out[i] = bf((c - sz[2 * g + 1]) * sz[2 * g]);
}

// NVFP4 (h3_core::quant::Nvfp4): E2M1 codes 2 per byte (low nibble = even
// column), one E4M3 scale per 16 columns (row-major), one alpha. Expands to
// bf16 at load; a Blackwell FP4 tensor-core GEMM path is future work.
__device__ __forceinline__ float e2m1(unsigned c) {
    const float mag[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    const float m = mag[c & 7];
    return (c & 8) ? -m : m;
}
__device__ __forceinline__ float e4m3(unsigned b) {
    const float sign = (b & 0x80) ? -1.0f : 1.0f;
    const int exp = (b >> 3) & 0xF;
    const float man = (float)(b & 7);
    if (exp == 0) return sign * man / 8.0f * 0.015625f;  // 2^-6
    if (exp == 15 && man == 7.0f) return __int_as_float(0x7fc00000);
    return sign * (1.0f + man / 8.0f) * exp2f((float)(exp - 7));
}
extern "C" __global__ void fw_dequant_nvfp4(u64 out_, u64 q_, u64 s_, long n, int cols, float alpha) {
    bf16_t* out = (bf16_t*)out_;
    const unsigned char* q = (const unsigned char*)q_;
    const unsigned char* s = (const unsigned char*)s_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long r = i / cols; const int c = (int)(i - r * cols);
    const unsigned code = (q[i >> 1] >> ((i & 1) * 4)) & 15u;
    const float sc = e4m3(s[r * (cols / 16) + c / 16]);
    out[i] = bf(e2m1(code) * sc * alpha);
}

// ---------------------------------------------------------------- 2-D conv (TAEH3)

// Implicit-GEMM conv2d: 64×64 tile, 16×16 threads, 4×4 outputs per thread.
// ep 0 none, 1 relu, 2 relu(+res), 3 +res.
extern "C" __global__ void fw_conv2d(u64 a_, u64 b_, u64 w_, u64 bias_, u64 res_, u64 out_,
                                  int frames, int cin_a, int cin_b, int cout, int h, int w,
                                  int up, int k, int has_bias, int ep) {
    __shared__ float xs[16][64];
    __shared__ float wsm[16][64];
    const bf16_t *a = (const bf16_t*)a_, *b = (const bf16_t*)b_, *wt = (const bf16_t*)w_;
    const bf16_t *bias = (const bf16_t*)bias_, *res = (const bf16_t*)res_;
    bf16_t* out = (bf16_t*)out_;
    const int oh = h * up, ow = w * up, kk2 = k * k, kt = (cin_a + cin_b) * kk2, pad = k / 2;
    const long plane = (long)oh * ow, mt = frames * plane;
    const long m0 = ((long)blockIdx.y + (long)blockIdx.z * gridDim.y) * 64;
    const int n0 = blockIdx.x * 64, tx = threadIdx.x, ty = threadIdx.y, tid = ty * 16 + tx;
    float acc[4][4] = {};
    for (int k0 = 0; k0 < kt; k0 += 16) {
        for (int i = 0; i < 4; ++i) {
            const int e = tid + i * 256, r = e / 16, kk = e % 16, gk = k0 + kk;
            float xv = 0.0f;
            const long m = m0 + r;
            if (m < mt && gk < kt) {
                const long f = m / plane, rr = m % plane;
                const int oy = rr / ow, ox = rr % ow;
                const int ci = gk / kk2, q = gk % kk2, ky = q / k, kx = q % k;
                const int yy = oy + ky - pad, xx = ox + kx - pad;
                if (yy >= 0 && xx >= 0 && yy < oh && xx < ow) {
                    const int sy = yy / up, sx = xx / up;
                    xv = ci < cin_a ? ld(a, ((f * cin_a + ci) * h + sy) * w + sx)
                                    : ld(b, ((f * cin_b + ci - cin_a) * h + sy) * w + sx);
                }
            }
            xs[kk][r] = xv;
            wsm[kk][r] = (n0 + r < cout && gk < kt) ? ld(wt, (long)(n0 + r) * kt + gk) : 0.0f;
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < 16; ++kk) {
            float av[4], bv[4];
            for (int i = 0; i < 4; ++i) { av[i] = xs[kk][ty + 16 * i]; bv[i] = wsm[kk][tx + 16 * i]; }
            for (int i = 0; i < 4; ++i) for (int j = 0; j < 4; ++j) acc[i][j] = fmaf(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
    }
    for (int i = 0; i < 4; ++i) {
        const long m = m0 + ty + 16 * i;
        if (m >= mt) continue;
        const long f = m / plane, px = m % plane;
        for (int j = 0; j < 4; ++j) {
            const int n = n0 + tx + 16 * j;
            if (n >= cout) continue;
            float v = acc[i][j] + (has_bias ? ld(bias, n) : 0.0f);
            const long o = (f * cout + n) * plane + px;
            if (ep == 1) v = fmaxf(v, 0.0f);
            if (ep == 2) v = fmaxf(v + ld(res, o), 0.0f);
            if (ep == 3) v += ld(res, o);
            out[o] = bf(v);
        }
    }
}

// Tensor-core implicit-GEMM conv2d (sm_80+): M = output pixels, N = cout,
// K = (cin_a + cin_b)·k² in (ci, ky, kx) order. Block tile 128 × 64 × 32, 4
// warps as 2 × 2, 64 × 32 per warp; each thread gathers one pixel row of A.
#define CV_BM 128
#define CV_BN 64
#define CV_BK 32
#define CV_LD 20   // u32 words per smem row: 32 bf16 + 8 pad

extern "C" __global__ void __launch_bounds__(128) fw_conv2d_mma(
        u64 a_, u64 b_, u64 w_, u64 bias_, u64 res_, u64 out_,
        int frames, int cin_a, int cin_b, int cout, int h, int w,
        int up, int k, int has_bias, int ep) {
    __shared__ unsigned As[CV_BM * CV_LD];
    __shared__ unsigned Bs[CV_BN * CV_LD];
    const unsigned short *a = (const unsigned short*)a_, *b = (const unsigned short*)b_;
    const unsigned short *wt = (const unsigned short*)w_;
    const bf16_t *bias = (const bf16_t*)bias_, *res = (const bf16_t*)res_;
    bf16_t* out = (bf16_t*)out_;
    const int oh = h * up, ow = w * up, kk2 = k * k, kt = (cin_a + cin_b) * kk2, pad = k / 2;
    const long plane = (long)oh * ow, mt = (long)frames * plane;
    const long m0 = ((long)blockIdx.y + (long)blockIdx.z * gridDim.y) * CV_BM;
    const int n0 = blockIdx.x * CV_BN;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 1, wn = warp >> 1, g = lane >> 2, t = lane & 3;

    const long m = m0 + tid;
    const bool mok = m < mt;
    long fa = 0, fb = 0;
    int oy = 0, ox = 0;
    if (mok) {
        const long f = m / plane, rr = m - f * plane;
        oy = (int)(rr / ow); ox = (int)(rr - (long)oy * ow);
        fa = f * cin_a * (long)h * w;
        fb = f * (long)cin_b * h * w;
    }
    const long hw = (long)h * w;
    const int bn = tid >> 1, bh = (tid & 1) * 16;
    const bool bok = n0 + bn < cout;
    const unsigned short* wrow = wt + (long)(n0 + bn) * kt;

    unsigned ra[16], rb[8];
    auto fetch = [&](int k0) {
        int ci = k0 / kk2, q = k0 - ci * kk2, ky = q / k, kx = q - ky * k;
        #pragma unroll
        for (int p = 0; p < 16; ++p) {
            unsigned v[2];
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int gk = k0 + 2 * p + e;
                unsigned short x = 0;
                if (mok && gk < kt) {
                    const int yy = oy + ky - pad, xx = ox + kx - pad;
                    if (yy >= 0 && xx >= 0 && yy < oh && xx < ow) {
                        const int sy = up == 2 ? yy >> 1 : yy, sx = up == 2 ? xx >> 1 : xx;
                        x = ci < cin_a ? a[fa + ci * hw + (long)sy * w + sx]
                                       : b[fb + (ci - cin_a) * hw + (long)sy * w + sx];
                    }
                }
                v[e] = x;
                if (++kx == k) { kx = 0; if (++ky == k) { ky = 0; ++ci; } }
            }
            ra[p] = v[0] | (v[1] << 16);
        }
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int gk = k0 + bh + 2 * p;
            const unsigned lo = (bok && gk < kt) ? wrow[gk] : 0u;
            const unsigned hi = (bok && gk + 1 < kt) ? wrow[gk + 1] : 0u;
            rb[p] = lo | (hi << 16);
        }
    };

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.0f;

    fetch(0);
    for (int k0 = 0; k0 < kt; k0 += CV_BK) {
        #pragma unroll
        for (int p = 0; p < 16; ++p) As[tid * CV_LD + p] = ra[p];
        #pragma unroll
        for (int p = 0; p < 8; ++p) Bs[bn * CV_LD + (bh >> 1) + p] = rb[p];
        __syncthreads();
        if (k0 + CV_BK < kt) fetch(k0 + CV_BK);
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            const int kw = ks * 8 + t;
            unsigned bfr[4][2];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int n = wn * 32 + j * 8 + g;
                bfr[j][0] = Bs[n * CV_LD + kw];
                bfr[j][1] = Bs[n * CV_LD + kw + 4];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r0 = wm * 64 + i * 16 + g;
                const unsigned a0 = As[r0 * CV_LD + kw], a1 = As[(r0 + 8) * CV_LD + kw];
                const unsigned a2 = As[r0 * CV_LD + kw + 4], a3 = As[(r0 + 8) * CV_LD + kw + 4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) mma_bf16(acc[i][j], a0, a1, a2, a3, bfr[j][0], bfr[j][1]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int hrow = 0; hrow < 2; ++hrow) {
            const long mm = m0 + wm * 64 + i * 16 + g + hrow * 8;
            if (mm >= mt) continue;
            const long f = mm / plane, px = mm - f * plane;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                #pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int n = n0 + wn * 32 + j * 8 + 2 * t + c;
                    if (n >= cout) continue;
                    float v = acc[i][j][hrow * 2 + c] + (has_bias ? ld(bias, n) : 0.0f);
                    const long o = (f * cout + n) * plane + px;
                    if (ep == 1) v = fmaxf(v, 0.0f);
                    if (ep == 2) v = fmaxf(v + ld(res, o), 0.0f);
                    if (ep == 3) v += ld(res, o);
                    out[o] = bf(v);
                }
            }
        }
    }
}

// ---------------------------------------------------------------- 3-D conv (VAE, upscaler)

// Generic conv3d: one thread per output element. groups, strides, temporal
// zero padding (front/back), spatial zero or reflect padding. Weight
// [cout, (cin/groups)·kt·kh·kw]. The fallback for grouped / pre-Ampere.
extern "C" __global__ void fw_conv3d(u64 x_, u64 w_, u64 bias_, u64 res_, u64 out_,
                                     int cin, int cout, int groups, int t, int h, int w,
                                     int kt, int kh, int kw, int st, int sh, int sw,
                                     int pt0, int ph0, int pw0, int reflect, int ot, int oh, int ow, int has_bias, int ep) {
    const bf16_t *x = (const bf16_t*)x_, *wt = (const bf16_t*)w_, *bias = (const bf16_t*)bias_, *res = (const bf16_t*)res_;
    bf16_t* out = (bf16_t*)out_;
    const long n = (long)cout * ot * oh * ow;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int xo = (int)(i % ow), yo = (int)((i / ow) % oh), to = (int)((i / ((long)ow * oh)) % ot);
    const int co = (int)(i / ((long)ow * oh * ot));
    const int cg = cin / groups, og = cout / groups, g = co / og;
    const int klen = cg * kt * kh * kw;
    const bf16_t* wr = wt + (long)co * klen;
    float acc = has_bias ? ld(bias, co) : 0.0f;
    for (int dt = 0; dt < kt; ++dt) {
        const int ti = to * st + dt - pt0;
        if (ti < 0 || ti >= t) continue;
        for (int dy = 0; dy < kh; ++dy) {
            const int yi = pad_index(yo * sh + dy - ph0, h, reflect);
            if (yi < 0) continue;
            for (int dx = 0; dx < kw; ++dx) {
                const int xi = pad_index(xo * sw + dx - pw0, w, reflect);
                if (xi < 0) continue;
                for (int cl = 0; cl < cg; ++cl) {
                    const int ci = g * cg + cl;
                    acc = fmaf(ld(wr, ((cl * kt + dt) * kh + dy) * kw + dx), ld(x, (((long)ci * t + ti) * h + yi) * (long)w + xi), acc);
                }
            }
        }
    }
    if (ep == 1) acc = fmaxf(acc, 0.0f);
    if (ep == 2) acc = fmaxf(acc + ld(res, i), 0.0f);
    if (ep == 3) acc += ld(res, i);
    out[i] = bf(acc);
}

// Tensor-core implicit-GEMM conv3d (sm_80+, groups == 1): M = output voxels
// (ot·oh·ow), N = cout, K = cin·kt·kh·kw in (ci, dt, dy, dx) order; the same
// 128 × 64 × 32 tile and fragment scheme as fw_conv2d_mma. Each thread owns
// one output voxel of the A tile; (ci, dt, dy, dx) advance by counting.
extern "C" __global__ void __launch_bounds__(128) fw_conv3d_mma(
        u64 x_, u64 w_, u64 bias_, u64 res_, u64 out_,
        int cin, int cout, int t, int h, int w,
        int kt, int kh, int kw, int st, int sh, int sw,
        int pt0, int ph0, int pw0, int reflect, int ot, int oh, int ow, int has_bias, int ep) {
    __shared__ unsigned As[CV_BM * CV_LD];
    __shared__ unsigned Bs[CV_BN * CV_LD];
    const unsigned short *x = (const unsigned short*)x_, *wt = (const unsigned short*)w_;
    const bf16_t *bias = (const bf16_t*)bias_, *res = (const bf16_t*)res_;
    bf16_t* out = (bf16_t*)out_;
    const int khw = kh * kw, kvol = kt * khw, klen = cin * kvol;
    const long plane = (long)ot * oh * ow;   // M
    const long m0 = ((long)blockIdx.y + (long)blockIdx.z * gridDim.y) * CV_BM;
    const int n0 = blockIdx.x * CV_BN;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp & 1, wn = warp >> 1, g = lane >> 2, t2 = lane & 3;

    const long m = m0 + tid;
    const bool mok = m < plane;
    int to = 0, oy = 0, ox = 0;
    if (mok) {
        ox = (int)(m % ow); oy = (int)((m / ow) % oh); to = (int)(m / ((long)ow * oh));
    }
    const long thw = (long)t * h * w, hw = (long)h * w;
    const int bn = tid >> 1, bh = (tid & 1) * 16;
    const bool bok = n0 + bn < cout;
    const unsigned short* wrow = wt + (long)(n0 + bn) * klen;

    unsigned ra[16], rb[8];
    auto fetch = [&](int k0) {
        int ci = k0 / kvol, q = k0 - ci * kvol, dt = q / khw, q2 = q - dt * khw, dy = q2 / kw, dx = q2 - dy * kw;
        #pragma unroll
        for (int p = 0; p < 16; ++p) {
            unsigned v[2];
            #pragma unroll
            for (int e = 0; e < 2; ++e) {
                const int gk = k0 + 2 * p + e;
                unsigned short xv = 0;
                if (mok && gk < klen) {
                    const int ti = to * st + dt - pt0;
                    if (ti >= 0 && ti < t) {
                        const int yi = pad_index(oy * sh + dy - ph0, h, reflect);
                        const int xi = pad_index(ox * sw + dx - pw0, w, reflect);
                        if (yi >= 0 && xi >= 0) xv = x[ci * thw + (long)ti * hw + (long)yi * w + xi];
                    }
                }
                v[e] = xv;
                if (++dx == kw) { dx = 0; if (++dy == kh) { dy = 0; if (++dt == kt) { dt = 0; ++ci; } } }
            }
            ra[p] = v[0] | (v[1] << 16);
        }
        #pragma unroll
        for (int p = 0; p < 8; ++p) {
            const int gk = k0 + bh + 2 * p;
            const unsigned lo = (bok && gk < klen) ? wrow[gk] : 0u;
            const unsigned hi = (bok && gk + 1 < klen) ? wrow[gk + 1] : 0u;
            rb[p] = lo | (hi << 16);
        }
    };

    float acc[4][4][4];
    #pragma unroll
    for (int i = 0; i < 4; ++i)
        #pragma unroll
        for (int j = 0; j < 4; ++j)
            #pragma unroll
            for (int r = 0; r < 4; ++r) acc[i][j][r] = 0.0f;

    fetch(0);
    for (int k0 = 0; k0 < klen; k0 += CV_BK) {
        #pragma unroll
        for (int p = 0; p < 16; ++p) As[tid * CV_LD + p] = ra[p];
        #pragma unroll
        for (int p = 0; p < 8; ++p) Bs[bn * CV_LD + (bh >> 1) + p] = rb[p];
        __syncthreads();
        if (k0 + CV_BK < klen) fetch(k0 + CV_BK);
        #pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
            const int kwd = ks * 8 + t2;
            unsigned bfr[4][2];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int n = wn * 32 + j * 8 + g;
                bfr[j][0] = Bs[n * CV_LD + kwd];
                bfr[j][1] = Bs[n * CV_LD + kwd + 4];
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int r0 = wm * 64 + i * 16 + g;
                const unsigned a0 = As[r0 * CV_LD + kwd], a1 = As[(r0 + 8) * CV_LD + kwd];
                const unsigned a2 = As[r0 * CV_LD + kwd + 4], a3 = As[(r0 + 8) * CV_LD + kwd + 4];
                #pragma unroll
                for (int j = 0; j < 4; ++j) mma_bf16(acc[i][j], a0, a1, a2, a3, bfr[j][0], bfr[j][1]);
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int hrow = 0; hrow < 2; ++hrow) {
            const long mm = m0 + wm * 64 + i * 16 + g + hrow * 8;
            if (mm >= plane) continue;
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                #pragma unroll
                for (int c = 0; c < 2; ++c) {
                    const int n = n0 + wn * 32 + j * 8 + 2 * t2 + c;
                    if (n >= cout) continue;
                    float v = acc[i][j][hrow * 2 + c] + (has_bias ? ld(bias, n) : 0.0f);
                    const long o = (long)n * plane + mm;
                    if (ep == 1) v = fmaxf(v, 0.0f);
                    if (ep == 2) v = fmaxf(v + ld(res, o), 0.0f);
                    if (ep == 3) v += ld(res, o);
                    out[o] = bf(v);
                }
            }
        }
    }
}
