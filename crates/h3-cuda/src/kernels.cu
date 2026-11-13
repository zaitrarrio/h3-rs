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
// Hardware round-to-nearest-even (sm_80+); the integer sequence it replaces
// was ~11 instructions per value, a third of the attention loop's ALU work.
__device__ __forceinline__ bf16_t bf(float f) {
    unsigned short r;
    asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(r) : "f"(f));
    return r;
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
__device__ __forceinline__ float block_fsum256(float v, float* red) {
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
    __shared__ float red[9];
    bf16_t* h = (bf16_t*)h_;
    const bf16_t* a = (const bf16_t*)a_;
    const bf16_t* w = (const bf16_t*)w_;
    bf16_t* out = (bf16_t*)out_;
    const int row = blockIdx.x;
    if (row >= rows) return;
    const long base = (long)row * dim;
    float cache[RMS_REG];
    const bool reg = dim <= 256 * RMS_REG;
    // f32 accumulation: the inputs are bf16 and double runs at 1/32 rate on
    // GeForce / RTX workstation parts (this kernel was 5 ms/call on an A6000).
    // The loops are unrolled over RMS_REG so `cache` lives in registers, not
    // in a local-memory stack frame (it did: 96 bytes per thread).
    float ss = 0.0f;
    #pragma unroll
    for (int i = 0; i < RMS_REG; ++i) {
        const int j = threadIdx.x + 256 * i;
        if (j >= dim) break;
        float v = ld(h, base + j);
        if (mode == 1) {
            const float g = gate_at(gkind, (const bf16_t*)gvec_, (const float*)gtab_, (const unsigned*)gidx_, gslots, gslot, row, dim, j);
            v += ld(a, base + j) * g;
            h[base + j] = bf(v);
        }
        cache[i] = v;
        ss += v * v;
    }
    for (int j = threadIdx.x + 256 * RMS_REG; j < dim; j += 256) {   // dims beyond the cache
        float v = ld(h, base + j);
        if (mode == 1) {
            const float g = gate_at(gkind, (const bf16_t*)gvec_, (const float*)gtab_, (const unsigned*)gidx_, gslots, gslot, row, dim, j);
            v += ld(a, base + j) * g;
            h[base + j] = bf(v);
        }
        ss += v * v;
    }
    const float ms = block_fsum256(ss, red) / dim;
    const float inv = rsqrtf(ms + eps);
    const float* mt = (const float*)mtab_;
    const unsigned* mi = (const unsigned*)midx_;
    long mbase = 0;
    if (has_mod) mbase = (long)mi[row] * mslots * dim;
    #pragma unroll
    for (int i = 0; i < RMS_REG; ++i) {
        const int j = threadIdx.x + 256 * i;
        if (j >= dim) break;
        float y = cache[i] * inv;
        if (has_w) y *= ld(w, j);
        if (has_mod) y = y * (1.0f + mt[mbase + (long)mscale * dim + j]) + mt[mbase + (long)mshift * dim + j];
        out[base + j] = bf(y);
    }
    for (int j = threadIdx.x + 256 * RMS_REG; j < dim; j += 256) {
        float y = ld(h, base + j) * inv;
        if (has_w) y *= ld(w, j);
        if (has_mod) y = y * (1.0f + mt[mbase + (long)mscale * dim + j]) + mt[mbase + (long)mshift * dim + j];
        out[base + j] = bf(y);
    }
    (void)reg;
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

// One warp per head (hd % 32 == 0, hd <= 128): lane l owns channels
// [l·k, l·k+k) with k = hd/32; the sum of squares is a warp reduction and the
// RoPE partner channel is re-read from global (L1-resident). Replaces the
// shared-memory atomics of fw_head_split for the DiT's 128-wide heads.
extern "C" __global__ void fw_head_split_warp(u64 src_, u64 out_, int rows, int stride, int col_off, int heads, int hd,
                                              int has_norm, int has_w, u64 w_, float eps, int has_rope, u64 cos_, u64 sin_, int rot) {
    const bf16_t *src = (const bf16_t*)src_, *w = (const bf16_t*)w_;
    const float *cs = (const float*)cos_, *sn = (const float*)sin_;
    bf16_t* out = (bf16_t*)out_;
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, k = hd >> 5;
    const long base = (long)r * stride + col_off;
    const int half = rot / 2;
    for (int hh = warp; hh < heads; hh += 8) {
        const long hbase = base + (long)hh * hd;
        float x[4];
        float ss = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            x[i] = i < k ? ld(src, hbase + lane * k + i) : 0.0f;
            ss += x[i] * x[i];
        }
        float hinv = 1.0f;
        if (has_norm) {
            for (int o = 16; o > 0; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
            hinv = rsqrtf(ss / hd + eps);
        }
        const long o = ((long)hh * rows + r) * hd;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            if (i >= k) break;
            const int c = lane * k + i;
            if (has_rope && c >= half && c < rot) continue;  // written by the partner channel
            float x1 = x[i];
            if (has_norm) { x1 *= hinv; if (has_w) x1 *= ld(w, c); }
            if (has_rope && c < half) {
                float x2 = ld(src, hbase + c + half);
                if (has_norm) { x2 *= hinv; if (has_w) x2 *= ld(w, c + half); }
                const long ti = (long)r * half + c;
                const float co = cs[ti], si = sn[ti];
                out[o + c] = bf(x1 * co - x2 * si);
                out[o + c + half] = bf(x2 * co + x1 * si);
            } else {
                out[o + c] = bf(x1);
            }
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
    else { a = expf(a); inv = 1.0f / (expf(ld(be, c)) + 1e-9f); }  // precise exp: the fast intrinsic put `snake beta` at rel 3.0e-2 on driver 615
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
// m16n8k32 int8 × int8 → int32 (sm_80+): A = 16×32 row, B = 32×8 col.
__device__ __forceinline__ void mma_s8(int* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned mx_e4m3(float x);
// m16n8k32 e4m3 × e4m3 → f32 (sm_89+): A = 16×32 row, B = 32×8 col. Below
// sm_89 the host never launches a kernel that reaches it.
__device__ __forceinline__ void mma_e4m3(float* c, unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#else
    asm volatile("trap;");
#endif
}
// Two f32 → packed E4M3 (cvt.rn.satfinite, sm_89+; the twin of mx_e4m3 and
// of quant::e4m3_satfinite), hi in the upper byte.
__device__ __forceinline__ unsigned pack_e4m3x2(float lo, float hi) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
#else
    return mx_e4m3(lo) | (mx_e4m3(hi) << 8);
#endif
}
__device__ __forceinline__ unsigned pack_bf16(float lo, float hi) {
    unsigned r;
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));   // hi lands in the upper half
    return r;
}
// B fragments of Q·Kᵀ for one 8-key n-tile and two 16-dim k-steps in one
// ldmatrix.x4 (lanes 0-7 address keys at dims kk·16, 8-15 at +8, 16-23 at
// +16, 24-31 at +24): b0, b1 of k-step kk and of kk+1.
__device__ __forceinline__ void ldsm_k4(const bf16_t* krow8, int lane, unsigned& b0, unsigned& b1, unsigned& b2, unsigned& b3) {
    const unsigned addr = (unsigned)__cvta_generic_to_shared(krow8 + (lane >> 3) * 8);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));
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
// Mode 2 (paired tiles, WARPS = 8): block p covers query tiles 2p and 2p+1
// (64 tokens each, warps 0-3 and 4-7) over the union of their selections
// (usel[h][p], entry = tile | flags << 30, flags bit b = tile 2p+b selected
// it); a warp whose tile did not select the key tile skips that chunk's
// mma work but still takes part in the loads and barriers, so every K / V
// row is gathered once for two tiles instead of once per tile.
template <int D, int WARPS>
__device__ __forceinline__ void attn_mma_body(int mode, const bf16_t* q, const bf16_t* k, const bf16_t* v, bf16_t* out,
        int heads, int kv_heads, int sq, int sk, int causal, float scale,
        const unsigned* slot, const unsigned* sel, const unsigned* nsel, const float* pooled, const bf16_t* gate,
        int has_gate, int nb, int np, int dense_all, int te, int maxsel,
        const unsigned* usel, const unsigned* nusel, int npairs, int ustride, int dbg,
        bf16_t* ks, bf16_t* vs, unsigned char* kvalid) {
    // dbg (timing experiments only, output is garbage): 1 sequential keys
    // instead of the slot gather, 2 skip Q·Kᵀ, 4 skip P·V, 8 skip exp2,
    // 16 skip the K/V loads, 32 no token-index prefetch.
    constexpr int KS_STRIDE = D + 8;
    constexpr int KK = D / 16;   // k-steps of Q·Kᵀ
    constexpr int DT = D / 8;    // 8-wide dim tiles of the output
    constexpr int CH = D / 8;    // 16-byte chunks per key row
    constexpr int NTHR = 32 * WARPS;
    constexpr int RP = NTHR / CH; // key rows loaded per pass
    const int h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int sub = mode == 0 ? 1 : te / 64;
    const int half = mode == 2 ? (warp * 16) / te : 0;
    const int qt = mode == 0 ? blockIdx.x : (mode == 2 ? 2 * blockIdx.x + half : blockIdx.x / sub);
    const int qsub = mode == 1 ? blockIdx.x % sub : 0;
    const int kh = mode == 0 ? h / (heads / kv_heads) : h;
    const int lr0 = (mode == 2 ? (warp * 16) % te : warp * 16) + g, lr1 = lr0 + 8;
    int tok0, tok1;
    if (mode == 0) {
        tok0 = qt * 64 + lr0 < sq ? qt * 64 + lr0 : -1;
        tok1 = qt * 64 + lr1 < sq ? qt * 64 + lr1 : -1;
    } else if (qt < nb) {
        const unsigned a = slot[(long)qt * te + qsub * 64 + lr0], b = slot[(long)qt * te + qsub * 64 + lr1];
        tok0 = a == 0xffffffffu ? -1 : (int)a;
        tok1 = b == 0xffffffffu ? -1 : (int)b;
    } else {
        tok0 = tok1 = -1;   // the odd tile at the end has no partner
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
    } else if (mode == 2) {
        nchunks = (int)nusel[(long)h * npairs + blockIdx.x] * sub;
    } else {
        nchunks = (int)nsel[(long)h * nb + qt] * sub;
    }
    const bool all_tiles = mode == 1 && (qt < np || dense_all);
    const unsigned* urow = mode == 2 ? usel + ((long)h * npairs + blockIdx.x) * ustride : 0;
    const unsigned* srow = mode == 1 && !all_tiles ? sel + ((long)h * nb + qt) * maxsel : 0;
    const bf16_t* kbase = k + (long)kh * sk * D;
    const bf16_t* vbase = v + (long)kh * sk * D;
    const int ch = tid % CH, krow = tid / CH;
    // 32-bit index math: nb·te and seq·D stay far below 2^32.
    auto key_tok = [&](int c, int key) -> int {
        if (mode == 0 || (dbg & 1)) return c * MMA_BK + key < sk ? c * MMA_BK + key : -1;
        unsigned tile;
        if (mode == 2) tile = urow[c / sub] & 0x3fffffffu;
        else tile = all_tiles ? (unsigned)(c / sub) : srow[c / sub];
        const unsigned s2 = slot[tile * (unsigned)te + (unsigned)((c % sub) * 64 + key)];
        return s2 == 0xffffffffu ? -1 : (int)s2;
    };
    // Token indices of the keys this thread loads, resolved one chunk ahead
    // of the cp.async that needs them (two dependent L2 reads otherwise sit
    // on the critical path of every chunk).
    constexpr int NJ = MMA_BK / RP;
    int ntk[NJ];
    auto resolve = [&](int c) {
        #pragma unroll
        for (int j = 0; j < NJ; ++j) ntk[j] = key_tok(c, krow + RP * j);
    };
    auto load = [&](bf16_t* dst, const bf16_t* src, int c, unsigned char* valid) {
        #pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int key = krow + RP * j, tk = ntk[j];
            if (!(dbg & 16)) cp_async16(dst + key * KS_STRIDE + ch * 8, src + (unsigned)(tk < 0 ? 0 : tk) * (unsigned)D + ch * 8, tk >= 0);
            if (valid && ch == 0) valid[key] = tk >= 0;
        }
    };
    if (nchunks > 0) {
        resolve(0);
        load(ks, kbase, 0, kvalid);
        cp_async_commit();
        load(vs, vbase, 0, 0);
        cp_async_commit();
    }

    for (int c = 0; c < nchunks; ++c) {
        const unsigned char* kv = kvalid + (c & 1) * MMA_BK;
        // Paired mode: does this warp's tile select the key tile of chunk c?
        const bool mine = mode != 2 || (((urow[c / sub] >> 30) >> half) & 1u);
        if (c + 1 < nchunks && !(dbg & 32)) resolve(c + 1);   // overlaps the K_c wait and Q·Kᵀ
        cp_async_wait1();          // K_c landed (V_c may still be in flight)
        __syncthreads();
        float s[8][4];
        float mx0 = m0, mx1 = m1;
        if (mine) {
        #pragma unroll
        for (int nt = 0; nt < 8; ++nt) {
            s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
            const bf16_t* kr8 = ks + (nt * 8 + (lane & 7)) * KS_STRIDE;
            if (!(dbg & 2)) {
            #pragma unroll
            for (int kk = 0; kk < KK; kk += 2) {
                unsigned b0, b1, b2, b3;
                ldsm_k4(kr8 + kk * 16, lane, b0, b1, b2, b3);
                mma_bf16(s[nt], qa[kk][0], qa[kk][1], qa[kk][2], qa[kk][3], b0, b1);
                mma_bf16(s[nt], qa[kk + 1][0], qa[kk + 1][1], qa[kk + 1][2], qa[kk + 1][3], b2, b3);
            }
            }
        }
        // Scores stay unscaled (the scale rides in the exp's FMA). A chunk
        // whose 64 keys are all valid (every chunk but a tile's padded tail)
        // skips the per-key mask: one 16-bit smem read and a warp vote.
        const bool full = !(mode == 0 && causal) && __all_sync(0xffffffffu, *(const unsigned short*)(kv + 2 * lane) == 0x0101u);
        if (full) {
            #pragma unroll
            for (int nt = 0; nt < 8; ++nt) {
                mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
                mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
            }
        } else {
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
                s[nt][cc] = ok0 ? s[nt][cc] : -1e30f;
                s[nt][2 + cc] = ok1 ? s[nt][2 + cc] : -1e30f;
                mx0 = fmaxf(mx0, s[nt][cc]);
                mx1 = fmaxf(mx1, s[nt][2 + cc]);
            }
        }
        }
        }
        __syncthreads();           // every warp is done with ks
        if (c + 1 < nchunks) {
            if (dbg & 32) resolve(c + 1);
            load(ks, kbase, c + 1, kvalid + ((c + 1) & 1) * MMA_BK);
        }
        cp_async_commit();
        if (mine) {
        #pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
        }
        // The running max changes a handful of times per tile: rescale only
        // when some lane's did (warp-uniform branch, exact otherwise).
        if (__any_sync(0xffffffffu, mx0 != m0 || mx1 != m1)) {
            const float c0 = exp2f((m0 - mx0) * sl2), c1 = exp2f((m1 - mx1) * sl2);
            m0 = mx0; m1 = mx1;
            l0 *= c0; l1 *= c1;
            #pragma unroll
            for (int i = 0; i < DT; ++i) { o[i][0] *= c0; o[i][1] *= c0; o[i][2] *= c1; o[i][3] *= c1; }
        }
        }
        const float ms0 = -m0 * sl2, ms1 = -m1 * sl2;   // p = exp2(s·sl2 − m·sl2)
        cp_async_wait1();          // V_c landed (K_{c+1} may still be in flight)
        __syncthreads();
        if (mine) {
        #pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            float p[8];
            #pragma unroll
            for (int cc = 0; cc < 4; ++cc) {
                if (dbg & 8) { p[cc] = s[2 * kk][cc]; p[4 + cc] = s[2 * kk + 1][cc]; }
                else {
                    p[cc] = exp2f(fmaf(s[2 * kk][cc], sl2, cc < 2 ? ms0 : ms1));
                    p[4 + cc] = exp2f(fmaf(s[2 * kk + 1][cc], sl2, cc < 2 ? ms0 : ms1));
                }
            }
            l0 += p[0] + p[1] + p[4] + p[5];
            l1 += p[2] + p[3] + p[6] + p[7];
            const unsigned a0 = pack_bf16(p[0], p[1]), a1 = pack_bf16(p[2], p[3]);
            const unsigned a2 = pack_bf16(p[4], p[5]), a3 = pack_bf16(p[6], p[7]);
            const int vkey = kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
            if (!(dbg & 4)) {
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
        if (mode != 0 && has_gate) {
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
    attn_mma_body<D, 4>(0, (const bf16_t*)q_, (const bf16_t*)k_, (const bf16_t*)v_, (bf16_t*)out_, heads, kv_heads, \
                     sq, sk, causal, scale, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ks, vs, kvalid);         \
}                                                                                                                    \
extern "C" __global__ void __launch_bounds__(128) fw_vsa_fine_mma##D(u64 q_, u64 k_, u64 v_, u64 out_, int heads,   \
        int seq, float scale, u64 slot_, u64 sel_, u64 nsel_, u64 pooled_, u64 gate_, int has_gate, int nb, int np,  \
        int dense_all, int te, int maxsel, int dbg) {                                                                \
    __shared__ __align__(16) bf16_t ks[MMA_BK * (D + 8)];                                                            \
    __shared__ __align__(16) bf16_t vs[MMA_BK * (D + 8)];                                                            \
    __shared__ unsigned char kvalid[2 * MMA_BK];                                                                     \
    attn_mma_body<D, 4>(1, (const bf16_t*)q_, (const bf16_t*)k_, (const bf16_t*)v_, (bf16_t*)out_, heads, heads,    \
                     seq, seq, 0, scale, (const unsigned*)slot_, (const unsigned*)sel_, (const unsigned*)nsel_,      \
                     (const float*)pooled_, (const bf16_t*)gate_, has_gate, nb, np, dense_all, te, maxsel,           \
                     0, 0, 0, 0, dbg, ks, vs, kvalid);                                                               \
}                                                                                                                    \
extern "C" __global__ void __launch_bounds__(256) fw_vsa_fine_pair_mma##D(u64 q_, u64 k_, u64 v_, u64 out_,         \
        int heads, int seq, float scale, u64 slot_, u64 sel_, u64 nsel_, u64 pooled_, u64 gate_, int has_gate,       \
        int nb, int np, int dense_all, int te, int maxsel, u64 usel_, u64 nusel_, int npairs, int ustride, int dbg) {\
    __shared__ __align__(16) bf16_t ks[MMA_BK * (D + 8)];                                                            \
    __shared__ __align__(16) bf16_t vs[MMA_BK * (D + 8)];                                                            \
    __shared__ unsigned char kvalid[2 * MMA_BK];                                                                     \
    attn_mma_body<D, 8>(2, (const bf16_t*)q_, (const bf16_t*)k_, (const bf16_t*)v_, (bf16_t*)out_, heads, heads,    \
                     seq, seq, 0, scale, (const unsigned*)slot_, (const unsigned*)sel_, (const unsigned*)nsel_,      \
                     (const float*)pooled_, (const bf16_t*)gate_, has_gate, nb, np, dense_all, te, maxsel,           \
                     (const unsigned*)usel_, (const unsigned*)nusel_, npairs, ustride, dbg, ks, vs, kvalid);         \
}
ATTN_MMA_KERNELS(64)
ATTN_MMA_KERNELS(128)

// VSA fine stage, third body: the occupancy experiment. The v1 body spends
// two thirds of its time outside the mma work (docs/runs/2026-11-07-vast-l40s-
// vsa-phases): 35 KB of staging and ~170 registers allow two 4-warp blocks
// per SM, so loads, barriers and softmax bookkeeping never overlap with
// another block's tensor work. Here: 32-key chunks (17 KB staging, S tile 16
// registers), optionally K/V double-buffered with the next chunk's loads
// issued a whole chunk ahead and two barriers per chunk (DB), optionally Q
// fragments re-read from global each chunk instead of 32 resident
// registers (QREG = false). 64 query rows, 4 warps, mode 1 only.
// SWZ: rows stored unpadded (D halfs) with the 16-byte chunk index XOR-ed
// with (row & 7), which keeps ldmatrix and cp.async conflict-free and cuts
// the staging from 35 to 32 KB so three blocks fit an SM instead of two.
// Q8: INT8 Q·Kᵀ (SageAttention's recipe). q8 / k8 hold the int8 codes of Q
// and of K − mean(K) ([heads][seq][D] bytes), qsc / ksc one scale per
// (head, tile); K is staged as int8 (ks8, swizzled 16-byte chunks) and the
// scores come out of mma.s8 as int32, scaled by qsc·ksc. P·V stays bf16.
// PV8 (with Q8 and DB): FP8 P·V as in the dense fw_attention_pv8 kernels.
// v8t holds V − mean(V) as E4M3 codes per head, transposed to [D][nb·te]
// in tile-slot order with each 32-slot chunk in mma B-fragment key order
// (fw_attn_v8 with the slot table); the running maxima are integers in
// log2 units, P = exp2(s·log2e − ⌈m⌉ + 6) rounded to E4M3, and the
// epilogue applies vsc[h] / l and adds vmean back before the gate term.
template <int D, bool DB, bool QREG, bool SWZ, bool Q8, bool PV8 = false>
__device__ __forceinline__ void vsa_fine3_body(const bf16_t* q, const bf16_t* k, const bf16_t* v, bf16_t* out,
        int heads, int seq, float scale, const unsigned* slot, const unsigned* sel, const unsigned* nsel,
        const float* pooled, const bf16_t* gate, int has_gate, int nb, int np, int dense_all, int te, int maxsel,
        bf16_t* ks, bf16_t* vs, unsigned char* kvalid,
        const signed char* q8, const signed char* k8, const float* qsc, const float* ksc, unsigned char* ks8,
        const unsigned char* v8t = 0, const float* vmean = 0, const float* vsc = 0, unsigned char* vs8 = 0) {
    static_assert(!PV8 || (Q8 && DB), "PV8 needs Q8 and DB");
    constexpr int F3BK = 32;
    constexpr int KS_STRIDE = SWZ ? D : D + 8;
    // element offset of 16-byte chunk `chunk` of row `row` in the staging tile
    auto at = [](int row, int chunk) -> int { return row * KS_STRIDE + ((SWZ ? (chunk ^ (row & 7)) : chunk) << 3); };
    constexpr int KK = D / 16, DT = D / 8, CH = D / 8, RP = 128 / CH, NJ = F3BK / RP, NT = F3BK / 8;
    constexpr int NBUF = DB ? 2 : 1;
    const int h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    const int sub = te / 64;                     // 64-key halves of a tile, as v1
    const int csub = te / F3BK;                    // 32-key chunks per tile (a power of two)
    const int cshift = __ffs(csub) - 1, cmask = csub - 1;
    const int qt = blockIdx.x / sub, qsub = blockIdx.x % sub;
    const int lr0 = warp * 16 + g, lr1 = lr0 + 8;
    const unsigned a = slot[(long)qt * te + qsub * 64 + lr0], b = slot[(long)qt * te + qsub * 64 + lr1];
    const int tok0 = a == 0xffffffffu ? -1 : (int)a, tok1 = b == 0xffffffffu ? -1 : (int)b;
    const float sl2 = scale * 1.4426950408889634f;
    const unsigned* q0 = (const unsigned*)(q + ((long)h * seq + (tok0 < 0 ? 0 : tok0)) * D);
    const unsigned* q1 = (const unsigned*)(q + ((long)h * seq + (tok1 < 0 ? 0 : tok1)) * D);
    unsigned qa[QREG ? KK : 1][4];
    if (QREG) {
        #pragma unroll
        for (int kk = 0; kk < KK; ++kk) {
            const int c = (kk * 16 + t * 2) >> 1;
            qa[kk][0] = tok0 >= 0 ? q0[c] : 0u;
            qa[kk][1] = tok1 >= 0 ? q1[c] : 0u;
            qa[kk][2] = tok0 >= 0 ? q0[c + 4] : 0u;
            qa[kk][3] = tok1 >= 0 ? q1[c + 4] : 0u;
        }
    }
    constexpr int KK8 = D / 32, CH8 = D / 16, RP8 = 128 / CH8, NJ8 = F3BK / RP8;
    // int8 staging: row = key (D bytes), 16-byte chunk c of row r at chunk c ^ (r & (CH8 − 1)).
    auto at8 = [](int row, int chunk) -> int { return row * D + ((chunk ^ (row & (CH8 - 1))) << 4); };
    unsigned qa8[Q8 ? KK8 : 1][4];
    if constexpr (Q8) {
        const unsigned* q80 = (const unsigned*)(q8 + ((long)h * seq + (tok0 < 0 ? 0 : tok0)) * D);
        const unsigned* q81 = (const unsigned*)(q8 + ((long)h * seq + (tok1 < 0 ? 0 : tok1)) * D);
        #pragma unroll
        for (int kk = 0; kk < KK8; ++kk) {
            qa8[kk][0] = tok0 >= 0 ? q80[kk * 8 + t] : 0u;
            qa8[kk][1] = tok1 >= 0 ? q81[kk * 8 + t] : 0u;
            qa8[kk][2] = tok0 >= 0 ? q80[kk * 8 + 4 + t] : 0u;
            qa8[kk][3] = tok1 >= 0 ? q81[kk * 8 + 4 + t] : 0u;
        }
    }
    const float qs8 = Q8 ? qsc[(long)h * nb + qt] : 1.0f;
    float o[DT][4];
    #pragma unroll
    for (int i = 0; i < DT; ++i) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.0f;
    float m0 = -1e30f, m1 = -1e30f, l0 = 0.0f, l1 = 0.0f;
    const bool all_tiles = qt < np || dense_all;
    const int nchunks = (int)(all_tiles ? (unsigned)nb : nsel[(long)h * nb + qt]) * csub;
    const bf16_t* kbase = k + (long)h * seq * D;
    const bf16_t* vbase = v + (long)h * seq * D;
    const int ch = tid % CH, krow = tid / CH;
    const unsigned* srow = all_tiles ? 0 : sel + ((long)h * nb + qt) * maxsel;
    int ntk[NJ];
    const int ch8 = tid % CH8, krow8 = tid / CH8;
    int ntk8[Q8 ? NJ8 : 1];
    // The thread's NJ keys of chunk c are krow + RP·j: one tile lookup and one
    // base pointer per chunk, then immediate offsets.
    int rtile = 0;    // the tile of the chunk last resolved (its V8 slice)
    auto resolve = [&](int c) {
        const unsigned tile = all_tiles ? (unsigned)(c >> cshift) : srow[c >> cshift];
        rtile = (int)tile;
        const unsigned* sp = slot + tile * (unsigned)te + (unsigned)((c & cmask) * F3BK + krow);
        #pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const unsigned s2 = sp[RP * j];
            ntk[j] = s2 == 0xffffffffu ? -1 : (int)s2;
        }
        if constexpr (Q8) {
            const unsigned* sp8 = slot + tile * (unsigned)te + (unsigned)((c & cmask) * F3BK + krow8);
            #pragma unroll
            for (int j = 0; j < NJ8; ++j) {
                const unsigned s2 = sp8[RP8 * j];
                ntk8[j] = s2 == 0xffffffffu ? -1 : (int)s2;
            }
        }
    };
    // int8 K rows of chunk c into ks8 (zero-filled padding), with the valid mask.
    auto load8 = [&](unsigned char* dst, unsigned char* valid) {
        #pragma unroll
        for (int j = 0; j < (Q8 ? NJ8 : 1); ++j) {
            const int key = krow8 + RP8 * j, tk = ntk8[j];
            cp_async16(dst + at8(key, ch8), k8 + (long)h * seq * D + (unsigned)(tk < 0 ? 0 : tk) * (unsigned)D + ch8 * 16, tk >= 0);
            if (valid && ch8 == 0) valid[key] = tk >= 0;
        }
    };
    auto k8buf = [&](int c) { return ks8 + (c % NBUF) * F3BK * D; };
    auto load = [&](bf16_t* dst, const bf16_t* src, unsigned char* valid) {
        #pragma unroll
        for (int j = 0; j < NJ; ++j) {
            const int key = krow + RP * j, tk = ntk[j];
            cp_async16(dst + at(key, ch), src + (unsigned)(tk < 0 ? 0 : tk) * (unsigned)D + ch * 8, tk >= 0);
            if (valid && ch == 0) valid[key] = tk >= 0;
        }
    };
    auto kbuf = [&](int c) { return ks + (c % NBUF) * F3BK * KS_STRIDE; };
    auto vbuf = [&](int c) { return vs + (c % NBUF) * F3BK * KS_STRIDE; };
    auto valid_of = [&](int c) { return kvalid + (c % NBUF) * F3BK; };
    // FP8 V: the chunk (resolved just before) is D rows × 32 bytes of the
    // head's transposed codes, the two 16-byte halves swizzled by bit 2 of
    // the row (as the dense kernel); row stride nb·te slots.
    const long ntot = (long)nb * te;
    const unsigned char* v8h = PV8 ? v8t + (long)h * D * ntot : 0;
    auto at8v = [](int r, int half) -> int { return r * 32 + ((half ^ ((r >> 2) & 1)) << 4); };
    auto vs8buf = [&](int c) { return vs8 + (c % NBUF) * F3BK * D; };
    auto load_v8 = [&](int c, unsigned char* dst) {
        #pragma unroll
        for (int j = 0; j < (PV8 ? (2 * D) / 128 : 1); ++j) {
            const int idx = tid + 128 * j, r = idx >> 1, half = idx & 1;
            cp_async16(dst + at8v(r, half), v8h + (long)r * ntot + (long)rtile * te + (c & cmask) * F3BK + half * 16, true);
        }
    };
    if (nchunks > 0) {
        resolve(0);
        if constexpr (Q8) load8(k8buf(0), valid_of(0)); else load(kbuf(0), kbase, valid_of(0));
        cp_async_commit();
        if constexpr (PV8) load_v8(0, vs8buf(0)); else load(vbuf(0), vbase, 0);
        cp_async_commit();
    }
    for (int c = 0; c < nchunks; ++c) {
        const unsigned char* kv = valid_of(c);
        if (c + 1 < nchunks) resolve(c + 1);
        if (DB) {
            // K_c landed (V_c, K_{c+1}, V_{c+1} may be in flight: at most 3 newer groups
            // once the next chunk's loads are issued below; here at most 1).
            asm volatile("cp.async.wait_group 1;\n" ::);
            __syncthreads();
            if (c + 1 < nchunks) {
                if constexpr (Q8) load8(k8buf(c + 1), valid_of(c + 1)); else load(kbuf(c + 1), kbase, valid_of(c + 1));
                cp_async_commit();
                if constexpr (PV8) load_v8(c + 1, vs8buf(c + 1)); else load(vbuf(c + 1), vbase, 0);
                cp_async_commit();
            }
        } else {
            cp_async_wait1();
            __syncthreads();
        }
        float s[NT][4];
        if constexpr (Q8) {
            // INT8 Q·Kᵀ: ldmatrix.x2 over the int8 rows gives the two 32-bit B
            // halves of a k32 step; the int32 sums scale by qsc·ksc of the
            // key's tile (K was shifted by its mean, which softmax ignores).
            const int tile_c = all_tiles ? (c >> cshift) : (int)srow[c >> cshift];
            const float f8 = qs8 * ksc[(long)h * nb + tile_c];
            const unsigned k8base_l = (unsigned)__cvta_generic_to_shared(k8buf(c)) + (lane & 7) * D;
            const int rsw = (lane & 7) & (CH8 - 1), mhalf = (lane >> 3) & 1;
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                int acc[4] = {0, 0, 0, 0};
                #pragma unroll
                for (int kk = 0; kk < KK8; ++kk) {
                    const unsigned addr = k8base_l + nt * 8 * D + (((2 * kk + mhalf) ^ rsw) << 4);
                    unsigned b0, b1;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(b0), "=r"(b1) : "r"(addr));
                    mma_s8(acc, qa8[kk][0], qa8[kk][1], qa8[kk][2], qa8[kk][3], b0, b1);
                }
                s[nt][0] = (float)acc[0] * f8; s[nt][1] = (float)acc[1] * f8;
                s[nt][2] = (float)acc[2] * f8; s[nt][3] = (float)acc[3] * f8;
            }
        } else {
            const bf16_t* kcur = kbuf(c);
            // ldmatrix addresses: row (lane & 7) of n-tile nt, chunk 2kk + (lane >> 3),
            // swizzled by (lane & 7). The chunk's bits split as 2kk (bits 2-3) and
            // lane >> 3 (bits 0-1), so the swizzle separates into a per-lane
            // constant and (2kk ^ (lane & 4)), both cheap.
            const unsigned kbase_l = (unsigned)__cvta_generic_to_shared(kcur) + (lane & 7) * (KS_STRIDE * 2)
                                   + (SWZ ? ((((lane >> 3) ^ lane) & 3) << 4) : ((lane >> 3) << 4));
            const unsigned kx = SWZ ? ((lane & 4) << 4) : 0u;
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.0f;
                #pragma unroll
                for (int kk = 0; kk < KK; kk += 2) {
                    unsigned b[4];
                    {
                        const unsigned addr = kbase_l + nt * 8 * KS_STRIDE * 2 + (((unsigned)(kk * 2) << 4) ^ kx);
                        asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                                     : "=r"(b[0]), "=r"(b[1]), "=r"(b[2]), "=r"(b[3]) : "r"(addr));
                    }
                    #pragma unroll
                    for (int u = 0; u < 2; ++u) {
                        unsigned a0, a1, a2, a3;
                        if (QREG) { a0 = qa[kk + u][0]; a1 = qa[kk + u][1]; a2 = qa[kk + u][2]; a3 = qa[kk + u][3]; }
                        else {
                            const int cc = ((kk + u) * 16 + t * 2) >> 1;
                            a0 = tok0 >= 0 ? __ldg(q0 + cc) : 0u;
                            a1 = tok1 >= 0 ? __ldg(q1 + cc) : 0u;
                            a2 = tok0 >= 0 ? __ldg(q0 + cc + 4) : 0u;
                            a3 = tok1 >= 0 ? __ldg(q1 + cc + 4) : 0u;
                        }
                        mma_bf16(s[nt], a0, a1, a2, a3, b[2 * u], b[2 * u + 1]);
                    }
                }
            }
        }
        float mx0 = PV8 ? -1e30f : m0, mx1 = PV8 ? -1e30f : m1;    // PV8: raw units here, m in log2 integers
        const bool full = __all_sync(0xffffffffu, *(const unsigned char*)(kv + lane) == 1u);
        if (full) {
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
                mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
            }
        } else {
        #pragma unroll
        for (int nt = 0; nt < NT; ++nt) {
            #pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
                const int key = nt * 8 + t * 2 + cc;
                const bool ok = kv[key];
                s[nt][cc] = ok ? s[nt][cc] : -1e30f;
                s[nt][2 + cc] = ok ? s[nt][2 + cc] : -1e30f;
                mx0 = fmaxf(mx0, s[nt][cc]);
                mx1 = fmaxf(mx1, s[nt][2 + cc]);
            }
        }
        }
        if (!DB) {
            __syncthreads();           // every warp is done with ks
            if (c + 1 < nchunks) { if constexpr (Q8) load8(k8buf(c + 1), valid_of(c + 1)); else load(kbuf(c + 1), kbase, valid_of(c + 1)); }
            cp_async_commit();
        }
        #pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, off));
            mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, off));
        }
        if constexpr (PV8) {
            // Integer maxima in log2 units: the rescales below are exact powers of two.
            mx0 = fmaxf(m0, mx0 > -1e29f ? ceilf(mx0 * sl2) : -1e30f);
            mx1 = fmaxf(m1, mx1 > -1e29f ? ceilf(mx1 * sl2) : -1e30f);
        }
        if (__any_sync(0xffffffffu, mx0 != m0 || mx1 != m1)) {
            const float c0 = PV8 ? exp2f(m0 - mx0) : exp2f((m0 - mx0) * sl2);
            const float c1 = PV8 ? exp2f(m1 - mx1) : exp2f((m1 - mx1) * sl2);
            m0 = mx0; m1 = mx1;
            l0 *= c0; l1 *= c1;
            #pragma unroll
            for (int i = 0; i < DT; ++i) { o[i][0] *= c0; o[i][1] *= c0; o[i][2] *= c1; o[i][3] *= c1; }
        }
        // PV8: a row without a key yet keeps p = 0 (6 + 1e30 would overflow the fma).
        const float ms0 = PV8 ? (m0 > -1e29f ? 6.0f - m0 : -1e30f) : -m0 * sl2;
        const float ms1 = PV8 ? (m1 > -1e29f ? 6.0f - m1 : -1e30f) : -m1 * sl2;
        if (DB) {
            // V_c landed (K_{c+1}, V_{c+1} may be in flight).
            if (c + 1 < nchunks) asm volatile("cp.async.wait_group 2;\n" ::);
            else asm volatile("cp.async.wait_group 0;\n" ::);
        } else {
            cp_async_wait1();
        }
        __syncthreads();
        if constexpr (PV8) {
            // One k32 e4m3 step per chunk: A = P in the dense kernel's key order
            // (rows g / g+8: n-tiles 0,1 then 2,3), B = ldmatrix.x2 over the
            // transposed V8 rows (lanes 0-7 the first 16 bytes, 8-15 the second).
            float pp[NT][4];
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                #pragma unroll
                for (int cc = 0; cc < 4; ++cc) pp[nt][cc] = exp2f(fmaf(s[nt][cc], sl2, cc < 2 ? ms0 : ms1));
            }
            l0 += pp[0][0] + pp[0][1] + pp[1][0] + pp[1][1] + pp[2][0] + pp[2][1] + pp[3][0] + pp[3][1];
            l1 += pp[0][2] + pp[0][3] + pp[1][2] + pp[1][3] + pp[2][2] + pp[2][3] + pp[3][2] + pp[3][3];
            const unsigned a0 = pack_e4m3x2(pp[0][0], pp[0][1]) | (pack_e4m3x2(pp[1][0], pp[1][1]) << 16);
            const unsigned a1 = pack_e4m3x2(pp[0][2], pp[0][3]) | (pack_e4m3x2(pp[1][2], pp[1][3]) << 16);
            const unsigned a2 = pack_e4m3x2(pp[2][0], pp[2][1]) | (pack_e4m3x2(pp[3][0], pp[3][1]) << 16);
            const unsigned a3 = pack_e4m3x2(pp[2][2], pp[2][3]) | (pack_e4m3x2(pp[3][2], pp[3][3]) << 16);
            const unsigned vs8_l = (unsigned)__cvta_generic_to_shared(vs8buf(c));
            const int lr = lane & 7, lhalf = (lane >> 3) & 1;
            #pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
                const int r = dt * 8 + lr;
                const unsigned addr = vs8_l + r * 32 + ((lhalf ^ ((r >> 2) & 1)) << 4);
                unsigned b0, b1;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(b0), "=r"(b1) : "r"(addr));
                mma_e4m3(o[dt], a0, a1, a2, a3, b0, b1);
            }
        } else {
        const bf16_t* vcur = vbuf(c);
        // V rows are keys kk·16 + (lane & 7) + 8·((lane >> 3) & 1), chunk dt + (lane >> 4)
        // with dt even: the swizzle splits into a per-lane term and (dt ^ (lane & 6)).
        const unsigned vbase_l = (unsigned)__cvta_generic_to_shared(vcur) + ((lane & 7) + 8 * ((lane >> 3) & 1)) * (KS_STRIDE * 2)
                               + (SWZ ? ((((lane >> 4) ^ lane) & 1) << 4) : ((lane >> 4) << 4));
        const unsigned vx = SWZ ? ((lane & 6) << 4) : 0u;
        #pragma unroll
        for (int kk = 0; kk < F3BK / 16; ++kk) {
            float p[8];
            #pragma unroll
            for (int cc = 0; cc < 4; ++cc) {
                p[cc] = exp2f(fmaf(s[2 * kk][cc], sl2, cc < 2 ? ms0 : ms1));
                p[4 + cc] = exp2f(fmaf(s[2 * kk + 1][cc], sl2, cc < 2 ? ms0 : ms1));
            }
            l0 += p[0] + p[1] + p[4] + p[5];
            l1 += p[2] + p[3] + p[6] + p[7];
            const unsigned a0 = pack_bf16(p[0], p[1]), a1 = pack_bf16(p[2], p[3]);
            const unsigned a2 = pack_bf16(p[4], p[5]), a3 = pack_bf16(p[6], p[7]);
            #pragma unroll
            for (int dt = 0; dt < DT; dt += 2) {
                const unsigned addr = vbase_l + kk * 16 * KS_STRIDE * 2 + (((unsigned)dt << 4) ^ vx);
                unsigned b0, b1, b2, b3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));
                mma_bf16(o[dt], a0, a1, a2, a3, b0, b1);
                mma_bf16(o[dt + 1], a0, a1, a2, a3, b2, b3);
            }
        }
        }
        if (!DB) {
            __syncthreads();           // every warp is done with vs
            if (c + 1 < nchunks) load(vbuf(c + 1), vbase, 0);
            cp_async_commit();
        }
    }
    #pragma unroll
    for (int off = 1; off < 4; off <<= 1) {
        l0 += __shfl_xor_sync(0xffffffffu, l0, off);
        l1 += __shfl_xor_sync(0xffffffffu, l1, off);
    }
    const float vscale = PV8 ? vsc[h] : 1.0f;
    const float i0 = l0 > 0.0f ? vscale / l0 : 0.0f, i1 = l1 > 0.0f ? vscale / l1 : 0.0f;
    #pragma unroll
    for (int dt = 0; dt < DT; ++dt) {
        const int c = dt * 8 + t * 2;
        float a0 = o[dt][0] * i0, a1 = o[dt][1] * i0, b0 = o[dt][2] * i1, b1 = o[dt][3] * i1;
        if constexpr (PV8) {
            const float m0v = vmean[(long)h * D + c], m1v = vmean[(long)h * D + c + 1];
            a0 += m0v; a1 += m1v; b0 += m0v; b1 += m1v;
        }
        if (has_gate) {
            const float* cr = pooled + ((long)h * nb + qt) * D + c;
            if (tok0 >= 0) { const bf16_t* gr = gate + ((long)h * seq + tok0) * D + c; a0 += cr[0] * ld(gr, 0); a1 += cr[1] * ld(gr, 1); }
            if (tok1 >= 0) { const bf16_t* gr = gate + ((long)h * seq + tok1) * D + c; b0 += cr[0] * ld(gr, 0); b1 += cr[1] * ld(gr, 1); }
        }
        if (tok0 >= 0) *(unsigned*)(out + ((long)tok0 * heads + h) * D + c) = pack_bf16(a0, a1);
        if (tok1 >= 0) *(unsigned*)(out + ((long)tok1 * heads + h) * D + c) = pack_bf16(b0, b1);
    }
}

// a: single-buffered, Q re-read, four blocks per SM asked for (regs ≤ 128).
// b: double-buffered, Q resident, two blocks per SM.
// c: double-buffered, Q re-read, three blocks per SM asked for.
#define VSA_FINE3_KERNELS(D)                                                                                         \
extern "C" __global__ void __launch_bounds__(128, 4) fw_vsa_fine3a_mma##D(u64 q_, u64 k_, u64 v_, u64 out_,         \
        int heads, int seq, float scale, u64 slot_, u64 sel_, u64 nsel_, u64 pooled_, u64 gate_, int has_gate,       \
        int nb, int np, int dense_all, int te, int maxsel) {                                                         \
    __shared__ __align__(16) bf16_t ks[32 * (D + 8)];                                                                \
    __shared__ __align__(16) bf16_t vs[32 * (D + 8)];                                                                \
    __shared__ unsigned char kvalid[32];                                                                             \
    vsa_fine3_body<D, false, false, false, false>((const bf16_t*)q_, (const bf16_t*)k_, (const bf16_t*)v_, (bf16_t*)out_, heads,   \
        seq, scale, (const unsigned*)slot_, (const unsigned*)sel_, (const unsigned*)nsel_, (const float*)pooled_,    \
        (const bf16_t*)gate_, has_gate, nb, np, dense_all, te, maxsel, ks, vs, kvalid, 0, 0, 0, 0, 0);                               \
}                                                                                                                    \
extern "C" __global__ void __launch_bounds__(128, 2) fw_vsa_fine3b_mma##D(u64 q_, u64 k_, u64 v_, u64 out_,         \
        int heads, int seq, float scale, u64 slot_, u64 sel_, u64 nsel_, u64 pooled_, u64 gate_, int has_gate,       \
        int nb, int np, int dense_all, int te, int maxsel) {                                                         \
    __shared__ __align__(16) bf16_t ks[2 * 32 * (D + 8)];                                                            \
    __shared__ __align__(16) bf16_t vs[2 * 32 * (D + 8)];                                                            \
    __shared__ unsigned char kvalid[64];                                                                             \
    vsa_fine3_body<D, true, true, false, false>((const bf16_t*)q_, (const bf16_t*)k_, (const bf16_t*)v_, (bf16_t*)out_, heads,     \
        seq, scale, (const unsigned*)slot_, (const unsigned*)sel_, (const unsigned*)nsel_, (const float*)pooled_,    \
        (const bf16_t*)gate_, has_gate, nb, np, dense_all, te, maxsel, ks, vs, kvalid, 0, 0, 0, 0, 0);                               \
}                                                                                                                    \
extern "C" __global__ void __launch_bounds__(128, 3) fw_vsa_fine3c_mma##D(u64 q_, u64 k_, u64 v_, u64 out_,         \
        int heads, int seq, float scale, u64 slot_, u64 sel_, u64 nsel_, u64 pooled_, u64 gate_, int has_gate,       \
        int nb, int np, int dense_all, int te, int maxsel) {                                                         \
    __shared__ __align__(16) bf16_t ks[2 * 32 * (D + 8)];                                                            \
    __shared__ __align__(16) bf16_t vs[2 * 32 * (D + 8)];                                                            \
    __shared__ unsigned char kvalid[64];                                                                             \
    vsa_fine3_body<D, true, false, false, false>((const bf16_t*)q_, (const bf16_t*)k_, (const bf16_t*)v_, (bf16_t*)out_, heads,    \
        seq, scale, (const unsigned*)slot_, (const unsigned*)sel_, (const unsigned*)nsel_, (const float*)pooled_,    \
        (const bf16_t*)gate_, has_gate, nb, np, dense_all, te, maxsel, ks, vs, kvalid, 0, 0, 0, 0, 0);                               \
}
VSA_FINE3_KERNELS(64)
VSA_FINE3_KERNELS(128)

// d: b with swizzled, unpadded staging (32 KB) and three blocks per SM.
#define VSA_FINE3D_KERNEL(D)                                                                                         \
extern "C" __global__ void __launch_bounds__(128, 3) fw_vsa_fine3d_mma##D(u64 q_, u64 k_, u64 v_, u64 out_,         \
        int heads, int seq, float scale, u64 slot_, u64 sel_, u64 nsel_, u64 pooled_, u64 gate_, int has_gate,       \
        int nb, int np, int dense_all, int te, int maxsel) {                                                         \
    __shared__ __align__(16) bf16_t ks[2 * 32 * D];                                                                  \
    __shared__ __align__(16) bf16_t vs[2 * 32 * D];                                                                  \
    __shared__ unsigned char kvalid[64];                                                                             \
    vsa_fine3_body<D, true, true, true, false>((const bf16_t*)q_, (const bf16_t*)k_, (const bf16_t*)v_, (bf16_t*)out_,     \
        heads, seq, scale, (const unsigned*)slot_, (const unsigned*)sel_, (const unsigned*)nsel_,                    \
        (const float*)pooled_, (const bf16_t*)gate_, has_gate, nb, np, dense_all, te, maxsel, ks, vs, kvalid,        \
        0, 0, 0, 0, 0);                                                                                              \
}
VSA_FINE3D_KERNEL(64)
VSA_FINE3D_KERNEL(128)

// e / f: 3d with INT8 Q·Kᵀ (q8 / k8 codes, per-tile scales); e asks for three
// blocks per SM, f for four (the int8 path holds half the Q registers and
// stages K in a quarter of the bytes).
#define VSA_FINE3Q8_KERNEL(D, NAME, BLOCKS)                                                                           \
extern "C" __global__ void __launch_bounds__(128, BLOCKS) NAME##D(u64 q_, u64 k_, u64 v_, u64 out_,                  \
        int heads, int seq, float scale, u64 slot_, u64 sel_, u64 nsel_, u64 pooled_, u64 gate_, int has_gate,       \
        int nb, int np, int dense_all, int te, int maxsel, u64 q8_, u64 k8_, u64 qsc_, u64 ksc_) {                   \
    __shared__ __align__(16) unsigned char ks8[2 * 32 * D];                                                          \
    __shared__ __align__(16) bf16_t vs[2 * 32 * D];                                                                  \
    __shared__ unsigned char kvalid[64];                                                                             \
    vsa_fine3_body<D, true, true, true, true>((const bf16_t*)q_, (const bf16_t*)k_, (const bf16_t*)v_, (bf16_t*)out_, \
        heads, seq, scale, (const unsigned*)slot_, (const unsigned*)sel_, (const unsigned*)nsel_,                    \
        (const float*)pooled_, (const bf16_t*)gate_, has_gate, nb, np, dense_all, te, maxsel, (bf16_t*)0, vs, kvalid, \
        (const signed char*)q8_, (const signed char*)k8_, (const float*)qsc_, (const float*)ksc_, ks8);              \
}
VSA_FINE3Q8_KERNEL(64, fw_vsa_fine3e_mma, 3)
VSA_FINE3Q8_KERNEL(128, fw_vsa_fine3e_mma, 3)
VSA_FINE3Q8_KERNEL(64, fw_vsa_fine3f_mma, 4)
VSA_FINE3Q8_KERNEL(128, fw_vsa_fine3f_mma, 4)

// g: e with FP8 P·V (v8t transposed codes in slot order, per-head mean and
// scale); sm_89+ only (mma.e4m3), 16 KB of staging.
#define VSA_FINE3PV8_KERNEL(D, NAME, BLOCKS)                                                                          \
extern "C" __global__ void __launch_bounds__(128, BLOCKS) NAME##D(u64 q_, u64 k_, u64 v_, u64 out_,                  \
        int heads, int seq, float scale, u64 slot_, u64 sel_, u64 nsel_, u64 pooled_, u64 gate_, int has_gate,       \
        int nb, int np, int dense_all, int te, int maxsel, u64 q8_, u64 k8_, u64 qsc_, u64 ksc_, u64 v8t_,           \
        u64 vmean_, u64 vsc_) {                                                                                      \
    __shared__ __align__(16) unsigned char ks8[2 * 32 * D];                                                          \
    __shared__ __align__(16) unsigned char vs8[2 * 32 * D];                                                          \
    __shared__ unsigned char kvalid[64];                                                                             \
    vsa_fine3_body<D, true, true, true, true, true>((const bf16_t*)q_, (const bf16_t*)k_, (const bf16_t*)v_,         \
        (bf16_t*)out_, heads, seq, scale, (const unsigned*)slot_, (const unsigned*)sel_, (const unsigned*)nsel_,     \
        (const float*)pooled_, (const bf16_t*)gate_, has_gate, nb, np, dense_all, te, maxsel, (bf16_t*)0,            \
        (bf16_t*)0, kvalid, (const signed char*)q8_, (const signed char*)k8_, (const float*)qsc_,                    \
        (const float*)ksc_, ks8, (const unsigned char*)v8t_, (const float*)vmean_, (const float*)vsc_, vs8);         \
}
VSA_FINE3PV8_KERNEL(64, fw_vsa_fine3g_mma, 3)
VSA_FINE3PV8_KERNEL(128, fw_vsa_fine3g_mma, 3)

// Per-head mean of K over the sequence (the shift that makes the int8
// rounding of K well conditioned; softmax does not see it). Two fixed-shape
// stages so the sum order never depends on scheduling: grid (nch, heads)
// writes one partial row sum per token chunk, grid (heads) folds the chunks.
// One block per head over 38k tokens took 2.0 ms; the pair takes ~0.1 ms.
extern "C" __global__ void fw_vsa_kmean(u64 k_, u64 part_, int seq, int d, int nch) {
    const bf16_t* k = (const bf16_t*)k_ + (long)blockIdx.y * seq * d;
    float* part = (float*)part_ + ((long)blockIdx.y * nch + blockIdx.x) * d;
    const int cs = (seq + nch - 1) / nch, t0 = blockIdx.x * cs, t1 = min(seq, t0 + cs);
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
        int tok = t0;
        for (; tok + 4 <= t1; tok += 4) {
            a0 += ld(k, (long)tok * d + c); a1 += ld(k, (long)(tok + 1) * d + c);
            a2 += ld(k, (long)(tok + 2) * d + c); a3 += ld(k, (long)(tok + 3) * d + c);
        }
        for (; tok < t1; ++tok) a0 += ld(k, (long)tok * d + c);
        part[c] = (a0 + a1) + (a2 + a3);
    }
}
extern "C" __global__ void fw_vsa_kmean2(u64 part_, u64 kmean_, int seq, int d, int nch) {
    const float* part = (const float*)part_ + (long)blockIdx.x * nch * d;
    float* kmean = (float*)kmean_ + (long)blockIdx.x * d;
    for (int c = threadIdx.x; c < d; c += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < nch; ++i) acc += part[(long)i * d + c];
        kmean[c] = acc / (float)seq;
    }
}
// INT8 codes of Q and K − kmean per (head, tile) with one scale each
// (max |x| / 127, codes round to nearest even, [−127, 127]); the host twin is
// h3_core::vsa::qk8_round. Grid (nb, heads), 256 threads, d % 4 == 0, d ≤ 128,
// tile ≤ 128 rows: warp w holds rows w, w+8, … in registers across the
// max pass and the rounding pass, so Q and K are read once (not twice).
extern "C" __global__ void __launch_bounds__(256) fw_vsa_qk8(u64 q_, u64 k_, u64 slot_, u64 bsz_, u64 kmean_, u64 q8_, u64 k8_,
                                                             u64 qsc_, u64 ksc_, int heads, int seq, int d, int nb, int te) {
    __shared__ float red[2][8];
    __shared__ float sc[2];
    const bf16_t *q = (const bf16_t*)q_, *k = (const bf16_t*)k_;
    const unsigned *slot = (const unsigned*)slot_, *bsz = (const unsigned*)bsz_;
    const float* kmean = (const float*)kmean_ + (long)blockIdx.y * d;
    signed char *q8 = (signed char*)q8_, *k8 = (signed char*)k8_;
    const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const unsigned n = bsz[t];
    const int d4 = d >> 2;
    const bool act = lane < d4;
    float km0 = 0.0f, km1 = 0.0f, km2 = 0.0f, km3 = 0.0f;
    if (act) { km0 = kmean[lane * 4]; km1 = kmean[lane * 4 + 1]; km2 = kmean[lane * 4 + 2]; km3 = kmean[lane * 4 + 3]; }
    uint2 wq[16], wk[16];
    long rows[16];
    float qm = 0.0f, kmx = 0.0f;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        const unsigned i = warp + 8u * j;
        rows[j] = -1;
        wq[j] = make_uint2(0u, 0u); wk[j] = make_uint2(0u, 0u);
        if (act && i < n) {
            rows[j] = ((long)h * seq + slot[(long)t * te + i]) * d + lane * 4;
            wq[j] = *(const uint2*)(q + rows[j]); wk[j] = *(const uint2*)(k + rows[j]);
            qm = fmaxf(qm, fmaxf(fmaxf(fabsf(__uint_as_float(wq[j].x << 16)), fabsf(__uint_as_float(wq[j].x & 0xffff0000u))),
                                 fmaxf(fabsf(__uint_as_float(wq[j].y << 16)), fabsf(__uint_as_float(wq[j].y & 0xffff0000u)))));
            kmx = fmaxf(kmx, fmaxf(fmaxf(fabsf(__uint_as_float(wk[j].x << 16) - km0), fabsf(__uint_as_float(wk[j].x & 0xffff0000u) - km1)),
                                   fmaxf(fabsf(__uint_as_float(wk[j].y << 16) - km2), fabsf(__uint_as_float(wk[j].y & 0xffff0000u) - km3))));
        }
    }
    for (int o = 16; o > 0; o >>= 1) { qm = fmaxf(qm, __shfl_xor_sync(0xffffffffu, qm, o)); kmx = fmaxf(kmx, __shfl_xor_sync(0xffffffffu, kmx, o)); }
    if (lane == 0) { red[0][warp] = qm; red[1][warp] = kmx; }
    __syncthreads();
    if (tid == 0) {
        float qmax = red[0][0], kmax = red[1][0];
        for (int w = 1; w < 8; ++w) { qmax = fmaxf(qmax, red[0][w]); kmax = fmaxf(kmax, red[1][w]); }
        sc[0] = qmax / 127.0f; sc[1] = kmax / 127.0f;
        ((float*)qsc_)[(long)h * nb + t] = sc[0];
        ((float*)ksc_)[(long)h * nb + t] = sc[1];
    }
    __syncthreads();
    const float qi = sc[0] > 0.0f ? 1.0f / sc[0] : 0.0f, ki = sc[1] > 0.0f ? 1.0f / sc[1] : 0.0f;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        if (rows[j] < 0) continue;
        const float qv[4] = {__uint_as_float(wq[j].x << 16), __uint_as_float(wq[j].x & 0xffff0000u), __uint_as_float(wq[j].y << 16), __uint_as_float(wq[j].y & 0xffff0000u)};
        const float kv[4] = {__uint_as_float(wk[j].x << 16) - km0, __uint_as_float(wk[j].x & 0xffff0000u) - km1,
                             __uint_as_float(wk[j].y << 16) - km2, __uint_as_float(wk[j].y & 0xffff0000u) - km3};
        unsigned pq = 0u, pk = 0u;
        #pragma unroll
        for (int c = 0; c < 4; ++c) {
            const int cq = max(-127, min(127, __float2int_rn(qv[c] * qi)));
            const int ck = max(-127, min(127, __float2int_rn(kv[c] * ki)));
            pq |= ((unsigned)cq & 0xffu) << (8 * c);
            pk |= ((unsigned)ck & 0xffu) << (8 * c);
        }
        *(unsigned*)(q8 + rows[j]) = pq;
        *(unsigned*)(k8 + rows[j]) = pk;
    }
}

// The union of two consecutive query tiles' selections (sorted, deduplicated),
// entry = tile | flags << 30. One warp per (pair, head); lane 0 merges (the
// lists are ≤ maxsel long, a few hundred steps). Prefix / dense queries
// attend to every tile: their list is the range 0..nb.
extern "C" __global__ void fw_vsa_pair(u64 sel_, u64 nsel_, u64 usel_, u64 nusel_, int nb, int np, int dense_all,
                                       int maxsel, int ustride, int npairs) {
    const unsigned *sel = (const unsigned*)sel_, *nsel = (const unsigned*)nsel_;
    unsigned *usel = (unsigned*)usel_, *nusel = (unsigned*)nusel_;
    const int p = blockIdx.x, h = blockIdx.y;
    if (threadIdx.x != 0) return;
    const int qa = 2 * p, qb = qa + 1;
    const bool alla = qa < np || dense_all, allb = qb < nb && (qb < np || dense_all);
    const unsigned na = alla ? (unsigned)nb : nsel[(long)h * nb + qa];
    const unsigned nbv = qb >= nb ? 0u : (allb ? (unsigned)nb : nsel[(long)h * nb + qb]);
    const unsigned* la = sel + ((long)h * nb + qa) * maxsel;
    const unsigned* lb = sel + ((long)h * nb + (qb < nb ? qb : qa)) * maxsel;
    unsigned* o = usel + ((long)h * npairs + p) * ustride;
    unsigned i = 0, j = 0, n = 0;
    while (i < na || j < nbv) {
        const unsigned ta = i < na ? (alla ? i : la[i]) : 0xffffffffu;
        const unsigned tb = j < nbv ? (allb ? j : lb[j]) : 0xffffffffu;
        if (ta < tb) { o[n++] = ta | (1u << 30); ++i; }
        else if (tb < ta) { o[n++] = tb | (2u << 30); ++j; }
        else { o[n++] = ta | (3u << 30); ++i; ++j; }
    }
    nusel[(long)h * npairs + p] = n;
}

// Second-generation body (v2): 32 query rows per warp (two m16 tiles), so
// every K / V fragment read from shared memory feeds two mma instructions
// instead of one. The v1 body above is shared-memory-bandwidth bound at
// ~half the tensor peak: per m16n8k16 it reads 256 B of K (or V) for 2048
// MACs, which on GA10x is exactly the 128 B/clk the SM can deliver. Chunks
// are 32 keys (MMA2_BK) so the S tile stays at 32 registers per m-tile and
// the K + V staging is 17 KB per block: 4-5 blocks per SM. WARPS warps =
// 32·WARPS query rows per block: 4 (128 rows) for dense attention and for
// VSA tiles of 128 / 256, 2 (64 rows) for 64-token VSA tiles.
#define MMA2_BK 32

// Q8 (mode 0 only): INT8 Q·Kᵀ as in the VSA fine stage. q8 / k8 hold the int8
// codes of Q and of K − mean(K) ([heads][sq][D] / [kv_heads][sk][D] bytes),
// qsc / ksc one scale per (head, 64-row tile); K chunks stage as int8 (ks8,
// 16-byte chunks swizzled by the row) and the scores come out of mma.s8 as
// int32, scaled by the two tiles' qsc·ksc. P·V stays bf16.
// PV8 (with Q8, mode 0): FP8 P·V as in h3_core::vsa::attention_pv8_reference.
// v8t holds V − mean(V) as E4M3 codes, transposed per KV head to [D][sk
// padded to 32] with the keys of every 32-key chunk in mma B-fragment order
// (fw_attn_v8), vmean [kv_heads][D] and vsc [kv_heads] its mean and scale.
// Running maxima are rounded up to integers (rescales exact powers of two),
// P = exp2(s − ⌈m⌉ + 6) rounded to E4M3 for the mma and summed exactly for
// l, and the epilogue applies vsc / l and adds the mean back.
template <int D, int WARPS, bool Q8 = false, bool PV8 = false>
__device__ __forceinline__ void attn_mma2_body(int mode, const bf16_t* q, const bf16_t* k, const bf16_t* v, bf16_t* out,
        int heads, int kv_heads, int sq, int sk, int causal, float scale,
        const unsigned* slot, const unsigned* sel, const unsigned* nsel, const float* pooled, const bf16_t* gate,
        int has_gate, int nb, int np, int dense_all, int te, int maxsel,
        bf16_t* ks, bf16_t* vs, unsigned char* kvalid,
        const signed char* q8 = 0, const signed char* k8 = 0, const float* qsc = 0, const float* ksc = 0,
        unsigned char* ks8 = 0, const unsigned char* v8t = 0, const float* vmean = 0, const float* vsc = 0,
        unsigned char* vs8 = 0) {
    static_assert(!PV8 || Q8, "PV8 needs Q8");
    constexpr int KS_STRIDE = D + 8;
    constexpr int KK = D / 16;        // k-steps of Q·Kᵀ
    constexpr int DT = D / 8;         // 8-wide dim tiles of the output
    constexpr int CH = D / 8;         // 16-byte chunks per key row
    constexpr int NTHR = 32 * WARPS;
    constexpr int QR = 32 * WARPS;    // query rows per block
    constexpr int RP = NTHR / CH;     // key rows loaded per pass
    constexpr int NT = MMA2_BK / 8;   // n-tiles of S per chunk (4)
    const int h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t = lane & 3;
    // VSA: a tile of te keys is te/32 chunks; its te queries are te/QR blocks.
    const int csub = mode == 0 ? 1 : te / MMA2_BK;
    const int qsubs = mode == 0 ? 1 : (te > QR ? te / QR : 1);
    const int qt = mode == 0 ? blockIdx.x : blockIdx.x / qsubs;
    const int qsub = mode == 0 ? 0 : blockIdx.x % qsubs;
    const int kh = mode == 0 ? h / (heads / kv_heads) : h;
    // Rows: m-tile mt of this warp covers block rows warp·32 + mt·16 + {g, g+8}.
    int tok[2][2];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        #pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
            const int lr = warp * 32 + mt * 16 + hh * 8 + g;
            if (mode == 0) {
                tok[mt][hh] = qt * QR + lr < sq ? qt * QR + lr : -1;
            } else {
                const int qi = qsub * QR + lr;
                const unsigned a = qi < te ? slot[(long)qt * te + qi] : 0xffffffffu;
                tok[mt][hh] = a == 0xffffffffu ? -1 : (int)a;
            }
        }
    }
    const float sl2 = scale * 1.4426950408889634f;

    unsigned qa[2][Q8 ? 1 : KK][4];
    if constexpr (!Q8) {
        #pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const unsigned* q0 = (const unsigned*)(q + ((long)h * sq + (tok[mt][0] < 0 ? 0 : tok[mt][0])) * D);
            const unsigned* q1 = (const unsigned*)(q + ((long)h * sq + (tok[mt][1] < 0 ? 0 : tok[mt][1])) * D);
            #pragma unroll
            for (int kk = 0; kk < KK; ++kk) {
                const int c = (kk * 16 + t * 2) >> 1;
                qa[mt][kk][0] = tok[mt][0] >= 0 ? q0[c] : 0u;
                qa[mt][kk][1] = tok[mt][1] >= 0 ? q1[c] : 0u;
                qa[mt][kk][2] = tok[mt][0] >= 0 ? q0[c + 4] : 0u;
                qa[mt][kk][3] = tok[mt][1] >= 0 ? q1[c + 4] : 0u;
            }
        }
    }
    // INT8: k32 steps; a thread's A fragment of step kk is bytes kk·32 + 4t.. of
    // its two rows (two 32-bit words each, 16 bytes apart).
    constexpr int KK8 = D / 32, CH8 = D / 16, RP8 = NTHR / CH8, NJ8 = MMA2_BK / RP8;
    auto at8 = [](int row, int chunk) -> int { return row * D + ((chunk ^ (row & (CH8 - 1))) << 4); };
    unsigned qa8[2][Q8 ? KK8 : 1][4];
    float qs8 = 1.0f;
    if constexpr (Q8) {
        #pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const unsigned* q80 = (const unsigned*)(q8 + ((long)h * sq + (tok[mt][0] < 0 ? 0 : tok[mt][0])) * D);
            const unsigned* q81 = (const unsigned*)(q8 + ((long)h * sq + (tok[mt][1] < 0 ? 0 : tok[mt][1])) * D);
            #pragma unroll
            for (int kk = 0; kk < KK8; ++kk) {
                qa8[mt][kk][0] = tok[mt][0] >= 0 ? q80[kk * 8 + t] : 0u;
                qa8[mt][kk][1] = tok[mt][1] >= 0 ? q81[kk * 8 + t] : 0u;
                qa8[mt][kk][2] = tok[mt][0] >= 0 ? q80[kk * 8 + 4 + t] : 0u;
                qa8[mt][kk][3] = tok[mt][1] >= 0 ? q81[kk * 8 + 4 + t] : 0u;
            }
        }
        // The warp's 32 rows lie inside one 64-row scale tile.
        const int nqt = (sq + 63) / 64;
        qs8 = qsc[(long)h * nqt + min((qt * QR + warp * 32) / 64, nqt - 1)];
    }
    float o[2][DT][4];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int i = 0; i < DT; ++i) o[mt][i][0] = o[mt][i][1] = o[mt][i][2] = o[mt][i][3] = 0.0f;
    float m[2][2], l[2][2];
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) { m[mt][0] = m[mt][1] = -1e30f; l[mt][0] = l[mt][1] = 0.0f; }

    int nchunks;
    if (mode == 0) {
        nchunks = (sk + MMA2_BK - 1) / MMA2_BK;
        if (causal) {
            const int maxrow = min(sq - 1, qt * QR + QR - 1);
            const int maxkey = maxrow + sk - sq;            // inclusive
            nchunks = min(nchunks, maxkey / MMA2_BK + 1);
            if (maxkey < 0) nchunks = 0;
        }
    } else {
        nchunks = (int)nsel[(long)h * nb + qt] * csub;
    }
    const bool all_tiles = mode == 1 && (qt < np || dense_all);
    const bf16_t* kbase = k + (long)kh * sk * D;
    const bf16_t* vbase = v + (long)kh * sk * D;
    const int ch = tid % CH, krow = tid / CH;
    auto key_tok = [&](int c, int key) -> int {
        if (mode == 0) return c * MMA2_BK + key < sk ? c * MMA2_BK + key : -1;
        const unsigned tile = all_tiles ? (unsigned)(c / csub) : sel[((long)h * nb + qt) * maxsel + c / csub];
        const unsigned s2 = slot[(long)tile * te + (c % csub) * MMA2_BK + key];
        return s2 == 0xffffffffu ? -1 : (int)s2;
    };
    auto load = [&](bf16_t* dst, const bf16_t* src, int c, unsigned char* valid) {
        #pragma unroll
        for (int j = 0; j < MMA2_BK / RP; ++j) {
            const int key = krow + RP * j, tk = key_tok(c, key);
            cp_async16(dst + key * KS_STRIDE + ch * 8, src + (long)(tk < 0 ? 0 : tk) * D + ch * 8, tk >= 0);
            if (valid && ch == 0) valid[key] = tk >= 0;
        }
    };
    // int8 K rows of chunk c into ks8 (zero-filled padding), with the valid mask.
    const int ch8 = tid % CH8, krow8 = tid / CH8;
    const signed char* k8base = Q8 ? k8 + (long)kh * sk * D : 0;
    auto load8 = [&](int c, unsigned char* valid) {
        #pragma unroll
        for (int j = 0; j < (Q8 ? NJ8 : 1); ++j) {
            const int key = krow8 + RP8 * j, tk = key_tok(c, key);
            cp_async16(ks8 + at8(key, ch8), k8base + (long)(tk < 0 ? 0 : tk) * D + ch8 * 16, tk >= 0);
            if (valid && ch8 == 0) valid[key] = tk >= 0;
        }
    };
    // FP8 V: chunk c of the transposed codes is D rows × 32 bytes (two 16-byte
    // halves per row, swizzled by bit 2 of the row so ldmatrix.x2 over eight
    // rows hits eight different banks). Zero codes pad keys ≥ sk.
    const int skp = (sk + MMA2_BK - 1) / MMA2_BK * MMA2_BK;
    const unsigned char* v8base = PV8 ? v8t + (long)kh * D * skp : 0;
    auto at8v = [](int r, int half) -> int { return r * 32 + ((half ^ ((r >> 2) & 1)) << 4); };
    auto load_v8 = [&](int c) {
        #pragma unroll
        for (int j = 0; j < (PV8 ? (2 * D) / NTHR : 1); ++j) {
            const int idx = tid + NTHR * j, r = idx >> 1, half = idx & 1;
            cp_async16(vs8 + at8v(r, half), v8base + (long)r * skp + c * MMA2_BK + half * 16, true);
        }
    };
    if (nchunks > 0) {
        if constexpr (Q8) load8(0, kvalid); else load(ks, kbase, 0, kvalid);
        cp_async_commit();
        if constexpr (PV8) load_v8(0); else load(vs, vbase, 0, 0);
        cp_async_commit();
    }

    for (int c = 0; c < nchunks; ++c) {
        const unsigned char* kv = kvalid + (c & 1) * MMA2_BK;
        cp_async_wait1();          // K_c landed (V_c may still be in flight)
        __syncthreads();
        float s[2][NT][4];
        if constexpr (Q8) {
            // The chunk's 32 keys lie inside one 64-row K scale tile.
            const int nkt = (sk + 63) / 64;
            const float f8 = qs8 * ksc[(long)kh * nkt + min((c * MMA2_BK) / 64, nkt - 1)];
            const unsigned k8base_l = (unsigned)__cvta_generic_to_shared(ks8) + (lane & 7) * D;
            const int rsw = (lane & 7) & (CH8 - 1), mhalf = (lane >> 3) & 1;
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                int a0[4] = {0, 0, 0, 0}, a1[4] = {0, 0, 0, 0};
                #pragma unroll
                for (int kk = 0; kk < KK8; ++kk) {
                    const unsigned addr = k8base_l + nt * 8 * D + (((2 * kk + mhalf) ^ rsw) << 4);
                    unsigned b0, b1;
                    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(b0), "=r"(b1) : "r"(addr));
                    mma_s8(a0, qa8[0][kk][0], qa8[0][kk][1], qa8[0][kk][2], qa8[0][kk][3], b0, b1);
                    mma_s8(a1, qa8[1][kk][0], qa8[1][kk][1], qa8[1][kk][2], qa8[1][kk][3], b0, b1);
                }
                #pragma unroll
                for (int i = 0; i < 4; ++i) { s[0][nt][i] = (float)a0[i] * f8; s[1][nt][i] = (float)a1[i] * f8; }
            }
        } else {
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                s[0][nt][0] = s[0][nt][1] = s[0][nt][2] = s[0][nt][3] = 0.0f;
                s[1][nt][0] = s[1][nt][1] = s[1][nt][2] = s[1][nt][3] = 0.0f;
                const bf16_t* kr = ks + (nt * 8 + g) * KS_STRIDE;
                #pragma unroll
                for (int kk = 0; kk < KK; ++kk) {
                    const unsigned b0 = *(const unsigned*)(kr + kk * 16 + t * 2);
                    const unsigned b1 = *(const unsigned*)(kr + kk * 16 + 8 + t * 2);
                    mma_bf16(s[0][nt], qa[0][kk][0], qa[0][kk][1], qa[0][kk][2], qa[0][kk][3], b0, b1);
                    mma_bf16(s[1][nt], qa[1][kk][0], qa[1][kk][1], qa[1][kk][2], qa[1][kk][3], b0, b1);
                }
            }
        }
        float mx[2][2];
        #pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            mx[mt][0] = m[mt][0]; mx[mt][1] = m[mt][1];
            #pragma unroll
            for (int nt = 0; nt < NT; ++nt) {
                #pragma unroll
                for (int cc = 0; cc < 2; ++cc) {
                    const int key = nt * 8 + t * 2 + cc;
                    bool ok0 = kv[key], ok1 = ok0;
                    if (mode == 0 && causal) {
                        const int j = c * MMA2_BK + key;
                        ok0 = ok0 && tok[mt][0] >= 0 && j <= tok[mt][0] + sk - sq;
                        ok1 = ok1 && tok[mt][1] >= 0 && j <= tok[mt][1] + sk - sq;
                    }
                    s[mt][nt][cc] = ok0 ? s[mt][nt][cc] * sl2 : -1e30f;
                    s[mt][nt][2 + cc] = ok1 ? s[mt][nt][2 + cc] * sl2 : -1e30f;
                    mx[mt][0] = fmaxf(mx[mt][0], s[mt][nt][cc]);
                    mx[mt][1] = fmaxf(mx[mt][1], s[mt][nt][2 + cc]);
                }
            }
        }
        __syncthreads();           // every warp is done with ks
        if (c + 1 < nchunks) {
            if constexpr (Q8) load8(c + 1, kvalid + ((c + 1) & 1) * MMA2_BK); else load(ks, kbase, c + 1, kvalid + ((c + 1) & 1) * MMA2_BK);
        }
        cp_async_commit();
        #pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            #pragma unroll
            for (int off = 1; off < 4; off <<= 1) {
                mx[mt][0] = fmaxf(mx[mt][0], __shfl_xor_sync(0xffffffffu, mx[mt][0], off));
                mx[mt][1] = fmaxf(mx[mt][1], __shfl_xor_sync(0xffffffffu, mx[mt][1], off));
            }
            if constexpr (PV8) {
                // Finite maxima only: a fully masked row keeps -1e30 and p = 0.
                mx[mt][0] = mx[mt][0] > -1e29f ? ceilf(mx[mt][0]) : mx[mt][0];
                mx[mt][1] = mx[mt][1] > -1e29f ? ceilf(mx[mt][1]) : mx[mt][1];
            }
            const float c0 = exp2f(m[mt][0] - mx[mt][0]), c1 = exp2f(m[mt][1] - mx[mt][1]);
            m[mt][0] = mx[mt][0]; m[mt][1] = mx[mt][1];
            l[mt][0] *= c0; l[mt][1] *= c1;
            #pragma unroll
            for (int i = 0; i < DT; ++i) { o[mt][i][0] *= c0; o[mt][i][1] *= c0; o[mt][i][2] *= c1; o[mt][i][3] *= c1; }
        }
        cp_async_wait1();          // V_c landed (K_{c+1} may still be in flight)
        __syncthreads();
        if constexpr (PV8) {
            // One k32 step covers the chunk. A fragment of row g: bytes (keys)
            // 4t..4t+3 of the virtual k order, i.e. keys {2t, 2t+1, 8+2t, 9+2t}
            // of the chunk (S n-tiles 0 and 1 of this thread) for a0 / a1 and
            // the same +16 (n-tiles 2 and 3) for a2 / a3. fw_attn_v8 stores V
            // in that key order, so B comes straight from ldmatrix over the
            // transposed rows: lanes 0-7 the first 16 bytes of rows d0..d0+7,
            // lanes 8-15 the second.
            unsigned a[2][4];
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                float p[4][4];
                #pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    #pragma unroll
                    for (int cc = 0; cc < 4; ++cc) {
                        p[nt][cc] = exp2f(s[mt][nt][cc] - (cc < 2 ? m[mt][0] : m[mt][1]) + 6.0f);
                    }
                }
                l[mt][0] += p[0][0] + p[0][1] + p[1][0] + p[1][1] + p[2][0] + p[2][1] + p[3][0] + p[3][1];
                l[mt][1] += p[0][2] + p[0][3] + p[1][2] + p[1][3] + p[2][2] + p[2][3] + p[3][2] + p[3][3];
                a[mt][0] = pack_e4m3x2(p[0][0], p[0][1]) | (pack_e4m3x2(p[1][0], p[1][1]) << 16);
                a[mt][1] = pack_e4m3x2(p[0][2], p[0][3]) | (pack_e4m3x2(p[1][2], p[1][3]) << 16);
                a[mt][2] = pack_e4m3x2(p[2][0], p[2][1]) | (pack_e4m3x2(p[3][0], p[3][1]) << 16);
                a[mt][3] = pack_e4m3x2(p[2][2], p[2][3]) | (pack_e4m3x2(p[3][2], p[3][3]) << 16);
            }
            const unsigned vs8_l = (unsigned)__cvta_generic_to_shared(vs8);
            const int lr = lane & 7, lhalf = (lane >> 3) & 1;
            #pragma unroll
            for (int dt = 0; dt < DT; ++dt) {
                const int r = dt * 8 + lr;
                const unsigned addr = vs8_l + r * 32 + ((lhalf ^ ((r >> 2) & 1)) << 4);
                unsigned b0, b1;
                asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(b0), "=r"(b1) : "r"(addr));
                mma_e4m3(o[0][dt], a[0][0], a[0][1], a[0][2], a[0][3], b0, b1);
                mma_e4m3(o[1][dt], a[1][0], a[1][1], a[1][2], a[1][3], b0, b1);
            }
        } else
        #pragma unroll
        for (int kk = 0; kk < MMA2_BK / 16; ++kk) {
            unsigned a[2][4];
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                float p[8];
                #pragma unroll
                for (int cc = 0; cc < 4; ++cc) {
                    p[cc] = exp2f(s[mt][2 * kk][cc] - (cc < 2 ? m[mt][0] : m[mt][1]));
                    p[4 + cc] = exp2f(s[mt][2 * kk + 1][cc] - (cc < 2 ? m[mt][0] : m[mt][1]));
                }
                l[mt][0] += p[0] + p[1] + p[4] + p[5];
                l[mt][1] += p[2] + p[3] + p[6] + p[7];
                a[mt][0] = pack_bf16(p[0], p[1]); a[mt][1] = pack_bf16(p[2], p[3]);
                a[mt][2] = pack_bf16(p[4], p[5]); a[mt][3] = pack_bf16(p[6], p[7]);
            }
            const int vkey = kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8;
            #pragma unroll
            for (int dt = 0; dt < DT; dt += 2) {
                const unsigned addr = (unsigned)__cvta_generic_to_shared(vs + vkey * KS_STRIDE + dt * 8 + (lane >> 4) * 8);
                unsigned b0, b1, b2, b3;
                asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                             : "=r"(b0), "=r"(b1), "=r"(b2), "=r"(b3) : "r"(addr));
                mma_bf16(o[0][dt], a[0][0], a[0][1], a[0][2], a[0][3], b0, b1);
                mma_bf16(o[0][dt + 1], a[0][0], a[0][1], a[0][2], a[0][3], b2, b3);
                mma_bf16(o[1][dt], a[1][0], a[1][1], a[1][2], a[1][3], b0, b1);
                mma_bf16(o[1][dt + 1], a[1][0], a[1][1], a[1][2], a[1][3], b2, b3);
            }
        }
        __syncthreads();           // every warp is done with vs
        if (c + 1 < nchunks) { if constexpr (PV8) load_v8(c + 1); else load(vs, vbase, c + 1, 0); }
        cp_async_commit();
    }
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        #pragma unroll
        for (int off = 1; off < 4; off <<= 1) {
            l[mt][0] += __shfl_xor_sync(0xffffffffu, l[mt][0], off);
            l[mt][1] += __shfl_xor_sync(0xffffffffu, l[mt][1], off);
        }
        const float vscale = PV8 ? vsc[kh] : 1.0f;
        const float i0 = l[mt][0] > 0.0f ? vscale / l[mt][0] : 0.0f, i1 = l[mt][1] > 0.0f ? vscale / l[mt][1] : 0.0f;
        const int tok0 = tok[mt][0], tok1 = tok[mt][1];
        #pragma unroll
        for (int dt = 0; dt < DT; ++dt) {
            const int cidx = dt * 8 + t * 2;
            float a0 = o[mt][dt][0] * i0, a1 = o[mt][dt][1] * i0, b0 = o[mt][dt][2] * i1, b1 = o[mt][dt][3] * i1;
            if constexpr (PV8) {
                // The oracle adds the mean back on every row, empty ones too.
                const float m0 = vmean[(long)kh * D + cidx], m1 = vmean[(long)kh * D + cidx + 1];
                a0 += m0; a1 += m1; b0 += m0; b1 += m1;
            }
            if (mode == 1 && has_gate) {
                const float* cr = pooled + ((long)h * nb + qt) * D + cidx;
                if (tok0 >= 0) { const bf16_t* gr = gate + ((long)h * sq + tok0) * D + cidx; a0 += cr[0] * ld(gr, 0); a1 += cr[1] * ld(gr, 1); }
                if (tok1 >= 0) { const bf16_t* gr = gate + ((long)h * sq + tok1) * D + cidx; b0 += cr[0] * ld(gr, 0); b1 += cr[1] * ld(gr, 1); }
            }
            if (tok0 >= 0) *(unsigned*)(out + ((long)tok0 * heads + h) * D + cidx) = pack_bf16(a0, a1);
            if (tok1 >= 0) *(unsigned*)(out + ((long)tok1 * heads + h) * D + cidx) = pack_bf16(b0, b1);
        }
    }
}

#define ATTN_MMA2_KERNELS(D)                                                                                         \
extern "C" __global__ void __launch_bounds__(128) fw_attention_mma2_##D(u64 q_, u64 k_, u64 v_, u64 out_, int heads, \
        int kv_heads, int sq, int sk, int causal, float scale) {                                                     \
    __shared__ __align__(16) bf16_t ks[MMA2_BK * (D + 8)];                                                           \
    __shared__ __align__(16) bf16_t vs[MMA2_BK * (D + 8)];                                                           \
    __shared__ unsigned char kvalid[2 * MMA2_BK];                                                                    \
    attn_mma2_body<D, 4>(0, (const bf16_t*)q_, (const bf16_t*)k_, (const bf16_t*)v_, (bf16_t*)out_, heads,          \
                         kv_heads, sq, sk, causal, scale, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, ks, vs, kvalid);          \
}                                                                                                                    \
extern "C" __global__ void __launch_bounds__(64) fw_vsa_fine_mma2_##D##_w2(u64 q_, u64 k_, u64 v_, u64 out_,         \
        int heads, int seq, float scale, u64 slot_, u64 sel_, u64 nsel_, u64 pooled_, u64 gate_, int has_gate,       \
        int nb, int np, int dense_all, int te, int maxsel) {                                                         \
    __shared__ __align__(16) bf16_t ks[MMA2_BK * (D + 8)];                                                           \
    __shared__ __align__(16) bf16_t vs[MMA2_BK * (D + 8)];                                                           \
    __shared__ unsigned char kvalid[2 * MMA2_BK];                                                                    \
    attn_mma2_body<D, 2>(1, (const bf16_t*)q_, (const bf16_t*)k_, (const bf16_t*)v_, (bf16_t*)out_, heads, heads,   \
                         seq, seq, 0, scale, (const unsigned*)slot_, (const unsigned*)sel_, (const unsigned*)nsel_,  \
                         (const float*)pooled_, (const bf16_t*)gate_, has_gate, nb, np, dense_all, te, maxsel, ks,   \
                         vs, kvalid);                                                                                 \
}                                                                                                                    \
extern "C" __global__ void __launch_bounds__(128) fw_vsa_fine_mma2_##D##_w4(u64 q_, u64 k_, u64 v_, u64 out_,        \
        int heads, int seq, float scale, u64 slot_, u64 sel_, u64 nsel_, u64 pooled_, u64 gate_, int has_gate,       \
        int nb, int np, int dense_all, int te, int maxsel) {                                                         \
    __shared__ __align__(16) bf16_t ks[MMA2_BK * (D + 8)];                                                           \
    __shared__ __align__(16) bf16_t vs[MMA2_BK * (D + 8)];                                                           \
    __shared__ unsigned char kvalid[2 * MMA2_BK];                                                                    \
    attn_mma2_body<D, 4>(1, (const bf16_t*)q_, (const bf16_t*)k_, (const bf16_t*)v_, (bf16_t*)out_, heads, heads,   \
                         seq, seq, 0, scale, (const unsigned*)slot_, (const unsigned*)sel_, (const unsigned*)nsel_,  \
                         (const float*)pooled_, (const bf16_t*)gate_, has_gate, nb, np, dense_all, te, maxsel, ks,   \
                         vs, kvalid);                                                                                 \
}
ATTN_MMA2_KERNELS(64)
ATTN_MMA2_KERNELS(128)

#define ATTN_QK8_KERNEL(D)                                                                                           \
extern "C" __global__ void __launch_bounds__(128) fw_attention_qk8_##D(u64 q_, u64 k_, u64 v_, u64 out_, int heads, \
        int kv_heads, int sq, int sk, int causal, float scale, u64 q8_, u64 k8_, u64 qsc_, u64 ksc_) {              \
    __shared__ __align__(16) unsigned char ks8[MMA2_BK * D];                                                         \
    __shared__ __align__(16) bf16_t vs[MMA2_BK * (D + 8)];                                                           \
    __shared__ unsigned char kvalid[2 * MMA2_BK];                                                                    \
    attn_mma2_body<D, 4, true>(0, (const bf16_t*)q_, (const bf16_t*)k_, (const bf16_t*)v_, (bf16_t*)out_, heads,    \
                         kv_heads, sq, sk, causal, scale, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (bf16_t*)0, vs, kvalid,   \
                         (const signed char*)q8_, (const signed char*)k8_, (const float*)qsc_, (const float*)ksc_,   \
                         ks8);                                                                                       \
}
ATTN_QK8_KERNEL(64)
ATTN_QK8_KERNEL(128)

#define ATTN_PV8_KERNEL(D)                                                                                           \
extern "C" __global__ void __launch_bounds__(128) fw_attention_pv8_##D(u64 q_, u64 k_, u64 v_, u64 out_, int heads, \
        int kv_heads, int sq, int sk, int causal, float scale, u64 q8_, u64 k8_, u64 qsc_, u64 ksc_, u64 v8t_,       \
        u64 vmean_, u64 vsc_) {                                                                                      \
    __shared__ __align__(16) unsigned char ks8[MMA2_BK * D];                                                         \
    __shared__ __align__(16) unsigned char vs8[MMA2_BK * D];                                                         \
    __shared__ unsigned char kvalid[2 * MMA2_BK];                                                                    \
    attn_mma2_body<D, 4, true, true>(0, (const bf16_t*)q_, (const bf16_t*)k_, (const bf16_t*)v_, (bf16_t*)out_,    \
                         heads, kv_heads, sq, sk, causal, scale, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, (bf16_t*)0,        \
                         (bf16_t*)0, kvalid, (const signed char*)q8_, (const signed char*)k8_, (const float*)qsc_,   \
                         (const float*)ksc_, ks8, (const unsigned char*)v8t_, (const float*)vmean_,                  \
                         (const float*)vsc_, vs8);                                                                   \
}
ATTN_PV8_KERNEL(64)
ATTN_PV8_KERNEL(128)

// h3_core::vsa::attn_v8_round, device side. fw_attn_v8_max: one block per
// (chunk of 64 keys, KV head), 256 threads, writes max|v − mean| per block to
// part[head][chunk]; fw_attn_v8: one block per (32-key chunk, KV head), d
// threads (thread = channel): reduces the chunk maxima to vsc[head] =
// max / 448 and writes the chunk's E4M3 codes of (v − mean) / vsc transposed
// into v8t[head][channel][skp] with the 32 keys in mma B-fragment order:
// position 4j + i holds key {2j, 2j+1, 8+2j, 9+2j}[i] for j < 4 and the same
// + 16 for j ≥ 4. Keys ≥ sk are code 0. With `slot` (the VSA fine stage) the
// positions are tile slots: position p holds the token slot[p] (0xffffffff
// = empty, code 0), and skp is the slot count.
extern "C" __global__ void __launch_bounds__(256) fw_attn_v8_max(u64 v_, u64 vmean_, u64 part_, int sk, int d) {
    __shared__ float red[8];
    const bf16_t* v = (const bf16_t*)v_;
    const float* vm = (const float*)vmean_ + (long)blockIdx.y * d;
    const int row0 = blockIdx.x * 64, rows = min(64, sk - row0), n = rows * d;
    const long base = ((long)blockIdx.y * sk + row0) * d;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    float mx = 0.0f;
    for (int i = tid; i < n; i += 256) mx = fmaxf(mx, fabsf(ld(v, base + i) - vm[i % d]));
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    if (tid == 0) {
        float m = red[0];
        for (int w = 1; w < 8; ++w) m = fmaxf(m, red[w]);
        ((float*)part_)[(long)blockIdx.y * gridDim.x + blockIdx.x] = m;
    }
}
extern "C" __global__ void __launch_bounds__(128) fw_attn_v8(u64 v_, u64 vmean_, u64 part_, int nparts, u64 vsc_, u64 v8t_, int sk, int d, u64 slot_) {
    __shared__ float scs;
    const bf16_t* v = (const bf16_t*)v_;
    const int h = blockIdx.y, c = blockIdx.x, tid = threadIdx.x;
    const float* vm = (const float*)vmean_ + (long)h * d;
    const int skp = gridDim.x * MMA2_BK;
    if (tid == 0) {
        const float* part = (const float*)part_ + (long)h * nparts;
        float m = 0.0f;
        for (int i = 0; i < nparts; ++i) m = fmaxf(m, part[i]);
        scs = m / 448.0f;
        if (c == 0) ((float*)vsc_)[h] = scs;
    }
    __syncthreads();
    if (tid >= d) return;
    const float inv = scs > 0.0f ? 1.0f / scs : 0.0f, mean = vm[tid];
    unsigned char* dst = (unsigned char*)v8t_ + ((long)h * d + tid) * skp + (long)c * MMA2_BK;
    unsigned words[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        unsigned w = 0;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int jj = j & 3, key = (j >> 2) * 16 + (i < 2 ? 2 * jj + i : 8 + 2 * jj + (i - 2));
            const int pos = c * MMA2_BK + key;
            int tk;
            if (slot_) { const unsigned s2 = ((const unsigned*)slot_)[pos]; tk = s2 == 0xffffffffu ? -1 : (int)s2; }
            else tk = pos < sk ? pos : -1;
            const unsigned code = tk >= 0 ? mx_e4m3((ld(v, ((long)h * sk + tk) * d + tid) - mean) * inv) : 0u;
            w |= code << (8 * i);
        }
        words[j] = w;
    }
    uint4* o = (uint4*)dst;
    uint4 w0; w0.x = words[0]; w0.y = words[1]; w0.z = words[2]; w0.w = words[3];
    uint4 w1; w1.x = words[4]; w1.y = words[5]; w1.z = words[6]; w1.w = words[7];
    o[0] = w0;
    o[1] = w1;
}

// h3_core::vsa::attn_qk8_round: one 64-row tile of X ([heads][seq][d] bf16)
// per block (tile, head), 256 threads: the tile's scale max|x − mean| / 127
// (mean = kmean[head] for K, none for Q) to sc[head][tile], then the int8
// codes (round to nearest even, clamped to ±127) to x8.
extern "C" __global__ void __launch_bounds__(256) fw_attn_qk8_quant(u64 x_, u64 kmean_, u64 x8_, u64 sc_, int seq, int d, int has_mean) {
    __shared__ float red[8];
    __shared__ float scs;
    const bf16_t* x = (const bf16_t*)x_;
    const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const float* km = has_mean ? (const float*)kmean_ + (long)h * d : 0;
    signed char* x8 = (signed char*)x8_;
    const int row0 = t * 64, rows = min(64, seq - row0), n = rows * d;
    const long base = ((long)h * seq + row0) * d;
    float mx = 0.0f;
    for (int i = tid; i < n; i += 256) {
        const float v = ld(x, base + i) - (km ? km[i % d] : 0.0f);
        mx = fmaxf(mx, fabsf(v));
    }
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if (lane == 0) red[warp] = mx;
    __syncthreads();
    if (tid == 0) {
        float m = red[0];
        for (int w = 1; w < 8; ++w) m = fmaxf(m, red[w]);
        scs = m / 127.0f;
        ((float*)sc_)[(long)h * gridDim.x + t] = scs;
    }
    __syncthreads();
    const float inv = scs > 0.0f ? 1.0f / scs : 0.0f;
    for (int i = tid; i < n; i += 256) {
        const float v = ld(x, base + i) - (km ? km[i % d] : 0.0f);
        x8[base + i] = (signed char)max(-127, min(127, __float2int_rn(v * inv)));
    }
}

// ---------------------------------------------------------------- VSA

// Tile means (f32): block (tile, head), 128 threads. pooled is [3][heads][nb][d].
// Each warp sums every fourth slot of the tile with 8-byte (4 × bf16) loads
// per lane, so one key row is one coalesced read per warp; the four partial
// sums meet in shared memory. d % 4 == 0 and d <= 128 (scalar fallback below).
extern "C" __global__ void __launch_bounds__(128) fw_vsa_pool(u64 q_, u64 k_, u64 v_, u64 slot_, u64 bsz_, u64 pooled_, int heads, int seq, int d, int nb, int te) {
    __shared__ float part[3][4][128];
    const bf16_t *q = (const bf16_t*)q_, *k = (const bf16_t*)k_, *v = (const bf16_t*)v_;
    const unsigned *slot = (const unsigned*)slot_, *bsz = (const unsigned*)bsz_;
    float* pooled = (float*)pooled_;
    const int t = blockIdx.x, h = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const unsigned n = bsz[t];
    const int d4 = d >> 2;
    float sq[4] = {0.f, 0.f, 0.f, 0.f}, sk[4] = {0.f, 0.f, 0.f, 0.f}, sv[4] = {0.f, 0.f, 0.f, 0.f};
    if (lane < d4) {
        for (unsigned i = warp; i < n; i += 4) {
            const long row = ((long)h * seq + slot[(long)t * te + i]) * d + lane * 4;
            const uint2 wq = *(const uint2*)(q + row), wk = *(const uint2*)(k + row), wv = *(const uint2*)(v + row);
            sq[0] += __uint_as_float(wq.x << 16); sq[1] += __uint_as_float(wq.x & 0xffff0000u);
            sq[2] += __uint_as_float(wq.y << 16); sq[3] += __uint_as_float(wq.y & 0xffff0000u);
            sk[0] += __uint_as_float(wk.x << 16); sk[1] += __uint_as_float(wk.x & 0xffff0000u);
            sk[2] += __uint_as_float(wk.y << 16); sk[3] += __uint_as_float(wk.y & 0xffff0000u);
            sv[0] += __uint_as_float(wv.x << 16); sv[1] += __uint_as_float(wv.x & 0xffff0000u);
            sv[2] += __uint_as_float(wv.y << 16); sv[3] += __uint_as_float(wv.y & 0xffff0000u);
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) { part[0][warp][lane * 4 + j] = sq[j]; part[1][warp][lane * 4 + j] = sk[j]; part[2][warp][lane * 4 + j] = sv[j]; }
    }
    __syncthreads();
    const int c = tid;
    if (c >= d) return;
    const float inv = 1.0f / (float)(n ? n : 1);
    const long o = ((long)h * nb + t) * d + c, plane = (long)heads * nb * d;
    pooled[o] = (part[0][0][c] + part[0][1][c] + part[0][2][c] + part[0][3][c]) * inv;
    pooled[plane + o] = (part[1][0][c] + part[1][1][c] + part[1][2][c] + part[1][3][c]) * inv;
    pooled[2 * plane + o] = (part[2][0][c] + part[2][1][c] + part[2][2][c] + part[2][3][c]) * inv;
}

// Scalar pool for head dims that are not a multiple of 4 (or > 128).
extern "C" __global__ void fw_vsa_pool_scalar(u64 q_, u64 k_, u64 v_, u64 slot_, u64 bsz_, u64 pooled_, int heads, int seq, int d, int nb, int te) {
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

// Four consecutive query tiles per block (nb <= 2048): the pooled K̄ rows are
// read once for four score rows and the pooled V̄ rows once for four gate
// branches, which is what the one-tile kernel above spends its time on
// (every block streams 2·nb·d floats from L2). Selection and compaction
// run per query tile as above.
#define VSA4_MAX_TILES 2048
extern "C" __global__ void __launch_bounds__(256) fw_vsa_coarse4(u64 pooled_, u64 pooled_out_, u64 sel_, u64 nsel_, u64 reg_,
        int heads, int d, int nb, int np, int nreg, int dense_all, int maxsel, int has_gate, float scale) {
    __shared__ float sc[4][VSA4_MAX_TILES];
    __shared__ unsigned char keep[VSA4_MAX_TILES];
    __shared__ float qrow[4][128];
    __shared__ float part[4][128];
    __shared__ unsigned hist[256];
    __shared__ unsigned wsum[8];
    __shared__ float red[9];
    __shared__ unsigned sh_prefix, sh_need;
    const float* pooled = (const float*)pooled_;
    float* pooled_out = (float*)pooled_out_;
    unsigned* sel = (unsigned*)sel_;
    unsigned* nsel = (unsigned*)nsel_;
    const unsigned* reg = (const unsigned*)reg_;
    const int q0 = blockIdx.x * 4, h = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int nq = min(4, nb - q0);
    const long plane = (long)heads * nb * d;
    for (int qq = 0; qq < 4; ++qq)
        if (tid < d) qrow[qq][tid] = qq < nq ? pooled[((long)h * nb + q0 + qq) * d + tid] : 0.0f;
    __syncthreads();

    // 1. scores: one warp per key tile, four query rows at once.
    const int d4 = d >> 2;
    for (int kt = warp; kt < nb; kt += 8) {
        const float4* kr = (const float4*)(pooled + plane + ((long)h * nb + kt) * d);
        float dot0 = 0.f, dot1 = 0.f, dot2 = 0.f, dot3 = 0.f;
        for (int c = lane; c < d4; c += 32) {
            const float4 kv = kr[c];
            dot0 += qrow[0][4 * c] * kv.x + qrow[0][4 * c + 1] * kv.y + qrow[0][4 * c + 2] * kv.z + qrow[0][4 * c + 3] * kv.w;
            dot1 += qrow[1][4 * c] * kv.x + qrow[1][4 * c + 1] * kv.y + qrow[1][4 * c + 2] * kv.z + qrow[1][4 * c + 3] * kv.w;
            dot2 += qrow[2][4 * c] * kv.x + qrow[2][4 * c + 1] * kv.y + qrow[2][4 * c + 2] * kv.z + qrow[2][4 * c + 3] * kv.w;
            dot3 += qrow[3][4 * c] * kv.x + qrow[3][4 * c + 1] * kv.y + qrow[3][4 * c + 2] * kv.z + qrow[3][4 * c + 3] * kv.w;
        }
        for (int o = 16; o > 0; o >>= 1) {
            dot0 += __shfl_xor_sync(0xffffffffu, dot0, o);
            dot1 += __shfl_xor_sync(0xffffffffu, dot1, o);
            dot2 += __shfl_xor_sync(0xffffffffu, dot2, o);
            dot3 += __shfl_xor_sync(0xffffffffu, dot3, o);
        }
        if (lane == 0) { sc[0][kt] = dot0 * scale; sc[1][kt] = dot1 * scale; sc[2][kt] = dot2 * scale; sc[3][kt] = dot3 * scale; }
    }
    __syncthreads();

    // 2 + 3. per query tile: marks, then stable compaction into sel.
    for (int qq = 0; qq < nq; ++qq) {
        const int qt = q0 + qq;
        const bool all = qt < np || dense_all;
        if (all) {
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
                    vsa_mark_topk(sc[qq], keep, lo, hi, need, hist, wsum, &sh_prefix, &sh_need);
                }
            }
        }
        unsigned* o = sel + ((long)h * nb + qt) * maxsel;
        const unsigned lt = (1u << lane) - 1u;
        unsigned base_sel = all ? (unsigned)nb : 0u;
        for (int c0 = all ? nb : 0; c0 < nb; c0 += 256) {
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
        __syncthreads();
    }

    // 4. pooled branch for the four rows, V̄ read once.
    if (!has_gate) return;
    float zt[4];
    for (int qq = 0; qq < 4; ++qq) {
        float mx = -3.0e38f;
        for (int kt = tid; kt < nb; kt += 256) mx = fmaxf(mx, sc[qq][kt]);
        for (int s2 = 16; s2 > 0; s2 >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, s2));
        if (lane == 0) red[warp] = mx;
        __syncthreads();
        float m = red[0];
        for (int i = 1; i < 8; ++i) m = fmaxf(m, red[i]);
        __syncthreads();
        float z = 0.f;
        for (int kt = tid; kt < nb; kt += 256) { const float e = __expf(sc[qq][kt] - m); sc[qq][kt] = e; z += e; }
        zt[qq] = block_sum256(z, red);   // syncs, so sc[qq][] holds the weights
    }
    const int c = tid & 127, hf = tid >> 7, mid = (nb + 1) >> 1;
    float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
    if (c < d) {
        const float* vp = pooled + 2 * plane + (long)h * nb * d + c;
        for (int kt = hf ? mid : 0; kt < (hf ? nb : mid); ++kt) {
            const float vv = vp[(long)kt * d];
            a0 = fmaf(sc[0][kt], vv, a0); a1 = fmaf(sc[1][kt], vv, a1); a2 = fmaf(sc[2][kt], vv, a2); a3 = fmaf(sc[3][kt], vv, a3);
        }
        if (hf) { part[0][c] = a0; part[1][c] = a1; part[2][c] = a2; part[3][c] = a3; }
    }
    __syncthreads();
    if (!hf && c < d) {
        float* ob = pooled_out + ((long)h * nb + q0) * d + c;
        if (nq > 0) ob[0] = (a0 + part[0][c]) / zt[0];
        if (nq > 1) ob[d] = (a1 + part[1][c]) / zt[1];
        if (nq > 2) ob[2 * d] = (a2 + part[2][c]) / zt[2];
        if (nq > 3) ob[3 * d] = (a3 + part[3][c]) / zt[3];
    }
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

// ---------------------------------------------------------------- MXFP8 (h3_core::quant::Mxfp8)
// f32 -> E4M3 as cvt.rn.satfinite: round to nearest even, saturate past 448,
// NaN -> 0x7F. Bit-exact twin of `quant::e4m3_satfinite`.
__device__ __forceinline__ unsigned mx_e4m3(float x) {
    unsigned u = __float_as_uint(x);
    unsigned sign = (u >> 24) & 0x80u;
    unsigned a = u & 0x7FFFFFFFu;
    if (a > 0x7F800000u) return sign | 0x7Fu;
    if (a >= 0x43E00000u) return sign | 0x7Eu;
    if (a < 0x3C800000u) return sign | (unsigned)rintf(__uint_as_float(a) * 512.0f);
    int e = (int)(a >> 23) - 127;
    unsigned m = (a >> 20) & 7u;
    unsigned rem = a & 0xFFFFFu;
    if (rem > 0x80000u || (rem == 0x80000u && (m & 1u))) { m += 1u; if (m == 8u) { m = 0u; e += 1; } }
    return sign | ((unsigned)(e + 7) << 3) | m;
}
__device__ __forceinline__ float mx_e4m3_decode(unsigned b) {
    float sign = (b & 0x80u) ? -1.0f : 1.0f;
    int e = (int)((b >> 3) & 0xFu);
    float m = (float)(b & 7u);
    if (e == 15 && (b & 7u) == 7u) return __int_as_float(0x7FC00000);
    if (e == 0) return sign * m / 512.0f;
    return sign * (1.0f + m / 8.0f) * exp2f((float)(e - 7));
}
// E8M0 block exponent: the smallest power of two keeping amax within 448. Returns the biased byte; *inv = 2^-e.
__device__ __forceinline__ unsigned mx_e8m0(float amax, float* inv) {
    int bits = __float_as_int(amax);
    int e0 = ((bits >> 23) & 0xFF) - 135;
    float threshold = __int_as_float(((e0 + 135) << 23) | 0x600000);
    int e = e0 + (amax > threshold ? 1 : 0);
    if (e < -127) e = -127;
    *inv = __int_as_float((127 - e) << 23);
    return (unsigned)(e + 127);
}
// cuBLASLt VEC32_UE8M0 (torch SWIZZLE_32_4_4) position of the scale of (row, group).
__device__ __forceinline__ long mx_scale_offset(long row, long group, long column_blocks) {
    long tile = (row / 128) * column_blocks + (group / 4);
    return tile * 512 + (row % 32) * 16 + ((row % 128) / 32) * 4 + (group % 4);
}
// One warp per 32-column block of x[rows, k] (bf16): block max, E8M0 scale, E4M3 codes.
// mode 0: codes (u8 [rows, k]) + swizzled scales (u8) for the FP8 GEMM;
// mode 1: the decoded values back to bf16 in `out` (the emulation path; equals the CPU reference exactly,
//         since E4M3 × 2^e is representable in bf16).
extern "C" __global__ void fw_mx_quant(u64 x_, long blocks, int k, u64 codes_, u64 scales_, u64 out_, int mode) {
    const bf16_t* x = (const bf16_t*)x_;
    const long blk = ((long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (blk >= blocks) return;
    const int lane = threadIdx.x & 31;
    const int groups = k >> 5;
    const long row = blk / groups;
    const int g = (int)(blk - row * groups);
    const long idx = row * k + (long)g * 32 + lane;
    const float v = ld(x, idx);
    float amax = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float inv;
    const unsigned byte = mx_e8m0(amax, &inv);
    const unsigned code = mx_e4m3(v * inv);
    if (mode == 0) {
        ((unsigned char*)codes_)[idx] = (unsigned char)code;
        if (lane == 0) ((unsigned char*)scales_)[mx_scale_offset(row, g, (groups + 3) >> 2)] = (unsigned char)byte;
    } else {
        ((bf16_t*)out_)[idx] = bf(mx_e4m3_decode(code) * __int_as_float((int)byte << 23));
    }
}
// Eight consecutive elements per thread (one 16-byte load), four threads per
// 32-column block: the same math as fw_mx_quant for 16-byte-aligned rows.
// No early return: the 4-lane reductions need every lane of the warp.
extern "C" __global__ void fw_mx_quant8(u64 x_, long total, int k, u64 codes_, u64 scales_, u64 out_, int mode) {
    const long i0 = ((long)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    const bool valid = i0 < total;
    uint4 raw = make_uint4(0u, 0u, 0u, 0u);
    if (valid) raw = *(const uint4*)((const bf16_t*)x_ + i0);
    float v[8];
    v[0] = __uint_as_float(raw.x << 16); v[1] = __uint_as_float(raw.x & 0xFFFF0000u);
    v[2] = __uint_as_float(raw.y << 16); v[3] = __uint_as_float(raw.y & 0xFFFF0000u);
    v[4] = __uint_as_float(raw.z << 16); v[5] = __uint_as_float(raw.z & 0xFFFF0000u);
    v[6] = __uint_as_float(raw.w << 16); v[7] = __uint_as_float(raw.w & 0xFFFF0000u);
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) amax = fmaxf(amax, fabsf(v[j]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    if (!valid) return;
    float inv;
    const unsigned byte = mx_e8m0(amax, &inv);
    if (mode == 0) {
        unsigned c[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) c[j] = mx_e4m3(v[j] * inv);
        uint2 packed = make_uint2(c[0] | (c[1] << 8) | (c[2] << 16) | (c[3] << 24), c[4] | (c[5] << 8) | (c[6] << 16) | (c[7] << 24));
        *(uint2*)((unsigned char*)codes_ + i0) = packed;
        if ((threadIdx.x & 3) == 0) {
            const long row = i0 / k; const int g = (int)((i0 - row * k) >> 5);
            ((unsigned char*)scales_)[mx_scale_offset(row, g, ((k >> 5) + 3) >> 2)] = (unsigned char)byte;
        }
    } else {
        const float scale = __int_as_float((int)byte << 23);
        unsigned o[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) o[j] = bf(mx_e4m3_decode(mx_e4m3(v[j] * inv)) * scale);
        *(uint4*)((bf16_t*)out_ + i0) = make_uint4(o[0] | (o[1] << 16), o[2] | (o[3] << 16), o[4] | (o[5] << 16), o[6] | (o[7] << 16));
    }
}
// Expand MXFP8 weight codes + swizzled scales to bf16 (the emulation path on pre-Blackwell cards).
extern "C" __global__ void fw_dequant_mxfp8(u64 out_, u64 codes_, u64 scales_, long n, int k) {
    bf16_t* out = (bf16_t*)out_;
    const unsigned char* q = (const unsigned char*)codes_;
    const unsigned char* sc = (const unsigned char*)scales_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long row = i / k; const int c = (int)(i - row * k);
    const int groups = k >> 5;
    const unsigned e = sc[mx_scale_offset(row, c >> 5, (groups + 3) >> 2)];
    out[i] = bf(mx_e4m3_decode(q[i]) * __int_as_float((int)e << 23));
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
extern "C" __global__ void fw_dequant_nvfp4(u64 out_, u64 q_, u64 s_, long n, int cols, float alpha, int first_high) {
    bf16_t* out = (bf16_t*)out_;
    const unsigned char* q = (const unsigned char*)q_;
    const unsigned char* s = (const unsigned char*)s_;
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long r = i / cols; const int c = (int)(i - r * cols);
    const int shift = ((int)(i & 1) ^ first_high) * 4;   // ComfyUI packs the first code in the high nibble
    const unsigned code = (q[i >> 1] >> shift) & 15u;
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
