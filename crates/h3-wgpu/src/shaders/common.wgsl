// Shared by every kernel. `{W_TYPE}` / `{W_LOAD}` are filled per weight
// storage: f16 pairs packed in u32 (core WGSL `unpack2x16float`, so no
// `shader-f16` extension is needed), int8 / 4-bit codes, or plain f32.
fn tanh_safe(u: f32) -> f32 {
    // tanh via exp, clamped: WGSL tanh overflows to NaN on some drivers.
    let e = exp(-2.0 * abs(u));
    return sign(u) * (1.0 - e) / (1.0 + e);
}
fn gelu_tanh(x: f32) -> f32 {
    let c = 0.7978845608;
    return 0.5 * x * (1.0 + tanh_safe(c * (x + 0.044715 * x * x * x)));
}
// erf, Abramowitz–Stegun 7.1.26 (|error| < 1.5e-7).
fn erf_approx(x: f32) -> f32 {
    let s = sign(x);
    let a = abs(x);
    let t = 1.0 / (1.0 + 0.3275911 * a);
    let y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-a * a);
    return s * y;
}
fn gelu_erf(x: f32) -> f32 {
    return 0.5 * x * (1.0 + erf_approx(x * 0.70710678118));
}
fn silu(x: f32) -> f32 {
    return x / (1.0 + exp(-x));
}
// Epilogue activation codes: 0 none, 1 gelu_tanh, 2 gelu_erf, 3 silu.
fn act(v: f32, a: u32) -> f32 {
    if (a == 1u) { return gelu_tanh(v); }
    if (a == 2u) { return gelu_erf(v); }
    if (a == 3u) { return silu(v); }
    return v;
}
// Linear workgroup index of a (possibly 2-D folded) dispatch.
fn wg_index(wid: vec3<u32>, nwg: vec3<u32>) -> u32 {
    return wid.x + wid.y * nwg.x;
}
