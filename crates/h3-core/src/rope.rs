//! Rotary tables. Three users, one convention: *rotate-half* over the first
//! `rot` channels of a head, `y[j] = x[j]·cos[j] − x[j + rot/2]·sin[j]`,
//! `y[j + rot/2] = x[j + rot/2]·cos[j] + x[j]·sin[j]`, channels past `rot`
//! untouched. Every table stores `rot / 2` angles per row.
//!
//! * The DiT (`MiniMaxH3RotaryPosEmbed`): 16 frequencies per `(t, h, w)` axis,
//!   `θ = 10⁴`, 48 angles rotating 96 of 128 channels, over the packed
//!   sequence's real-valued positions (f32, as the reference).
//! * The video VAE's ViT decoder: `2π·pos·θ^(−k/8)`, `θ = 100`, 8 per axis, 24
//!   angles rotating 48 of 64, over length-normalised `[−1, 1)` grids.
//! * Qwen3-VL: 64 angles over 128 channels, interleaved multimodal sections.

/// `cos` / `sin` of shape `[rows, half]`.
pub struct RopeTable {
    pub cos: Vec<f32>,
    pub sin: Vec<f32>,
    pub half: usize,
}

/// H3 DiT table for packed positions `[rows][3]`.
pub fn h3_rope(positions: &[[f64; 3]], rope_freq_dim: usize, theta: f64) -> RopeTable {
    let inv: Vec<f32> = (0..rope_freq_dim)
        .map(|i| (1.0 / theta.powf(2.0 * i as f64 / (2 * rope_freq_dim) as f64)) as f32)
        .collect();
    let half = 3 * rope_freq_dim;
    let mut cos = Vec::with_capacity(positions.len() * half);
    let mut sin = Vec::with_capacity(positions.len() * half);
    for p in positions {
        for axis in 0..3 {
            let pos = p[axis] as f32;
            for &f in &inv {
                let a = pos * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
    }
    RopeTable { cos, sin, half }
}

/// ViT decoder table: `rot` rotated channels (48), three axes, `(t, h, w)`
/// token order, then `suffix` rows (register + cls tokens) at position 0.
pub fn vae_decoder_rope(
    grid: (usize, usize, usize),
    rot: usize,
    theta: f64,
    suffix: usize,
) -> RopeTable {
    let per_axis = rot / 6;
    let inv: Vec<f64> = (0..per_axis)
        .map(|k| 1.0 / theta.powf(k as f64 * 6.0 / rot as f64))
        .collect();
    let half = rot / 2;
    let coords = |n: usize| -> Vec<f64> {
        (0..n)
            .map(|i| 2.0 * ((i as f64 + 0.5) / n as f64) - 1.0)
            .collect()
    };
    let (ct, ch, cw) = (coords(grid.0), coords(grid.1), coords(grid.2));
    let n = grid.0 * grid.1 * grid.2 + suffix;
    let mut cos = Vec::with_capacity(n * half);
    let mut sin = Vec::with_capacity(n * half);
    let mut push = |p: [f64; 3]| {
        for axis in 0..3 {
            for &f in &inv {
                let a = (std::f64::consts::TAU * p[axis] * f) as f32;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
    };
    for &t in &ct {
        for &h in &ch {
            for &w in &cw {
                push([t, h, w]);
            }
        }
    }
    for _ in 0..suffix {
        push([0.0; 3]);
    }
    RopeTable { cos, sin, half }
}

/// The fast (Sol-H3) VAE decoder's window-local table for latents
/// `t ∈ [t0, t1)` of an `(h, w)` token grid, `(t, y, x)` order: each axis id
/// is the token's index inside its window after the layer's shift,
/// normalised to `(i + 0.5) / size · 2 − 1` (time windows of `block`
/// latents shifted by `st`, spatial windows of `window` tokens shifted by
/// `s`), on the base decoder's frequencies (Sana `NonCausalViTDecoder._rope`).
#[allow(clippy::too_many_arguments)]
pub fn vae_fast_rope(
    t: (usize, usize),
    hw: (usize, usize),
    rot: usize,
    theta: f64,
    block: usize,
    window: usize,
    s: usize,
    st: usize,
) -> RopeTable {
    let per_axis = rot / 6;
    let inv: Vec<f64> = (0..per_axis)
        .map(|k| 1.0 / theta.powf(k as f64 * 6.0 / rot as f64))
        .collect();
    let half = rot / 2;
    let id = |i: usize, shift: usize, size: usize| ((((i + shift) % size) as f64) + 0.5) / size as f64 * 2.0 - 1.0;
    let n = (t.1 - t.0) * hw.0 * hw.1;
    let mut cos = Vec::with_capacity(n * half);
    let mut sin = Vec::with_capacity(n * half);
    for ti in t.0..t.1 {
        for y in 0..hw.0 {
            for x in 0..hw.1 {
                let p = [id(ti, st, block), id(y, s, window), id(x, s, window)];
                for axis in p {
                    for &f in &inv {
                        let a = (std::f64::consts::TAU * axis * f) as f32;
                        cos.push(a.cos());
                        sin.push(a.sin());
                    }
                }
            }
        }
    }
    RopeTable { cos, sin, half }
}

/// Qwen3-VL text table. `positions[i] = [t, h, w]` (all equal for text
/// tokens). Frequency `j < 64`: for `j < 3·section[1]`, axis `j % 3`
/// (0 = t, 1 = h, 2 = w); otherwise t.
pub fn qwen3_mrope(
    positions: &[[u32; 3]],
    head_dim: usize,
    theta: f64,
    section: [usize; 3],
) -> RopeTable {
    let half = head_dim / 2;
    let inv: Vec<f32> = (0..half)
        .map(|i| (1.0 / theta.powf(2.0 * i as f64 / head_dim as f64)) as f32)
        .collect();
    let stop_h = section[1] * 3;
    let stop_w = section[2] * 3;
    let mut cos = Vec::with_capacity(positions.len() * half);
    let mut sin = Vec::with_capacity(positions.len() * half);
    for p in positions {
        for (j, &f) in inv.iter().enumerate() {
            let axis = if j % 3 == 1 && j < stop_h {
                1
            } else if j % 3 == 2 && j < stop_w {
                2
            } else {
                0
            };
            let a = p[axis] as f32 * f;
            cos.push(a.cos());
            sin.push(a.sin());
        }
    }
    RopeTable { cos, sin, half }
}

/// Qwen3-VL vision table: `(row, col)` patch coordinates, `rot = head_dim`,
/// angles `[row·f_k (k < rot/4), col·f_k]`, `f_k = 10⁴^(−2k/(rot/2))`.
pub fn qwen3_vision_rope(coords: &[[u32; 2]], head_dim: usize) -> RopeTable {
    let quarter = head_dim / 4;
    let inv: Vec<f32> = (0..quarter)
        .map(|k| (1.0 / 10000f64.powf(2.0 * k as f64 / (head_dim / 2) as f64)) as f32)
        .collect();
    let half = head_dim / 2;
    let mut cos = Vec::with_capacity(coords.len() * half);
    let mut sin = Vec::with_capacity(coords.len() * half);
    for c in coords {
        for axis in 0..2 {
            for &f in &inv {
                let a = c[axis] as f32 * f;
                cos.push(a.cos());
                sin.push(a.sin());
            }
        }
    }
    RopeTable { cos, sin, half }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h3_rope_shapes() {
        let t = h3_rope(&[[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], 16, 10000.0);
        assert_eq!(t.half, 48);
        assert_eq!(t.cos.len(), 96);
        assert!(t.cos[..48].iter().all(|&c| c == 1.0));
        // Row 1: temporal angle at frequency 0 is 1 rad; height axis starts at index 16.
        assert!((t.cos[48] - 1f32.cos()).abs() < 1e-6);
        assert!((t.cos[48 + 16] - 2f32.cos()).abs() < 1e-6);
        assert!((t.cos[48 + 32] - 3f32.cos()).abs() < 1e-6);
    }

    #[test]
    fn vae_rope_grid_is_length_normalised() {
        let t = vae_decoder_rope((1, 2, 2), 48, 100.0, 5);
        assert_eq!(t.half, 24);
        assert_eq!(t.cos.len(), 9 * 24);
        // t coordinate for a single frame is 0 → cos 1 for the first 8 angles.
        assert!(t.cos[..8].iter().all(|&c| (c - 1.0).abs() < 1e-6));
        // h = −0.5 for the first row: angle 2π·(−0.5)·1 = −π → cos −1.
        assert!((t.cos[8] + 1.0).abs() < 1e-5);
        // suffix rows at the origin
        assert!(t.cos[4 * 24..].iter().all(|&c| (c - 1.0).abs() < 1e-6));
    }

    #[test]
    fn mrope_sections() {
        let t = qwen3_mrope(&[[1, 2, 3]], 128, 5e6, [24, 20, 20]);
        assert_eq!(t.half, 64);
        // j = 0 → t, j = 1 → h, j = 2 → w, j = 63 → t (past the h/w sections).
        assert!((t.cos[0] - 1f32.cos()).abs() < 1e-6);
        let f1 = (1.0 / 5e6f64.powf(2.0 / 128.0)) as f32;
        assert!((t.cos[1] - (2.0 * f1).cos()).abs() < 1e-6);
        let f2 = (1.0 / 5e6f64.powf(4.0 / 128.0)) as f32;
        assert!((t.cos[2] - (3.0 * f2).cos()).abs() < 1e-6);
        let f63 = (1.0 / 5e6f64.powf(126.0 / 128.0)) as f32;
        assert!((t.cos[63] - (1.0 * f63).cos()).abs() < 1e-6);
    }
}
