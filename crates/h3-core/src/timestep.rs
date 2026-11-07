//! Timestep conditioning, computed on the host.
//!
//! H3 feeds clean time `t = 1 − σ ∈ [0, 1]` to Diffusers' `Timesteps(256,
//! flip_sin_to_cos=True, downscale_freq_shift=0)` and a two-layer SiLU MLP;
//! the result `temb ∈ R^2688` drives every AdaLN projection. The sinusoid and
//! the MLP are a few mat-vecs per unique timestep, so they run here in f32 and
//! only the finished tables reach the device (see `h3-model`).

/// `[cos(t·f_i), sin(t·f_i)]`, `f_i = exp(−ln(10⁴)·i / (dim/2))`.
pub fn sinusoid(t: f32, dim: usize) -> Vec<f32> {
    let half = dim / 2;
    let mut out = vec![0.0; dim];
    for i in 0..half {
        let f = (-(10000f32.ln()) * i as f32 / half as f32).exp();
        out[i] = (t * f).cos();
        out[half + i] = (t * f).sin();
    }
    out
}

/// `y = W x + b` for a row-major `[out, in]` matrix, f64 accumulation.
pub fn matvec(w: &[f32], b: Option<&[f32]>, x: &[f32]) -> Vec<f32> {
    let n_in = x.len();
    let n_out = w.len() / n_in;
    (0..n_out)
        .map(|o| {
            let row = &w[o * n_in..(o + 1) * n_in];
            let acc: f64 = row.iter().zip(x).map(|(&a, &b)| a as f64 * b as f64).sum();
            acc as f32 + b.map_or(0.0, |b| b[o])
        })
        .collect()
}

pub fn silu(x: &[f32]) -> Vec<f32> {
    x.iter().map(|&v| v / (1.0 + (-v).exp())).collect()
}

/// Host copy of `time_embedder` (`linear_1 [hidden, freq]`, `linear_2
/// [embed, hidden]`) and the optional rank-reduction basis `adaln_basis
/// [rank, embed]` of a Trim checkpoint.
pub struct TimeEmbedder {
    pub freq_dim: usize,
    pub l1w: Vec<f32>,
    pub l1b: Vec<f32>,
    pub l2w: Vec<f32>,
    pub l2b: Vec<f32>,
    pub basis: Option<Vec<f32>>,
    /// ComfyUI's single-file Trim checkpoints replace the embedder and the
    /// basis by `adaln_t_table [grid, rank]`: the AdaLN input is the table
    /// linearly interpolated at `t · (grid − 1)` (no SiLU).
    pub table: Option<AdalnTable>,
}

/// `adaln_t_table [grid, width]`, row-major.
pub struct AdalnTable {
    pub grid: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

impl AdalnTable {
    /// Linear interpolation at `t ∈ [0, 1]` (clamped), as ComfyUI's
    /// `torch.lerp(table[i0], table[i0 + 1], pos − i0)` with `pos = t·(grid−1)`,
    /// `i0 = min(floor(pos), grid − 2)`.
    pub fn at(&self, t: f32) -> Vec<f32> {
        let pos = t.clamp(0.0, 1.0) * (self.grid - 1) as f32;
        let i0 = (pos.floor() as usize).min(self.grid - 2);
        let f = pos - i0 as f32;
        let (a, b) = (&self.data[i0 * self.width..(i0 + 1) * self.width], &self.data[(i0 + 1) * self.width..(i0 + 2) * self.width]);
        a.iter().zip(b).map(|(&x, &y)| x + (y - x) * f).collect()
    }
}

impl TimeEmbedder {
    /// An embedder that only holds a table (no MLP weights).
    pub fn from_table(table: AdalnTable) -> Self {
        Self { freq_dim: 0, l1w: Vec::new(), l1b: Vec::new(), l2w: Vec::new(), l2b: Vec::new(), basis: None, table: Some(table) }
    }

    /// `temb(t)`: the MLP output (`[time_embed_dim]`).
    pub fn temb(&self, t: f32) -> Vec<f32> {
        let e = sinusoid(t, self.freq_dim);
        let h = silu(&matvec(&self.l1w, Some(&self.l1b), &e));
        matvec(&self.l2w, Some(&self.l2b), &h)
    }

    /// What the AdaLN projections consume: `silu(temb)` for the full-rank
    /// checkpoints, `basis · silu(temb)` (no further SiLU) for a rank-reduced
    /// one, the interpolated table row for a table checkpoint.
    pub fn adaln_input(&self, t: f32) -> Vec<f32> {
        if let Some(tab) = &self.table {
            return tab.at(t);
        }
        let s = silu(&self.temb(t));
        match &self.basis {
            Some(b) => matvec(b, None, &s),
            None => s,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_interpolates_linearly_and_clamps() {
        let tab = AdalnTable { grid: 3, width: 2, data: vec![0.0, 10.0, 1.0, 20.0, 4.0, 40.0] };
        assert_eq!(tab.at(0.0), vec![0.0, 10.0]);
        assert_eq!(tab.at(0.25), vec![0.5, 15.0]);
        assert_eq!(tab.at(0.5), vec![1.0, 20.0]);
        assert_eq!(tab.at(0.75), vec![2.5, 30.0]);
        assert_eq!(tab.at(1.0), vec![4.0, 40.0]);
        assert_eq!(tab.at(7.0), vec![4.0, 40.0]);
        let te = TimeEmbedder::from_table(tab);
        assert_eq!(te.adaln_input(0.25), vec![0.5, 15.0]);
    }

    #[test]
    fn sinusoid_is_cos_then_sin() {
        let e = sinusoid(0.0, 8);
        assert_eq!(&e[..4], &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(&e[4..], &[0.0, 0.0, 0.0, 0.0]);
        let e = sinusoid(0.5, 8);
        assert!((e[0] - 0.5f32.cos()).abs() < 1e-7 && (e[4] - 0.5f32.sin()).abs() < 1e-7);
    }
}
