//! Sampling schedules (FastVideo `scheduling_minimax_h3.py`, `layers/pdd.py`,
//! `open_h3_regenerate_2k/sampling.py`).
//!
//! H3 is a rectified flow on a *clean-time* clock: the transformer receives
//! `t = 1 − σ` and predicts a velocity `v` with `x0 = x + σ·v`. Video and
//! audio share one base noise clock and each applies its own rational shift
//! `σ = s·b / (1 + (s − 1)·b)` once (12 / 3 for the release, 10 / 3 for
//! FastH3 V2 and Trim). One step is then
//!
//! ```text
//! Euler    x ← (σ'/σ)·x + (1 − σ'/σ)·x0                 (base H3, DMD students, PDD blocks)
//! Renoise  x ← (1 − σ')·x0 + σ'·ε, x0 when σ' = 0       (Pika H3 Flex)
//! ```
//!
//! Every sigma is computed in f32 as the reference does.

use crate::{Error, Result};

/// `s·b / (1 + (s − 1)·b)`.
pub fn shift_sigma(base: f32, shift: f32) -> f32 {
    shift * base / (1.0 + (shift - 1.0) * base)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StepRule {
    Euler,
    Renoise,
}

/// The two modality shifts a checkpoint's `scheduler/` and `audio_scheduler/`
/// record.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Shifts {
    pub video: f32,
    pub audio: f32,
}

impl Shifts {
    /// MiniMax-H3, FastH3 V1, FastH3 Omni, Pika H3 Flex.
    pub const H3: Shifts = Shifts { video: 12.0, audio: 3.0 };
    /// FastH3 V2 and Trim (`vshift10`).
    pub const FASTH3_V2: Shifts = Shifts { video: 10.0, audio: 3.0 };
}

/// One modality's sigma grid: `steps + 1` strictly decreasing values ending at 0.
#[derive(Debug, Clone, PartialEq)]
pub struct Schedule {
    pub sigmas: Vec<f32>,
    pub rule: StepRule,
}

/// What one denoising step needs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepCoeffs {
    /// Noise level of the current sample; the transformer sees `1 − sigma`.
    pub sigma: f32,
    pub next: f32,
    pub rule: StepRule,
}

impl Schedule {
    /// Explicit sigmas (already shifted), validated.
    pub fn from_sigmas(sigmas: Vec<f32>, rule: StepRule) -> Result<Self> {
        if sigmas.len() < 2 || sigmas.windows(2).any(|w| w[1] >= w[0]) || *sigmas.last().unwrap() != 0.0 {
            return Err(Error::Config(format!(
                "sigmas must hold ≥ 2 strictly decreasing values ending at 0, got {sigmas:?}"
            )));
        }
        Ok(Self { sigmas, rule })
    }

    /// A distilled ladder: integer rungs on the 1000-step clock, strictly
    /// decreasing in `(0, 1000]`, shifted once by `shift`, then 0.
    pub fn from_rungs(rungs: &[u32], shift: f32, rule: StepRule) -> Result<Self> {
        if rungs.is_empty() || rungs.iter().any(|&r| r == 0 || r > 1000) || rungs.windows(2).any(|w| w[1] >= w[0]) {
            return Err(Error::Config(format!("rungs must be strictly decreasing in (0, 1000], got {rungs:?}")));
        }
        let mut sigmas: Vec<f32> = rungs.iter().map(|&r| shift_sigma(r as f32 / 1000.0, shift)).collect();
        sigmas.push(0.0);
        Self::from_sigmas(sigmas, rule)
    }

    /// Base H3: `points` grid points evenly spaced on the base clock from 1 to 0
    /// (`points − 1` forwards; the release runs 50 points).
    pub fn uniform(points: usize, shift: f32) -> Result<Self> {
        if points < 2 {
            return Err(Error::Config("a uniform schedule needs ≥ 2 grid points".into()));
        }
        let mut sigmas: Vec<f32> = linspace_f32(1.0, 0.0, points).into_iter().map(|b| shift_sigma(b, shift)).collect();
        sigmas.dedup();
        Self::from_sigmas(sigmas, StepRule::Euler)
    }

    pub fn steps(&self) -> usize {
        self.sigmas.len() - 1
    }

    /// Clean time the transformer receives at step `i`.
    pub fn timestep(&self, i: usize) -> f32 {
        1.0 - self.sigmas[i]
    }

    pub fn step(&self, i: usize) -> StepCoeffs {
        StepCoeffs {
            sigma: self.sigmas[i],
            next: self.sigmas[i + 1],
            rule: self.rule,
        }
    }
}

/// `torch.linspace` in f32: symmetric fill from both ends.
pub fn linspace_f32(start: f32, end: f32, n: usize) -> Vec<f32> {
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f32;
    (0..n)
        .map(|i| {
            if i < n / 2 {
                start + step * i as f32
            } else {
                end - step * (n - 1 - i) as f32
            }
        })
        .collect()
}

pub fn linspace_f64(start: f64, end: f64, n: usize) -> Vec<f64> {
    if n == 1 {
        return vec![start];
    }
    let step = (end - start) / (n - 1) as f64;
    (0..n)
        .map(|i| {
            if i < n / 2 {
                start + step * i as f64
            } else {
                end - step * (n - 1 - i) as f64
            }
        })
        .collect()
}

/// Distilled ladders, as each checkpoint's `fastvideo_inference.json` /
/// `manifest.json` records them.
pub const FASTH3_V1_RUNGS: [u32; 4] = [999, 749, 500, 250];
pub const FASTH3_V2_RUNGS: [u32; 8] = [999, 874, 749, 624, 500, 375, 250, 125];
pub const PIKA_KNOTS: [u32; 6] = [1000, 750, 500, 250, 100, 40];
/// Base H3's uniform grid: 50 points, 49 forwards.
pub const H3_BASE_POINTS: usize = 50;

/// Keyframe / reference rows are the clean latent at clean time 0.999:
/// `0.999·clean + 0.001·ε`, and their row timestep is `max(t_video, 0.999)`.
pub const KEYFRAME_NOISE_AUG: f32 = 0.999;

// ---- Parallel Decoding Distillation (FastH3 Omni) ---------------------------

/// FastGen's rectified-flow schedules end their finite time domain at 0.999.
pub const PDD_GRID_MAX_T: f64 = 0.999;

/// `f_s(u) = s·u·M / (u·(s − 1) + M)` on `[0, M]`, `M = max_t` a fixed point.
pub fn pdd_shift(u: f64, shift: f64, max_t: f64) -> f64 {
    if shift == 1.0 {
        return u;
    }
    shift * u * max_t / (u * (shift - 1.0) + max_t)
}

/// `f_s(b) − f_s(a)` without cancellation: `s·M²·(b − a) / (D(a)·D(b))`.
pub fn pdd_shift_delta(a: f64, b: f64, shift: f64, max_t: f64) -> f64 {
    if shift == 1.0 {
        return b - a;
    }
    let da = max_t + (shift - 1.0) * a;
    let db = max_t + (shift - 1.0) * b;
    shift * max_t * max_t * (b - a) / (da * db)
}

/// One runtime partition of a PDD student's fine grid.
#[derive(Debug, Clone, PartialEq)]
pub struct PddPlan {
    /// `u_0 = 0.999 > … > u_N = 0`, f64.
    pub fine_grid: Vec<f64>,
    /// Node indices of the fused blocks, `0 = i_0 < … < i_B = N`.
    pub indices: Vec<usize>,
    /// Per-modality sigmas at the nodes (the Euler grid, f32).
    pub video_sigmas: Vec<f32>,
    pub audio_sigmas: Vec<f32>,
    /// Per-modality `σ(u_{j+1}) − σ(u_j)` for every fine interval (negative).
    pub video_weights: Vec<f64>,
    pub audio_weights: Vec<f64>,
}

impl PddPlan {
    pub fn new(indices: &[usize], shifts: Shifts) -> Result<Self> {
        if indices.len() < 2 || indices[0] != 0 || indices.windows(2).any(|w| w[1] <= w[0]) {
            return Err(Error::Config(format!(
                "pdd_step_indices must increase strictly from 0, got {indices:?}"
            )));
        }
        let steps = *indices.last().unwrap();
        if steps < 2 {
            return Err(Error::Config("a PDD grid needs ≥ 2 intervals".into()));
        }
        let fine_grid: Vec<f64> = linspace_f64(PDD_GRID_MAX_T, 0.0, steps + 1)
            .into_iter()
            .map(|u| u.min(PDD_GRID_MAX_T))
            .collect();
        let weights = |shift: f64| -> Vec<f64> {
            fine_grid
                .windows(2)
                .map(|w| pdd_shift_delta(w[0], w[1], shift, PDD_GRID_MAX_T))
                .collect()
        };
        let nodes = |shift: f64| -> Vec<f32> {
            indices
                .iter()
                .map(|&i| pdd_shift(fine_grid[i], shift, PDD_GRID_MAX_T) as f32)
                .collect()
        };
        let (vw, aw) = (weights(shifts.video as f64), weights(shifts.audio as f64));
        if vw.iter().chain(&aw).any(|w| !w.is_finite() || *w == 0.0) {
            return Err(Error::Config("PDD integration weights must be finite and non-zero".into()));
        }
        Ok(Self {
            video_sigmas: nodes(shifts.video as f64),
            audio_sigmas: nodes(shifts.audio as f64),
            fine_grid,
            indices: indices.to_vec(),
            video_weights: vw,
            audio_weights: aw,
        })
    }

    pub fn pdd_steps(&self) -> usize {
        self.fine_grid.len() - 1
    }
    pub fn num_steps(&self) -> usize {
        self.indices.len() - 1
    }
    /// Half-open fine-grid block of runtime step `i`.
    pub fn block(&self, i: usize) -> (usize, usize) {
        (self.indices[i], self.indices[i + 1])
    }
    /// Normalised fusion weights of block `[start, end)`: the fused head is
    /// `Σ_j w_j/Σw · head_j` and the ordinary Euler step over the block's two
    /// node sigmas applies the block's total increment.
    pub fn fused_weights(weights: &[f64], start: usize, end: usize) -> Vec<f32> {
        let total: f64 = weights[start..end].iter().sum();
        weights[start..end].iter().map(|w| (w / total) as f32).collect()
    }
    /// The per-modality Euler schedules over the node sigmas.
    pub fn schedules(&self) -> (Schedule, Schedule) {
        (
            Schedule { sigmas: self.video_sigmas.clone(), rule: StepRule::Euler },
            Schedule { sigmas: self.audio_sigmas.clone(), rule: StepRule::Euler },
        )
    }
}

// ---- Open-H3-Regenerate-2K -------------------------------------------------

pub const REGEN_REFINE_SHIFT: f64 = 6.0;
/// The released three-step refine grid.
pub const REGEN_SIGMAS_3: [f64; 4] = [0.909375, 6.0 / 7.0, 2.0 / 3.0, 0.0];
pub const REGEN_TWO_STEP_START: f64 = 0.7;

/// Euler grid from `start` to 0: even spacing in unshifted sigma, then the
/// shift `s·x / (1 + (s − 1)·x)`.
pub fn regen_shifted_grid(start: f64, steps: usize, shift: f64) -> Result<Vec<f64>> {
    if !(0.0 < start && start < 1.0) || steps < 1 {
        return Err(Error::Config(format!("refine start {start} must be in (0, 1) with steps ≥ 1")));
    }
    let raw = start / (shift - (shift - 1.0) * start);
    let mut g = vec![start];
    for i in 1..steps {
        let x = raw * (1.0 - i as f64 / steps as f64);
        g.push(shift * x / (1.0 + (shift - 1.0) * x));
    }
    g.push(0.0);
    Ok(g)
}

/// Sigma grid of `steps` refine steps (3 or 2 as released, or any count from an
/// explicit start sigma).
pub fn regen_sigmas(steps: usize, start: Option<f64>) -> Result<Vec<f64>> {
    match (steps, start) {
        (_, Some(s)) => regen_shifted_grid(s, steps, REGEN_REFINE_SHIFT),
        (3, None) => Ok(REGEN_SIGMAS_3.to_vec()),
        (2, None) => regen_shifted_grid(REGEN_TWO_STEP_START, 2, REGEN_REFINE_SHIFT),
        _ => Err(Error::Config(format!("Open-H3-Regenerate-2K supports 2 or 3 refine steps, got {steps}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn v2_ladder() {
        let s = Schedule::from_rungs(&FASTH3_V2_RUNGS, 10.0, StepRule::Euler).unwrap();
        assert_eq!(s.steps(), 8);
        assert!((s.sigmas[0] - shift_sigma(0.999, 10.0)).abs() < 1e-7);
        assert!((s.sigmas[4] - 10.0 * 0.5 / 5.5).abs() < 1e-6);
        assert_eq!(*s.sigmas.last().unwrap(), 0.0);
        assert!((s.timestep(4) - (1.0 - 10.0 * 0.5 / 5.5)).abs() < 1e-6);
        let a = Schedule::from_rungs(&FASTH3_V2_RUNGS, 3.0, StepRule::Euler).unwrap();
        assert!(a.sigmas[4] > 0.74 && a.sigmas[4] < 0.76);
    }

    #[test]
    fn uniform_has_points_minus_one_steps() {
        let s = Schedule::uniform(50, 12.0).unwrap();
        assert_eq!(s.steps(), 49);
        assert_eq!(s.sigmas[0], 1.0);
        assert!(Schedule::uniform(1, 12.0).is_err());
    }

    #[test]
    fn pika_renoise_knots() {
        let s = Schedule::from_rungs(&PIKA_KNOTS, 12.0, StepRule::Renoise).unwrap();
        assert_eq!(s.steps(), 6);
        assert_eq!(s.sigmas[0], 1.0);
        assert!((s.sigmas[5] - shift_sigma(0.04, 12.0)).abs() < 1e-7);
    }

    #[test]
    fn pdd_omni_plan() {
        let p = PddPlan::new(&[0, 4, 8, 12, 16, 20, 24, 28, 32], Shifts::H3).unwrap();
        assert_eq!(p.pdd_steps(), 32);
        assert_eq!(p.num_steps(), 8);
        assert_eq!(p.block(0), (0, 4));
        // u = M is a fixed point of every shift.
        assert!((p.video_sigmas[0] - 0.999).abs() < 1e-6);
        assert_eq!(*p.video_sigmas.last().unwrap(), 0.0);
        assert!(p.video_weights.iter().all(|w| *w < 0.0));
        let f = PddPlan::fused_weights(&p.video_weights, 0, 4);
        assert!((f.iter().sum::<f32>() - 1.0).abs() < 1e-6);
        // Consistency: node sigma differences equal the summed interval weights.
        let sum: f64 = p.video_weights[4..8].iter().sum();
        assert!((sum - (p.video_sigmas[2] as f64 - p.video_sigmas[1] as f64)).abs() < 1e-6);
        assert!(PddPlan::new(&[0, 4, 3], Shifts::H3).is_err());
    }

    #[test]
    fn regen_grids() {
        assert_eq!(regen_sigmas(3, None).unwrap(), REGEN_SIGMAS_3.to_vec());
        let two = regen_sigmas(2, None).unwrap();
        assert_eq!(two.len(), 3);
        assert!((two[0] - 0.7).abs() < 1e-12 && (two[1] - 0.494118).abs() < 1e-5 && two[2] == 0.0);
        // The released 3-step grid is a fixed tuple, not the shifted grid; an
        // explicit start always uses the shifted grid.
        let g = regen_sigmas(3, Some(0.9)).unwrap();
        assert_eq!(g.len(), 4);
        assert!(g.windows(2).all(|w| w[0] > w[1]) && g[0] == 0.9 && g[3] == 0.0);
    }
}
