//! Deterministic standard-normal noise: the same seed gives the same latents on
//! every backend and every platform, so backend parity can be checked end to
//! end. SplitMix64 → Box–Muller. (Not torch's generator: a seed here does not
//! reproduce a FastVideo seed.)

pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `(0, 1]`.
    fn unit(&mut self) -> f64 {
        ((self.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64
    }

    pub fn normal(&mut self, n: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(n + 1);
        while out.len() < n {
            let (u1, u2) = (self.unit(), self.unit());
            let r = (-2.0 * u1.ln()).sqrt();
            let th = std::f64::consts::TAU * u2;
            out.push((r * th.cos()) as f32);
            out.push((r * th.sin()) as f32);
        }
        out.truncate(n);
        out
    }

    /// Uniform in `[-a, a)`; for synthetic weights.
    pub fn uniform(&mut self, n: usize, a: f32) -> Vec<f32> {
        (0..n)
            .map(|_| ((self.unit() * 2.0 - 1.0) as f32) * a)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn normal_moments() {
        let v = super::Rng::new(7).normal(200_000);
        let m = v.iter().map(|&x| x as f64).sum::<f64>() / v.len() as f64;
        let var = v.iter().map(|&x| (x as f64 - m).powi(2)).sum::<f64>() / v.len() as f64;
        assert!(m.abs() < 0.01 && (var - 1.0).abs() < 0.02, "{m} {var}");
    }
}
