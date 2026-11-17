//! Paths the shared conformance harness does not reach: the one-row-per-
//! thread attention kernel (head dims that are not multiples of 16, as the
//! Qwen3-VL vision tower's 72), same-buffer `copy`, and `fill`.

use h3_core::{Backend, rng::Rng};

fn open() -> Option<h3_wgpu::Wgpu> {
    let mut opts = h3_wgpu::WgpuOptions::from_env();
    if std::env::var("CI").is_ok() {
        opts.allow_software = true;
    }
    h3_wgpu::Wgpu::with_options(opts).ok()
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())) + 1e-6;
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
}

#[test]
fn fallback_attention_and_buffer_ops() {
    let Some(be) = open() else { return };
    let cpu = h3_cpu::Cpu;
    let mut rng = Rng::new(7);
    for &(heads, kv, sq, sk, d, causal) in &[
        (4usize, 2usize, 70usize, 70usize, 72usize, false),
        (3, 1, 33, 40, 24, true),
        (2, 2, 130, 130, 72, true),
    ] {
        let q = rng.normal(heads * sq * d);
        let k = rng.normal(kv * sk * d);
        let v = rng.normal(kv * sk * d);
        let run = |be: &dyn Run| be.attn(&q, &k, &v, heads, kv, sq, sk, d, causal);
        let got = run(&W(&be));
        let want = run(&W(&cpu));
        let e = rel(&got, &want);
        assert!(e < 2e-3, "attention d{d} causal={causal}: rel {e}");
    }
    // Same-buffer copy and fill.
    let x: Vec<f32> = (0..100).map(|i| i as f32).collect();
    let g = be.upload(&x);
    be.copy(&g, 10, &g, 50, 20);
    be.fill(&g, 0, 5, -1.0);
    let out = be.download(&g);
    assert_eq!(&out[50..70], &x[10..30]);
    assert!(out[..5].iter().all(|&v| v == -1.0));
    assert_eq!(out[5], 5.0);
}

trait Run {
    #[allow(clippy::too_many_arguments)]
    fn attn(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        heads: usize,
        kv: usize,
        sq: usize,
        sk: usize,
        d: usize,
        causal: bool,
    ) -> Vec<f32>;
}
struct W<'a, B: Backend>(&'a B);
impl<B: Backend> Run for W<'_, B> {
    fn attn(
        &self,
        q: &[f32],
        k: &[f32],
        v: &[f32],
        heads: usize,
        kv: usize,
        sq: usize,
        sk: usize,
        d: usize,
        causal: bool,
    ) -> Vec<f32> {
        let be = self.0;
        let o = be.alloc(sq * heads * d);
        be.attention(
            &be.upload(q),
            &be.upload(k),
            &be.upload(v),
            heads,
            kv,
            sq,
            sk,
            d,
            causal,
            &o,
        );
        be.download(&o)
    }
}
