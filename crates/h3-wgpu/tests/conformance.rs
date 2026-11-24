//! Every op and the tiny models against the CPU reference
//! (`h3_model::check::run_all`). Runs on whatever adapter wgpu finds; in CI
//! that is lavapipe:
//!
//! ```text
//! VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json H3_ALLOW_SOFTWARE=1 cargo test -p h3-wgpu
//! ```
//!
//! Tolerance 2e-3 (relative to the output's max magnitude): weights are
//! stored as f16 pairs, which is ~5e-4 per weight and accumulates to ~1e-3
//! over a long reduction; the dequantized-int formats are exact.

fn open() -> Option<h3_wgpu::Wgpu> {
    let mut opts = h3_wgpu::WgpuOptions::from_env();
    if std::env::var("CI").is_ok() {
        opts.allow_software = true;
    }
    match h3_wgpu::Wgpu::with_options(opts) {
        Ok(b) => Some(b),
        Err(e) => {
            eprintln!("skipping: {e}");
            None
        }
    }
}

#[test]
fn wgpu_matches_cpu_reference() {
    let Some(be) = open() else { return };
    eprintln!("{}", h3_core::Backend::name(&be));
    let results = h3_model::check::run_all(&be, 2e-3);
    let mut bad = Vec::new();
    for r in &results {
        eprintln!(
            "{:<56} rel {:.2e} {}",
            r.name,
            r.rel_err,
            if r.ok { "ok" } else { "FAIL" }
        );
        if !r.ok {
            bad.push(r.name.clone());
        }
    }
    assert!(bad.is_empty(), "failed: {bad:?}");
}

/// The f16-math GEMM (where the adapter has `shader-f16`) at f16 tolerance.
#[test]
fn wgpu_f16_matches_cpu_reference() {
    let mut opts = h3_wgpu::WgpuOptions::from_env();
    opts.f16 = true;
    let be = match h3_wgpu::Wgpu::with_options(opts) {
        Ok(b) if b.f16() => b,
        Ok(_) => {
            eprintln!("skipping: adapter has no shader-f16");
            return;
        }
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let results = h3_model::check::run_ops(&be, 1e-2);
    let bad: Vec<_> = results
        .iter()
        .filter(|r| !r.ok)
        .map(|r| format!("{} {:.2e}", r.name, r.rel_err))
        .collect();
    assert!(bad.is_empty(), "failed: {bad:?}");
}

/// The row-chunked GEMM / head_split paths that a > 2 GiB activation takes
/// on a real card, forced here by a 64 KiB binding cap.
#[test]
fn wgpu_chunked_bindings_match_cpu_reference() {
    let mut opts = h3_wgpu::WgpuOptions::from_env();
    if std::env::var("CI").is_ok() {
        opts.allow_software = true;
    }
    opts.max_binding = Some(65536);
    let be = match h3_wgpu::Wgpu::with_options(opts) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let results = h3_model::check::run_all(&be, 2e-3);
    let bad: Vec<_> = results
        .iter()
        .filter(|r| !r.ok)
        .map(|r| r.name.clone())
        .collect();
    assert!(bad.is_empty(), "failed: {bad:?}");
}

/// The previous kernels (`linear3.wgsl` for every GEMM, `flash.wgsl`),
/// still selectable with `H3_WGPU_LINEAR=3` / `H3_WGPU_FLASH=1` for A/B runs.
#[test]
fn wgpu_previous_kernels_match_cpu_reference() {
    let mut opts = h3_wgpu::WgpuOptions::from_env();
    if std::env::var("CI").is_ok() {
        opts.allow_software = true;
    }
    opts.linear = 3;
    opts.flash = 1;
    let be = match h3_wgpu::Wgpu::with_options(opts) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("skipping: {e}");
            return;
        }
    };
    let results = h3_model::check::run_ops(&be, 2e-3);
    let bad: Vec<_> = results
        .iter()
        .filter(|r| !r.ok)
        .map(|r| format!("{} {:.2e}", r.name, r.rel_err))
        .collect();
    assert!(bad.is_empty(), "failed: {bad:?}");
}

/// linear4 sums every output over k in linear3's order with the same fma:
/// the two kernels agree bit for bit (f32 and int4 weights, a bias, GELU,
/// ragged edges in every dimension).
#[test]
fn wgpu_linear4_equals_linear3_exactly() {
    use h3_core::{Backend, backend::Act, quant::Q4, rng::Rng};
    let open = |linear: u32| {
        let mut opts = h3_wgpu::WgpuOptions::from_env();
        if std::env::var("CI").is_ok() {
            opts.allow_software = true;
        }
        opts.linear = linear;
        h3_wgpu::Wgpu::with_options(opts)
    };
    let (Ok(a), Ok(b)) = (open(3), open(4)) else {
        eprintln!("skipping: no adapter");
        return;
    };
    let (m, k, n) = (300, 200, 261);
    let mut r = Rng::new(11);
    let x = r.normal(m * k);
    let w = r.uniform(n * k, 0.3);
    let bias = r.normal(n);
    let q4 = Q4::quantize(&w, n, k, 64, false);
    let run = |be: &h3_wgpu::Wgpu, quant: bool| {
        let wd = if quant {
            be.weight_q4(&q4)
        } else {
            be.weight(&w, n, k)
        };
        let (xd, bd, yd) = (be.upload(&x), be.upload(&bias), be.alloc(m * n));
        be.linear(&xd, 0, m, &wd, Some(&bd), Act::GeluTanh, &yd, 0);
        be.download(&yd)
    };
    for quant in [false, true] {
        let (ya, yb) = (run(&a, quant), run(&b, quant));
        let diff = ya
            .iter()
            .zip(&yb)
            .filter(|(p, q)| p.to_bits() != q.to_bits())
            .count();
        assert_eq!(
            diff, 0,
            "linear4 differs from linear3 in {diff} outputs (int4: {quant})"
        );
    }
}
