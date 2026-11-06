//! Parity of the Qwen3-VL conditioner with transformers, on the fixture from
//! `scripts/reference/make_fixtures.py` (skipped when absent).

use h3_core::{config::Qwen3VlConfig, weights::{SafetensorsDir, WeightSource}};
use h3_cpu::Cpu;
use h3_text::{present::{Presentation, VisionBlock}, qwen3vl::Qwen3Vl, vision::VisionInput};
use std::path::PathBuf;

#[test]
fn hf_parity_qwen3vl() {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3vl_tiny.safetensors");
    if !p.is_file() {
        eprintln!("skipped: {} is absent (run scripts/reference/make_fixtures.py)", p.display());
        return;
    }
    let src = SafetensorsDir::open(&p).unwrap();
    let cfg = Qwen3VlConfig::tiny();
    let be = Cpu;
    let m = Qwen3Vl::open(&be, cfg.clone(), SafetensorsDir::open(&p).unwrap(), true, true).unwrap();
    let ids: Vec<u32> = src.get("input.ids", &[11]).unwrap().iter().map(|v| *v as u32).collect();
    let pres = Presentation { ids, tags: vec![1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1], blocks: vec![VisionBlock { start: 3, tokens: 4, video: false, grid: (1, 4, 4) }] };
    let vi = VisionInput { patches: src.get("input.pixels", &[16, 1536]).unwrap(), grid: (1, 4, 4) };
    let enc = m.encode(&be, &pres, &[&vi]).unwrap();
    let want = src.get("expected.hidden2", &[11, cfg.hidden]).unwrap();
    let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())) + 1e-6;
    let err = enc.embeds.iter().zip(&want).fold(0f32, |m, (a, b)| m.max((a - b).abs())) / scale;
    assert!(err < 2e-3, "{err:.2e}");
}
