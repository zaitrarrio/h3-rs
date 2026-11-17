//! Parity of the Qwen3-VL conditioner with transformers, on the fixture from
//! `scripts/reference/make_fixtures.py` (skipped when absent).

use h3_core::{
    config::Qwen3VlConfig,
    weights::{SafetensorsDir, WeightSource},
};
use h3_cpu::Cpu;
use h3_text::{
    present::{Presentation, VisionBlock},
    qwen3vl::Qwen3Vl,
    vision::VisionInput,
};
use std::path::PathBuf;

#[test]
fn hf_parity_qwen3vl() {
    let p =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/qwen3vl_tiny.safetensors");
    if !p.is_file() {
        eprintln!(
            "skipped: {} is absent (run scripts/reference/make_fixtures.py)",
            p.display()
        );
        return;
    }
    let src = SafetensorsDir::open(&p).unwrap();
    let cfg = Qwen3VlConfig::tiny();
    let be = Cpu;
    let ids: Vec<u32> = src
        .get("input.ids", &[11])
        .unwrap()
        .iter()
        .map(|v| *v as u32)
        .collect();
    let pres = Presentation {
        ids,
        tags: vec![1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        blocks: vec![VisionBlock {
            start: 3,
            tokens: 4,
            video: false,
            grid: (1, 4, 4),
        }],
    };
    let vi = VisionInput {
        patches: src.get("input.pixels", &[16, 1536]).unwrap(),
        grid: (1, 4, 4),
    };
    // transformers' hidden_states[k]: the embeddings with the merged image
    // tokens in place (k = 0), then each block's output as the block returned
    // it, which is *before* the DeepStack injection that follows blocks 0 and
    // 1 in this tiny config (transformers records the block's return value).
    // FastVideo's own H3 text stack (`minimax_h3_qwen3_vl.py`) injects and then
    // returns at `output_hidden_state_index`, as the conditioner here does, so
    // the expected rows get `deepstack{k-1}` added at the image tokens. The
    // real conditioner reads layer 50, past the last DeepStack index, where
    // the two readings are the same anyway.
    let mut errs = Vec::new();
    for layer in 0..=cfg.output_layer {
        let c = Qwen3VlConfig {
            output_layer: layer,
            ..cfg.clone()
        };
        let m = Qwen3Vl::open(&be, c, SafetensorsDir::open(&p).unwrap(), true, true).unwrap();
        let enc = m.encode(&be, &pres, &[&vi]).unwrap();
        let mut want = src
            .get(&format!("expected.hidden{layer}"), &[11, cfg.hidden])
            .unwrap();
        if let Some(di) = layer
            .checked_sub(1)
            .and_then(|prev| cfg.vision_deepstack.iter().position(|&d| d == prev))
        {
            let deep = src
                .get(&format!("expected.deepstack{di}"), &[4, cfg.hidden])
                .unwrap();
            for (row, d) in deep.chunks(cfg.hidden).enumerate() {
                for (w, x) in want[(3 + row) * cfg.hidden..(4 + row) * cfg.hidden]
                    .iter_mut()
                    .zip(d)
                {
                    *w += x;
                }
            }
        }
        let scale = want.iter().fold(0f32, |m, v| m.max(v.abs())) + 1e-6;
        errs.push(
            enc.embeds
                .iter()
                .zip(&want)
                .fold(0f32, |m, (a, b)| m.max((a - b).abs()))
                / scale,
        );
    }
    assert!(
        errs.iter().all(|e| *e < 2e-3),
        "relative error per layer (embeddings first): {:?}",
        errs.iter().map(|e| format!("{e:.2e}")).collect::<Vec<_>>()
    );
}
