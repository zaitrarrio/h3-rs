//! Parity with the PyTorch reference on fixtures made by
//! `scripts/reference/make_fixtures.py`. Each test skips (printing why) when
//! its fixture is absent, and fails on a mismatch when it is present, so a
//! checkout without the fixtures still builds and the fixtures are not
//! something a `cargo test` run can silently miss.

use h3_core::{
    config::{AudioVaeConfig, H3Config, VideoVaeConfig},
    packing::{self, Anchor},
    weights::{SafetensorsDir, WeightSource},
};
use h3_core::Backend;
use h3_cpu::Cpu;
use h3_model::{
    audio_vae::{AudioLatents, AudioVae},
    dit::{Attention, H3Dit, LoadOptions},
    vae::{VideoVae, Volume},
};
use std::path::PathBuf;

fn fixture(name: &str) -> Option<SafetensorsDir> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures").join(name);
    if !p.is_file() {
        eprintln!("skipped: {} is absent (run scripts/reference/make_fixtures.py)", p.display());
        return None;
    }
    Some(SafetensorsDir::open(&p).expect("fixture opens"))
}

fn rel(a: &[f32], b: &[f32]) -> f32 {
    let scale = b.iter().fold(0f32, |m, v| m.max(v.abs())) + 1e-6;
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / scale
}

#[test]
fn hf_parity_dit() {
    let Some(src) = fixture("h3_dit_tiny.safetensors") else { return };
    let be = Cpu;
    let cfg = H3Config::tiny();
    let dit = H3Dit::load(&be, cfg.clone(), &src, &LoadOptions::default(), &[]).unwrap();
    let tags: Vec<u8> = src.get("input.text_tags", &[6]).unwrap().iter().map(|v| *v as u8).collect();
    let layout = packing::build_layout(&tags, 2, 4, 6, 5, cfg.patch, &[Anchor::First]).unwrap();
    let text = dit.refine_text(&be, &src.get("input.text", &[6, cfg.text_dim]).unwrap());
    let ws = dit.workspace(&be, &layout, usize::MAX, Attention::Dense { qk8: false, pv8: false }).unwrap();
    let tables = dit.adaln_tables(&be, &src, &[], &[0.25, 0.7, 0.999, 1.0], None).unwrap();
    let (unique, inverse) = packing::row_timesteps(&layout, 0.25, 0.7, 0.999, 1.0);
    let mods = dit.step_mods(&be, &tables, &unique, &inverse, &layout.token_tags, false).unwrap();
    let nv = layout.video_indices.len();
    let na = layout.audio_indices.len();
    let video = be.upload(&src.get("input.video_rows", &[nv, cfg.video_patch_dim()]).unwrap());
    let audio = be.upload(&src.get("input.audio_rows", &[na, cfg.audio_in_channels]).unwrap());
    let (vo, ao) = (be.alloc(nv * cfg.video_patch_dim()), be.alloc(na * cfg.audio_in_channels));
    dit.forward(&be, &ws, &video, &audio, &text, &mods, None, &vo, &ao);
    let ev = src.get("expected.video", &[nv, cfg.video_patch_dim()]).unwrap();
    let ea = src.get("expected.audio", &[na, cfg.audio_in_channels]).unwrap();
    let (rv, ra) = (rel(&be.download(&vo), &ev), rel(&be.download(&ao), &ea));
    assert!(rv < 2e-3 && ra < 2e-3, "video {rv:.2e} audio {ra:.2e}");
}

#[test]
fn hf_parity_video_vae() {
    let Some(src) = fixture("h3_vae_tiny.safetensors") else { return };
    let be = Cpu;
    let vae = VideoVae::load(&be, VideoVaeConfig::tiny(), &src, true, true).unwrap();
    let px = Volume::from_data(3, 17, 32, 48, src.get("input.pixels", &[3, 17, 32, 48]).unwrap());
    let m = vae.encode_pixels(&be, &px, None).unwrap();
    let em = src.get("expected.moments", &[8, 2, 2, 3]).unwrap();
    assert!(rel(&m.data, &em) < 2e-3);
    let z = Volume::from_data(4, 2, 2, 3, src.get("input.latent", &[4, 2, 2, 3]).unwrap());
    let mut frames = Vec::new();
    vae.decode(&be, &z, None, &mut |v| frames.push(v.clone())).unwrap();
    let out = Volume::cat_t(&frames);
    let ep = src.get("expected.pixels", &[3, 5, 32, 48]).unwrap();
    // The reference returns ImageNet-normalised pixels; ours are denormalised and clamped.
    let mut want = Volume::from_data(3, 5, 32, 48, ep);
    h3_model::vae::denormalize_pixels(&mut want);
    assert!(rel(&out.data, &want.data) < 2e-3);
}

#[test]
fn hf_parity_audio_vae() {
    let Some(src) = fixture("h3_avae_tiny.safetensors") else { return };
    let be = Cpu;
    let cfg = AudioVaeConfig::tiny();
    let vae = AudioVae::load(&be, cfg.clone(), &src, true, true).unwrap();
    let wave = src.get("input.wave", &[95]).unwrap();
    let post = vae.encode(&be, &wave).unwrap();
    assert!(rel(&post.mean.data, &src.get("expected.mean", &[4, 10]).unwrap()) < 2e-3);
    assert!(rel(&post.logs.data, &src.get("expected.logs", &[4, 10]).unwrap()) < 2e-3);
    let z = AudioLatents { channels: 4, frames: 6, data: src.get("input.latent", &[4, 6]).unwrap() };
    let out = vae.decode(&be, &z).unwrap();
    assert!(rel(&out, &src.get("expected.wave", &[60]).unwrap()) < 2e-3);
}
