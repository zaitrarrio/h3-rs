//! Upscale raw interleaved RGB frames on the CPU backend with
//! `h3_model::esrgan` (for byte comparison against fastwan-rs's
//! `upscale_raw`, which matches a numpy SRVGGNetCompact):
//! `esrgan_raw WEIGHTS.safetensors IN.rgb FRAMES H W OUT.rgb`.
use h3_core::weights::SafetensorsDir;
use h3_cpu::Cpu;
use h3_model::esrgan::{Esrgan, EsrganConfig};
use std::path::Path;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let [_, w, input, frames, h, wd, out] = &a[..] else {
        panic!("usage: esrgan_raw WEIGHTS IN.rgb FRAMES H W OUT.rgb")
    };
    let (frames, h, wd): (usize, usize, usize) = (
        frames.parse().unwrap(),
        h.parse().unwrap(),
        wd.parse().unwrap(),
    );
    let src = SafetensorsDir::open(Path::new(w)).unwrap();
    let up = Esrgan::load(&Cpu, EsrganConfig::anime_video_v3(), &src).unwrap();
    let rgb = std::fs::read(input).unwrap();
    let t = std::time::Instant::now();
    let mut px = Vec::new();
    up.upscale(&Cpu, &rgb, frames, h, wd, 1, &mut |_, _, b| {
        px.extend(b);
        Ok(())
    })
    .unwrap();
    eprintln!(
        "{frames}x{h}x{wd} -> x{} in {:.2}s",
        up.cfg.scale,
        t.elapsed().as_secs_f64()
    );
    std::fs::write(out, px).unwrap();
}
