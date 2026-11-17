//! TAEH3 (madebyollin/taehv `taeh3.safetensors`): the tiny preview decoder
//! for H3 latents, fastwan-rs's TAEHV port with H3's frame rule.
//!
//! ```text
//! clamp tanh(x/3)·3 → conv 24→256, relu
//! 3 stages: 3×MemBlock(n_s) → upsample ×2 → TGrow(stride) → conv n_s→n_{s+1}
//! relu → conv 64→12 → pixel-shuffle 2 → clamp [0, 1]
//! MemBlock(x, past) = relu(conv(relu(conv(relu(conv(cat[x, past]))))) + x)
//! ```
//!
//! Three exact rewrites from fastwan-rs: TGrow (a 1×1 conv) runs before the
//! nearest upsample; the upsample is fused into the next conv's addressing;
//! the MemBlock concat is read from two buffers. Each latent frame yields 4 raw
//! frames; raw frame `i` (counted from the clip start) is kept when
//! `i mod 20 ≥ 3`, so 5 latents give 17 frames and 37 give 124. The decoder
//! consumes the DiT's normalised latents directly.

use h3_core::{
    Result,
    backend::{Backend, ConvEpilogue, ConvShape},
    config::TaeH3Config,
    weights::WeightSource,
};

struct ConvW<B: Backend> {
    w: B::Weight,
    b: Option<B::Buf>,
}

struct Stage<B: Backend> {
    mems: Vec<[ConvW<B>; 3]>,
    grow: ConvW<B>,
    conv: ConvW<B>,
}

pub struct TaeH3<B: Backend> {
    pub cfg: TaeH3Config,
    first: ConvW<B>,
    stages: Vec<Stage<B>>,
    last: ConvW<B>,
}

/// Carried between chunks: one saved input frame per MemBlock and the raw
/// frame counter that drives the keep rule.
pub struct DecodeState<B: Backend> {
    memory: Vec<Option<B::Buf>>,
    raw_index: usize,
}

/// Decoded frames of one chunk: `[frames, 3, H, W]` in `[0, 1]`.
pub struct Frames {
    pub frames: usize,
    pub height: usize,
    pub width: usize,
    pub data: Vec<f32>,
}

fn conv_w<B: Backend>(
    be: &B,
    src: &dyn WeightSource,
    key: &str,
    cout: usize,
    cin: usize,
    k: usize,
    bias: bool,
) -> Result<ConvW<B>> {
    Ok(ConvW {
        w: be.weight(
            &src.get(&format!("{key}.weight"), &[cout, cin, k, k])?,
            cout,
            cin * k * k,
        ),
        b: if bias {
            Some(be.upload(&src.get(&format!("{key}.bias"), &[cout])?))
        } else {
            None
        },
    })
}

impl<B: Backend> TaeH3<B> {
    pub fn load(be: &B, cfg: TaeH3Config, src: &dyn WeightSource) -> Result<Self> {
        let nf = cfg.widths;
        let mut stages = Vec::new();
        for s in 0..3 {
            let base = 3 + 6 * s;
            let mems = (0..cfg.mem_blocks)
                .map(|m| {
                    let p = format!("decoder.{}", base + m);
                    Ok([
                        conv_w(be, src, &format!("{p}.conv.0"), nf[s], 2 * nf[s], 3, true)?,
                        conv_w(be, src, &format!("{p}.conv.2"), nf[s], nf[s], 3, true)?,
                        conv_w(be, src, &format!("{p}.conv.4"), nf[s], nf[s], 3, true)?,
                    ])
                })
                .collect::<Result<Vec<_>>>()?;
            stages.push(Stage {
                mems,
                grow: conv_w(
                    be,
                    src,
                    &format!("decoder.{}.conv", base + 4),
                    nf[s] * cfg.tgrow[s],
                    nf[s],
                    1,
                    false,
                )?,
                conv: conv_w(
                    be,
                    src,
                    &format!("decoder.{}", base + 5),
                    nf[s + 1],
                    nf[s],
                    3,
                    false,
                )?,
            });
        }
        Ok(Self {
            first: conv_w(be, src, "decoder.1", nf[0], cfg.latent_channels, 3, true)?,
            last: conv_w(be, src, "decoder.22", cfg.out_channels(), nf[3], 3, true)?,
            stages,
            cfg,
        })
    }

    pub fn state(&self) -> DecodeState<B> {
        DecodeState {
            memory: (0..3 * self.cfg.mem_blocks).map(|_| None).collect(),
            raw_index: 0,
        }
    }

    /// Frames a decode of `latent_frames` keeps (raw index rule).
    pub fn kept_frames(latent_frames: usize, t_upscale: usize) -> usize {
        (0..latent_frames * t_upscale)
            .filter(|&i| TaeH3Config::keeps_raw_frame(i))
            .count()
    }

    /// Decode `frames` latent frames given frame-major (`[frames, C, h, w]`).
    pub fn decode_chunk(
        &self,
        be: &B,
        st: &mut DecodeState<B>,
        z: &[f32],
        frames: usize,
        h: usize,
        w: usize,
    ) -> Frames {
        let cfg = &self.cfg;
        let nf = cfg.widths;
        assert_eq!(z.len(), frames * cfg.latent_channels * h * w);
        let (mut f, mut h, mut w) = (frames, h, w);
        let x0 = be.upload(z);
        be.tanh_clamp(&x0, z.len(), 3.0);
        let shape = |f, cin_a, cin_b, cout, h, w, upsample, k| ConvShape {
            frames: f,
            cin_a,
            cin_b,
            cout,
            h,
            w,
            upsample,
            k,
        };
        let conv = |x: &B::Buf,
                    b: Option<&B::Buf>,
                    s: ConvShape,
                    cw: &ConvW<B>,
                    ep: ConvEpilogue<'_, B::Buf>| {
            let out = be.alloc(s.out_len());
            be.conv2d(x, b, &s, &cw.w, cw.b.as_ref(), ep, &out);
            out
        };
        let mut x = conv(
            &x0,
            None,
            shape(f, cfg.latent_channels, 0, nf[0], h, w, false, 3),
            &self.first,
            ConvEpilogue::Relu,
        );
        for (si, stage) in self.stages.iter().enumerate() {
            let n = nf[si];
            for (mi, [c0, c2, c4]) in stage.mems.iter().enumerate() {
                let fsz = n * h * w;
                let past = be.alloc(f * fsz);
                let slot = &mut st.memory[si * cfg.mem_blocks + mi];
                if let Some(prev) = slot.as_ref() {
                    be.copy(prev, 0, &past, 0, fsz);
                }
                if f > 1 {
                    be.copy(&x, 0, &past, fsz, (f - 1) * fsz);
                }
                let keep = be.alloc(fsz);
                be.copy(&x, (f - 1) * fsz, &keep, 0, fsz);
                *slot = Some(keep);
                let t1 = conv(
                    &x,
                    Some(&past),
                    shape(f, n, n, n, h, w, false, 3),
                    c0,
                    ConvEpilogue::Relu,
                );
                let t2 = conv(
                    &t1,
                    None,
                    shape(f, n, 0, n, h, w, false, 3),
                    c2,
                    ConvEpilogue::Relu,
                );
                x = conv(
                    &t2,
                    None,
                    shape(f, n, 0, n, h, w, false, 3),
                    c4,
                    ConvEpilogue::AddRelu(&x),
                );
            }
            let stride = cfg.tgrow[si];
            let g = conv(
                &x,
                None,
                shape(f, n, 0, n * stride, h, w, false, 1),
                &stage.grow,
                ConvEpilogue::None,
            );
            f *= stride;
            let ep = if si == 2 {
                ConvEpilogue::Relu
            } else {
                ConvEpilogue::None
            };
            x = conv(
                &g,
                None,
                shape(f, n, 0, nf[si + 1], h, w, true, 3),
                &stage.conv,
                ep,
            );
            h *= 2;
            w *= 2;
        }
        let rgb = conv(
            &x,
            None,
            shape(f, nf[3], 0, cfg.out_channels(), h, w, false, 3),
            &self.last,
            ConvEpilogue::None,
        );
        let p = cfg.patch;
        let (oh, ow) = (h * p, w * p);
        let px = be.alloc(f * 3 * oh * ow);
        be.pixel_shuffle(&rgb, f, 3, h, w, p, &px);
        be.clamp(&px, f * 3 * oh * ow, 0.0, 1.0);
        let raw = be.download(&px);
        let keep: Vec<usize> = (0..f)
            .filter(|&i| TaeH3Config::keeps_raw_frame(st.raw_index + i))
            .collect();
        st.raw_index += f;
        let per = 3 * oh * ow;
        let mut data = Vec::with_capacity(keep.len() * per);
        for &i in &keep {
            data.extend_from_slice(&raw[i * per..(i + 1) * per]);
        }
        Frames {
            frames: keep.len(),
            height: oh,
            width: ow,
            data,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3_core::weights::Synthetic;
    use h3_cpu::Cpu;

    #[test]
    fn frame_rule_and_tiny_decode() {
        assert_eq!(TaeH3::<Cpu>::kept_frames(5, 4), 17);
        assert_eq!(TaeH3::<Cpu>::kept_frames(37, 4), 124);
        assert_eq!(TaeH3::<Cpu>::kept_frames(2, 4), 5);
        let be = Cpu;
        let cfg = TaeH3Config::tiny();
        let tae = TaeH3::load(&be, cfg.clone(), &Synthetic { seed: 3 }).unwrap();
        let (h, w) = (2, 3);
        let z: Vec<f32> = (0..7 * cfg.latent_channels * h * w)
            .map(|i| ((i * 31) % 17) as f32 * 0.1 - 0.8)
            .collect();
        let mut st = tae.state();
        let per = cfg.latent_channels * h * w;
        let a = tae.decode_chunk(&be, &mut st, &z[..5 * per], 5, h, w);
        let b = tae.decode_chunk(&be, &mut st, &z[5 * per..], 2, h, w);
        assert_eq!((a.frames, b.frames), (17, 5));
        assert_eq!((a.height, a.width), (h * 16, w * 16));
        assert!(
            a.data
                .iter()
                .chain(&b.data)
                .all(|v| (0.0..=1.0).contains(v))
        );
        // Chunking is exact: one 7-frame decode equals the two chunks.
        let mut st2 = tae.state();
        let c = tae.decode_chunk(&be, &mut st2, &z, 7, h, w);
        assert_eq!(c.frames, 22);
        let joined: Vec<f32> = a.data.iter().chain(&b.data).copied().collect();
        let err = joined
            .iter()
            .zip(&c.data)
            .fold(0f32, |m, (x, y)| m.max((x - y).abs()));
        assert!(err < 1e-5, "{err}");
    }
}
