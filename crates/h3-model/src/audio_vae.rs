//! `AutoencoderKLMiniMaxH3Audio`: a DAC encoder (hop 800 → 40 Hz), a causal
//! attention projection to 32 latent channels, and a BigVGAN decoder with
//! alias-free Snake activations. Stereo is two mono streams. The encoder
//! convolutions and the decoder run on the backend; the small attention
//! projection runs on the host in f32 (it is ~0.1% of the work and the
//! reference keeps it in fp32 too).

use crate::linear::Linear;
use h3_core::{
    Error, Result,
    backend::{Backend, Conv1dShape, ConvEpilogue, ConvT1dShape, Snake},
    config::AudioVaeConfig,
    rng::Rng,
    timestep::matvec,
    weights::WeightSource,
};
use h3_cpu::gelu_tanh;

/// Filter length of the anti-aliasing resamplers.
pub const AA_KERNEL: usize = 12;
pub const AA_RATIO: usize = 2;

struct Conv<B: Backend> {
    w: B::Weight,
    b: Option<B::Buf>,
    cin: usize,
    cout: usize,
    k: usize,
}

fn conv<B: Backend>(
    be: &B,
    src: &dyn WeightSource,
    key: &str,
    cout: usize,
    cin: usize,
    k: usize,
    bias: bool,
) -> Result<Conv<B>> {
    let w = src.conv_weight(key, &[cout, cin, k])?;
    Ok(Conv {
        w: be.weight(&w, cout, cin * k),
        b: if bias {
            Some(be.upload(&src.get(&format!("{key}.bias"), &[cout])?))
        } else {
            None
        },
        cin,
        cout,
        k,
    })
}

/// A transposed convolution (`[cin, cout, k]` as stored).
fn conv_t<B: Backend>(
    be: &B,
    src: &dyn WeightSource,
    key: &str,
    cin: usize,
    cout: usize,
    k: usize,
) -> Result<Conv<B>> {
    let w = src.conv_weight(key, &[cin, cout, k])?;
    Ok(Conv {
        w: be.weight(&w, cin, cout * k),
        b: Some(be.upload(&src.get(&format!("{key}.bias"), &[cout])?)),
        cin,
        cout,
        k,
    })
}

struct Unit<B: Backend> {
    alpha1: B::Buf,
    conv1: Conv<B>,
    dilation: usize,
    alpha2: B::Buf,
    conv2: Conv<B>,
}

struct EncBlock<B: Backend> {
    units: Vec<Unit<B>>,
    alpha: B::Buf,
    down: Conv<B>,
    stride: usize,
}

pub struct Encoder<B: Backend> {
    conv_in: Conv<B>,
    blocks: Vec<EncBlock<B>>,
    alpha: B::Buf,
    conv_out: Conv<B>,
}

/// The host-side causal attention projection (`pre_block`).
pub struct PreBlock {
    norm1: (Vec<f32>, Vec<f32>),
    qkv: Vec<f32>,
    qkv_b: Vec<f32>,
    attn_proj: (Vec<f32>, Vec<f32>),
    proj: (Vec<f32>, Vec<f32>),
    norm2: (Vec<f32>, Vec<f32>),
    norm3: (Vec<f32>, Vec<f32>),
    mlp_norm: (Vec<f32>, Vec<f32>),
    w0: (Vec<f32>, Vec<f32>),
    w1: (Vec<f32>, Vec<f32>),
    w2: (Vec<f32>, Vec<f32>),
    heads: usize,
    mean_proj: (Vec<f32>, Vec<f32>),
    logs_proj: (Vec<f32>, Vec<f32>),
}

struct Activation<B: Backend> {
    alpha: B::Buf,
    beta: B::Buf,
    /// The 12-tap filters as depthwise `[c, k]` weights.
    up: B::Weight,
    down: B::Weight,
}

fn depthwise<B: Backend>(be: &B, filter: &[f32], c: usize) -> B::Weight {
    let w: Vec<f32> = (0..c).flat_map(|_| filter.iter().copied()).collect();
    be.weight(&w, c, filter.len())
}

struct Amp<B: Backend> {
    convs1: Vec<(Conv<B>, usize)>,
    convs2: Vec<Conv<B>>,
    acts: Vec<Activation<B>>,
}

pub struct Decoder<B: Backend> {
    dec_in: Linear<B>,
    conv_pre: Conv<B>,
    ups: Vec<(Conv<B>, usize)>,
    resblocks: Vec<Amp<B>>,
    act_post: Activation<B>,
    conv_post: Conv<B>,
}

pub struct AudioVae<B: Backend> {
    pub cfg: AudioVaeConfig,
    encoder: Option<Encoder<B>>,
    pre: Option<PreBlock>,
    decoder: Option<Decoder<B>>,
}

/// Latents `[channels, frames]` of one mono stream.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioLatents {
    pub channels: usize,
    pub frames: usize,
    pub data: Vec<f32>,
}

impl AudioLatents {
    pub fn zeros(channels: usize, frames: usize) -> Self {
        Self {
            channels,
            frames,
            data: vec![0.0; channels * frames],
        }
    }
    /// `[frames, channels]` rows (the DiT's audio row layout).
    pub fn rows(&self) -> Vec<f32> {
        let mut v = Vec::with_capacity(self.data.len());
        for f in 0..self.frames {
            for c in 0..self.channels {
                v.push(self.data[c * self.frames + f]);
            }
        }
        v
    }
    pub fn from_rows(rows: &[f32], channels: usize) -> Self {
        let frames = rows.len() / channels;
        let mut data = vec![0.0; channels * frames];
        for f in 0..frames {
            for c in 0..channels {
                data[c * frames + f] = rows[f * channels + c];
            }
        }
        Self {
            channels,
            frames,
            data,
        }
    }
    pub fn channel_affine(&mut self, scale: &[f32], shift: &[f32]) {
        for c in 0..self.channels {
            for v in &mut self.data[c * self.frames..(c + 1) * self.frames] {
                *v = *v * scale[c] + shift[c];
            }
        }
    }
}

/// Mean and log-std of the audio posterior.
pub struct AudioPosterior {
    pub mean: AudioLatents,
    pub logs: AudioLatents,
}

impl AudioPosterior {
    pub fn mode(&self) -> AudioLatents {
        self.mean.clone()
    }
    pub fn sample(&self, rng: &mut Rng) -> AudioLatents {
        let eps = rng.normal(self.mean.data.len());
        let mut v = self.mean.clone();
        for ((x, &ls), e) in v.data.iter_mut().zip(&self.logs.data).zip(eps) {
            *x += ls.exp() * e;
        }
        v
    }
}

fn layer_norm(x: &[f32], w: &[f32], b: &[f32]) -> Vec<f32> {
    let n = x.len() as f64;
    let mean = x.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = x.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let inv = 1.0 / (var + 1e-5).sqrt();
    x.iter()
        .enumerate()
        .map(|(i, &v)| ((v as f64 - mean) * inv) as f32 * w[i] + b[i])
        .collect()
}

impl PreBlock {
    fn load(src: &dyn WeightSource, cfg: &AudioVaeConfig) -> Result<Self> {
        let (ld, lc) = (cfg.latent_dim, cfg.latent_channels);
        let pair = |k: &str, o: usize, i: usize| -> Result<(Vec<f32>, Vec<f32>)> {
            Ok((
                src.get(&format!("{k}.weight"), &[o, i])?,
                src.get(&format!("{k}.bias"), &[o])?,
            ))
        };
        let vecs = |k: &str, n: usize| -> Result<(Vec<f32>, Vec<f32>)> {
            Ok((
                src.get(&format!("{k}.weight"), &[n])?,
                src.get(&format!("{k}.bias"), &[n])?,
            ))
        };
        let mut qkv_b = src.get("pre_block.attn.q_bias", &[ld])?;
        qkv_b.extend(src.get("pre_block.attn.zero_k_bias", &[ld])?);
        qkv_b.extend(src.get("pre_block.attn.v_bias", &[ld])?);
        Ok(Self {
            norm1: vecs("pre_block.norm1", ld)?,
            qkv: src.get("pre_block.attn.qkv.weight", &[3 * ld, ld])?,
            qkv_b,
            attn_proj: pair("pre_block.attn.proj", lc, lc)?,
            proj: pair("pre_block.proj", lc, ld)?,
            norm2: vecs("pre_block.norm2", lc)?,
            norm3: vecs("pre_block.norm3", ld)?,
            mlp_norm: vecs("pre_block.mlp.norm", lc)?,
            w0: pair("pre_block.mlp.w0", 2 * lc, lc)?,
            w1: pair("pre_block.mlp.w1", 2 * lc, lc)?,
            w2: pair("pre_block.mlp.w2", lc, 2 * lc)?,
            heads: cfg.heads,
            mean_proj: (
                src.get("mean_proj.weight", &[lc, lc, 1])?,
                src.get("mean_proj.bias", &[lc])?,
            ),
            logs_proj: (
                src.get("logs_proj.weight", &[lc, lc, 1])?,
                src.get("logs_proj.bias", &[lc])?,
            ),
        })
    }

    /// `[frames, latent_dim]` trunk features → `(mean, logs)` `[lc, frames]`.
    fn forward(&self, x: &[Vec<f32>], lc: usize) -> (AudioLatents, AudioLatents) {
        let s = x.len();
        let ld = x[0].len();
        let heads = self.heads;
        let hd = ld / heads;
        // Causal attention over normed inputs.
        let n1: Vec<Vec<f32>> = x
            .iter()
            .map(|r| layer_norm(r, &self.norm1.0, &self.norm1.1))
            .collect();
        let qkv: Vec<Vec<f32>> = n1
            .iter()
            .map(|r| matvec(&self.qkv, Some(&self.qkv_b), r))
            .collect();
        let scale = 1.0 / (hd as f32).sqrt();
        let mut pooled = vec![vec![0.0f32; hd]; s]; // mean over heads
        for i in 0..s {
            for h in 0..heads {
                let q = &qkv[i][h * hd..(h + 1) * hd];
                let mut scores = Vec::with_capacity(i + 1);
                let mut mx = f32::NEG_INFINITY;
                for j in 0..=i {
                    let k = &qkv[j][ld + h * hd..ld + (h + 1) * hd];
                    let sc = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
                    mx = mx.max(sc);
                    scores.push(sc);
                }
                let mut den = 0.0;
                for sc in &mut scores {
                    *sc = (*sc - mx).exp();
                    den += *sc;
                }
                for (j, p) in scores.iter().enumerate() {
                    let v = &qkv[j][2 * ld + h * hd..2 * ld + (h + 1) * hd];
                    for c in 0..hd {
                        pooled[i][c] += p / den * v[c] / heads as f32;
                    }
                }
            }
        }
        // adaptive_avg_pool1d(hd → lc), then proj.
        let bin = hd / lc;
        let attn_out: Vec<Vec<f32>> = pooled
            .iter()
            .map(|p| {
                let pooled: Vec<f32> = (0..lc)
                    .map(|o| p[o * bin..(o + 1) * bin].iter().sum::<f32>() / bin as f32)
                    .collect();
                matvec(&self.attn_proj.0, Some(&self.attn_proj.1), &pooled)
            })
            .collect();
        let mut h: Vec<Vec<f32>> = x
            .iter()
            .zip(&attn_out)
            .map(|(r, a)| {
                let p = matvec(
                    &self.proj.0,
                    Some(&self.proj.1),
                    &layer_norm(r, &self.norm3.0, &self.norm3.1),
                );
                p.iter().zip(a).map(|(x, y)| x + y).collect()
            })
            .collect();
        for r in &mut h {
            let n2 = layer_norm(r, &self.norm2.0, &self.norm2.1);
            let nm = layer_norm(&n2, &self.mlp_norm.0, &self.mlp_norm.1);
            let g = matvec(&self.w0.0, Some(&self.w0.1), &nm);
            let u = matvec(&self.w1.0, Some(&self.w1.1), &nm);
            let gu: Vec<f32> = g.iter().zip(&u).map(|(a, b)| gelu_tanh(*a) * b).collect();
            let m = matvec(&self.w2.0, Some(&self.w2.1), &gu);
            r.iter_mut().zip(m).for_each(|(a, b)| *a += b);
        }
        let mut mean = AudioLatents::zeros(lc, s);
        let mut logs = AudioLatents::zeros(lc, s);
        for (f, r) in h.iter().enumerate() {
            let m = matvec(&self.mean_proj.0, Some(&self.mean_proj.1), r);
            let l = matvec(&self.logs_proj.0, Some(&self.logs_proj.1), r);
            for c in 0..lc {
                mean.data[c * s + f] = m[c];
                logs.data[c * s + f] = l[c];
            }
        }
        (mean, logs)
    }
}

impl<B: Backend> AudioVae<B> {
    pub fn load(
        be: &B,
        cfg: AudioVaeConfig,
        src: &dyn WeightSource,
        encoder: bool,
        decoder: bool,
    ) -> Result<Self> {
        let enc = if encoder {
            Some(Self::load_encoder(be, &cfg, src)?)
        } else {
            None
        };
        let pre = if encoder {
            Some(PreBlock::load(src, &cfg)?)
        } else {
            None
        };
        let dec = if decoder {
            Some(Self::load_decoder(be, &cfg, src)?)
        } else {
            None
        };
        Ok(Self {
            cfg,
            encoder: enc,
            pre,
            decoder: dec,
        })
    }

    fn load_encoder(be: &B, cfg: &AudioVaeConfig, src: &dyn WeightSource) -> Result<Encoder<B>> {
        let alpha =
            |k: &str, c: usize| -> Result<B::Buf> { Ok(be.upload(&src.get(k, &[1, c, 1])?)) };
        let mut blocks = Vec::new();
        let mut c = cfg.encoder_dim;
        for (i, &stride) in cfg.encoder_rates.iter().enumerate() {
            let p = format!("encoder.block.{}", i + 1);
            let mut units = Vec::new();
            for (j, dil) in [1usize, 3, 9].into_iter().enumerate() {
                let q = format!("{p}.block.{j}");
                units.push(Unit {
                    alpha1: alpha(&format!("{q}.block.0.alpha"), c)?,
                    conv1: conv(be, src, &format!("{q}.block.1"), c, c, 7, true)?,
                    dilation: dil,
                    alpha2: alpha(&format!("{q}.block.2.alpha"), c)?,
                    conv2: conv(be, src, &format!("{q}.block.3"), c, c, 1, true)?,
                });
            }
            blocks.push(EncBlock {
                units,
                alpha: alpha(&format!("{p}.block.3.alpha"), c)?,
                down: conv(be, src, &format!("{p}.block.4"), 2 * c, c, 2 * stride, true)?,
                stride,
            });
            c *= 2;
        }
        let n = cfg.encoder_rates.len() + 1;
        Ok(Encoder {
            conv_in: conv(be, src, "encoder.block.0", cfg.encoder_dim, 1, 7, true)?,
            blocks,
            alpha: alpha(&format!("encoder.block.{n}.alpha"), c)?,
            conv_out: conv(
                be,
                src,
                &format!("encoder.block.{}", n + 1),
                cfg.latent_dim,
                c,
                3,
                true,
            )?,
        })
    }

    fn load_decoder(be: &B, cfg: &AudioVaeConfig, src: &dyn WeightSource) -> Result<Decoder<B>> {
        let act = |p: &str, c: usize| -> Result<Activation<B>> {
            Ok(Activation {
                alpha: be.upload(&src.get(&format!("{p}.act.alpha"), &[c])?),
                beta: be.upload(&src.get(&format!("{p}.act.beta"), &[c])?),
                up: depthwise(
                    be,
                    &src.get(&format!("{p}.upsample.filter"), &[1, 1, AA_KERNEL])?
                        .iter()
                        .map(|v| v * AA_RATIO as f32)
                        .collect::<Vec<_>>(),
                    c,
                ),
                down: depthwise(
                    be,
                    &src.get(
                        &format!("{p}.downsample.lowpass.filter"),
                        &[1, 1, AA_KERNEL],
                    )?,
                    c,
                ),
            })
        };
        let dd = cfg.decoder_dim;
        let mut ups = Vec::new();
        let mut resblocks = Vec::new();
        let mut c = dd;
        let nk = cfg.resblock_kernels.len();
        for (i, (&rate, &k)) in cfg
            .decoder_rates
            .iter()
            .zip(&cfg.decoder_kernels)
            .enumerate()
        {
            ups.push((
                conv_t(be, src, &format!("decoder.ups.{i}.0"), c, c / 2, k)?,
                rate,
            ));
            c /= 2;
            for (j, &rk) in cfg.resblock_kernels.iter().enumerate() {
                let p = format!("decoder.resblocks.{}", i * nk + j);
                let dils = &cfg.resblock_dilations[j];
                let mut convs1 = Vec::new();
                let mut convs2 = Vec::new();
                let mut acts = Vec::new();
                for (d, &dil) in dils.iter().enumerate() {
                    convs1.push((
                        conv(be, src, &format!("{p}.convs1.{d}"), c, c, rk, true)?,
                        dil,
                    ));
                    convs2.push(conv(be, src, &format!("{p}.convs2.{d}"), c, c, rk, true)?);
                }
                for a in 0..2 * dils.len() {
                    acts.push(act(&format!("{p}.activations.{a}"), c)?);
                }
                resblocks.push(Amp {
                    convs1,
                    convs2,
                    acts,
                });
            }
        }
        let dec_in_w = src.get(
            "dec_in_proj.weight",
            &[cfg.latent_dim, cfg.latent_channels, 1],
        )?;
        Ok(Decoder {
            dec_in: Linear {
                w: be.weight(&dec_in_w, cfg.latent_dim, cfg.latent_channels),
                b: Some(be.upload(&src.get("dec_in_proj.bias", &[cfg.latent_dim])?)),
                rows: cfg.latent_dim,
                cols: cfg.latent_channels,
                lora: None,
            },
            conv_pre: conv(be, src, "decoder.conv_pre", dd, cfg.latent_dim, 7, true)?,
            ups,
            resblocks,
            act_post: act("decoder.activation_post", c)?,
            conv_post: conv(be, src, "decoder.conv_post", 1, c, 7, false)?,
        })
    }

    pub fn normalize_latents(&self, z: &mut AudioLatents) {
        let scale: Vec<f32> = self.cfg.latents_std.iter().map(|s| 1.0 / s).collect();
        let shift: Vec<f32> = self
            .cfg
            .latents_mean
            .iter()
            .zip(&self.cfg.latents_std)
            .map(|(m, s)| -m / s)
            .collect();
        z.channel_affine(&scale, &shift);
    }
    pub fn denormalize_latents(&self, z: &mut AudioLatents) {
        z.channel_affine(&self.cfg.latents_std, &self.cfg.latents_mean);
    }

    fn c1(
        &self,
        be: &B,
        x: &B::Buf,
        c: &Conv<B>,
        len: usize,
        stride: usize,
        dilation: usize,
        pad: (usize, usize),
        ep: ConvEpilogue<'_, B::Buf>,
    ) -> (B::Buf, usize) {
        let s = Conv1dShape {
            cin: c.cin,
            cout: c.cout,
            groups: 1,
            len,
            k: c.k,
            stride,
            dilation,
            pad,
        };
        let out = be.alloc(s.out_len());
        be.conv1d(x, &s, &c.w, c.b.as_ref(), ep, &out);
        (out, s.out_len_1d())
    }

    /// Posterior of one mono waveform (samples in `[−1, 1]`), right-padded to a
    /// multiple of the hop: latents at 40 Hz.
    pub fn encode(&self, be: &B, wave: &[f32]) -> Result<AudioPosterior> {
        let enc = self
            .encoder
            .as_ref()
            .ok_or_else(|| Error::Msg("the audio VAE encoder is not loaded".into()))?;
        let pre = self.pre.as_ref().unwrap();
        let hop = self.cfg.hop();
        let len = wave.len().div_ceil(hop) * hop;
        let mut x = wave.to_vec();
        x.resize(len, 0.0);
        let gx = be.upload(&x);
        let (mut cur, mut n) =
            self.c1(be, &gx, &enc.conv_in, len, 1, 1, (3, 3), ConvEpilogue::None);
        let mut c = self.cfg.encoder_dim;
        for blk in &enc.blocks {
            for u in &blk.units {
                let y = be.alloc(c * n);
                be.copy(&cur, 0, &y, 0, c * n);
                be.snake(&y, c, n, Snake::Alpha(&u.alpha1));
                let pad = 3 * u.dilation;
                let (y2, n2) = self.c1(
                    be,
                    &y,
                    &u.conv1,
                    n,
                    1,
                    u.dilation,
                    (pad, pad),
                    ConvEpilogue::None,
                );
                debug_assert_eq!(n2, n);
                be.snake(&y2, c, n, Snake::Alpha(&u.alpha2));
                let (out, _) = self.c1(be, &y2, &u.conv2, n, 1, 1, (0, 0), ConvEpilogue::Add(&cur));
                cur = out;
            }
            be.snake(&cur, c, n, Snake::Alpha(&blk.alpha));
            let pad = blk.stride.div_ceil(2);
            let (out, n2) = self.c1(
                be,
                &cur,
                &blk.down,
                n,
                blk.stride,
                1,
                (pad, pad),
                ConvEpilogue::None,
            );
            cur = out;
            n = n2;
            c *= 2;
        }
        be.snake(&cur, c, n, Snake::Alpha(&enc.alpha));
        let (out, n2) = self.c1(be, &cur, &enc.conv_out, n, 1, 1, (1, 1), ConvEpilogue::None);
        let ld = self.cfg.latent_dim;
        let host = be.download(&out);
        let rows: Vec<Vec<f32>> = (0..n2)
            .map(|f| (0..ld).map(|ch| host[ch * n2 + f]).collect())
            .collect();
        let (mean, logs) = pre.forward(&rows, self.cfg.latent_channels);
        Ok(AudioPosterior { mean, logs })
    }

    fn activation(
        &self,
        be: &B,
        x: &B::Buf,
        c: usize,
        len: usize,
        a: &Activation<B>,
    ) -> (B::Buf, usize) {
        // UpSample1d: replicate pad `pad`, transposed conv stride 2, ×ratio, crop.
        // The crop is symmetric (`pl == pr`), so it is the transposed conv's own
        // `pad`: the kernel writes the kept samples only, and the ×ratio lives in
        // the filter (`depthwise`, exact in bf16). The earlier per-channel crop
        // copies were 36 k launches and 46 % of the audio decode on an RTX PRO 6000.
        let k = AA_KERNEL;
        let pad = k / AA_RATIO - 1;
        let pl = pad * AA_RATIO + (k - AA_RATIO) / 2;
        let pr = pad * AA_RATIO + (k - AA_RATIO + 1) / 2;
        debug_assert_eq!(pl, pr);
        let padded = be.alloc(c * (len + 2 * pad));
        be.replicate_pad1d(x, c, len, pad, pad, &padded);
        let ts = ConvT1dShape {
            cin: c,
            cout: c,
            groups: c,
            len: len + 2 * pad,
            k,
            stride: AA_RATIO,
            pad: pl,
        };
        let up_len = ts.out_len_1d();
        let up = be.alloc(c * up_len);
        be.conv_transpose1d(&padded, &ts, &a.up, None, &up);
        be.snake(
            &up,
            c,
            up_len,
            Snake::AlphaBeta {
                alpha: &a.alpha,
                beta: &a.beta,
            },
        );
        // DownSample1d: replicate pad (k/2 − 1, k/2), depthwise conv stride 2.
        let (dl, dr) = (k / 2 - 1, k / 2);
        let dp = be.alloc(c * (up_len + dl + dr));
        be.replicate_pad1d(&up, c, up_len, dl, dr, &dp);
        let ds = Conv1dShape {
            cin: c,
            cout: c,
            groups: c,
            len: up_len + dl + dr,
            k,
            stride: AA_RATIO,
            dilation: 1,
            pad: (0, 0),
        };
        let out = be.alloc(ds.out_len());
        be.conv1d(&dp, &ds, &a.down, None, ConvEpilogue::None, &out);
        (out, ds.out_len_1d())
    }

    /// Decode one mono latent stream (denormalised `[lc, frames]`) to
    /// `frames · hop` samples in `[−1, 1]`.
    pub fn decode(&self, be: &B, z: &AudioLatents) -> Result<Vec<f32>> {
        let dec = self
            .decoder
            .as_ref()
            .ok_or_else(|| Error::Msg("the audio VAE decoder is not loaded".into()))?;
        let cfg = &self.cfg;
        let n0 = z.frames;
        // dec_in_proj (1×1 conv): rows are frames.
        let rows = be.upload(&z.rows());
        let proj = be.alloc(n0 * cfg.latent_dim);
        be.linear(
            &rows,
            0,
            n0,
            &dec.dec_in.w,
            dec.dec_in.b.as_ref(),
            h3_core::backend::Act::None,
            &proj,
            0,
        );
        // [frames, ld] → [ld, frames]: unpatchify with a unit patch is a transpose.
        let xt = be.alloc(cfg.latent_dim * n0);
        be.unpatchify(
            &proj,
            0,
            cfg.latent_dim,
            0,
            cfg.latent_dim,
            1,
            1,
            n0,
            (1, 1, 1),
            &xt,
        );
        let (mut cur, mut n) =
            self.c1(be, &xt, &dec.conv_pre, n0, 1, 1, (3, 3), ConvEpilogue::None);
        let mut c = cfg.decoder_dim;
        let nk = cfg.resblock_kernels.len();
        for (i, (up, rate)) in dec.ups.iter().enumerate() {
            let pad = (up.k - rate) / 2;
            let ts = ConvT1dShape {
                cin: c,
                cout: c / 2,
                groups: 1,
                len: n,
                k: up.k,
                stride: *rate,
                pad,
            };
            let y = be.alloc(ts.out_len());
            be.conv_transpose1d(&cur, &ts, &up.w, up.b.as_ref(), &y);
            c /= 2;
            n = ts.out_len_1d();
            let mut sum: Option<B::Buf> = None;
            for j in 0..nk {
                let amp = &dec.resblocks[i * nk + j];
                let mut h = be.alloc(c * n);
                be.copy(&y, 0, &h, 0, c * n);
                for (d, ((conv1, dil), conv2)) in amp.convs1.iter().zip(&amp.convs2).enumerate() {
                    let (a1, l1) = self.activation(be, &h, c, n, &amp.acts[2 * d]);
                    debug_assert_eq!(l1, n);
                    let p1 = (conv1.k * dil - dil) / 2;
                    let (r1, _) = self.c1(be, &a1, conv1, n, 1, *dil, (p1, p1), ConvEpilogue::None);
                    let (a2, _) = self.activation(be, &r1, c, n, &amp.acts[2 * d + 1]);
                    let p2 = (conv2.k - 1) / 2;
                    let (r2, _) = self.c1(be, &a2, conv2, n, 1, 1, (p2, p2), ConvEpilogue::Add(&h));
                    h = r2;
                }
                match &sum {
                    None => sum = Some(h),
                    Some(s) => be.axpby(s, 0, 1.0, &h, 0, 1.0, c * n),
                }
            }
            cur = sum.unwrap();
            be.axpby(&cur, 0, 1.0 / nk as f32, &cur, 0, 0.0, c * n);
        }
        let (a, _) = self.activation(be, &cur, c, n, &dec.act_post);
        let (out, n2) = self.c1(be, &a, &dec.conv_post, n, 1, 1, (3, 3), ConvEpilogue::None);
        be.clamp(&out, n2, -1.0, 1.0);
        let mut v = be.download(&out);
        v.truncate(n2);
        Ok(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use h3_core::weights::Synthetic;
    use h3_cpu::Cpu;

    #[test]
    fn tiny_audio_round_trip_shapes() {
        let be = Cpu;
        let cfg = AudioVaeConfig::tiny();
        let vae = AudioVae::load(&be, cfg.clone(), &Synthetic { seed: 9 }, true, true).unwrap();
        let hop = cfg.hop();
        assert_eq!(hop, 10);
        let wave: Vec<f32> = (0..95).map(|i| ((i as f32) * 0.3).sin() * 0.5).collect();
        let post = vae.encode(&be, &wave).unwrap();
        assert_eq!(
            (post.mean.channels, post.mean.frames),
            (cfg.latent_channels, 10)
        );
        let mut z = post.mode();
        vae.normalize_latents(&mut z);
        assert_eq!(AudioLatents::from_rows(&z.rows(), cfg.latent_channels), z);
        vae.denormalize_latents(&mut z);
        let out = vae.decode(&be, &z).unwrap();
        assert_eq!(out.len(), 10 * hop);
        assert!(out.iter().all(|v| (-1.0..=1.0).contains(v)));
    }
}
