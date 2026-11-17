//! Architecture constants. Numbers follow each checkpoint's `config.json`
//! (`MiniMaxAI/MiniMax-H3`, read 2026-11-06) and FastVideo's
//! `configs/models/*/minimax_h3*.py`.

use crate::{Error, Result};

/// The H3 Omni-Transformer (`MiniMaxH3Transformer3DModel`). Batch is always 1:
/// every released checkpoint is CFG-distilled, so a second batch row never
/// exists.
#[derive(Debug, Clone, PartialEq)]
pub struct H3Config {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub layers: usize,
    pub refiner_layers: usize,
    pub ffn_dim: usize,
    /// Video latent channels (the VAE's 24).
    pub in_channels: usize,
    /// Audio latent channels (the audio VAE's 32).
    pub audio_in_channels: usize,
    /// `(t, h, w)` patch; `(1, 2, 2)` for every release.
    pub patch: (usize, usize, usize),
    /// Qwen3-VL hidden size.
    pub text_dim: usize,
    pub freq_dim: usize,
    pub time_embed_hidden: usize,
    pub time_embed_dim: usize,
    /// Rotary frequencies per axis; the rotated prefix of a head is `6·this`.
    pub rope_freq_dim: usize,
    pub rope_theta: f64,
    pub norm_eps: f32,
    pub qk_norm_eps: f32,
    pub final_norm_eps: f32,
    /// Rank-reduced AdaLN (FastH3 Trim): `adaln_proj.linear` is `[6·3·hidden,
    /// rank]` and `adaln_basis [rank, time_embed_dim]` precedes it.
    pub adaln_rank: Option<usize>,
    /// Parallel Decoding Distillation heads on `proj_out` / `audio_proj_out`
    /// (FastH3 Omni): the output projections are `pdd_steps` times wider.
    pub pdd_steps: Option<usize>,
    /// Trim: which base blocks the 42 kept blocks came from (informational).
    pub kept_block_indices: Option<Vec<usize>>,
}

impl H3Config {
    pub fn dim(&self) -> usize {
        self.heads * self.head_dim
    }
    /// Width of the rotated prefix of every head (96 of 128).
    pub fn rotary_dim(&self) -> usize {
        6 * self.rope_freq_dim
    }
    pub fn video_patch_dim(&self) -> usize {
        self.in_channels * self.patch.0 * self.patch.1 * self.patch.2
    }
    /// Rows of one block's modulation table per `(timestep, modality)`:
    /// shift/scale/gate for attention and FFN.
    pub const MOD_SLOTS: usize = 6;
    pub const MODALITIES: usize = 3;

    /// `MiniMaxAI/MiniMax-H3` (`transformer/` and `transformer_ref/`), FastH3
    /// V1 / V2, Pika H3 Flex: the released 50-block, 33 B configuration.
    pub fn h3() -> Self {
        Self {
            hidden: 5376,
            heads: 56,
            head_dim: 128,
            layers: 50,
            refiner_layers: 2,
            ffn_dim: 14336,
            in_channels: 24,
            audio_in_channels: 32,
            patch: (1, 2, 2),
            text_dim: 5120,
            freq_dim: 256,
            time_embed_hidden: 5376,
            time_embed_dim: 2688,
            rope_freq_dim: 16,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            final_norm_eps: 1e-5,
            adaln_rank: None,
            pdd_steps: None,
            kept_block_indices: None,
        }
    }

    /// `FastVideo/FastVideo-FastH3-Trim-8-Step-NVFP4`: 42 kept blocks,
    /// rank-16 AdaLN.
    pub fn fasth3_trim() -> Self {
        Self {
            layers: 42,
            adaln_rank: Some(16),
            kept_block_indices: Some(vec![
                0, 1, 2, 3, 4, 5, 8, 10, 11, 12, 14, 17, 18, 19, 20, 21, 24, 25, 26, 27, 28, 29,
                30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49,
            ]),
            ..Self::h3()
        }
    }

    /// `FastVideo/FastVideo-FastH3-Omni-8-Step-V1` (`transformer_ref`): PDD
    /// student with 32 heads.
    pub fn fasth3_omni() -> Self {
        Self {
            pdd_steps: Some(32),
            ..Self::h3()
        }
    }

    /// A graph small enough for the CPU reference in a unit test. `head_dim`
    /// 16 with `rope_freq_dim` 2 rotates 12 of 16 channels, so the pass-through
    /// tail is exercised.
    pub fn tiny() -> Self {
        Self {
            hidden: 32,
            heads: 2,
            head_dim: 16,
            layers: 2,
            refiner_layers: 1,
            ffn_dim: 48,
            in_channels: 4,
            audio_in_channels: 6,
            patch: (1, 2, 2),
            text_dim: 24,
            freq_dim: 16,
            time_embed_hidden: 32,
            time_embed_dim: 24,
            rope_freq_dim: 2,
            rope_theta: 10000.0,
            norm_eps: 1e-5,
            qk_norm_eps: 1e-5,
            final_norm_eps: 1e-5,
            adaln_rank: None,
            pdd_steps: None,
            kept_block_indices: None,
        }
    }

    /// [`H3Config::tiny`] with `head_dim` 32: the smallest graph the GPU
    /// flash / VSA kernels take (head_dim a multiple of 16).
    pub fn tiny_vsa() -> Self {
        Self {
            heads: 2,
            head_dim: 32,
            hidden: 64,
            rope_freq_dim: 4,
            ..Self::tiny()
        }
    }

    /// Read a Diffusers `transformer/config.json` (`MiniMaxH3Transformer3DModel`).
    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let g = |k: &str| -> Result<usize> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .ok_or_else(|| Error::Config(format!("transformer config.json lacks `{k}`")))
        };
        let f = |k: &str, d: f64| v.get(k).and_then(|x| x.as_f64()).unwrap_or(d);
        let patch = v
            .get("patch_size")
            .and_then(|p| p.as_array())
            .map(|p| {
                let x: Vec<usize> = p
                    .iter()
                    .filter_map(|a| a.as_u64())
                    .map(|a| a as usize)
                    .collect();
                (x[0], x[1], x[2])
            })
            .unwrap_or((1, 2, 2));
        let c = Self {
            hidden: g("hidden_size")?,
            heads: g("num_attention_heads")?,
            head_dim: g("attention_head_dim")?,
            layers: g("num_layers")?,
            refiner_layers: g("num_refiner_layers").or_else(|_| g("token_refiner_num_layers"))?,
            ffn_dim: g("ffn_dim").or_else(|_| g("ffn_hidden_size"))?,
            in_channels: g("in_channels").or_else(|_| g("latents_dim"))?,
            audio_in_channels: g("audio_in_channels").or_else(|_| g("audio_latents_dim"))?,
            patch,
            text_dim: g("text_dim")?,
            freq_dim: g("freq_dim").or_else(|_| g("timestep_input_dim"))?,
            time_embed_hidden: g("time_embed_hidden_dim")
                .or_else(|_| g("time_embed_hidden_size"))?,
            time_embed_dim: g("time_embed_dim")?,
            rope_freq_dim: g("rope_freq_dim").or_else(|_| g("rope_inv_freq_len"))?,
            rope_theta: f("rope_theta", 10000.0),
            norm_eps: f("norm_eps", 1e-5) as f32,
            qk_norm_eps: f("qk_norm_eps", 1e-5) as f32,
            final_norm_eps: f("final_norm_eps", 1e-5) as f32,
            adaln_rank: v
                .get("adaln_rank")
                .and_then(|x| x.as_u64())
                .map(|x| x as usize),
            pdd_steps: v
                .get("pdd_steps")
                .and_then(|x| x.as_u64())
                .map(|x| x as usize),
            kept_block_indices: v
                .get("kept_block_indices")
                .and_then(|x| x.as_array())
                .map(|a| {
                    a.iter()
                        .filter_map(|x| x.as_u64())
                        .map(|x| x as usize)
                        .collect()
                }),
        };
        if c.rotary_dim() > c.head_dim || c.rotary_dim() % 2 != 0 {
            return Err(Error::Config(format!(
                "rotary width {} must be even and ≤ head_dim {}",
                c.rotary_dim(),
                c.head_dim
            )));
        }
        Ok(c)
    }

    /// Parameter count of the DiT as released (AdaLN projections included,
    /// VSA gates excluded): the sum of the manifest's shapes.
    pub fn params(&self) -> usize {
        crate::weights::manifest_params(&crate::weights::h3_manifest(self, false))
    }

    /// Parameters a resident DiT needs once the AdaLN tables are precomputed
    /// (what the device holds): everything but `adaln_proj` / `adaln_basis`
    /// and `norm_out.linear`.
    pub fn params_without_adaln(&self) -> usize {
        crate::weights::h3_manifest(self, false)
            .iter()
            .filter(|(n, _)| !(n.contains("adaln") || n.starts_with("norm_out.linear")))
            .map(|(_, s)| s.iter().product::<usize>())
            .sum()
    }
}

/// `AutoencoderKLMiniMaxH3` (`vae/config.json`): causal 3-D CNN encoder, ViT
/// decoder; 16× spatial, 4× temporal, 24 latent channels.
#[derive(Debug, Clone, PartialEq)]
pub struct VideoVaeConfig {
    pub in_channels: usize,
    pub latent_channels: usize,
    pub block_out_channels: Vec<usize>,
    pub layers_per_block: usize,
    pub spatial_down: Vec<usize>,
    pub temporal_down: Vec<usize>,
    pub norm_groups: usize,
    pub norm_eps: f32,
    pub decoder_layers: usize,
    pub decoder_heads: usize,
    pub decoder_head_dim: usize,
    pub decoder_register_tokens: usize,
    pub decoder_ffn_mult: usize,
    pub decoder_rope_theta: f64,
    pub decoder_rope_dim_ratio: f64,
    pub decoder_norm_eps: f32,
    pub clip_length: usize,
    pub token_drop: usize,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

pub const H3_LATENTS_MEAN: [f32; 24] = [
    0.858090341091156,
    -0.9606591463088989,
    1.0661640167236328,
    -0.5090325474739075,
    -0.2727581858634949,
    -1.3675414323806763,
    -0.2553254961967468,
    -0.26907554268836975,
    -0.5376840829849243,
    -0.0464097298681736,
    0.6657370328903198,
    0.19690127670764923,
    -0.5460608005523682,
    -0.4035342037677765,
    -0.23683024942874908,
    0.25928452610969543,
    -0.30133944749832153,
    0.211341992020607,
    -1.1206848621368408,
    0.3581933379173279,
    -0.04225143790245056,
    0.2604829967021942,
    0.22864092886447906,
    0.7056031823158264,
];
pub const H3_LATENTS_STD: [f32; 24] = [
    1.2223774194717407,
    1.2767263650894165,
    1.6831774711608887,
    1.7549455165863037,
    1.5636216402053833,
    2.194143533706665,
    0.9653137922286987,
    1.0569885969161987,
    0.841948926448822,
    0.7729952931404114,
    1.8955937623977661,
    0.946841835975647,
    0.7996809482574463,
    0.44988900423049927,
    0.7197399735450745,
    0.6936293244361877,
    2.961095094680786,
    2.7694199085235596,
    3.0496184825897217,
    2.1088054180145264,
    3.276226282119751,
    3.1627357006073,
    2.2816812992095947,
    2.6127843856811523,
];
/// ImageNet pixel statistics the VAE normalises with.
pub const PIXEL_MEAN: [f32; 3] = [0.485, 0.456, 0.406];
pub const PIXEL_STD: [f32; 3] = [0.229, 0.224, 0.225];

impl VideoVaeConfig {
    pub fn h3() -> Self {
        Self {
            in_channels: 3,
            latent_channels: 24,
            block_out_channels: vec![128, 256, 256, 512, 512, 1024],
            layers_per_block: 2,
            spatial_down: vec![2, 2, 2, 2, 1, 1],
            temporal_down: vec![1, 2, 2, 1, 1, 1],
            norm_groups: 32,
            norm_eps: 1e-6,
            decoder_layers: 36,
            decoder_heads: 32,
            decoder_head_dim: 64,
            decoder_register_tokens: 4,
            decoder_ffn_mult: 4,
            decoder_rope_theta: 100.0,
            decoder_rope_dim_ratio: 0.75,
            decoder_norm_eps: 1e-5,
            clip_length: 17,
            token_drop: 3,
            latents_mean: H3_LATENTS_MEAN.to_vec(),
            latents_std: H3_LATENTS_STD.to_vec(),
        }
    }

    /// The Trim checkpoint's light VAE: the same encoder, a 26-block decoder.
    pub fn fasth3_trim() -> Self {
        Self {
            decoder_layers: 26,
            ..Self::h3()
        }
    }

    pub fn tiny() -> Self {
        Self {
            in_channels: 3,
            latent_channels: 4,
            block_out_channels: vec![8, 8, 16, 16, 16, 16],
            layers_per_block: 1,
            spatial_down: vec![2, 2, 2, 2, 1, 1],
            temporal_down: vec![1, 2, 2, 1, 1, 1],
            norm_groups: 4,
            norm_eps: 1e-6,
            decoder_layers: 2,
            decoder_heads: 2,
            decoder_head_dim: 16,
            decoder_register_tokens: 2,
            decoder_ffn_mult: 2,
            decoder_rope_theta: 100.0,
            decoder_rope_dim_ratio: 0.75,
            decoder_norm_eps: 1e-5,
            clip_length: 17,
            token_drop: 3,
            latents_mean: vec![0.1, -0.2, 0.3, 0.0],
            latents_std: vec![1.1, 0.9, 1.3, 1.0],
        }
    }

    pub fn from_json(v: &serde_json::Value) -> Result<Self> {
        let mut c = Self::h3();
        let g = |k: &str| v.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
        let arr = |k: &str| -> Option<Vec<usize>> {
            v.get(k).and_then(|x| x.as_array()).map(|a| {
                a.iter()
                    .filter_map(|x| x.as_u64())
                    .map(|x| x as usize)
                    .collect()
            })
        };
        let farr = |k: &str| -> Option<Vec<f32>> {
            v.get(k).and_then(|x| x.as_array()).map(|a| {
                a.iter()
                    .filter_map(|x| x.as_f64())
                    .map(|x| x as f32)
                    .collect()
            })
        };
        if let Some(x) = g("latent_channels") {
            c.latent_channels = x;
        }
        if let Some(x) = arr("block_out_channels") {
            c.block_out_channels = x;
        }
        if let Some(x) = g("layers_per_block") {
            c.layers_per_block = x;
        }
        if let Some(x) = arr("spatial_downsample_factors") {
            c.spatial_down = x;
        }
        if let Some(x) = arr("temporal_downsample_factors") {
            c.temporal_down = x;
        }
        if let Some(x) = g("norm_num_groups") {
            c.norm_groups = x;
        }
        if let Some(x) = g("decoder_num_layers") {
            c.decoder_layers = x;
        }
        if let Some(x) = g("decoder_num_attention_heads") {
            c.decoder_heads = x;
        }
        if let Some(x) = g("decoder_attention_head_dim") {
            c.decoder_head_dim = x;
        }
        if let Some(x) = g("decoder_num_register_tokens") {
            c.decoder_register_tokens = x;
        }
        if let Some(x) = g("decoder_ffn_mult") {
            c.decoder_ffn_mult = x;
        }
        if let Some(x) = g("clip_length") {
            c.clip_length = x;
        }
        if let Some(x) = g("token_drop") {
            c.token_drop = x;
        }
        if let Some(x) = v.get("decoder_rope_theta").and_then(|x| x.as_f64()) {
            c.decoder_rope_theta = x;
        }
        if let Some(x) = v.get("decoder_rope_dim_ratio").and_then(|x| x.as_f64()) {
            c.decoder_rope_dim_ratio = x;
        }
        if let Some(x) = farr("latents_mean") {
            c.latents_mean = x;
        }
        if let Some(x) = farr("latents_std") {
            c.latents_std = x;
        }
        if c.latents_mean.len() != c.latent_channels || c.latents_std.len() != c.latent_channels {
            return Err(Error::Config(
                "vae latents_mean/std do not match latent_channels".into(),
            ));
        }
        Ok(c)
    }

    pub fn spatial_ratio(&self) -> usize {
        self.spatial_down.iter().product()
    }
    pub fn temporal_ratio(&self) -> usize {
        self.temporal_down.iter().product()
    }
    pub fn decoder_dim(&self) -> usize {
        self.decoder_heads * self.decoder_head_dim
    }
    /// Rotated channels of a decoder head (48 of 64).
    pub fn decoder_rotary_dim(&self) -> usize {
        (self.decoder_head_dim as f64 * self.decoder_rope_dim_ratio) as usize
    }
    /// Frames the causal padding adds in front of each 17-frame clip: `(−17) mod 4 = 3`.
    pub fn frame_pre_padding(&self) -> usize {
        (self.temporal_ratio() - self.clip_length % self.temporal_ratio()) % self.temporal_ratio()
    }
    /// Latent frames per clip: `ceil(17 / 4) = 5`.
    pub fn tokens_chunk(&self) -> usize {
        self.clip_length.div_ceil(self.temporal_ratio())
    }
    /// Latent frames a decode call borrows from the next chunk: `(−3) mod 5 = 2`.
    pub fn token_overlap(&self) -> usize {
        (self.tokens_chunk() - self.token_drop % self.tokens_chunk()) % self.tokens_chunk()
    }
    /// Pixel frames two consecutive decoded chunks blend over: `max(2·4 − 3, 0) = 5`.
    pub fn frame_overlap(&self) -> usize {
        (self.token_overlap() * self.temporal_ratio()).saturating_sub(self.frame_pre_padding())
    }
}

/// `AutoencoderKLMiniMaxH3Audio` (`audio_vae/config.json`): DAC encoder, causal
/// attention projection to 32 latent channels at 40 Hz, BigVGAN decoder.
#[derive(Debug, Clone, PartialEq)]
pub struct AudioVaeConfig {
    pub encoder_dim: usize,
    pub encoder_rates: Vec<usize>,
    pub latent_dim: usize,
    pub latent_channels: usize,
    pub decoder_dim: usize,
    pub decoder_rates: Vec<usize>,
    pub decoder_kernels: Vec<usize>,
    pub heads: usize,
    pub resblock_kernels: Vec<usize>,
    pub resblock_dilations: Vec<Vec<usize>>,
    pub sample_rate: usize,
    pub latents_mean: Vec<f32>,
    pub latents_std: Vec<f32>,
}

pub const AUDIO_LATENTS_MEAN: [f32; 32] = [
    -0.020211687488382354,
    0.3876466479950502,
    -0.04398279799186767,
    -0.28591514936373,
    0.08179686214561671,
    -0.35782641352446604,
    0.040623809960919084,
    -0.01552534501956604,
    -0.223362481667332,
    0.1821006842509091,
    0.2941778783780663,
    -0.07901167601970885,
    -0.056815072777201,
    -0.3699028221860095,
    -0.31616315591624855,
    0.5905951377425391,
    -0.052139568068853864,
    0.013673160263486295,
    -0.03691647864630577,
    0.09732660653298163,
    -0.3394662328788498,
    -0.30685677538541667,
    -0.24504598907458763,
    -0.034698524462007344,
    0.02868032184767538,
    -0.21217779266454084,
    -0.1678263169941987,
    0.3221287889040614,
    -0.1223055851554907,
    0.4356604928128464,
    -0.0502599202236253,
    0.3979258376211797,
];
pub const AUDIO_LATENTS_STD: [f32; 32] = [
    1.6895524230479284,
    2.76263727217653,
    1.7945344281264435,
    1.6801681847309828,
    1.6390226546605453,
    2.7788298348882177,
    1.7659090095747236,
    1.6199757612137327,
    2.6336525640336896,
    1.8539356672817833,
    2.5056497896915633,
    1.811019237886178,
    1.9579657790720237,
    1.6685498243529284,
    1.4922469314453364,
    3.298670198067373,
    1.9491804496832168,
    1.8720003270431442,
    1.8334080103291832,
    1.6488070416529093,
    1.6176957696319716,
    1.9131449234774398,
    1.5695245398428617,
    1.6943659940415912,
    1.8318420762504692,
    1.5540637421583379,
    1.9344930328968526,
    1.599198216109855,
    1.718045989838149,
    1.6307219190837705,
    1.8661226051202384,
    1.5613768203168363,
];

impl AudioVaeConfig {
    pub fn h3() -> Self {
        Self {
            encoder_dim: 64,
            encoder_rates: vec![2, 4, 4, 5, 5],
            latent_dim: 2048,
            latent_channels: 32,
            decoder_dim: 1024,
            decoder_rates: vec![5, 5, 2, 2, 2, 2, 2],
            decoder_kernels: vec![9, 9, 4, 4, 4, 4, 4],
            heads: 8,
            resblock_kernels: vec![3, 7, 11],
            resblock_dilations: vec![vec![1, 3, 5], vec![1, 3, 5], vec![1, 3, 5]],
            sample_rate: 32000,
            latents_mean: AUDIO_LATENTS_MEAN.to_vec(),
            latents_std: AUDIO_LATENTS_STD.to_vec(),
        }
    }

    pub fn tiny() -> Self {
        Self {
            encoder_dim: 4,
            encoder_rates: vec![2, 5],
            latent_dim: 24,
            latent_channels: 6,
            decoder_dim: 8,
            decoder_rates: vec![5, 2],
            decoder_kernels: vec![9, 4],
            heads: 2,
            resblock_kernels: vec![3, 7],
            resblock_dilations: vec![vec![1, 3], vec![1, 3]],
            sample_rate: 320,
            latents_mean: vec![0.0, 0.1, -0.1, 0.2, 0.05, -0.3],
            latents_std: vec![1.0, 1.5, 0.8, 1.2, 0.9, 1.1],
        }
    }

    /// Samples per latent frame: `Π encoder_rates = 800` (40 Hz at 32 kHz).
    pub fn hop(&self) -> usize {
        self.encoder_rates.iter().product()
    }
}

/// TAEH3 (madebyollin/taehv `taeh3.safetensors`): the tiny preview decoder for
/// H3 latents. Same network as TAEHV with a ×2 pixel-shuffle head; 16× spatial,
/// 4 raw frames per latent frame of which 3 of every 20 are dropped.
#[derive(Debug, Clone, PartialEq)]
pub struct TaeH3Config {
    pub latent_channels: usize,
    pub widths: [usize; 4],
    pub tgrow: [usize; 3],
    pub mem_blocks: usize,
    pub patch: usize,
}

impl TaeH3Config {
    pub fn taeh3() -> Self {
        Self {
            latent_channels: 24,
            widths: [256, 128, 64, 64],
            tgrow: [1, 2, 2],
            mem_blocks: 3,
            patch: 2,
        }
    }
    pub fn tiny() -> Self {
        Self {
            latent_channels: 4,
            widths: [8, 6, 4, 4],
            tgrow: [1, 2, 2],
            mem_blocks: 3,
            patch: 2,
        }
    }
    pub fn t_upscale(&self) -> usize {
        self.tgrow.iter().product()
    }
    pub fn spatial_scale(&self) -> usize {
        8 * self.patch
    }
    pub fn out_channels(&self) -> usize {
        3 * self.patch * self.patch
    }
    /// Raw decoded frames `f` (index `start·4 + i`) that are kept: H3's 17
    /// frames per 5 latents are the raw frames `≥ 3 (mod 20)`.
    pub fn keeps_raw_frame(raw_index: usize) -> bool {
        raw_index % 20 >= 3
    }
}

/// Open-H3-Regenerate-2K's latent upscaler (3-D conv, 345 M parameters).
#[derive(Debug, Clone, PartialEq)]
pub struct UpscalerConfig {
    pub in_channels: usize,
    pub channels: usize,
    pub in_blocks: usize,
    pub out_blocks: usize,
    pub temporal_every: usize,
    pub temporal_kernel: usize,
    pub emb_dim: usize,
    pub norm_groups: usize,
}

impl UpscalerConfig {
    pub fn regen_2k() -> Self {
        Self {
            in_channels: 24,
            channels: 512,
            in_blocks: 12,
            out_blocks: 12,
            temporal_every: 2,
            temporal_kernel: 5,
            emb_dim: 64,
            norm_groups: 32,
        }
    }
    pub fn tiny() -> Self {
        Self {
            in_channels: 4,
            channels: 8,
            in_blocks: 2,
            out_blocks: 2,
            temporal_every: 2,
            temporal_kernel: 5,
            emb_dim: 6,
            norm_groups: 4,
        }
    }
    /// Blocks per side including the temporal convs inserted every
    /// `temporal_every` resblocks (index 0, 2, 4, …).
    pub fn block_layout(&self, count: usize) -> Vec<bool> {
        let mut v = Vec::new();
        for i in 0..count {
            v.push(false);
            if self.temporal_every > 0 && i % self.temporal_every == 0 {
                v.push(true);
            }
        }
        v
    }
}

/// Qwen3-VL-32B as H3 uses it: the language stack to hidden state 50 (no
/// final norm), the 27-block vision tower with DeepStack.
#[derive(Debug, Clone, PartialEq)]
pub struct Qwen3VlConfig {
    pub vocab: usize,
    pub hidden: usize,
    pub intermediate: usize,
    pub layers: usize,
    /// Layers that are built and run; the hidden state after this many layers
    /// is the conditioning.
    pub output_layer: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rms_eps: f32,
    pub rope_theta: f64,
    pub mrope_section: [usize; 3],
    pub vision_depth: usize,
    pub vision_hidden: usize,
    pub vision_intermediate: usize,
    pub vision_heads: usize,
    pub vision_patch: usize,
    pub vision_merge: usize,
    pub vision_temporal_patch: usize,
    pub vision_out_hidden: usize,
    pub vision_num_position_embeddings: usize,
    pub vision_deepstack: Vec<usize>,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_id: u32,
    pub vision_end_id: u32,
    /// H3 truncates text conditioning at this many tokens.
    pub text_len: usize,
}

impl Qwen3VlConfig {
    pub fn h3() -> Self {
        Self {
            vocab: 151936,
            hidden: 5120,
            intermediate: 25600,
            layers: 64,
            output_layer: 50,
            heads: 64,
            kv_heads: 8,
            head_dim: 128,
            rms_eps: 1e-6,
            rope_theta: 5_000_000.0,
            mrope_section: [24, 20, 20],
            vision_depth: 27,
            vision_hidden: 1152,
            vision_intermediate: 4304,
            vision_heads: 16,
            vision_patch: 16,
            vision_merge: 2,
            vision_temporal_patch: 2,
            vision_out_hidden: 5120,
            vision_num_position_embeddings: 2304,
            vision_deepstack: vec![8, 16, 24],
            image_token_id: 151655,
            video_token_id: 151656,
            vision_start_id: 151652,
            vision_end_id: 151653,
            text_len: 1024,
        }
    }

    pub fn tiny() -> Self {
        Self {
            vocab: 64,
            hidden: 32,
            intermediate: 48,
            layers: 3,
            output_layer: 2,
            heads: 4,
            kv_heads: 2,
            head_dim: 8,
            rms_eps: 1e-6,
            rope_theta: 5_000_000.0,
            mrope_section: [2, 1, 1],
            vision_depth: 2,
            vision_hidden: 16,
            vision_intermediate: 24,
            vision_heads: 2,
            vision_patch: 16,
            vision_merge: 2,
            vision_temporal_patch: 2,
            vision_out_hidden: 32,
            vision_num_position_embeddings: 16,
            vision_deepstack: vec![0, 1],
            image_token_id: 62,
            video_token_id: 63,
            vision_start_id: 60,
            vision_end_id: 61,
            text_len: 64,
        }
    }

    pub fn vision_head_dim(&self) -> usize {
        self.vision_hidden / self.vision_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn h3_is_33b_params() {
        let p = H3Config::h3().params();
        // The safetensors headers sum to 33.123 B.
        assert!((33_100_000_000..33_150_000_000).contains(&p), "{p}");
        let q = H3Config::h3().params_without_adaln();
        assert!((20_000_000_000..20_200_000_000).contains(&q), "{q}");
    }

    #[test]
    fn trim_and_omni_shapes() {
        let t = H3Config::fasth3_trim();
        assert_eq!(t.layers, 42);
        assert_eq!(t.kept_block_indices.as_ref().unwrap().len(), 42);
        let o = H3Config::fasth3_omni();
        assert_eq!(o.pdd_steps, Some(32));
        assert_eq!(H3Config::h3().rotary_dim(), 96);
        assert_eq!(H3Config::tiny().rotary_dim(), 12);
    }

    /// The tiny models form one pipeline (`h3 serve --synthetic`, the checker).
    #[test]
    fn tiny_models_agree() {
        let d = H3Config::tiny_vsa();
        assert_eq!(d.in_channels, VideoVaeConfig::tiny().latent_channels);
        assert_eq!(d.in_channels, TaeH3Config::tiny().latent_channels);
        assert_eq!(d.audio_in_channels, AudioVaeConfig::tiny().latent_channels);
        assert_eq!(
            H3Config::h3().in_channels,
            VideoVaeConfig::h3().latent_channels
        );
        assert_eq!(
            H3Config::h3().audio_in_channels,
            AudioVaeConfig::h3().latent_channels
        );
    }

    #[test]
    fn vae_chunking_constants() {
        let v = VideoVaeConfig::h3();
        assert_eq!((v.spatial_ratio(), v.temporal_ratio()), (16, 4));
        assert_eq!(v.frame_pre_padding(), 3);
        assert_eq!(v.tokens_chunk(), 5);
        assert_eq!(v.token_overlap(), 2);
        assert_eq!(v.frame_overlap(), 5);
        assert_eq!(v.decoder_rotary_dim(), 48);
        assert_eq!(AudioVaeConfig::h3().hop(), 800);
    }

    #[test]
    fn config_json_round_trip() {
        let j: serde_json::Value = serde_json::from_str(
            r#"{"hidden_size":5376,"num_attention_heads":56,"attention_head_dim":128,"num_layers":42,
            "num_refiner_layers":2,"ffn_dim":14336,"in_channels":24,"audio_in_channels":32,"patch_size":[1,2,2],
            "text_dim":5120,"freq_dim":256,"time_embed_hidden_dim":5376,"time_embed_dim":2688,"rope_freq_dim":16,
            "rope_theta":10000.0,"adaln_rank":16}"#,
        )
        .unwrap();
        let c = H3Config::from_json(&j).unwrap();
        assert_eq!(c.layers, 42);
        assert_eq!(c.adaln_rank, Some(16));
    }
}
