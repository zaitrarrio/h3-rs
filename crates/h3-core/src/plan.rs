//! Memory plan: what a generation keeps resident, for a given device, before
//! anything is allocated. H3's 33 B parameters do not fit a consumer card in
//! bf16 (66 GB), so the plan decides four things: the weight format, whether
//! the block weights are resident or streamed from host memory per step,
//! whether the AdaLN tables are resident or recomputed per step, and the FFN
//! token chunk — the only activation that scales with `ffn_dim`.

use crate::{
    config::{H3Config, VideoVaeConfig},
    packing::Layout,
    quant::WeightQuant,
};

#[derive(Debug, Clone, Copy)]
pub struct Device {
    pub budget_bytes: u64,
    /// Largest single buffer binding the device allows.
    pub max_binding_bytes: u64,
    /// Bytes per native weight element (2 for f16/bf16).
    pub weight_bytes: u64,
    /// Bytes per activation element (4 for f32, 2 for bf16).
    pub act_bytes: u64,
}

impl Device {
    pub fn gib(budget: f64, weight_bytes: u64, act_bytes: u64) -> Self {
        Self { budget_bytes: (budget * (1u64 << 30) as f64) as u64, max_binding_bytes: u64::MAX, weight_bytes, act_bytes }
    }
}

/// Bytes per weight element of a format (codes + scales).
pub fn bytes_per_weight(q: WeightQuant, native: u64) -> f64 {
    match q {
        WeightQuant::Native => native as f64,
        WeightQuant::Int8 => 1.0 + 4.0 / 128.0,
        WeightQuant::Int4 { group, .. } => 0.5 + 4.0 / group as f64,
        WeightQuant::Nvfp4 => 0.5 + 1.0 / 16.0,
    }
}

#[derive(Debug, Clone)]
pub struct Plan {
    pub tokens: usize,
    pub weights: WeightQuant,
    /// Bytes of one block's linears on the device.
    pub block_bytes: u64,
    /// Bytes of all resident block linears (0 when streamed).
    pub block_weights: u64,
    pub stream_blocks: bool,
    pub other_weights: u64,
    pub adaln_entries: usize,
    pub adaln_tables: u64,
    pub adaln_resident: bool,
    pub ffn_chunk: usize,
    pub activations: u64,
    pub decode_peak: u64,
    pub largest_binding: u64,
    pub peak: u64,
    pub fits: bool,
    pub notes: Vec<String>,
}

fn block_linear_params(cfg: &H3Config, gate: bool) -> u64 {
    let (d, qkv, f) = (cfg.hidden as u64, cfg.dim() as u64, cfg.ffn_dim as u64);
    3 * qkv * d + d * qkv + 2 * f * d + d * f + if gate { qkv * d } else { 0 }
}

/// Activation bytes of one step at `ffn_chunk` tokens.
fn step_acts(cfg: &H3Config, s: u64, chunk: u64, a: u64) -> u64 {
    let (d, qkv) = (cfg.hidden as u64, cfg.dim() as u64);
    let elems = s * d // residual stream
        + s * d // normed
        + s * 3 * qkv // fused qkv projection
        + 3 * s * qkv // q, k, v as [heads, S, d]
        + s * qkv // attention output
        + s * qkv // gate
        + s * d // block output
        + chunk * 2 * cfg.ffn_dim as u64 // FFN hidden, one chunk
        + 2 * s * cfg.video_patch_dim().max(cfg.text_dim) as u64; // in/out rows
    let rope = s * cfg.rotary_dim() as u64 * 4; // cos + sin f32, rot/2 each
    elems * a + rope + s * 4
}

#[allow(clippy::too_many_arguments)]
pub fn plan(
    cfg: &H3Config,
    vae: &VideoVaeConfig,
    layout: &Layout,
    unique_timesteps: usize,
    quant: WeightQuant,
    with_gate: bool,
    dev: &Device,
) -> Plan {
    let s = layout.sequence_length as u64;
    let a = dev.act_bytes;
    let d = cfg.hidden as u64;
    let mut notes = Vec::new();
    let block_bytes = (block_linear_params(cfg, with_gate) as f64 * bytes_per_weight(quant, dev.weight_bytes)) as u64;
    let adaln_in = cfg.adaln_rank.unwrap_or(cfg.time_embed_dim) as u64;
    let mods = (H3Config::MOD_SLOTS * H3Config::MODALITIES) as u64 * d;
    // Everything that is not a block linear or an AdaLN weight: refiner, norms,
    // boundary projections, time embedder, norm_out.
    let other_params = cfg.params_without_adaln() as u64 - cfg.layers as u64 * block_linear_params(cfg, false)
        + if with_gate { 0 } else { 0 };
    let other_weights = other_params * dev.weight_bytes + cfg.layers as u64 * mods * dev.weight_bytes;
    let adaln_entries = unique_timesteps * H3Config::MODALITIES;
    // f32 tables: [entries, 6, d] per block plus the final norm's [entries, 2, d].
    let adaln_tables = adaln_entries as u64 * (cfg.layers as u64 * mods + 2 * d) * 4;
    let _ = adaln_in;

    let all_blocks = cfg.layers as u64 * block_bytes;
    let mut chunk = s;
    let fixed_with = |blocks: u64, tables: u64| blocks + other_weights + tables;
    let fits_chunk = |fixed: u64, c: u64| {
        c * 2 * cfg.ffn_dim as u64 * a <= dev.max_binding_bytes && fixed + step_acts(cfg, s, c, a) <= dev.budget_bytes
    };
    // Prefer: resident blocks + resident tables; then stream tables; then
    // stream blocks (one block resident at a time, double-buffered).
    let (stream_blocks, adaln_resident, block_weights) = if fits_chunk(fixed_with(all_blocks, adaln_tables), 256) {
        (false, true, all_blocks)
    } else if fits_chunk(fixed_with(all_blocks, 0), 256) {
        notes.push("AdaLN tables recomputed on the host each step".into());
        (false, false, all_blocks)
    } else {
        notes.push(format!(
            "block weights streamed from host memory: {} MiB per block, {} blocks per step",
            block_bytes >> 20,
            cfg.layers
        ));
        (true, false, 2 * block_bytes)
    };
    let tables = if adaln_resident { adaln_tables } else { adaln_entries as u64 * (mods + 2 * d) * 4 };
    let fixed = fixed_with(block_weights, tables);
    while chunk > 256 && !fits_chunk(fixed, chunk) {
        chunk = (chunk / 2).div_ceil(256) * 256;
    }
    let activations = step_acts(cfg, s, chunk, a);
    if chunk < s {
        notes.push(format!("FFN runs in {} chunks of {chunk} tokens", s.div_ceil(chunk)));
    }

    // ViT decoder: a chunk of 7 latent frames at h/16 × w/16 tokens, 2048 wide,
    // 8192-wide FFN; output 4 frames × 16² px per token in f32.
    let dd = vae.decoder_dim() as u64;
    let toks = (vae.tokens_chunk() + vae.token_overlap()) as u64 * (layout.latent_height * layout.latent_width) as u64;
    let decoder_weights = (vae.decoder_layers as u64 * (4 * dd * dd + 3 * vae.decoder_ffn_mult as u64 * dd * dd)) * dev.weight_bytes;
    let decode_acts = toks * (dd * 4 + 2 * vae.decoder_ffn_mult as u64 * dd) * a
        + toks * (vae.in_channels * vae.temporal_ratio() * vae.spatial_ratio() * vae.spatial_ratio()) as u64 * 4 * 2;
    let decode_peak = decoder_weights + decode_acts;
    let largest_binding = (s * 3 * cfg.dim() as u64 * a)
        .max(chunk * 2 * cfg.ffn_dim as u64 * a)
        .max(block_bytes / 2)
        .max(toks * 2 * vae.decoder_ffn_mult as u64 * dd * a);
    if largest_binding > dev.max_binding_bytes {
        notes.push(format!(
            "largest buffer {} MiB exceeds the device binding cap {} MiB",
            largest_binding >> 20,
            dev.max_binding_bytes >> 20
        ));
    }
    let peak = (fixed + activations).max(if stream_blocks { decode_peak } else { block_weights + other_weights + decode_peak });
    Plan {
        tokens: s as usize,
        weights: quant,
        block_bytes,
        block_weights,
        stream_blocks,
        other_weights,
        adaln_entries,
        adaln_tables,
        adaln_resident,
        ffn_chunk: chunk as usize,
        activations,
        decode_peak,
        largest_binding,
        peak,
        fits: peak <= dev.budget_bytes && largest_binding <= dev.max_binding_bytes,
        notes,
    }
}

impl std::fmt::Display for Plan {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mib = |b: u64| b as f64 / (1u64 << 20) as f64;
        writeln!(f, "tokens          {}", self.tokens)?;
        writeln!(f, "weights         {} ({:.0} MiB per block)", self.weights, mib(self.block_bytes))?;
        writeln!(
            f,
            "block weights   {:>9.1} MiB{}",
            mib(self.block_weights),
            if self.stream_blocks { " (streamed, two resident)" } else { " (resident)" }
        )?;
        writeln!(f, "other weights   {:>9.1} MiB", mib(self.other_weights))?;
        writeln!(
            f,
            "AdaLN tables    {:>9.1} MiB ({} entries{})",
            mib(self.adaln_tables),
            self.adaln_entries,
            if self.adaln_resident { ", resident" } else { ", per step" }
        )?;
        writeln!(f, "activations     {:>9.1} MiB (FFN chunk {})", mib(self.activations), self.ffn_chunk)?;
        writeln!(f, "decode peak     {:>9.1} MiB", mib(self.decode_peak))?;
        writeln!(f, "largest buffer  {:>9.1} MiB", mib(self.largest_binding))?;
        writeln!(f, "peak            {:>9.1} MiB  fits: {}", mib(self.peak), self.fits)?;
        for n in &self.notes {
            writeln!(f, "note: {n}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::packing::{Geometry, TEXT_TAG, build_layout};

    fn layout_768p() -> Layout {
        let g = Geometry::H3_768P;
        let (t, h, w) = g.latent(16);
        build_layout(&vec![TEXT_TAG; 512], t, h, w, g.audio_latents(), (1, 2, 2), &[]).unwrap()
    }

    #[test]
    fn bf16_768p_fits_an_80gb_card_resident() {
        let p = plan(&H3Config::h3(), &VideoVaeConfig::h3(), &layout_768p(), 2, WeightQuant::Native, true, &Device::gib(80.0, 2, 2));
        assert!(p.fits, "{p}");
        assert!(!p.stream_blocks && p.adaln_resident, "{p}");
        assert_eq!(p.ffn_chunk, p.tokens);
    }

    #[test]
    fn int4_768p_fits_a_24gb_card() {
        let q = WeightQuant::Int4 { group: 64, hqq: true, ffn_only: false };
        let p = plan(&H3Config::h3(), &VideoVaeConfig::h3(), &layout_768p(), 2, q, true, &Device::gib(24.0, 2, 2));
        assert!(p.fits, "{p}");
        assert!(!p.stream_blocks, "{p}");
    }

    #[test]
    fn eight_gb_streams_blocks_and_chunks_the_ffn() {
        let mut dev = Device::gib(8.0, 2, 4);
        dev.max_binding_bytes = 1 << 30;
        let p = plan(&H3Config::h3(), &VideoVaeConfig::h3(), &layout_768p(), 2, WeightQuant::Int8, true, &dev);
        assert!(p.stream_blocks, "{p}");
        assert!(p.ffn_chunk < p.tokens, "{p}");
        assert!(p.ffn_chunk as u64 * 2 * 14336 * 4 <= 1 << 30);
    }
}
