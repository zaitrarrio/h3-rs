//! The Qwen3-VL conditioner's placement (`--encoder`, `--encoder-weights`)
//! and the prompt-embedding cache (`--prompt-cache`), shared by `serve`,
//! `generate`, `encode` and `regen2k`.
//!
//! Placement ([`h3_text::Placement`]): `stream` reads the 50 language layers
//! from the checkpoint for every prompt; `host` keeps a page-locked copy in
//! RAM and uploads it per prompt; `resident` keeps them on the device.
//! `auto` picks, for an engine that encodes many prompts (`serve`,
//! `encode`): resident when the device's free memory (after everything else
//! is loaded) holds the layers plus `H3_ENCODER_RESERVE_GB` (default 16) for
//! the generation's activations, else host when `MemAvailable` holds them
//! plus 16 GB, else stream; and stream for a one-prompt run (`generate`,
//! `regen2k`), where reading the layers once is all any placement does.
//!
//! The cache keeps each encoded prompt (f32, exact) under a key that hashes
//! the whitespace-normalised prompt, the tokenizer file, the text encoder's
//! files (names, sizes, safetensors headers), the output layer and the
//! weight format: a different encoder or format never reuses an entry.

use crate::gallery;
use anyhow::{Context, Result};
use clap::ValueEnum;
use h3_core::{Backend, config::Qwen3VlConfig, export, weights::SafetensorsDir};
use h3_model::pipeline::Prompt;
use h3_text::{EncoderWeights, Placement};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq, Debug)]
pub enum EncoderMode {
    Auto,
    Stream,
    Host,
    Resident,
    /// No text encoder (`serve`: gallery and cache prompts only).
    None,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq, Debug, Default)]
pub enum EncoderFormat {
    /// The checkpoint's bf16.
    #[default]
    Bf16,
    /// Int8 weight-only, half the bytes; moves the embeddings slightly.
    Int8,
}

impl EncoderFormat {
    pub fn weights(self) -> EncoderWeights {
        match self {
            EncoderFormat::Bf16 => EncoderWeights::Bf16,
            EncoderFormat::Int8 => EncoderWeights::Int8,
        }
    }
}

#[derive(clap::Args, Clone, Debug)]
pub struct EncoderArgs {
    /// Where Qwen3-VL's language layers live: auto (resident when the
    /// device has room, else a host copy when RAM has room, else stream;
    /// stream for one-prompt runs), stream, host, resident, or none.
    #[arg(long, value_enum)]
    pub encoder: Option<EncoderMode>,
    /// The language layers' format.
    #[arg(long, value_enum, default_value_t = EncoderFormat::Bf16)]
    pub encoder_weights: EncoderFormat,
    /// Keep encoded prompts here and reuse them (any later run with the same
    /// encoder, tokenizer and format).
    #[arg(long)]
    pub prompt_cache: Option<PathBuf>,
    /// The pre-quantized int8 layers (`h3 quantize-encoder`), read with
    /// `--encoder-weights int8` instead of quantizing the bf16 checkpoint at
    /// load (default: `<model-dir>/text_encoder_int8` when it exists).
    #[arg(long)]
    pub encoder_int8: Option<PathBuf>,
}

impl EncoderArgs {
    /// The int8 tree to read, if the format is int8 and one exists.
    pub fn int8_tree(&self, model_dir: &Path) -> Option<PathBuf> {
        if self.encoder_weights != EncoderFormat::Int8 {
            return None;
        }
        self.encoder_int8
            .clone()
            .or_else(|| Some(model_dir.join("text_encoder_int8")).filter(|d| d.is_dir()))
    }
}

fn mem_available() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let line = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

const GB: u64 = 1 << 30;

/// The placement for `mode` (`many`: the engine encodes many prompts).
pub fn choose<B: Backend>(
    be: &B,
    model_dir: &Path,
    mode: EncoderMode,
    format: EncoderFormat,
    many: bool,
) -> Result<Placement> {
    let pick = match mode {
        EncoderMode::Stream | EncoderMode::None => Placement::Stream,
        EncoderMode::Host => Placement::Host,
        EncoderMode::Resident => Placement::Resident,
        EncoderMode::Auto if !many => Placement::Stream,
        EncoderMode::Auto => {
            let src = SafetensorsDir::open(&model_dir.join("text_encoder"))?;
            let need = h3_text::language_bytes(&src, &Qwen3VlConfig::h3(), format.weights());
            let reserve = std::env::var("H3_ENCODER_RESERVE_GB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .unwrap_or(16)
                * GB;
            let free = be.memory_free().map(|f| f as u64);
            let ram = mem_available();
            let p = if free.is_some_and(|f| f >= need + reserve) {
                Placement::Resident
            } else if ram.is_some_and(|r| r >= need + 16 * GB) {
                Placement::Host
            } else {
                Placement::Stream
            };
            let gb =
                |v: Option<u64>| v.map_or("?".into(), |v| format!("{:.1}", v as f64 / GB as f64));
            eprintln!(
                "encoder auto: {:.1} GB of {:?} layers; device free {} GB (reserve {} GB), RAM available {} GB -> {p:?}",
                need as f64 / GB as f64,
                format,
                gb(free),
                reserve / GB,
                gb(ram),
            );
            p
        }
    };
    Ok(pick)
}

/// What identifies an encoder's output: tokenizer bytes, the text
/// encoder's files (names, sizes, headers), the output layer, the format.
pub fn identity(model_dir: &Path, tokenizer: &Path, format: EncoderFormat) -> Result<String> {
    let mut h = Sha256::new();
    h.update(b"h3-prompt-cache v1\n");
    h.update(std::fs::read(tokenizer).with_context(|| format!("read {}", tokenizer.display()))?);
    let dir = model_dir.join("text_encoder");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&dir)
        .with_context(|| format!("read {}", dir.display()))?
        .flatten()
        .map(|e| e.path())
        .filter(|p| {
            p.extension()
                .is_some_and(|e| e == "safetensors" || e == "json")
        })
        .collect();
    files.sort();
    for f in files {
        let meta = std::fs::metadata(&f)?;
        h.update(f.file_name().unwrap_or_default().as_encoded_bytes());
        h.update(meta.len().to_le_bytes());
        // The safetensors header (dtypes, shapes, offsets) or the json itself.
        use std::io::Read;
        let mut file = std::fs::File::open(&f)?;
        let mut head = vec![0u8; 8];
        if f.extension().is_some_and(|e| e == "safetensors") && file.read_exact(&mut head).is_ok() {
            let n = u64::from_le_bytes(head.clone().try_into().unwrap()).min(64 << 20) as usize;
            let mut hdr = vec![0u8; n];
            file.read_exact(&mut hdr)?;
            h.update(&hdr);
        } else {
            let mut all = Vec::new();
            std::fs::File::open(&f)?.read_to_end(&mut all)?;
            h.update(&all);
        }
    }
    h.update(Qwen3VlConfig::h3().output_layer.to_le_bytes());
    h.update(format!("{format:?}").as_bytes());
    Ok(hex(&h.finalize()))
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Encoded prompts by (identity, prompt): the last few in memory, all of
/// them on disk when a directory is given.
pub struct PromptCache {
    dir: Option<PathBuf>,
    identity: String,
    text_dim: usize,
    mem: Mutex<Vec<(String, Prompt)>>,
}

const MEM_ENTRIES: usize = 32;

impl PromptCache {
    pub fn new(dir: Option<PathBuf>, identity: String, text_dim: usize) -> Result<Self> {
        if let Some(d) = &dir {
            std::fs::create_dir_all(d).with_context(|| format!("create {}", d.display()))?;
        }
        Ok(Self {
            dir,
            identity,
            text_dim,
            mem: Mutex::new(Vec::new()),
        })
    }

    fn key(&self, prompt: &str) -> String {
        let mut h = Sha256::new();
        h.update(self.identity.as_bytes());
        h.update(b"\n");
        h.update(gallery::norm(prompt).as_bytes());
        hex(&h.finalize())
    }

    fn path(&self, key: &str) -> Option<PathBuf> {
        self.dir
            .as_ref()
            .map(|d| d.join(format!("{key}.safetensors")))
    }

    pub fn get(&self, prompt: &str) -> Option<Prompt> {
        let key = self.key(prompt);
        {
            let mut mem = self.mem.lock().unwrap();
            if let Some(i) = mem.iter().position(|(k, _)| *k == key) {
                let e = mem.remove(i);
                let p = e.1.clone();
                mem.push(e);
                return Some(p);
            }
        }
        let p = gallery::load_prompt(&self.path(&key)?, self.text_dim).ok()?;
        self.remember(key, &p);
        Some(p)
    }

    fn remember(&self, key: String, p: &Prompt) {
        let mut mem = self.mem.lock().unwrap();
        mem.retain(|(k, _)| *k != key);
        mem.push((key, p.clone()));
        if mem.len() > MEM_ENTRIES {
            mem.remove(0);
        }
    }

    pub fn put(&self, prompt: &str, p: &Prompt) -> Result<()> {
        let key = self.key(prompt);
        if let Some(path) = self.path(&key) {
            let tokens = p.rows(self.text_dim);
            let tmp = path.with_extension(format!("tmp-{}", std::process::id()));
            export::write(
                &tmp,
                &[
                    (
                        "prompt_embeds".into(),
                        safetensors::Dtype::F32,
                        vec![tokens, self.text_dim],
                        export::f32_bytes(&p.embeds[..tokens * self.text_dim]),
                    ),
                    (
                        "token_tags".into(),
                        safetensors::Dtype::U8,
                        vec![tokens],
                        p.tags[..tokens].to_vec(),
                    ),
                ],
                [("prompt".to_string(), prompt.to_string())].into(),
            )?;
            std::fs::rename(&tmp, &path)?;
        }
        self.remember(key, p);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cache_round_trips_and_keys_on_identity_and_prompt() {
        let dir = std::env::temp_dir().join(format!("h3-pcache-{}", std::process::id()));
        let dim = 4;
        let p = Prompt {
            embeds: (0..3 * dim).map(|i| i as f32 * 0.25 - 1.0).collect(),
            tags: vec![h3_core::packing::TEXT_TAG; 3],
        };
        let c = PromptCache::new(Some(dir.clone()), "id-a".into(), dim).unwrap();
        assert!(c.get("a fox").is_none());
        c.put("a fox", &p).unwrap();
        // Whitespace does not matter, the identity does.
        let hit = c.get("a   fox ").unwrap();
        assert_eq!((hit.embeds, hit.tags), (p.embeds.clone(), p.tags.clone()));
        let fresh = PromptCache::new(Some(dir.clone()), "id-a".into(), dim).unwrap();
        assert_eq!(fresh.get("a fox").unwrap().embeds, p.embeds, "from disk");
        let other = PromptCache::new(Some(dir.clone()), "id-b".into(), dim).unwrap();
        assert!(other.get("a fox").is_none());
        assert!(c.get("a cat").is_none());
        // Memory only.
        let m = PromptCache::new(None, "id-a".into(), dim).unwrap();
        m.put("x", &p).unwrap();
        assert!(m.get("x").is_some());
        std::fs::remove_dir_all(&dir).ok();
    }
}
