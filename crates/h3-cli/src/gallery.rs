//! A prompt gallery: Qwen3-VL hidden states computed once (`h3 encode`) so
//! the engine can run without the 64 GB conditioner (`h3 serve --gallery`).
//!
//! A directory with `gallery.json` and one safetensors file per prompt:
//!
//! ```text
//! gallery.json        {"prompts": [{"prompt": "...", "tokens": 33, "file": "01-….safetensors"}]}
//! 01-….safetensors    prompt_embeds [tokens, 5120] F16 (F32 if a value does not fit),
//!                     token_tags [tokens] U8; metadata {"prompt": "...", "tokens": "33"}
//! ```
//!
//! Only the prompt's real tokens are stored (~330 KB each in f16 for a
//! 33-token prompt). Galleries hold text-only prompts: keyframes and
//! references need the conditioner.

use anyhow::{Context, Result, bail};
use h3_core::{export, weights::SafetensorsDir, weights::WeightSource};
use h3_model::pipeline::Prompt;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

#[derive(Clone, Serialize, Deserialize)]
pub struct Entry {
    pub prompt: String,
    pub tokens: usize,
    pub file: String,
}

#[derive(Default, Serialize, Deserialize)]
struct Index {
    prompts: Vec<Entry>,
}

pub struct Gallery {
    dir: PathBuf,
    pub entries: Vec<Entry>,
}

/// Whitespace-insensitive key: the page may reflow a prompt.
pub fn norm(p: &str) -> String {
    p.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn slug(p: &str) -> String {
    let mut s = String::new();
    for w in p
        .split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|w| !w.is_empty())
    {
        if s.len() + w.len() > 40 {
            break;
        }
        if !s.is_empty() {
            s.push('-');
        }
        s.push_str(&w.to_ascii_lowercase());
    }
    if s.is_empty() { "prompt".into() } else { s }
}

impl Gallery {
    pub fn open(dir: &Path) -> Result<Self> {
        let path = dir.join("gallery.json");
        let idx: Index = serde_json::from_slice(
            &std::fs::read(&path).with_context(|| format!("read {}", path.display()))?,
        )
        .with_context(|| format!("parse {}", path.display()))?;
        if idx.prompts.is_empty() {
            bail!("{} lists no prompts", path.display());
        }
        for e in &idx.prompts {
            if !dir.join(&e.file).is_file() {
                bail!("{}: missing {}", path.display(), e.file);
            }
        }
        Ok(Self {
            dir: dir.to_path_buf(),
            entries: idx.prompts,
        })
    }

    pub fn find(&self, prompt: &str) -> Option<&Entry> {
        let k = norm(prompt);
        self.entries.iter().find(|e| norm(&e.prompt) == k)
    }

    pub fn prompt(&self, e: &Entry, text_dim: usize) -> Result<Prompt> {
        load_prompt(&self.dir.join(&e.file), text_dim)
    }

    /// Store one prompt and list it in `gallery.json`, replacing an entry with
    /// the same prompt.
    pub fn add(dir: &Path, prompt: &str, p: &Prompt, text_dim: usize) -> Result<Entry> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("gallery.json");
        let mut idx: Index = match std::fs::read(&path) {
            Ok(b) => {
                serde_json::from_slice(&b).with_context(|| format!("parse {}", path.display()))?
            }
            Err(_) => Index::default(),
        };
        let tokens = p.rows(text_dim);
        let rows = &p.embeds[..tokens * text_dim];
        let (dt, bytes) = if rows.iter().all(|v| v.is_finite() && v.abs() <= 65504.0) {
            (safetensors::Dtype::F16, export::f16_bytes(rows))
        } else {
            (safetensors::Dtype::F32, export::f32_bytes(rows))
        };
        let pos = idx
            .prompts
            .iter()
            .position(|e| norm(&e.prompt) == norm(prompt));
        let n = pos.map_or(idx.prompts.len() + 1, |i| i + 1);
        let file = format!("{n:02}-{}.safetensors", slug(prompt));
        export::write(
            &dir.join(&file),
            &[
                ("prompt_embeds".into(), dt, vec![tokens, text_dim], bytes),
                (
                    "token_tags".into(),
                    safetensors::Dtype::U8,
                    vec![tokens],
                    p.tags.clone(),
                ),
            ],
            [
                ("prompt".to_string(), prompt.to_string()),
                ("tokens".to_string(), tokens.to_string()),
            ]
            .into(),
        )?;
        let e = Entry {
            prompt: prompt.to_string(),
            tokens,
            file,
        };
        match pos {
            Some(i) => {
                let old = std::mem::replace(&mut idx.prompts[i], e.clone());
                if old.file != e.file {
                    let _ = std::fs::remove_file(dir.join(old.file));
                }
            }
            None => idx.prompts.push(e.clone()),
        }
        std::fs::write(&path, serde_json::to_string_pretty(&idx)?)?;
        Ok(e)
    }
}

/// `prompt_embeds [L, text_dim]` (+ optional `token_tags [L]`; all text when absent).
pub fn load_prompt(path: &Path, text_dim: usize) -> Result<Prompt> {
    let st = SafetensorsDir::open(path)?;
    let name = st
        .names()
        .find(|n| n.contains("embed"))
        .map(str::to_string)
        .context("no `prompt_embeds` tensor in the embeddings file")?;
    let (_, shape, _) = st.raw(&name)?;
    let rows = shape.iter().product::<usize>() / text_dim;
    if rows * text_dim != shape.iter().product::<usize>() {
        bail!("{name} is {shape:?}, not [L, {text_dim}]");
    }
    let embeds = st.get(&name, &shape)?;
    let tags = if st.contains("token_tags") {
        st.get("token_tags", &[rows])?
            .iter()
            .map(|v| *v as u8)
            .collect()
    } else {
        vec![h3_core::packing::TEXT_TAG; rows]
    };
    Ok(Prompt { embeds, tags })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_find_and_load_round_trip() {
        let dir = std::env::temp_dir().join(format!("h3-gallery-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let dim = 8;
        let mk = |n: usize| Prompt {
            embeds: (0..n * dim).map(|i| (i % 97) as f32 * 0.01 - 0.4).collect(),
            tags: (0..n).map(|i| (i % 2) as u8).collect(),
        };
        Gallery::add(&dir, "A red fox, in snow!", &mk(3), dim).unwrap();
        Gallery::add(&dir, "a  market", &mk(2), dim).unwrap();
        Gallery::add(&dir, "A red fox,   in snow!", &mk(4), dim).unwrap();
        let g = Gallery::open(&dir).unwrap();
        assert_eq!(g.entries.len(), 2);
        let e = g.find("A red fox, in\nsnow!").unwrap();
        assert_eq!(
            (e.tokens, e.file.as_str()),
            (4, "01-a-red-fox-in-snow.safetensors")
        );
        let got = g.prompt(e, dim).unwrap();
        assert_eq!(got.tags, vec![0, 1, 0, 1]);
        for (a, b) in got.embeds.iter().zip(&mk(4).embeds) {
            assert!((a - b).abs() <= 1e-3 * b.abs().max(1.0));
        }
        assert!(g.find("a dog").is_none());
        std::fs::remove_dir_all(&dir).ok();
    }
}
