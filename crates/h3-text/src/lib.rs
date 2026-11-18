//! Prompt → Qwen3-VL hidden states, natively: the conditioner MiniMax H3
//! packs as its text rows. Text-only prompts, FL2VA keyframes and Ref2VA
//! references are presented exactly as the reference pipeline does
//! ([`present`]); images go through the Qwen3-VL processor rules
//! ([`vision`]); the model runs on any backend with its language layers
//! streamed from the memory-mapped checkpoint ([`qwen3vl`]).

pub mod image;
pub mod present;
pub mod qwen3vl;
pub mod vision;

use h3_core::backend::Backend;
use h3_core::{Result, config::Qwen3VlConfig, packing::TEXT_TAG, weights::SafetensorsDir};
use h3_model::pipeline::Prompt;
use image::Image;
use present::{Presentation, RefTokens, Tokenizer};
use std::path::Path;
use vision::VisionInput;

pub use qwen3vl::{EncoderWeights, Placement, Qwen3Vl, language_bytes, write_int8_tree};

/// A reference as the presentation needs it: pixels already prepared.
pub enum PresentedRef {
    /// Reference image at its canvas (short edge 2048 for the release).
    Image(Image),
    /// 24-fps frames at the target canvas and whether a soundtrack exists.
    Video {
        frames: Vec<Image>,
        has_audio: bool,
    },
    Audio,
}

pub struct Conditioner<B: Backend> {
    pub tokenizer: Tokenizer,
    pub model: Qwen3Vl<B>,
}

impl<B: Backend> Conditioner<B> {
    /// `tokenizer`: `tokenizer/tokenizer.json`; `text_encoder`: the checkpoint's
    /// `text_encoder/` directory.
    pub fn open(
        be: &B,
        tokenizer: &Path,
        text_encoder: &Path,
        cfg: Qwen3VlConfig,
        placement: Placement,
        weights: EncoderWeights,
        int8_tree: Option<&Path>,
        vision: bool,
    ) -> Result<Self> {
        let tree = match int8_tree {
            Some(d) => Some(SafetensorsDir::open(d)?),
            None => None,
        };
        Ok(Self {
            tokenizer: Tokenizer::from_file(tokenizer)?,
            model: Qwen3Vl::open(
                be,
                cfg,
                SafetensorsDir::open(text_encoder)?,
                placement,
                weights,
                tree,
                vision,
            )?,
        })
    }

    fn finish(&self, be: &B, p: &Presentation, inputs: &[&VisionInput]) -> Result<Prompt> {
        let enc = self.model.encode(be, p, inputs)?;
        Ok(Prompt {
            embeds: enc.embeds,
            tags: p.tags.clone(),
        })
    }

    /// T2VA: the prompt alone.
    pub fn encode_text(&self, be: &B, prompt: &str) -> Result<Prompt> {
        let p = present::present_fl2va(&self.tokenizer, prompt, &[])?;
        debug_assert!(p.tags.iter().all(|&t| t == TEXT_TAG));
        self.finish(be, &p, &[])
    }

    /// FL2VA: keyframes at the canvas, in anchor order.
    pub fn encode_fl2va(&self, be: &B, prompt: &str, keyframes: &[Image]) -> Result<Prompt> {
        let inputs: Vec<VisionInput> = keyframes
            .iter()
            .map(vision::preprocess_image)
            .collect::<Result<_>>()?;
        let p = present::present_fl2va(&self.tokenizer, prompt, &inputs)?;
        let refs: Vec<&VisionInput> = inputs.iter().collect();
        self.finish(be, &p, &refs)
    }

    /// Ref2VA: ordered references.
    pub fn encode_ref2va(&self, be: &B, prompt: &str, refs: &[PresentedRef]) -> Result<Prompt> {
        let mut inputs: Vec<VisionInput> = Vec::new();
        let mut stamps: Vec<Vec<f64>> = Vec::new();
        for r in refs {
            match r {
                PresentedRef::Image(im) => inputs.push(vision::preprocess_image(im)?),
                PresentedRef::Video { frames, .. } => {
                    let (idx, ts) = present::sample_frames(frames.len());
                    let sampled: Vec<Image> = idx.iter().map(|&i| frames[i].clone()).collect();
                    inputs.push(vision::preprocess_video(&sampled)?);
                    stamps.push(ts);
                }
                PresentedRef::Audio => {}
            }
        }
        let mut ii = 0;
        let mut si = 0;
        let mut toks: Vec<RefTokens<'_>> = Vec::new();
        for r in refs {
            match r {
                PresentedRef::Image(_) => {
                    toks.push(RefTokens::Image(&inputs[ii]));
                    ii += 1;
                }
                PresentedRef::Video { has_audio, .. } => {
                    toks.push(RefTokens::Video {
                        input: &inputs[ii],
                        timestamps: &stamps[si],
                        has_audio: *has_audio,
                    });
                    ii += 1;
                    si += 1;
                }
                PresentedRef::Audio => toks.push(RefTokens::Audio),
            }
        }
        let p = present::present_ref2va(&self.tokenizer, prompt, &toks)?;
        let refs: Vec<&VisionInput> = inputs.iter().collect();
        self.finish(be, &p, &refs)
    }
}
