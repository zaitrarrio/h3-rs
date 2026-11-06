//! The H3 prompt presentation: ordered reference labels and vision
//! placeholder blocks, then the prompt, each piece tokenised on its own with
//! `add_special_tokens=False` (`minimax_h3_conditioning.py`).

use crate::vision::{TEMPORAL_PATCH, VisionInput};
use h3_core::{
    Error, Result,
    packing::{FPS, TEXT_TAG, VIDEO_TAG},
};

pub const VISION_START: &str = "<|vision_start|>";
pub const VISION_END: &str = "<|vision_end|>";
pub const IMAGE_PAD: &str = "<|image_pad|>";
pub const VIDEO_PAD: &str = "<|video_pad|>";
/// Reference videos are shown to Qwen at 2 fps.
pub const QWEN_VIDEO_SAMPLE_FPS: f64 = 2.0;

pub struct Tokenizer {
    inner: tokenizers::Tokenizer,
    pub vision_start: u32,
    pub vision_end: u32,
    pub image_pad: u32,
    pub video_pad: u32,
}

impl Tokenizer {
    pub fn from_file(path: &std::path::Path) -> Result<Self> {
        let inner = tokenizers::Tokenizer::from_file(path).map_err(|e| Error::Msg(format!("tokenizer {}: {e}", path.display())))?;
        Self::from_inner(inner)
    }

    pub fn from_inner(inner: tokenizers::Tokenizer) -> Result<Self> {
        let id = |s: &str| inner.token_to_id(s).ok_or_else(|| Error::Msg(format!("tokenizer lacks `{s}`")));
        Ok(Self { vision_start: id(VISION_START)?, vision_end: id(VISION_END)?, image_pad: id(IMAGE_PAD)?, video_pad: id(VIDEO_PAD)?, inner })
    }

    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let enc = self.inner.encode(text, false).map_err(|e| Error::Msg(format!("tokenize: {e}")))?;
        Ok(enc.get_ids().to_vec())
    }
}

/// What a reference contributes to the presentation.
pub enum RefTokens<'a> {
    Image(&'a VisionInput),
    /// Sampled-frame input and its block timestamps (seconds).
    Video { input: &'a VisionInput, timestamps: &'a [f64], has_audio: bool },
    Audio,
}

/// Token ids and tags of a presentation; `vision` lists `(start row, merged
/// tokens, is_video, grid)` of every placeholder block in order.
pub struct Presentation {
    pub ids: Vec<u32>,
    pub tags: Vec<u8>,
    pub blocks: Vec<VisionBlock>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VisionBlock {
    /// Row of the first pad token.
    pub start: usize,
    pub tokens: usize,
    pub video: bool,
    /// `(t, h, w)` patch grid of the input this block came from.
    pub grid: (usize, usize, usize),
}

fn emit(p: &mut Presentation, ids: Vec<u32>, tag: u8) {
    p.tags.extend(std::iter::repeat_n(tag, ids.len()));
    p.ids.extend(ids);
}

fn vision(p: &mut Presentation, tok: &Tokenizer, pad: u32, count: usize, video: bool, grid: (usize, usize, usize)) {
    let start = p.ids.len() + 1;
    let mut ids = vec![tok.vision_start];
    ids.extend(std::iter::repeat_n(pad, count));
    ids.push(tok.vision_end);
    emit(p, ids, VIDEO_TAG);
    p.blocks.push(VisionBlock { start, tokens: count, video, grid });
}

/// FL2VA: `<Picture N>: ` + image block per keyframe, then the prompt.
pub fn present_fl2va(tok: &Tokenizer, prompt: &str, keyframes: &[VisionInput]) -> Result<Presentation> {
    let mut p = Presentation { ids: vec![], tags: vec![], blocks: vec![] };
    for (i, kf) in keyframes.iter().enumerate() {
        let ids = tok.encode(&format!("<Picture {}>: ", i + 1))?;
        emit(&mut p, ids, TEXT_TAG);
        vision(&mut p, tok, tok.image_pad, kf.merged_tokens(), false, kf.grid);
    }
    emit(&mut p, tok.encode(prompt)?, TEXT_TAG);
    Ok(p)
}

/// Ref2VA: `build_ref2va_presentation`.
pub fn present_ref2va(tok: &Tokenizer, prompt: &str, refs: &[RefTokens<'_>]) -> Result<Presentation> {
    let mut p = Presentation { ids: vec![], tags: vec![], blocks: vec![] };
    let (mut ni, mut nv, mut na) = (0, 0, 0);
    for r in refs {
        let has_audio = matches!(r, RefTokens::Audio | RefTokens::Video { has_audio: true, .. });
        if has_audio {
            na += 1;
            emit(&mut p, tok.encode(&format!("<Audio {na}>: "))?, TEXT_TAG);
        }
        match r {
            RefTokens::Image(v) => {
                ni += 1;
                emit(&mut p, tok.encode(&format!("<Picture {ni}>: "))?, TEXT_TAG);
                vision(&mut p, tok, tok.image_pad, v.merged_tokens(), false, v.grid);
            }
            RefTokens::Video { input, timestamps, .. } => {
                nv += 1;
                emit(&mut p, tok.encode(&format!("<Video {nv}>: "))?, TEXT_TAG);
                if timestamps.len() != input.grid.0 {
                    return Err(Error::Geometry(format!("{} block timestamps for {} Qwen video blocks", timestamps.len(), input.grid.0)));
                }
                let per_block = input.grid.1 * input.grid.2 / 4;
                for &ts in timestamps.iter() {
                    emit(&mut p, tok.encode(&format!("<{ts:.1} seconds>"))?, TEXT_TAG);
                    vision(&mut p, tok, tok.video_pad, per_block, true, (1, input.grid.1, input.grid.2));
                }
            }
            RefTokens::Audio => {}
        }
    }
    emit(&mut p, tok.encode(prompt)?, TEXT_TAG);
    Ok(p)
}

/// `sample_reference_video_frames`: indices of the 24-fps frames shown at 2
/// fps and the timestamp of every temporal block of two frames.
pub fn sample_frames(num_frames: usize) -> (Vec<usize>, Vec<f64>) {
    let stride = FPS as f64 / QWEN_VIDEO_SAMPLE_FPS;
    let mut idx: Vec<usize> = Vec::new();
    let mut cursor = 0.0f64;
    while (cursor.round() as usize) < num_frames {
        let r = cursor.round() as usize;
        if idx.last().is_none_or(|&l| r > l) {
            idx.push(r);
        }
        cursor += stride;
    }
    let mut ts: Vec<f64> = (0..idx.len()).map(|i| i as f64 / QWEN_VIDEO_SAMPLE_FPS).collect();
    while ts.len() % TEMPORAL_PATCH != 0 {
        ts.push(*ts.last().unwrap());
    }
    let blocks = ts.chunks(TEMPORAL_PATCH).map(|c| (c[0] + c[TEMPORAL_PATCH - 1]) / 2.0).collect();
    (idx, blocks)
}

#[cfg(test)]
pub(crate) fn test_tokenizer() -> Tokenizer {
    use tokenizers::models::bpe::BPE;
    use tokenizers::{AddedToken, Tokenizer as T};
    let chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 <>:.,";
    let vocab: tokenizers::models::bpe::Vocab = chars.chars().enumerate().map(|(i, c)| (c.to_string(), i as u32)).collect();
    let bpe = BPE::builder().vocab_and_merges(vocab, vec![]).unk_token("a".into()).build().unwrap();
    let mut t = T::new(bpe);
    let special: Vec<AddedToken> = [VISION_START, VISION_END, IMAGE_PAD, VIDEO_PAD].iter().map(|s| AddedToken::from(*s, true)).collect();
    t.add_special_tokens(special).unwrap();
    Tokenizer::from_inner(t).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_sampling() {
        let (idx, ts) = sample_frames(124);
        assert_eq!(idx[..3], [0, 12, 24]);
        assert_eq!(idx.len(), 11);
        assert_eq!(ts.len(), 6);
        assert_eq!(ts[0], 0.25);
        assert_eq!(*ts.last().unwrap(), 5.0);
    }

    #[test]
    fn presentation_layout() {
        let tok = test_tokenizer();
        let kf = VisionInput { patches: vec![], grid: (1, 4, 6) };
        let p = present_fl2va(&tok, "a cat", &[kf]).unwrap();
        let label = tok.encode("<Picture 1>: ").unwrap();
        assert_eq!(p.ids[..label.len()], label[..]);
        assert_eq!(p.blocks, vec![VisionBlock { start: label.len() + 1, tokens: 6, video: false, grid: (1, 4, 6) }]);
        assert_eq!(p.ids[label.len()], tok.vision_start);
        assert_eq!(p.ids[label.len() + 7], tok.vision_end);
        assert_eq!(p.tags[label.len()], VIDEO_TAG);
        assert_eq!(*p.tags.last().unwrap(), TEXT_TAG);
        assert_eq!(p.ids.len(), label.len() + 8 + tok.encode("a cat").unwrap().len());
        let vi = VisionInput { patches: vec![], grid: (2, 4, 4) };
        let r = present_ref2va(&tok, "p", &[RefTokens::Audio, RefTokens::Video { input: &vi, timestamps: &[0.25, 1.25], has_audio: true }]).unwrap();
        assert_eq!(r.blocks.len(), 2);
        assert!(r.blocks.iter().all(|b| b.video && b.tokens == 4));
        assert!(present_ref2va(&tok, "p", &[RefTokens::Video { input: &vi, timestamps: &[0.25], has_audio: false }]).is_err());
    }
}
