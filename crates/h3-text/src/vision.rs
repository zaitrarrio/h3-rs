//! Qwen3-VL vision preprocessing as the released processor configures it:
//! `smart_resize` onto the 32-pixel lattice (patch 16 × merge 2), bicubic
//! resampling, `(x/255 − 0.5)/0.5`, patches in merge-block order with the
//! temporal patch of 2 (an image is its own second frame).

use crate::image::{Filter, Image};
use h3_core::{Error, Result};

pub const PATCH: usize = 16;
pub const MERGE: usize = 2;
pub const TEMPORAL_PATCH: usize = 2;
pub const FACTOR: usize = PATCH * MERGE;
/// The released image processor's pixel bounds.
pub const IMAGE_MIN_PIXELS: usize = 65536;
pub const IMAGE_MAX_PIXELS: usize = 16_777_216;
/// The released video processor's bounds (over all sampled frames).
pub const VIDEO_MIN_PIXELS: usize = 4096;
pub const VIDEO_MAX_PIXELS: usize = 25_165_824;
pub const PATCH_FEATURES: usize = 3 * TEMPORAL_PATCH * PATCH * PATCH;

/// Python's round-half-even for the processor's `round(x / factor)`.
fn round_half_even(x: f64) -> f64 {
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && (r as i64) % 2 != 0 { r - x.signum() } else { r }
}

/// Qwen2-VL `smart_resize` for one image.
pub fn smart_resize(height: usize, width: usize, min_pixels: usize, max_pixels: usize) -> Result<(usize, usize)> {
    let (h, w, f) = (height as f64, width as f64, FACTOR as f64);
    if h.max(w) / h.min(w) > 200.0 {
        return Err(Error::Geometry("absolute aspect ratio must be below 200".into()));
    }
    let mut hb = round_half_even(h / f) * f;
    let mut wb = round_half_even(w / f) * f;
    if hb * wb > max_pixels as f64 {
        let beta = (h * w / max_pixels as f64).sqrt();
        hb = f.max((h / beta / f).floor() * f);
        wb = f.max((w / beta / f).floor() * f);
    } else if hb * wb < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        hb = (h * beta / f).ceil() * f;
        wb = (w * beta / f).ceil() * f;
    }
    Ok((hb as usize, wb as usize))
}

/// Qwen3-VL video `smart_resize`: the budget is over all frames.
pub fn smart_resize_video(frames: usize, height: usize, width: usize, min_pixels: usize, max_pixels: usize) -> Result<(usize, usize)> {
    if frames < TEMPORAL_PATCH {
        return Err(Error::Geometry("a video needs at least two frames".into()));
    }
    let f = FACTOR as f64;
    let (mut h, mut w) = (height as f64, width as f64);
    if h < f || w < f {
        let s = (f / h).max(f / w);
        h = (h * s).trunc();
        w = (w * s).trunc();
    }
    if h.max(w) / h.min(w) > 200.0 {
        return Err(Error::Geometry("absolute aspect ratio must be below 200".into()));
    }
    let mut hb = round_half_even(h / f) * f;
    let mut wb = round_half_even(w / f) * f;
    let tb = round_half_even(frames as f64 / TEMPORAL_PATCH as f64) * TEMPORAL_PATCH as f64;
    if tb * hb * wb > max_pixels as f64 {
        let beta = (frames as f64 * h * w / max_pixels as f64).sqrt();
        hb = f.max((h / beta / f).floor() * f);
        wb = f.max((w / beta / f).floor() * f);
    } else if tb * hb * wb < min_pixels as f64 {
        let beta = (min_pixels as f64 / (frames as f64 * h * w)).sqrt();
        hb = (h * beta / f).ceil() * f;
        wb = (w * beta / f).ceil() * f;
    }
    Ok((hb as usize, wb as usize))
}

/// One preprocessed visual input: patches `[tokens, 1536]` and its `(t, h, w)`
/// patch grid.
pub struct VisionInput {
    pub patches: Vec<f32>,
    pub grid: (usize, usize, usize),
}

impl VisionInput {
    pub fn tokens(&self) -> usize {
        self.grid.0 * self.grid.1 * self.grid.2
    }
    /// Placeholder tokens after the 2×2 merge.
    pub fn merged_tokens(&self) -> usize {
        self.tokens() / (MERGE * MERGE)
    }
}

fn normalized(im: &Image) -> Vec<f32> {
    // [3, H, W] in (x/255 − 0.5)/0.5
    let n = im.width * im.height;
    let mut v = vec![0.0f32; 3 * n];
    for i in 0..n {
        for c in 0..3 {
            v[c * n + i] = (im.rgb[i * 3 + c] as f32 / 255.0 - 0.5) / 0.5;
        }
    }
    v
}

/// Patches of `frames` (already resized to a lattice size), in merge-block
/// order with temporal pairs; a single image is paired with itself.
fn patchify(frames: &[Vec<f32>], height: usize, width: usize) -> VisionInput {
    let mut frames: Vec<&Vec<f32>> = frames.iter().collect();
    if frames.len() % TEMPORAL_PATCH != 0 {
        frames.push(frames[frames.len() - 1]);
    }
    let (gh, gw) = (height / PATCH, width / PATCH);
    let gt = frames.len() / TEMPORAL_PATCH;
    let n = height * width;
    let mut out = Vec::with_capacity(gt * gh * gw * PATCH_FEATURES);
    for t in 0..gt {
        for br in 0..gh / MERGE {
            for bc in 0..gw / MERGE {
                for mr in 0..MERGE {
                    for mc in 0..MERGE {
                        let (py0, px0) = ((br * MERGE + mr) * PATCH, (bc * MERGE + mc) * PATCH);
                        for c in 0..3 {
                            for tp in 0..TEMPORAL_PATCH {
                                let fr = frames[t * TEMPORAL_PATCH + tp];
                                for py in 0..PATCH {
                                    let row = (py0 + py) * width + px0;
                                    out.extend_from_slice(&fr[c * n + row..c * n + row + PATCH]);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    VisionInput { patches: out, grid: (gt, gh, gw) }
}

/// Preprocess one image (`Qwen2VLImageProcessor` with the released bounds).
pub fn preprocess_image(im: &Image) -> Result<VisionInput> {
    let (h, w) = smart_resize(im.height, im.width, IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS)?;
    let r = im.resize(w, h, Filter::Bicubic);
    Ok(patchify(&[normalized(&r)], h, w))
}

/// Preprocess sampled video frames (`Qwen3VLVideoProcessor`, no frame
/// resampling: the caller already picked the 2 fps frames).
pub fn preprocess_video(frames: &[Image]) -> Result<VisionInput> {
    let (h, w) = smart_resize_video(frames.len(), frames[0].height, frames[0].width, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS)?;
    let fr: Vec<Vec<f32>> = frames.iter().map(|f| normalized(&f.resize(w, h, Filter::Bicubic))).collect();
    Ok(patchify(&fr, h, w))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_rules_and_patch_layout() {
        assert_eq!(smart_resize(768, 1344, IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS).unwrap(), (768, 1344));
        assert_eq!(smart_resize(2048, 3648, IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS).unwrap(), (2048, 3648));
        assert_eq!(smart_resize(100, 100, IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS).unwrap(), (256, 256));
        assert_eq!(smart_resize(5000, 5000, IMAGE_MIN_PIXELS, IMAGE_MAX_PIXELS).unwrap(), (4096, 4096));
        // 24 frames of 1344×768 fit the video budget; 30 do not.
        assert_eq!(smart_resize_video(24, 768, 1344, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS).unwrap(), (768, 1344));
        let (h, w) = smart_resize_video(30, 768, 1344, VIDEO_MIN_PIXELS, VIDEO_MAX_PIXELS).unwrap();
        assert!(h < 768 && w < 1344 && h % 32 == 0 && w % 32 == 0);
        // 256×256: already on the lattice and above the 65536-pixel minimum, so
        // no resampling touches the values. Red = x / 4, green = y / 4.
        let mut rgb = vec![0u8; 256 * 256 * 3];
        for y in 0..256 {
            for x in 0..256 {
                rgb[(y * 256 + x) * 3] = (x / 4) as u8;
                rgb[(y * 256 + x) * 3 + 1] = (y / 4) as u8;
            }
        }
        let im = Image::new(256, 256, rgb);
        let v = preprocess_image(&im).unwrap();
        assert_eq!(v.grid, (1, 16, 16));
        assert_eq!(v.merged_tokens(), 64);
        assert_eq!(v.patches.len(), 256 * PATCH_FEATURES);
        let norm = |p: u8| (p as f32 / 255.0 - 0.5) / 0.5;
        // Token 1 is merge cell (0, 1) of block (0, 0): pixels x in 16..32, y in 0..16.
        // Feature 0 = channel 0, temporal 0, pixel (0, 0) → red = 16 / 4.
        let f = &v.patches[PATCH_FEATURES..2 * PATCH_FEATURES];
        assert_eq!(f[0], norm(4));
        // The temporal copy equals the first.
        assert_eq!(f[0], f[PATCH * PATCH]);
        // Token 2 is merge cell (1, 0): green (channel 1, temporal 0) at pixel (0, 0) = 16 / 4.
        let g = &v.patches[2 * PATCH_FEATURES..3 * PATCH_FEATURES];
        assert_eq!(g[TEMPORAL_PATCH * PATCH * PATCH], norm(4));
        // Token 4 is block (0, 1), cell (0, 0): x in 32..48 → red 8.
        assert_eq!(v.patches[4 * PATCH_FEATURES], norm(8));
        // A 64×64 image is upscaled to the 65536-pixel minimum.
        let small = Image::new(64, 64, vec![7; 64 * 64 * 3]);
        assert_eq!(preprocess_image(&small).unwrap().grid, (1, 16, 16));
    }
}
