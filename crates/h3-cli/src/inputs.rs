//! Conditioning inputs once decoded (keyframe images, reference images /
//! frames / soundtracks, the continuation clip) and their sizing to the
//! model's contracts: keyframes at the canvas, reference images at the
//! reference short edge, reference videos at H3's canvas rule, the
//! continuation at the canvas. `generate` fills them from files and `serve`
//! from a request's base64 fields; both hand [`prepare`]'s result to the
//! pipeline, so the two paths cannot drift apart.

use anyhow::{Result, bail};
use h3_core::{
    packing::{self, Anchor, Geometry},
    variant::Task,
};
use h3_model::{
    pipeline::{Continuation, Reference},
    vae::Volume,
};
use h3_text::{
    PresentedRef,
    image::{Filter, Image},
};

/// The shortest reference video: one full causal-VAE chunk after the first
/// latent frame (`trim_reference_frames` keeps `17n + 5` frames, n ≥ 1).
pub const MIN_REFERENCE_FRAMES: usize = packing::FRAMES_PER_CHUNK + packing::LATENTS_PER_CHUNK;

/// One Ref2VA reference as decoded.
pub enum RefInput {
    Image(Image),
    Video {
        frames: Vec<Image>,
        /// The container's frame rate when it states one (H3 assumes 24).
        fps: Option<f64>,
        audio: Option<[Vec<f32>; 2]>,
    },
    /// 32 kHz stereo.
    Audio([Vec<f32>; 2]),
}

impl RefInput {
    pub fn kind(&self) -> &'static str {
        match self {
            RefInput::Image(_) => "image",
            RefInput::Video { .. } => "video",
            RefInput::Audio(_) => "audio",
        }
    }
}

/// Everything a request conditions on besides its prompt.
#[derive(Default)]
pub struct Inputs {
    /// In anchor order.
    pub keyframes: Vec<(Anchor, Image)>,
    pub refs: Vec<RefInput>,
    /// 5 / 22 / 39 frames and an optional 32 kHz stereo soundtrack.
    pub continuation: Option<(Vec<Image>, Option<[Vec<f32>; 2]>)>,
}

impl Inputs {
    pub fn is_empty(&self) -> bool {
        self.keyframes.is_empty() && self.refs.is_empty() && self.continuation.is_none()
    }

    /// The task these inputs imply: references or a continuation make a
    /// Ref2VA run, keyframes an FL2VA run, nothing a T2VA run.
    pub fn implied_task(&self) -> Task {
        if !self.refs.is_empty() || self.continuation.is_some() {
            Task::Ref2va
        } else if !self.keyframes.is_empty() {
            Task::Fl2va
        } else {
            Task::T2va
        }
    }
}

impl std::fmt::Debug for Inputs {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Inputs")
            .field(
                "keyframes",
                &self.keyframes.iter().map(|(a, _)| *a).collect::<Vec<_>>(),
            )
            .field(
                "refs",
                &self.refs.iter().map(RefInput::kind).collect::<Vec<_>>(),
            )
            .field(
                "continuation",
                &self.continuation.as_ref().map(|(fr, a)| (fr.len(), a.is_some())),
            )
            .finish()
    }
}

/// How references are sized.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Sizing {
    /// The release contracts: images at `ref_short_edge` (2048 for H3,
    /// Pika's tier value), videos at H3's canvas rule (768 short edge).
    Release { ref_short_edge: usize },
    /// Everything at the target canvas: the tiny synthetic pipeline, whose
    /// VAE cannot afford a 2048-pixel reference.
    Canvas,
}

/// Inputs sized for the pipeline and for the conditioner.
pub struct Prepared {
    pub keyframes: Vec<(Anchor, Volume)>,
    /// The keyframes at the canvas, for the conditioner (FL2VA).
    pub kf_images: Vec<Image>,
    pub references: Vec<Reference>,
    /// The references as the conditioner sees them (Ref2VA).
    pub presented: Vec<PresentedRef>,
    pub continuation: Option<Continuation>,
}

/// Size the inputs for canvas `g`. `pika`: Pika's keyframe rule (the first
/// frame is letterboxed to the canvas, not stretched).
pub fn prepare(inputs: Inputs, g: Geometry, pika: bool, sizing: Sizing) -> Result<Prepared> {
    let mut keyframes = Vec::new();
    let mut kf_images = Vec::new();
    for (anchor, im) in inputs.keyframes {
        let stretch = anchor == Anchor::First && !pika;
        let im = im.fit_canvas(g.width, g.height, stretch);
        keyframes.push((anchor, im.to_volume()));
        kf_images.push(im);
    }
    let mut references = Vec::new();
    let mut presented = Vec::new();
    for r in inputs.refs {
        match r {
            RefInput::Image(im) => {
                let im = match sizing {
                    Sizing::Release { ref_short_edge } => {
                        let (h, w) =
                            packing::reference_image_size(im.width, im.height, ref_short_edge)?;
                        im.resize(w, h, Filter::Lanczos)
                    }
                    Sizing::Canvas => im.fit_canvas(g.width, g.height, false),
                };
                references.push(Reference::Image(im.to_volume()));
                presented.push(PresentedRef::Image(im));
            }
            RefInput::Video { frames, fps, audio } => {
                if frames.len() < MIN_REFERENCE_FRAMES {
                    bail!(
                        "a reference video needs at least {MIN_REFERENCE_FRAMES} frames (17n + 5 at 24 fps), got {}",
                        frames.len()
                    );
                }
                if fps.is_some_and(|f| (f - 24.0).abs() > 1e-6) {
                    eprintln!(
                        "warning: a reference video is {:.2} fps; H3 treats reference frames as 24 fps (resample with ffmpeg -r 24)",
                        fps.unwrap()
                    );
                }
                let n = packing::trim_reference_frames(frames.len().min(g.frames));
                let (h, w) = match sizing {
                    Sizing::Release { .. } => packing::canvas_size(
                        frames[0].width as f64,
                        frames[0].height as f64,
                        packing::SHORT_EDGE,
                        packing::MAX_PIXELS,
                    )?,
                    Sizing::Canvas => (g.height, g.width),
                };
                let fr: Vec<Image> = frames[..n]
                    .iter()
                    .map(|f| f.resize(w, h, Filter::Lanczos))
                    .collect();
                references.push(Reference::Video {
                    pixels: Image::frames_to_volume(&fr),
                    audio: audio.clone(),
                });
                presented.push(PresentedRef::Video {
                    frames: fr,
                    has_audio: audio.is_some(),
                });
            }
            RefInput::Audio(a) => {
                references.push(Reference::Audio(a));
                presented.push(PresentedRef::Audio);
            }
        }
    }
    let continuation = match inputs.continuation {
        Some((frames, audio)) => {
            if !packing::CONTINUATION_FRAMES.contains(&frames.len()) {
                bail!(
                    "a continuation clip has 5, 22 or 39 frames, got {}",
                    frames.len()
                );
            }
            let fr: Vec<Image> = frames
                .iter()
                .map(|f| f.fit_canvas(g.width, g.height, false))
                .collect();
            Some(Continuation {
                pixels: Image::frames_to_volume(&fr),
                audio,
            })
        }
        None => None,
    };
    Ok(Prepared {
        keyframes,
        kf_images,
        references,
        presented,
        continuation,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn im(w: usize, h: usize) -> Image {
        Image::new(w, h, vec![100; w * h * 3])
    }

    #[test]
    fn inputs_imply_their_task() {
        let mut i = Inputs::default();
        assert_eq!(i.implied_task(), Task::T2va);
        i.keyframes.push((Anchor::Last, im(4, 4)));
        assert_eq!(i.implied_task(), Task::Fl2va);
        i.refs.push(RefInput::Image(im(4, 4)));
        assert_eq!(i.implied_task(), Task::Ref2va);
        assert!(!i.is_empty());
    }

    #[test]
    fn prepare_sizes_everything_to_its_contract() {
        let g = Geometry {
            width: 64,
            height: 32,
            frames: 124,
        };
        let inputs = Inputs {
            keyframes: vec![(Anchor::First, im(10, 10)), (Anchor::Last, im(100, 20))],
            refs: vec![
                RefInput::Image(im(30, 20)),
                RefInput::Video {
                    frames: vec![im(8, 8); 24],
                    fps: Some(24.0),
                    audio: None,
                },
                RefInput::Audio([vec![0.0; 100], vec![0.0; 100]]),
            ],
            continuation: Some((vec![im(8, 6); 5], None)),
        };
        let p = prepare(inputs, g, false, Sizing::Canvas).unwrap();
        assert_eq!(p.keyframes.len(), 2);
        for (_, v) in &p.keyframes {
            assert_eq!((v.t, v.h, v.w), (1, 32, 64));
        }
        assert_eq!(p.kf_images.len(), 2);
        assert_eq!(p.references.len(), 3);
        match &p.references[0] {
            Reference::Image(v) => assert_eq!((v.h, v.w), (32, 64)),
            _ => panic!("image first"),
        }
        match &p.references[1] {
            // 24 frames trim to 17 + 5.
            Reference::Video { pixels, audio } => {
                assert_eq!((pixels.t, pixels.h, pixels.w), (22, 32, 64));
                assert!(audio.is_none());
            }
            _ => panic!("video second"),
        }
        // A reference video shorter than a chunk is refused, not trimmed up.
        let inputs = Inputs {
            refs: vec![RefInput::Video {
                frames: vec![im(8, 8); 7],
                fps: None,
                audio: None,
            }],
            ..Default::default()
        };
        assert!(prepare(inputs, g, false, Sizing::Canvas).is_err());
        assert!(matches!(p.presented[2], PresentedRef::Audio));
        let c = p.continuation.unwrap();
        assert_eq!((c.pixels.t, c.pixels.h, c.pixels.w), (5, 32, 64));

        // The release rule puts a reference image on the 32-pixel grid at the
        // short edge.
        let inputs = Inputs {
            refs: vec![RefInput::Image(im(30, 20))],
            ..Default::default()
        };
        let p = prepare(
            inputs,
            g,
            false,
            Sizing::Release {
                ref_short_edge: 64,
            },
        )
        .unwrap();
        match &p.references[0] {
            Reference::Image(v) => assert_eq!((v.h, v.w), (64, 96)),
            _ => panic!("image"),
        }
        // A continuation of the wrong length is refused.
        let inputs = Inputs {
            continuation: Some((vec![im(8, 6); 6], None)),
            ..Default::default()
        };
        assert!(prepare(inputs, g, false, Sizing::Canvas).is_err());
    }
}
