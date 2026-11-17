//! Pixel ↔ latent ↔ packed-token geometry of one H3 request (FastVideo
//! `pipelines/basic/minimax_h3/packing.py`, `pika_h3_flex/buckets.py`).
//!
//! H3 denoises one packed sequence
//!
//! ```text
//! [ text (+ vision placeholder) tokens | condition video rows | audio rows | video rows ]       T2VA / FL2VA
//! [ text | ordered reference rows … | continuation clip rows | target audio | target video ]  Ref2VA
//! ```
//!
//! Every row carries a modality tag (video 0, text 1, audio 2) that selects its
//! AdaLN table and a real-valued `(t, h, w)` rotary position. Positions are
//! computed in f64 exactly as the reference does; the rotary tables are then
//! formed in f32.

use crate::{Error, Result};

pub const VIDEO_TAG: u8 = 0;
pub const TEXT_TAG: u8 = 1;
pub const AUDIO_TAG: u8 = 2;

pub const FPS: usize = 24;
pub const SHORT_EDGE: usize = 768;
pub const MAX_PIXELS: usize = 768 * 1344;
pub const CANVAS_MULTIPLE: usize = 32;
pub const MIN_ASPECT: f64 = 0.25;
pub const MAX_ASPECT: f64 = 4.0;
pub const MIN_DURATION_S: f64 = 5.0;
pub const MAX_DURATION_S: f64 = 15.0;
/// The causal VAE's 17-frame clip gives 5 latent frames; a request has
/// `17n + 5` frames and `5n + 2` latent frames.
pub const FRAMES_PER_CHUNK: usize = 17;
pub const LATENTS_PER_CHUNK: usize = 5;
pub const AUDIO_LATENTS_PER_SECOND: usize = 40;
pub const AUDIO_CHANNELS: usize = 2;
/// Reference images are presented to the conditioner at this short edge.
pub const REFERENCE_IMAGE_SHORT_EDGE: usize = 2048;
pub const MAX_REFERENCE_IMAGES: usize = 9;
pub const MAX_REFERENCE_VIDEOS: usize = 3;
pub const MAX_REFERENCE_AUDIOS: usize = 3;
pub const MAX_REFERENCES: usize = 12;
/// A continuation clip ends on a chunk boundary: 2, 7 or 12 latent frames.
pub const CONTINUATION_FRAMES: [usize; 3] = [5, 22, 39];

const ROPE_FRAME_RESCALE: f64 = 5.0 / 3.0;
const ROPE_FRAMES_PER_LATENT: [f64; 5] = [1.0, 4.0, 4.0, 4.0, 4.0];
const ROPE_SPATIAL_SCALE: f64 = 32.0;

/// Canvas `(height, width)` for an aspect ratio under a tier's short-edge and
/// area rules, snapped to the 32-pixel lattice (the release's 768p rule by
/// default; Pika passes its tier values).
pub fn canvas_size(
    aspect_w: f64,
    aspect_h: f64,
    short_edge: usize,
    max_pixels: usize,
) -> Result<(usize, usize)> {
    if aspect_w <= 0.0 || aspect_h <= 0.0 {
        return Err(Error::Geometry(format!(
            "aspect ratio must be positive, got {aspect_w}:{aspect_h}"
        )));
    }
    let ratio = aspect_w / aspect_h;
    if !(MIN_ASPECT..=MAX_ASPECT).contains(&ratio) {
        return Err(Error::Geometry(format!(
            "aspect ratios run from 1:4 to 4:1, got {aspect_w}:{aspect_h}"
        )));
    }
    let (mut w, mut h) = if ratio >= 1.0 {
        (short_edge as f64 * ratio, short_edge as f64)
    } else {
        (short_edge as f64, short_edge as f64 / ratio)
    };
    let area = w * h;
    if area > max_pixels as f64 {
        let s = (max_pixels as f64 / area).sqrt();
        w *= s;
        h *= s;
    }
    let m = CANVAS_MULTIPLE as f64;
    let snap = |v: f64| ((v / m).round() * m).max(m) as usize;
    Ok((snap(h), snap(w)))
}

/// Round a frame count up to the `17n + 5` grid.
pub fn align_num_frames(mut n: usize) -> usize {
    if n == 0 {
        n = 1;
    }
    while n % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK {
        n += 1;
    }
    n
}

pub const MIN_ALIGNED_FRAMES: usize = 124;
pub const MAX_ALIGNED_FRAMES: usize = 362;

/// Validate a requested frame count: aligned up, 5–15 s (124–362 frames).
pub fn resolve_num_frames(n: usize) -> Result<usize> {
    let a = align_num_frames(n);
    if !(MIN_ALIGNED_FRAMES..=MAX_ALIGNED_FRAMES).contains(&a) {
        return Err(Error::Geometry(format!(
            "H3 generates 5–15 s at 24 fps; aligned num_frames {a} is outside 124–362"
        )));
    }
    Ok(a)
}

/// `(n − 5) / 17 · 5 + 2` latent frames.
pub fn video_latent_frames(n: usize) -> Result<usize> {
    if n % FRAMES_PER_CHUNK != LATENTS_PER_CHUNK {
        return Err(Error::Geometry(format!(
            "num_frames must be 17n + 5, got {n}"
        )));
    }
    Ok((n - LATENTS_PER_CHUNK) / FRAMES_PER_CHUNK * LATENTS_PER_CHUNK + 2)
}

/// Audio latents per channel: `round(frames / 24 · 40)` (Python's half-even round).
pub fn audio_latents(frames: usize) -> usize {
    let x = frames as f64 / FPS as f64 * AUDIO_LATENTS_PER_SECOND as f64;
    let r = x.round();
    if (x - x.trunc()).abs() == 0.5 && (r as i64) % 2 != 0 {
        (r - 1.0) as usize
    } else {
        r as usize
    }
}

/// Pixel geometry of a target clip.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Geometry {
    pub width: usize,
    pub height: usize,
    /// `17n + 5` frames.
    pub frames: usize,
}

impl Geometry {
    /// 1344×768, 124 frames: the release's 16:9 default (5.17 s).
    pub const H3_768P: Geometry = Geometry {
        width: 1344,
        height: 768,
        frames: 124,
    };
    /// 832×480, 124 frames: the 16:9 480p bucket.
    pub const H3_480P: Geometry = Geometry {
        width: 832,
        height: 480,
        frames: 124,
    };

    pub fn validate(&self, spatial_ratio: usize) -> Result<()> {
        if self.width == 0
            || self.height == 0
            || self.width % CANVAS_MULTIPLE != 0
            || self.height % CANVAS_MULTIPLE != 0
        {
            return Err(Error::Geometry(format!(
                "{}x{} is not a positive multiple of 32",
                self.width, self.height
            )));
        }
        if self.width % spatial_ratio != 0 || self.height % spatial_ratio != 0 {
            return Err(Error::Geometry(format!(
                "{}x{} is not divisible by the VAE ratio {spatial_ratio}",
                self.width, self.height
            )));
        }
        video_latent_frames(self.frames)?;
        Ok(())
    }

    /// Latent `(T, H, W)` for a VAE with `spatial_ratio` (16).
    pub fn latent(&self, spatial_ratio: usize) -> (usize, usize, usize) {
        (
            video_latent_frames(self.frames).expect("validated"),
            self.height / spatial_ratio,
            self.width / spatial_ratio,
        )
    }

    pub fn audio_latents(&self) -> usize {
        audio_latents(self.frames)
    }

    pub fn duration_s(&self) -> f64 {
        self.frames as f64 / FPS as f64
    }
}

/// Which keyframe a FL2VA condition frame anchors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Anchor {
    First,
    Last,
}

/// One prepared reference's latent geometry (Ref2VA).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefGeom {
    Image {
        latent_h: usize,
        latent_w: usize,
    },
    /// Latent frames, height, width, and the audio latents of its soundtrack (0 if silent).
    Video {
        latent_t: usize,
        latent_h: usize,
        latent_w: usize,
        audio_latents: usize,
    },
    Audio {
        audio_latents: usize,
    },
}

/// Kinds of conditioning spans recorded on a Ref2VA layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SegmentKind {
    Audio,
    Image,
    Video,
    Continuation,
}

/// One packed joint sequence and the geometry needed to interpret it.
#[derive(Debug, Clone, PartialEq)]
pub struct Layout {
    pub sequence_length: usize,
    /// `(t, h, w)` rotary positions per row.
    pub position_ids: Vec<[f64; 3]>,
    pub token_tags: Vec<u8>,
    /// Rows of the video stream, condition rows first, then the target.
    pub video_indices: Vec<u32>,
    /// Rows of the audio stream, condition rows first, then the target (channel-major).
    pub audio_indices: Vec<u32>,
    pub text_indices: Vec<u32>,
    pub num_condition_video_rows: usize,
    pub num_condition_audio_rows: usize,
    pub num_video_latent_frames: usize,
    pub latent_height: usize,
    pub latent_width: usize,
    pub num_audio_latents: usize,
    /// Ref2VA: `(kind, rows, (t, h, w) latent shape)` of every conditioning
    /// span between the text and the target audio, in packed order.
    pub reference_segments: Vec<(SegmentKind, usize, (usize, usize, usize))>,
}

impl Layout {
    pub fn num_text_tokens(&self) -> usize {
        self.text_indices.len()
    }
    pub fn num_target_video_rows(&self) -> usize {
        self.video_indices.len() - self.num_condition_video_rows
    }
    pub fn num_target_audio_rows(&self) -> usize {
        self.audio_indices.len() - self.num_condition_audio_rows
    }
    /// Target video rows as one `(t, h, w)` token grid.
    pub fn video_grid(&self, patch: (usize, usize, usize)) -> (usize, usize, usize) {
        (
            self.num_video_latent_frames / patch.0,
            self.latent_height / patch.1,
            self.latent_width / patch.2,
        )
    }
}

fn spatial_position_grid(dim: usize, patch: usize, sqrt_area: f64) -> Vec<f64> {
    let ratio = dim as f64 / sqrt_area;
    let left = (1.0 - ratio) / 2.0;
    let n = dim / patch;
    let step = ratio / n as f64;
    (0..n)
        .map(|i| (left + i as f64 * step) * ROPE_SPATIAL_SCALE)
        .collect()
}

/// Per-latent-frame rotary time: frame `i` spans `5/3 · [1, 4, 4, 4, 4][i mod 5]`.
pub fn temporal_position_grid(num_latent_frames: usize, origin: f64) -> Vec<f64> {
    let mut out = Vec::with_capacity(num_latent_frames);
    let mut acc = 0.0;
    for i in 0..num_latent_frames {
        out.push(origin + acc);
        acc += ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % 5];
    }
    out
}

fn temporal_span(num_latent_frames: usize) -> f64 {
    (0..num_latent_frames)
        .map(|i| ROPE_FRAME_RESCALE * ROPE_FRAMES_PER_LATENT[i % 5])
        .sum()
}

/// Frame grid `(h, w)` positions for every token of one latent frame, row-major,
/// and the width grid.
fn frame_grid(latent_h: usize, latent_w: usize, ph: usize, pw: usize) -> (Vec<[f64; 2]>, Vec<f64>) {
    let sqrt_area = ((latent_h * latent_w) as f64).sqrt();
    let hg = spatial_position_grid(latent_h, ph, sqrt_area);
    let wg = spatial_position_grid(latent_w, pw, sqrt_area);
    let mut g = Vec::with_capacity(hg.len() * wg.len());
    for &h in &hg {
        for &w in &wg {
            g.push([h, w]);
        }
    }
    (g, wg)
}

fn fill_audio(
    pos: &mut [[f64; 3]],
    start: usize,
    num_audio_latents: usize,
    time0: f64,
    wg: &[f64],
) {
    for ch in 0..AUDIO_CHANNELS {
        let wpos = if ch == 0 { wg[0] } else { wg[wg.len() - 1] };
        for i in 0..num_audio_latents {
            pos[start + ch * num_audio_latents + i] = [time0 + i as f64, 0.0, wpos];
        }
    }
}

fn check_text_tags(text_tags: &[u8]) -> Result<()> {
    if text_tags.iter().any(|&t| t != TEXT_TAG && t != VIDEO_TAG) {
        return Err(Error::Geometry(
            "text token tags may be text or vision only".into(),
        ));
    }
    Ok(())
}

/// T2VA / FL2VA: `[text | condition keyframes | audio | video]`.
pub fn build_layout(
    text_tags: &[u8],
    latent_t: usize,
    latent_h: usize,
    latent_w: usize,
    num_audio_latents: usize,
    patch: (usize, usize, usize),
    anchors: &[Anchor],
) -> Result<Layout> {
    check_text_tags(text_tags)?;
    if latent_t % patch.0 != 0 || latent_h % patch.1 != 0 || latent_w % patch.2 != 0 {
        return Err(Error::Geometry(format!(
            "latent ({latent_t}, {latent_h}, {latent_w}) is not divisible by the patch {patch:?}"
        )));
    }
    let rows_per_frame = (latent_h / patch.1) * (latent_w / patch.2);
    let n_text = text_tags.len();
    let n_cond = anchors.len() * rows_per_frame;
    let n_audio = num_audio_latents * AUDIO_CHANNELS;
    let n_video = (latent_t / patch.0) * rows_per_frame;
    let total = n_text + n_cond + n_audio + n_video;
    let cond_start = n_text;
    let audio_start = cond_start + n_cond;
    let video_start = audio_start + n_audio;

    let mut pos = vec![[0.0f64; 3]; total];
    for i in 0..n_text {
        pos[i] = [i as f64, 0.0, 0.0];
    }
    let (grid, wg) = frame_grid(latent_h, latent_w, patch.1, patch.2);
    for (k, a) in anchors.iter().enumerate() {
        let t = match a {
            Anchor::First => n_text as f64,
            Anchor::Last => n_text as f64 + temporal_span(latent_t) - ROPE_FRAME_RESCALE,
        };
        for (j, g) in grid.iter().enumerate() {
            pos[cond_start + k * rows_per_frame + j] = [t, g[0], g[1]];
        }
    }
    fill_audio(&mut pos, audio_start, num_audio_latents, n_text as f64, &wg);
    let times = temporal_position_grid(latent_t, n_text as f64);
    for (f, &t) in times.iter().enumerate() {
        for (j, g) in grid.iter().enumerate() {
            pos[video_start + f * rows_per_frame + j] = [t, g[0], g[1]];
        }
    }
    let mut tags = vec![0u8; total];
    tags[..n_text].copy_from_slice(text_tags);
    for i in cond_start..audio_start {
        tags[i] = VIDEO_TAG;
    }
    for i in audio_start..video_start {
        tags[i] = AUDIO_TAG;
    }
    for i in video_start..total {
        tags[i] = VIDEO_TAG;
    }
    let video_indices: Vec<u32> = (cond_start..audio_start)
        .chain(video_start..total)
        .map(|i| i as u32)
        .collect();
    Ok(Layout {
        sequence_length: total,
        position_ids: pos,
        token_tags: tags,
        video_indices,
        audio_indices: (audio_start..video_start).map(|i| i as u32).collect(),
        text_indices: (0..n_text as u32).collect(),
        num_condition_video_rows: n_cond,
        num_condition_audio_rows: 0,
        num_video_latent_frames: latent_t,
        latent_height: latent_h,
        latent_width: latent_w,
        num_audio_latents,
        reference_segments: Vec::new(),
    })
}

/// Ref2VA: `[text | ordered references | continuation clip | target audio | target video]`.
///
/// `continuation` is `(latent frames, audio latents)` of a clip that the target
/// regenerates in place: its rows take the target's first frame / audio
/// coordinates after advancing the rotary clock as a reference would.
pub fn build_ref2va_layout(
    text_tags: &[u8],
    refs: &[RefGeom],
    latent_t: usize,
    latent_h: usize,
    latent_w: usize,
    num_audio_latents: usize,
    patch: (usize, usize, usize),
    continuation: Option<(usize, usize)>,
) -> Result<Layout> {
    if patch != (1, 2, 2) {
        return Err(Error::Geometry(format!(
            "Ref2VA requires patch (1, 2, 2), got {patch:?}"
        )));
    }
    check_text_tags(text_tags)?;
    if refs.is_empty() {
        return Err(Error::Geometry(
            "Ref2VA requires at least one reference".into(),
        ));
    }
    if latent_t == 0 || latent_h % 2 != 0 || latent_w % 2 != 0 || num_audio_latents == 0 {
        return Err(Error::Geometry(
            "Ref2VA target geometry must be positive and patch-aligned".into(),
        ));
    }
    let n_text = text_tags.len();
    let rows_of = |t: usize, h: usize, w: usize| -> Result<usize> {
        if t == 0 || h == 0 || w == 0 || h % 2 != 0 || w % 2 != 0 {
            return Err(Error::Geometry(format!(
                "visual reference geometry ({t}, {h}, {w}) is not patch-aligned"
            )));
        }
        Ok(t * (h / 2) * (w / 2))
    };
    let rows_per_target_frame = (latent_h / 2) * (latent_w / 2);
    let n_target_video = latent_t * rows_per_target_frame;
    let n_target_audio = num_audio_latents * AUDIO_CHANNELS;
    let mut n_ref_video = 0;
    let mut n_ref_audio = 0;
    for r in refs {
        match r {
            RefGeom::Image {
                latent_h: h,
                latent_w: w,
            } => n_ref_video += rows_of(1, *h, *w)?,
            RefGeom::Video {
                latent_t: t,
                latent_h: h,
                latent_w: w,
                audio_latents,
            } => {
                n_ref_video += rows_of(*t, *h, *w)?;
                n_ref_audio += audio_latents * AUDIO_CHANNELS;
            }
            RefGeom::Audio { audio_latents } => {
                if *audio_latents == 0 {
                    return Err(Error::Geometry("an audio reference has no latents".into()));
                }
                n_ref_audio += audio_latents * AUDIO_CHANNELS;
            }
        }
    }
    if let Some((ct, ca)) = continuation {
        if ct == 0 || ct >= latent_t {
            return Err(Error::Geometry(format!(
                "continuation clip of {ct} latent frames does not fit a target of {latent_t}"
            )));
        }
        n_ref_video += ct * rows_per_target_frame;
        n_ref_audio += ca * AUDIO_CHANNELS;
    }
    let total = n_text + n_ref_video + n_ref_audio + n_target_audio + n_target_video;
    let mut pos = vec![[0.0f64; 3]; total];
    for i in 0..n_text {
        pos[i] = [i as f64, 0.0, 0.0];
    }
    let (target_grid, target_wg) = frame_grid(latent_h, latent_w, 2, 2);
    let mut video_indices: Vec<u32> = Vec::new();
    let mut audio_indices: Vec<u32> = Vec::new();
    let mut segments = Vec::new();
    let mut cursor = n_text;
    let mut clock = n_text as f64;
    for r in refs {
        match r {
            RefGeom::Image {
                latent_h: h,
                latent_w: w,
            } => {
                let n = rows_of(1, *h, *w)?;
                segments.push((SegmentKind::Image, n, (1, *h, *w)));
                let (grid, _) = frame_grid(*h, *w, 2, 2);
                for (j, g) in grid.iter().enumerate() {
                    pos[cursor + j] = [clock, g[0], g[1]];
                }
                video_indices.extend((cursor..cursor + n).map(|i| i as u32));
                cursor += n;
                clock += 1.0;
            }
            RefGeom::Audio { audio_latents } => {
                let n = audio_latents * AUDIO_CHANNELS;
                segments.push((SegmentKind::Audio, n, (0, 0, 0)));
                fill_audio(&mut pos, cursor, *audio_latents, clock, &target_wg);
                audio_indices.extend((cursor..cursor + n).map(|i| i as u32));
                cursor += n;
                clock += *audio_latents as f64;
            }
            RefGeom::Video {
                latent_t: t,
                latent_h: h,
                latent_w: w,
                audio_latents,
            } => {
                let na = audio_latents * AUDIO_CHANNELS;
                let nv = rows_of(*t, *h, *w)?;
                if na > 0 {
                    segments.push((SegmentKind::Audio, na, (0, 0, 0)));
                }
                segments.push((SegmentKind::Video, nv, (*t, *h, *w)));
                let (grid, wg) = frame_grid(*h, *w, 2, 2);
                if na > 0 {
                    fill_audio(&mut pos, cursor, *audio_latents, clock, &wg);
                    audio_indices.extend((cursor..cursor + na).map(|i| i as u32));
                    cursor += na;
                }
                let times = temporal_position_grid(*t, clock);
                for (f, &tt) in times.iter().enumerate() {
                    for (j, g) in grid.iter().enumerate() {
                        pos[cursor + f * grid.len() + j] = [tt, g[0], g[1]];
                    }
                }
                video_indices.extend((cursor..cursor + nv).map(|i| i as u32));
                cursor += nv;
                clock += (*audio_latents as f64).max(temporal_span(*t));
            }
        }
    }
    if let Some((ct, ca)) = continuation {
        let na = ca * AUDIO_CHANNELS;
        let nv = ct * rows_per_target_frame;
        if na > 0 {
            segments.push((SegmentKind::Audio, na, (0, 0, 0)));
        }
        segments.push((SegmentKind::Continuation, nv, (ct, latent_h, latent_w)));
        // The clip advances the clock like a reference, then takes the
        // target's own first coordinates.
        clock += (ca as f64).max(temporal_span(ct));
        if na > 0 {
            fill_audio(&mut pos, cursor, ca, clock, &target_wg);
            audio_indices.extend((cursor..cursor + na).map(|i| i as u32));
            cursor += na;
        }
        let times = temporal_position_grid(ct, clock);
        for (f, &tt) in times.iter().enumerate() {
            for (j, g) in target_grid.iter().enumerate() {
                pos[cursor + f * rows_per_target_frame + j] = [tt, g[0], g[1]];
            }
        }
        video_indices.extend((cursor..cursor + nv).map(|i| i as u32));
        cursor += nv;
    }
    let audio_start = cursor;
    let video_start = audio_start + n_target_audio;
    fill_audio(&mut pos, audio_start, num_audio_latents, clock, &target_wg);
    let times = temporal_position_grid(latent_t, clock);
    for (f, &tt) in times.iter().enumerate() {
        for (j, g) in target_grid.iter().enumerate() {
            pos[video_start + f * rows_per_target_frame + j] = [tt, g[0], g[1]];
        }
    }
    audio_indices.extend((audio_start..video_start).map(|i| i as u32));
    video_indices.extend((video_start..total).map(|i| i as u32));
    debug_assert_eq!(video_start + n_target_video, total);

    let mut tags = vec![0u8; total];
    tags[..n_text].copy_from_slice(text_tags);
    for &i in &audio_indices {
        tags[i as usize] = AUDIO_TAG;
    }
    for &i in &video_indices {
        tags[i as usize] = VIDEO_TAG;
    }
    Ok(Layout {
        sequence_length: total,
        position_ids: pos,
        token_tags: tags,
        video_indices,
        audio_indices,
        text_indices: (0..n_text as u32).collect(),
        num_condition_video_rows: n_ref_video,
        num_condition_audio_rows: n_ref_audio,
        num_video_latent_frames: latent_t,
        latent_height: latent_h,
        latent_width: latent_w,
        num_audio_latents,
        reference_segments: segments,
    })
}

/// Per-row clean times of one step: target video `t_v`, condition video
/// `max(t_v, 0.999)`, target audio `t_a`, condition audio `1.0` (f32). Returns
/// the sorted unique times and each row's index into them.
pub fn row_timesteps(
    layout: &Layout,
    video_t: f32,
    audio_t: f32,
    cond_video_t: f32,
    cond_audio_t: f32,
) -> (Vec<f32>, Vec<u32>) {
    let mut rows = vec![video_t; layout.sequence_length];
    for &i in &layout.video_indices[..layout.num_condition_video_rows] {
        rows[i as usize] = cond_video_t;
    }
    for &i in &layout.audio_indices[layout.num_condition_audio_rows..] {
        rows[i as usize] = audio_t;
    }
    for &i in &layout.audio_indices[..layout.num_condition_audio_rows] {
        rows[i as usize] = cond_audio_t;
    }
    let mut unique: Vec<f32> = rows.clone();
    unique.sort_by(|a, b| a.total_cmp(b));
    unique.dedup();
    let inverse = rows
        .iter()
        .map(|t| unique.iter().position(|u| u == t).unwrap() as u32)
        .collect();
    (unique, inverse)
}

/// Pika H3 Flex buckets: two released tiers on the same 32-pixel lattice.
pub mod pika {
    use super::{Error, Result, canvas_size};

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Tier {
        pub name: &'static str,
        pub short_edge: usize,
        pub max_pixels: usize,
        pub reference_short_edge: usize,
    }

    pub const P480: Tier = Tier {
        name: "p480",
        short_edge: 480,
        max_pixels: 832 * 480,
        reference_short_edge: 1024,
    };
    pub const P768: Tier = Tier {
        name: "p768",
        short_edge: 768,
        max_pixels: 1344 * 768,
        reference_short_edge: 2048,
    };

    /// Landscape families first, then square, then the mirrored portraits.
    pub const ASPECTS: [(u32, u32); 11] = [
        (21, 9),
        (16, 9),
        (3, 2),
        (4, 3),
        (5, 4),
        (1, 1),
        (4, 5),
        (3, 4),
        (2, 3),
        (9, 16),
        (9, 21),
    ];
    pub const MIN_ALIGNED_FRAMES: usize = 120;
    pub const MAX_ALIGNED_FRAMES: usize = 362;

    pub fn tier(name: &str) -> Result<Tier> {
        match name.trim().to_ascii_lowercase().as_str() {
            "480p" | "p480" | "480" => Ok(P480),
            "768p" | "p768" | "768" => Ok(P768),
            other => Err(Error::Geometry(format!(
                "Pika H3 Flex samples the 480p and 768p tiers only, got {other}"
            ))),
        }
    }

    /// `(height, width)` of one tier's bucket for an aspect family.
    pub fn canvas_for_aspect(aspect: (u32, u32), t: Tier) -> Result<(usize, usize)> {
        canvas_size(aspect.0 as f64, aspect.1 as f64, t.short_edge, t.max_pixels)
    }

    /// Parse `"16:9"`.
    pub fn parse_aspect(s: &str) -> Result<(u32, u32)> {
        let (w, h) = s
            .split_once(':')
            .ok_or_else(|| Error::Geometry(format!("aspect must be W:H, got {s}")))?;
        let (w, h): (u32, u32) = (
            w.trim()
                .parse()
                .map_err(|_| Error::Geometry(format!("bad aspect {s}")))?,
            h.trim()
                .parse()
                .map_err(|_| Error::Geometry(format!("bad aspect {s}")))?,
        );
        if w == 0 || h == 0 {
            return Err(Error::Geometry(format!("aspect must be positive, got {s}")));
        }
        Ok((w, h))
    }

    /// Resolve a request canvas to `(height, width, tier)`: an exact bucket is
    /// kept; otherwise the tier follows the short edge (≤ 480 → p480) and the
    /// canvas snaps to the nearest aspect family.
    pub fn resolve_canvas(
        height: Option<usize>,
        width: Option<usize>,
        default_aspect: (u32, u32),
        default_tier: Tier,
    ) -> Result<(usize, usize, Tier)> {
        match (height, width) {
            (None, None) => {
                let (h, w) = canvas_for_aspect(default_aspect, default_tier)?;
                Ok((h, w, default_tier))
            }
            (Some(h), Some(w)) => {
                if h == 0 || w == 0 {
                    return Err(Error::Geometry(format!(
                        "canvas must be positive, got {h}x{w}"
                    )));
                }
                for t in [P480, P768] {
                    for a in ASPECTS {
                        if canvas_for_aspect(a, t)? == (h, w) {
                            return Ok((h, w, t));
                        }
                    }
                }
                let t = if h.min(w) <= P480.short_edge {
                    P480
                } else {
                    P768
                };
                let ratio = w as f64 / h as f64;
                if !(super::MIN_ASPECT - 1e-9..=super::MAX_ASPECT + 1e-9).contains(&ratio) {
                    return Err(Error::Geometry(format!(
                        "Pika H3 Flex supports 1:4 to 4:1, got {w}x{h}"
                    )));
                }
                let mut best: Option<(f64, (usize, usize))> = None;
                for a in ASPECTS {
                    let d = (ratio / (a.0 as f64 / a.1 as f64)).ln().abs();
                    if best.is_none_or(|b| d < b.0) {
                        best = Some((d, canvas_for_aspect(a, t)?));
                    }
                }
                let (bh, bw) = best.unwrap().1;
                Ok((bh, bw, t))
            }
            _ => Err(Error::Geometry(
                "height and width must be passed together, or neither".into(),
            )),
        }
    }

    /// Round up to the `17n + 5` grid inside 5–15 s (120–362 frames).
    pub fn resolve_frames(n: usize) -> Result<usize> {
        let a = super::align_num_frames(n);
        if !(MIN_ALIGNED_FRAMES..=MAX_ALIGNED_FRAMES).contains(&a) {
            return Err(Error::Geometry(format!(
                "Pika H3 Flex generates 5–15 s at 24 fps; aligned num_frames {a} is outside 120–362"
            )));
        }
        Ok(a)
    }
}

/// Reference-image canvas at `short_edge` on the 32-pixel lattice.
pub fn reference_image_size(
    width: usize,
    height: usize,
    short_edge: usize,
) -> Result<(usize, usize)> {
    if width == 0 || height == 0 || width > 4 * height || height > 4 * width {
        return Err(Error::Geometry(format!(
            "a reference image must be within 1:4 and 4:1, got {width}x{height}"
        )));
    }
    let s = short_edge as f64 / width.min(height) as f64;
    let m = CANVAS_MULTIPLE as f64;
    let snap = |v: f64| ((v / m).round() * m).max(m) as usize;
    Ok((snap(height as f64 * s), snap(width as f64 * s)))
}

/// Trim a reference video to complete causal-VAE chunks.
pub fn trim_reference_frames(n: usize) -> usize {
    if n < 1 {
        return LATENTS_PER_CHUNK;
    }
    ((n.saturating_sub(LATENTS_PER_CHUNK)) / FRAMES_PER_CHUNK).max(1) * FRAMES_PER_CHUNK
        + LATENTS_PER_CHUNK
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_and_latent_counts() {
        assert_eq!(align_num_frames(120), 124);
        assert_eq!(align_num_frames(240), 243);
        assert_eq!(align_num_frames(360), 362);
        assert_eq!(video_latent_frames(124).unwrap(), 37);
        assert_eq!(video_latent_frames(362).unwrap(), 107);
        assert_eq!(audio_latents(124), 207);
        assert_eq!(audio_latents(243), 405);
        assert!(video_latent_frames(120).is_err());
        assert_eq!(resolve_num_frames(5 * 24).unwrap(), 124);
        assert!(resolve_num_frames(60).is_err());
    }

    #[test]
    fn canvas_rules() {
        assert_eq!(
            canvas_size(16.0, 9.0, SHORT_EDGE, MAX_PIXELS).unwrap(),
            (768, 1344)
        );
        assert_eq!(
            canvas_size(9.0, 16.0, SHORT_EDGE, MAX_PIXELS).unwrap(),
            (1344, 768)
        );
        assert_eq!(
            canvas_size(1.0, 1.0, SHORT_EDGE, MAX_PIXELS).unwrap(),
            (768, 768)
        );
        // 21:9 hits the area cap and shrinks.
        let (h, w) = canvas_size(21.0, 9.0, SHORT_EDGE, MAX_PIXELS).unwrap();
        assert!(h * w <= MAX_PIXELS + 32 * 32 && h % 32 == 0 && w % 32 == 0 && w > h);
        assert_eq!(
            pika::canvas_for_aspect((16, 9), pika::P480).unwrap(),
            (480, 832)
        );
        assert_eq!(
            pika::resolve_canvas(None, None, (16, 9), pika::P768).unwrap(),
            (768, 1344, pika::P768)
        );
        let (h, w, t) = pika::resolve_canvas(Some(500), Some(900), (16, 9), pika::P768).unwrap();
        assert_eq!((h, w, t.name), (768, 1344, "p768"));
        assert_eq!(pika::resolve_frames(120).unwrap(), 124);
        assert_eq!(
            reference_image_size(1920, 1080, 2048).unwrap(),
            (2048, 3648)
        );
        assert_eq!(trim_reference_frames(48), 39);
        assert_eq!(trim_reference_frames(130), 124);
    }

    #[test]
    fn t2va_layout_768p() {
        let g = Geometry::H3_768P;
        g.validate(16).unwrap();
        let (t, h, w) = g.latent(16);
        assert_eq!((t, h, w), (37, 48, 84));
        let text = vec![TEXT_TAG; 10];
        let l = build_layout(&text, t, h, w, g.audio_latents(), (1, 2, 2), &[]).unwrap();
        let video = 37 * 24 * 42;
        assert_eq!(l.sequence_length, 10 + 2 * 207 + video);
        assert_eq!(l.video_indices.len(), video);
        assert_eq!(l.audio_indices.len(), 414);
        assert_eq!(l.position_ids[3], [3.0, 0.0, 0.0]);
        // First audio row: time = num_text, w = width grid[0].
        let a0 = l.position_ids[l.audio_indices[0] as usize];
        assert_eq!(a0[0], 10.0);
        assert_eq!(a0[1], 0.0);
        // Second channel row k has the same time as the first channel row k.
        let a_r = l.position_ids[l.audio_indices[207] as usize];
        assert_eq!(a_r[0], a0[0]);
        assert!(a_r[2] > a0[2]);
        // Video frame 1 starts 5/3 later than frame 0, frame 2 4·5/3 after that.
        let v0 = l.position_ids[l.video_indices[0] as usize];
        let v1 = l.position_ids[l.video_indices[24 * 42] as usize];
        let v2 = l.position_ids[l.video_indices[2 * 24 * 42] as usize];
        assert_eq!(v0[0], 10.0);
        assert!((v1[0] - v0[0] - 5.0 / 3.0).abs() < 1e-12);
        assert!((v2[0] - v1[0] - 20.0 / 3.0).abs() < 1e-12);
        // Spatial grid: 16:9 → width ratio above 1, centred around 16.
        assert!((v0[1] + v0[2]) < 32.0);
        assert_eq!(l.token_tags[l.audio_indices[0] as usize], AUDIO_TAG);
        assert_eq!(l.token_tags[l.video_indices[0] as usize], VIDEO_TAG);
        assert_eq!(l.video_grid((1, 2, 2)), (37, 24, 42));
    }

    #[test]
    fn fl2va_anchors_and_row_timesteps() {
        let text = vec![TEXT_TAG, TEXT_TAG, VIDEO_TAG];
        let l = build_layout(
            &text,
            7,
            4,
            6,
            20,
            (1, 2, 2),
            &[Anchor::First, Anchor::Last],
        )
        .unwrap();
        assert_eq!(l.num_condition_video_rows, 2 * 2 * 3);
        let first = l.position_ids[l.video_indices[0] as usize][0];
        let last = l.position_ids[l.video_indices[6] as usize][0];
        // Upstream's "last" anchor is `origin + span − 5/3`, not the last
        // frame's own time (33 here): a checkpoint contract, kept as is.
        assert_eq!(first, 3.0);
        assert!(
            (last - (3.0 + 5.0 / 3.0 * 22.0 - 5.0 / 3.0)).abs() < 1e-9,
            "{last}"
        );
        let (u, inv) = row_timesteps(&l, 0.3, 0.6, 0.999, 1.0);
        assert_eq!(u, vec![0.3, 0.6, 0.999]);
        assert_eq!(inv[l.video_indices[0] as usize], 2);
        assert_eq!(inv[l.audio_indices[0] as usize], 1);
        assert_eq!(inv[*l.video_indices.last().unwrap() as usize], 0);
        assert_eq!(inv[0], 0);
    }

    #[test]
    fn ref2va_layout_orders_segments() {
        let text = vec![TEXT_TAG; 4];
        let refs = vec![
            RefGeom::Image {
                latent_h: 4,
                latent_w: 4,
            },
            RefGeom::Video {
                latent_t: 2,
                latent_h: 4,
                latent_w: 6,
                audio_latents: 3,
            },
            RefGeom::Audio { audio_latents: 5 },
        ];
        let l = build_ref2va_layout(&text, &refs, 7, 4, 6, 20, (1, 2, 2), None).unwrap();
        let img = 4;
        let vid = 2 * 2 * 3;
        assert_eq!(l.num_condition_video_rows, img + vid);
        assert_eq!(l.num_condition_audio_rows, (3 + 5) * 2);
        assert_eq!(l.reference_segments.len(), 4);
        assert_eq!(l.reference_segments[0].0, SegmentKind::Image);
        assert_eq!(l.reference_segments[1].0, SegmentKind::Audio);
        assert_eq!(l.reference_segments[2].0, SegmentKind::Video);
        assert_eq!(l.sequence_length, 4 + img + 6 + vid + 10 + 40 + 7 * 6);
        // Clock: image at 4 (+1), video audio/video at 5, span max(3, 5/3+20/3) = 25/3, audio ref at 5+25/3 (+5),
        // target at 5 + 25/3 + 5.
        let t_target = l.position_ids[l.audio_indices[l.num_condition_audio_rows] as usize][0];
        assert!(
            (t_target - (5.0 + 25.0 / 3.0 + 5.0)).abs() < 1e-9,
            "{t_target}"
        );
        let with_clip =
            build_ref2va_layout(&text, &refs, 7, 4, 6, 20, (1, 2, 2), Some((2, 3))).unwrap();
        assert_eq!(with_clip.num_condition_video_rows, img + vid + 2 * 6);
        assert_eq!(
            with_clip.reference_segments.last().unwrap().0,
            SegmentKind::Continuation
        );
        // The clip takes the target's first coordinates.
        let clip_first = with_clip.position_ids[with_clip.video_indices[img + vid] as usize];
        let target_first = with_clip.position_ids[with_clip.video_indices[img + vid + 12] as usize];
        assert_eq!(clip_first, target_first);
        assert!(build_ref2va_layout(&text, &[], 7, 4, 6, 20, (1, 2, 2), None).is_err());
    }
}
