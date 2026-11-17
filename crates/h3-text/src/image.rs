//! 8-bit RGB images and PIL-compatible resampling (the reference prepares
//! keyframes with `Image.resize(..., LANCZOS)` and the Qwen processor with
//! bicubic + antialias; both are separable filters whose support grows with
//! the downscale factor, which is what PIL and torchvision implement).

use h3_model::vae::Volume;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Image {
    pub width: usize,
    pub height: usize,
    /// Row-major RGB.
    pub rgb: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Filter {
    /// Keys cubic, `a = −0.5` (PIL `BICUBIC`).
    Bicubic,
    /// `sinc(x)·sinc(x/3)`, support 3 (PIL `LANCZOS`).
    Lanczos,
}

fn kernel(f: Filter, x: f32) -> f32 {
    match f {
        Filter::Bicubic => {
            let a = -0.5f32;
            let x = x.abs();
            if x < 1.0 {
                ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
            } else if x < 2.0 {
                (((x - 5.0) * x + 8.0) * x - 4.0) * a
            } else {
                0.0
            }
        }
        Filter::Lanczos => {
            let x = x.abs();
            if x < 3.0 {
                let sinc = |v: f32| {
                    if v == 0.0 {
                        1.0
                    } else {
                        (std::f32::consts::PI * v).sin() / (std::f32::consts::PI * v)
                    }
                };
                sinc(x) * sinc(x / 3.0)
            } else {
                0.0
            }
        }
    }
}

fn support(f: Filter) -> f32 {
    match f {
        Filter::Bicubic => 2.0,
        Filter::Lanczos => 3.0,
    }
}

/// PIL's precomputed coefficients for one axis: for each output pixel the
/// first input index and normalised weights.
fn coefficients(f: Filter, in_size: usize, out_size: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = in_size as f32 / out_size as f32;
    let filter_scale = scale.max(1.0);
    let sup = support(f) * filter_scale;
    (0..out_size)
        .map(|xx| {
            let center = (xx as f32 + 0.5) * scale;
            let xmin = ((center - sup + 0.5).floor().max(0.0)) as usize;
            let xmax = ((center + sup + 0.5).floor() as usize).min(in_size);
            let mut w: Vec<f32> = (xmin..xmax)
                .map(|x| kernel(f, (x as f32 - center + 0.5) / filter_scale))
                .collect();
            let s: f32 = w.iter().sum();
            if s != 0.0 {
                w.iter_mut().for_each(|v| *v /= s);
            }
            (xmin, w)
        })
        .collect()
}

impl Image {
    pub fn new(width: usize, height: usize, rgb: Vec<u8>) -> Self {
        assert_eq!(rgb.len(), width * height * 3);
        Self { width, height, rgb }
    }

    /// Separable resample to `(width, height)`, horizontal pass first (PIL).
    pub fn resize(&self, width: usize, height: usize, f: Filter) -> Image {
        if (width, height) == (self.width, self.height) {
            return self.clone();
        }
        let xs = coefficients(f, self.width, width);
        let mut tmp = vec![0.0f32; height.max(self.height) * width * 3];
        // Horizontal: rows of the source → width columns.
        for y in 0..self.height {
            for (xo, (x0, w)) in xs.iter().enumerate() {
                for c in 0..3 {
                    let mut acc = 0.0;
                    for (k, &wk) in w.iter().enumerate() {
                        acc += wk * self.rgb[(y * self.width + x0 + k) * 3 + c] as f32;
                    }
                    tmp[(y * width + xo) * 3 + c] = acc;
                }
            }
        }
        let ys = coefficients(f, self.height, height);
        let mut out = vec![0u8; width * height * 3];
        for (yo, (y0, w)) in ys.iter().enumerate() {
            for x in 0..width {
                for c in 0..3 {
                    let mut acc = 0.0;
                    for (k, &wk) in w.iter().enumerate() {
                        acc += wk * tmp[((y0 + k) * width + x) * 3 + c];
                    }
                    out[(yo * width + x) * 3 + c] = (acc + 0.5).clamp(0.0, 255.0) as u8;
                }
            }
        }
        Image {
            width,
            height,
            rgb: out,
        }
    }

    pub fn crop(&self, left: usize, top: usize, width: usize, height: usize) -> Image {
        let mut rgb = Vec::with_capacity(width * height * 3);
        for y in top..top + height {
            rgb.extend_from_slice(
                &self.rgb[(y * self.width + left) * 3..(y * self.width + left + width) * 3],
            );
        }
        Image { width, height, rgb }
    }

    /// `prepare_keyframe_image`: stretch to the canvas, or cover-scale and
    /// centre-crop (Lanczos).
    pub fn fit_canvas(&self, width: usize, height: usize, stretch: bool) -> Image {
        if (self.width, self.height) == (width, height) {
            return self.clone();
        }
        if stretch {
            return self.resize(width, height, Filter::Lanczos);
        }
        let scale = (width as f64 / self.width as f64).max(height as f64 / self.height as f64);
        let rw = width.max((self.width as f64 * scale).round() as usize);
        let rh = height.max((self.height as f64 * scale).round() as usize);
        let left = (rw - width) / 2;
        let top = (rh - height) / 2;
        self.resize(rw, rh, Filter::Lanczos)
            .crop(left, top, width, height)
    }

    /// `[3, 1, H, W]` in `[0, 1]`.
    pub fn to_volume(&self) -> Volume {
        let mut data = vec![0.0f32; 3 * self.width * self.height];
        for i in 0..self.width * self.height {
            for c in 0..3 {
                data[c * self.width * self.height + i] = self.rgb[i * 3 + c] as f32 / 255.0;
            }
        }
        Volume::from_data(3, 1, self.height, self.width, data)
    }

    /// Frames `[3, T, H, W]` in `[0, 1]`.
    pub fn frames_to_volume(frames: &[Image]) -> Volume {
        let (w, h, t) = (frames[0].width, frames[0].height, frames.len());
        let mut data = vec![0.0f32; 3 * t * h * w];
        for (f, im) in frames.iter().enumerate() {
            assert_eq!((im.width, im.height), (w, h));
            for i in 0..w * h {
                for c in 0..3 {
                    data[(c * t + f) * h * w + i] = im.rgb[i * 3 + c] as f32 / 255.0;
                }
            }
        }
        Volume::from_data(3, t, h, w, data)
    }

    /// Decode a PNG / JPEG / PPM file is the CLI's job (it links `image`);
    /// this helper reads binary PPM (P6), which needs no dependency.
    pub fn from_ppm(bytes: &[u8]) -> Option<Image> {
        let mut fields = Vec::new();
        let mut i = 0;
        while fields.len() < 4 && i < bytes.len() {
            while i < bytes.len() && bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < bytes.len() && bytes[i] == b'#' {
                while i < bytes.len() && bytes[i] != b'\n' {
                    i += 1;
                }
                continue;
            }
            let s = i;
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() {
                i += 1;
            }
            fields.push(std::str::from_utf8(&bytes[s..i]).ok()?.to_string());
        }
        if fields.len() < 4 || fields[0] != "P6" || fields[3] != "255" {
            return None;
        }
        let (w, h): (usize, usize) = (fields[1].parse().ok()?, fields[2].parse().ok()?);
        i += 1;
        let rgb = bytes.get(i..i + w * h * 3)?.to_vec();
        Some(Image {
            width: w,
            height: h,
            rgb,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_preserves_flat_colour_and_cover_crops() {
        let im = Image::new(8, 4, vec![100; 8 * 4 * 3]);
        let r = im.resize(4, 2, Filter::Lanczos);
        assert!(r.rgb.iter().all(|&v| v == 100));
        let b = im.resize(16, 8, Filter::Bicubic);
        assert!(b.rgb.iter().all(|&v| (v as i32 - 100).abs() <= 1));
        // Cover: 8×4 → 4×4 keeps the centre 4 columns.
        let mut rgb = vec![0u8; 8 * 4 * 3];
        for y in 0..4 {
            for x in 2..6 {
                rgb[(y * 8 + x) * 3] = 255;
            }
        }
        let im = Image::new(8, 4, rgb);
        let c = im.fit_canvas(4, 4, false);
        assert_eq!((c.width, c.height), (4, 4));
        assert!(c.rgb.chunks(3).all(|p| p[0] == 255));
        let v = c.to_volume();
        assert_eq!((v.c, v.t, v.h, v.w), (3, 1, 4, 4));
        assert_eq!(v.data[0], 1.0);
        let ppm = Image::from_ppm(b"P6\n2 1\n255\n\x01\x02\x03\x04\x05\x06").unwrap();
        assert_eq!(ppm.rgb, vec![1, 2, 3, 4, 5, 6]);
    }
}
