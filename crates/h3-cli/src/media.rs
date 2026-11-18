//! Media in and out without ffmpeg in the loop: PNG / JPEG / PPM images,
//! Y4M video (`ffmpeg -i clip.mp4 clip.y4m` converts anything), PCM WAV audio.
//! `ffmpeg -i out.y4m -i out.wav -c:v libx264 -pix_fmt yuv420p -c:a aac out.mp4`
//! muxes a finished clip.

use anyhow::{Context, Result, bail};
use h3_text::image::Image;
use std::io::{Read, Write};
use std::path::Path;

pub fn read_image(path: &Path) -> Result<Image> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    decode_image(&bytes, None).with_context(|| format!("decode {}", path.display()))
}

/// Pixels per image a decoder accepts from an untrusted source (`h3 serve`):
/// 8192×4096, far above any canvas or reference contract.
pub const MAX_DECODED_PIXELS: u64 = 8192 * 4096;

/// Decode PNG / JPEG / PPM bytes; `format` pins the decoder (a request's
/// MIME type) instead of sniffing, and the dimensions are checked against
/// [`MAX_DECODED_PIXELS`] before any pixel is decoded.
pub fn decode_image(bytes: &[u8], format: Option<image::ImageFormat>) -> Result<Image> {
    if format.is_none()
        && let Some(im) = Image::from_ppm(bytes)
    {
        return Ok(im);
    }
    let reader = || {
        let mut r = image::ImageReader::new(std::io::Cursor::new(bytes));
        match format {
            Some(f) => {
                r.set_format(f);
                Ok(r)
            }
            None => r.with_guessed_format().context("image format"),
        }
    };
    let (w, h) = reader()?.into_dimensions().context("image header")?;
    if w == 0 || h == 0 || w as u64 * h as u64 > MAX_DECODED_PIXELS {
        bail!("{w}x{h} image: the limit is {MAX_DECODED_PIXELS} pixels");
    }
    let im = reader()?.decode().context("decode image")?.to_rgb8();
    Ok(Image::new(
        im.width() as usize,
        im.height() as usize,
        im.into_raw(),
    ))
}

#[allow(dead_code)]
pub fn write_png(path: &Path, im: &Image) -> Result<()> {
    let buf = image::RgbImage::from_raw(im.width as u32, im.height as u32, im.rgb.clone())
        .context("png buffer")?;
    buf.save(path)
        .with_context(|| format!("write {}", path.display()))
}

/// Frames of a `.y4m` file or of a directory of images (sorted by name), and
/// the frame rate when the file states one.
pub fn read_frames(path: &Path) -> Result<(Vec<Image>, Option<f64>)> {
    if path.is_dir() {
        let mut files: Vec<_> = std::fs::read_dir(path)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| {
                p.extension()
                    .is_some_and(|e| matches!(e.to_str(), Some("png" | "jpg" | "jpeg" | "ppm")))
            })
            .collect();
        files.sort();
        if files.is_empty() {
            bail!("{} holds no png/jpg/ppm frames", path.display());
        }
        let frames = files
            .iter()
            .map(|f| read_image(f))
            .collect::<Result<Vec<_>>>()?;
        return Ok((frames, None));
    }
    if path.extension().is_some_and(|e| e == "y4m") {
        return read_y4m(path);
    }
    let im = read_image(path)?;
    Ok((vec![im], None))
}

fn read_y4m(path: &Path) -> Result<(Vec<Image>, Option<f64>)> {
    let data = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    decode_y4m(&data).with_context(|| format!("{}", path.display()))
}

/// The frames of a YUV4MPEG2 stream (4:2:0 / 4:2:2 / 4:4:4, limited or
/// full range) as RGB, and its frame rate when the header states one.
pub fn decode_y4m(data: &[u8]) -> Result<(Vec<Image>, Option<f64>)> {
    let header = data
        .iter()
        .position(|&b| b == b'\n')
        .and_then(|nl| std::str::from_utf8(&data[..nl]).ok())
        .filter(|h| h.starts_with("YUV4MPEG2"))
        .context("not a YUV4MPEG2 stream")?;
    let (mut w, mut h, mut fps, mut chroma) = (0usize, 0usize, None, "420jpeg".to_string());
    for tag in header.split_whitespace().skip(1) {
        match tag.as_bytes()[0] {
            b'W' => w = tag[1..].parse()?,
            b'H' => h = tag[1..].parse()?,
            b'F' => {
                let (n, d) = tag[1..].split_once(':').context("y4m frame rate")?;
                fps = Some(n.parse::<f64>()? / d.parse::<f64>()?);
            }
            b'C' => chroma = tag[1..].to_string(),
            _ => {}
        }
    }
    if w == 0 || h == 0 {
        bail!("y4m without W/H");
    }
    let (cw, ch) = if chroma.starts_with("444") {
        (w, h)
    } else if chroma.starts_with("422") {
        (w.div_ceil(2), h)
    } else {
        (w.div_ceil(2), h.div_ceil(2))
    };
    let full = chroma.contains("XCOLORRANGE=FULL") || header.contains("XCOLORRANGE=FULL");
    let frame_bytes = w * h + 2 * cw * ch;
    let mut pos = header.len() + 1;
    let mut frames = Vec::new();
    while pos + 6 <= data.len() {
        let line_end = data[pos..]
            .iter()
            .position(|&b| b == b'\n')
            .context("y4m frame header")?
            + pos;
        if !data[pos..].starts_with(b"FRAME") {
            bail!("y4m: expected FRAME at byte {pos}");
        }
        pos = line_end + 1;
        if pos + frame_bytes > data.len() {
            bail!("y4m: truncated frame");
        }
        let y = &data[pos..pos + w * h];
        let u = &data[pos + w * h..pos + w * h + cw * ch];
        let v = &data[pos + w * h + cw * ch..pos + frame_bytes];
        let mut rgb = vec![0u8; w * h * 3];
        for yy in 0..h {
            for xx in 0..w {
                let (cx, cy) = (xx * cw / w, yy * ch / h);
                let (yv, uv, vv) = (
                    y[yy * w + xx] as f32,
                    u[cy * cw + cx] as f32 - 128.0,
                    v[cy * cw + cx] as f32 - 128.0,
                );
                let yv = if full {
                    yv
                } else {
                    (yv - 16.0) * 255.0 / 219.0
                };
                let (uv, vv) = if full {
                    (uv, vv)
                } else {
                    (uv * 255.0 / 224.0, vv * 255.0 / 224.0)
                };
                let r = yv + 1.402 * vv;
                let g = yv - 0.344_136 * uv - 0.714_136 * vv;
                let b = yv + 1.772 * uv;
                let o = (yy * w + xx) * 3;
                rgb[o] = r.round().clamp(0.0, 255.0) as u8;
                rgb[o + 1] = g.round().clamp(0.0, 255.0) as u8;
                rgb[o + 2] = b.round().clamp(0.0, 255.0) as u8;
            }
        }
        frames.push(Image::new(w, h, rgb));
        pos += frame_bytes;
    }
    Ok((frames, fps))
}

/// YUV4MPEG2, 4:4:4, BT.601 full range (plays in mpv / ffplay).
pub fn write_y4m(
    path: &Path,
    frames: usize,
    width: usize,
    height: usize,
    rgb: &[u8],
    fps: u32,
) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    write_y4m_to(&mut f, frames, width, height, rgb, fps)
}

/// [`write_y4m`] into any writer.
pub fn write_y4m_to(
    f: &mut impl Write,
    frames: usize,
    width: usize,
    height: usize,
    rgb: &[u8],
    fps: u32,
) -> Result<()> {
    writeln!(
        f,
        "YUV4MPEG2 W{width} H{height} F{fps}:1 Ip A1:1 C444 XCOLORRANGE=FULL"
    )?;
    let px = width * height;
    let mut planes = vec![0u8; 3 * px];
    for fr in 0..frames {
        let p = &rgb[fr * px * 3..(fr + 1) * px * 3];
        for i in 0..px {
            let (r, g, b) = (p[3 * i] as f32, p[3 * i + 1] as f32, p[3 * i + 2] as f32);
            planes[i] = (0.299 * r + 0.587 * g + 0.114 * b)
                .round()
                .clamp(0.0, 255.0) as u8;
            planes[px + i] = (128.0 - 0.168_736 * r - 0.331_264 * g + 0.5 * b)
                .round()
                .clamp(0.0, 255.0) as u8;
            planes[2 * px + i] = (128.0 + 0.5 * r - 0.418_688 * g - 0.081_312 * b)
                .round()
                .clamp(0.0, 255.0) as u8;
        }
        f.write_all(b"FRAME\n")?;
        f.write_all(&planes)?;
    }
    Ok(())
}

/// 16-bit PCM stereo WAV.
pub fn write_wav(path: &Path, stereo: &[Vec<f32>; 2], rate: u32) -> Result<()> {
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    write_wav_to(&mut f, stereo, rate)
}

/// [`write_wav`] into any writer.
pub fn write_wav_to(f: &mut impl Write, stereo: &[Vec<f32>; 2], rate: u32) -> Result<()> {
    let n = stereo[0].len().min(stereo[1].len());
    let data_len = (n * 4) as u32;
    f.write_all(b"RIFF")?;
    f.write_all(&(36 + data_len).to_le_bytes())?;
    f.write_all(b"WAVEfmt ")?;
    f.write_all(&16u32.to_le_bytes())?;
    f.write_all(&1u16.to_le_bytes())?; // PCM
    f.write_all(&2u16.to_le_bytes())?;
    f.write_all(&rate.to_le_bytes())?;
    f.write_all(&(rate * 4).to_le_bytes())?;
    f.write_all(&4u16.to_le_bytes())?;
    f.write_all(&16u16.to_le_bytes())?;
    f.write_all(b"data")?;
    f.write_all(&data_len.to_le_bytes())?;
    for i in 0..n {
        for ch in stereo {
            let v = (ch[i].clamp(-1.0, 1.0) * 32767.0).round() as i16;
            f.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

/// Read a PCM WAV (8/16/24/32-bit int or 32-bit float), any channel count, as
/// stereo at `target_rate` (linear resampling; mono is duplicated).
pub fn read_wav(path: &Path, target_rate: usize) -> Result<[Vec<f32>; 2]> {
    let mut data = Vec::new();
    std::fs::File::open(path)
        .with_context(|| format!("open {}", path.display()))?
        .read_to_end(&mut data)?;
    decode_wav(&data, target_rate).with_context(|| format!("{}", path.display()))
}

/// [`read_wav`] on bytes.
pub fn decode_wav(data: &[u8], target_rate: usize) -> Result<[Vec<f32>; 2]> {
    if data.len() < 12 || &data[..4] != b"RIFF" || &data[8..12] != b"WAVE" {
        bail!("not a WAV file");
    }
    let (mut pos, mut fmt, mut pcm) = (12usize, None, None);
    while pos + 8 <= data.len() {
        let id = &data[pos..pos + 4];
        let len = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = &data[pos + 8..(pos + 8 + len).min(data.len())];
        if id == b"fmt " {
            if body.len() < 16 {
                bail!("WAV fmt chunk is truncated");
            }
            let tag = u16::from_le_bytes([body[0], body[1]]);
            let channels = u16::from_le_bytes([body[2], body[3]]) as usize;
            let rate = u32::from_le_bytes(body[4..8].try_into().unwrap()) as usize;
            let bits = u16::from_le_bytes([body[14], body[15]]) as usize;
            fmt = Some((tag, channels, rate, bits));
        } else if id == b"data" {
            pcm = Some(body.to_vec());
        }
        pos += 8 + len + (len & 1);
    }
    let (tag, channels, rate, bits) = fmt.context("WAV without fmt chunk")?;
    let pcm = pcm.context("WAV without data chunk")?;
    let bytes = bits / 8;
    if bytes == 0 || channels == 0 || rate == 0 {
        bail!("WAV with {bits}-bit samples, {channels} channels at {rate} Hz");
    }
    let frames = pcm.len() / (bytes * channels);
    let sample = |i: usize, c: usize| -> f32 {
        let o = (i * channels + c) * bytes;
        match (tag, bits) {
            (3, 32) => f32::from_le_bytes(pcm[o..o + 4].try_into().unwrap()),
            (_, 8) => (pcm[o] as f32 - 128.0) / 128.0,
            (_, 16) => i16::from_le_bytes([pcm[o], pcm[o + 1]]) as f32 / 32768.0,
            (_, 24) => {
                (i32::from_le_bytes([0, pcm[o], pcm[o + 1], pcm[o + 2]]) >> 8) as f32 / 8_388_608.0
            }
            (_, 32) => {
                i32::from_le_bytes(pcm[o..o + 4].try_into().unwrap()) as f32 / 2_147_483_648.0
            }
            _ => 0.0,
        }
    };
    let mut out = [Vec::new(), Vec::new()];
    let n_out = (frames as f64 * target_rate as f64 / rate as f64).round() as usize;
    for (ch, o) in out.iter_mut().enumerate() {
        let src = ch.min(channels - 1);
        o.reserve(n_out);
        for i in 0..n_out {
            let t = i as f64 * rate as f64 / target_rate as f64;
            let i0 = (t.floor() as usize).min(frames.saturating_sub(1));
            let i1 = (i0 + 1).min(frames.saturating_sub(1));
            let f = (t - i0 as f64) as f32;
            o.push(sample(i0, src) * (1.0 - f) + sample(i1, src) * f);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn y4m_and_wav_round_trip() {
        let dir = std::env::temp_dir().join(format!("h3-media-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (w, h) = (4, 2);
        let mut rgb = Vec::new();
        for f in 0..3u8 {
            for i in 0..(w * h) as u8 {
                rgb.extend([i * 20, 255 - i * 20, f * 60]);
            }
        }
        let p = dir.join("a.y4m");
        write_y4m(&p, 3, w, h, &rgb, 24).unwrap();
        let (frames, fps) = read_frames(&p).unwrap();
        assert_eq!((frames.len(), fps), (3, Some(24.0)));
        let err = frames[1]
            .rgb
            .iter()
            .zip(&rgb[w * h * 3..2 * w * h * 3])
            .map(|(a, b)| (*a as i32 - *b as i32).abs())
            .max()
            .unwrap();
        assert!(err <= 3, "{err}");
        let wav = dir.join("a.wav");
        let s: Vec<f32> = (0..100).map(|i| (i as f32 * 0.1).sin() * 0.5).collect();
        write_wav(&wav, &[s.clone(), s.clone()], 1000).unwrap();
        let back = read_wav(&wav, 2000).unwrap();
        assert_eq!(back[0].len(), 200);
        assert!((back[1][20] - s[10]).abs() < 2e-3);
        std::fs::remove_dir_all(&dir).ok();
    }
}
