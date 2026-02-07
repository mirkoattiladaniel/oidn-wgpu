//! Denoise wgpu textures by copying to CPU, running OIDN, and copying back.

use crate::device::OidnDevice;
use crate::filter::{Quality, RtFilter};
use crate::Error;
use bytemuck::{cast_slice, cast_slice_mut};
use std::sync::mpsc;
use wgpu::util::DeviceExt;

/// Supported texture format for denoising input/output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DenoiseTextureFormat {
    /// RGBA 32-bit float (4 components). Alpha is preserved.
    Rgba32Float,
    /// RGBA 16-bit float (4 components). Alpha is preserved.
    Rgba16Float,
}

impl DenoiseTextureFormat {
    /// Converts from a wgpu texture format if it is supported for denoising.
    pub fn from_wgpu(format: wgpu::TextureFormat) -> Option<Self> {
        format.try_into().ok()
    }

    fn bytes_per_pixel(self) -> u32 {
        match self {
            DenoiseTextureFormat::Rgba32Float => 16,
            DenoiseTextureFormat::Rgba16Float => 8,
        }
    }
}

impl TryFrom<wgpu::TextureFormat> for DenoiseTextureFormat {
    type Error = ();

    fn try_from(format: wgpu::TextureFormat) -> Result<Self, Self::Error> {
        match format {
            wgpu::TextureFormat::Rgba32Float => Ok(Self::Rgba32Float),
            wgpu::TextureFormat::Rgba16Float => Ok(Self::Rgba16Float),
            _ => Err(()),
        }
    }
}

/// Options for denoising a wgpu texture.
#[derive(Clone, Debug)]
pub struct DenoiseOptions {
    /// Quality vs performance: `Fast`, `Balanced`, or `High`.
    pub quality: Quality,
    /// `true` if the image is HDR (linear, possibly > 1.0).
    pub hdr: bool,
    /// `true` if the image is sRGB-encoded LDR.
    pub srgb: bool,
    /// Input scale for HDR (e.g. exposure). `None` = auto.
    pub input_scale: Option<f32>,
}

impl Default for DenoiseOptions {
    fn default() -> Self {
        Self {
            quality: Quality::Default,
            hdr: true,
            srgb: false,
            input_scale: None,
        }
    }
}

/// Denoises a wgpu texture by readback → OIDN (CPU) → upload.
///
/// Input and output can be the same texture for in-place denoising, or different.
/// Supported formats: [`DenoiseTextureFormat::Rgba32Float`], [`DenoiseTextureFormat::Rgba16Float`].
/// Only RGB is denoised; alpha is preserved.
///
/// **Texture usage:** `input` must have [`TextureUsages::COPY_SRC`](wgpu::TextureUsages::COPY_SRC);
/// `output` must have [`TextureUsages::COPY_DST`](wgpu::TextureUsages::COPY_DST).
///
/// This is a blocking call: it submits copy commands, waits for readback, runs OIDN, then uploads.
///
/// # Errors
///
/// Returns [`Error::InvalidDimensions`] if texture sizes or array layers are incompatible, or
/// [`Error::BufferMapFailed`] if wgpu buffer mapping fails. OIDN execution errors are returned
/// as [`Error::OidnError`] or other [`Error`] variants.
pub fn denoise_texture(
    device: &OidnDevice,
    wgpu_device: &wgpu::Device,
    wgpu_queue: &wgpu::Queue,
    input: &wgpu::Texture,
    output: &wgpu::Texture,
    format: DenoiseTextureFormat,
    options: &DenoiseOptions,
) -> Result<(), Error> {
    let size = input.size();
    if size.depth_or_array_layers != 1 {
        return Err(Error::InvalidDimensions);
    }
    let out_size = output.size();
    if out_size.width != size.width
        || out_size.height != size.height
        || out_size.depth_or_array_layers != 1
    {
        return Err(Error::InvalidDimensions);
    }
    let w = size.width;
    let h = size.height;
    let bpp = format.bytes_per_pixel();
    let bytes_per_row = w * bpp;
    let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = (bytes_per_row + alignment - 1) / alignment * alignment;
    let buffer_size = padded_bytes_per_row as u64 * h as u64;

    // Create staging buffer for readback
    let read_buffer = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("oidn_wgpu readback"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: input,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &read_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(h),
            },
        },
        size,
    );
    wgpu_queue.submit(Some(encoder.finish()));

    // Map and read
    let slice = read_buffer.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    loop {
        let _ = wgpu_device.poll(wgpu::PollType::wait_indefinitely());
        match rx.try_recv() {
            Ok(Ok(())) => break,
            Ok(Err(_)) => return Err(Error::BufferMapFailed(wgpu::BufferAsyncError)),
            Err(mpsc::TryRecvError::Disconnected) => return Err(Error::BufferMapFailed(wgpu::BufferAsyncError)),
            Err(mpsc::TryRecvError::Empty) => std::thread::sleep(std::time::Duration::from_micros(100)),
        }
    }

    let mapped = slice.get_mapped_range();
    let raw: &[u8] = &mapped;

    let n_pixels = (w * h) as usize;
    let mut color_f32 = vec![0.0f32; n_pixels * 3];
    let alpha_f32: Vec<f32> = match format {
        DenoiseTextureFormat::Rgba32Float => {
            let floats: &[f32] = cast_slice(raw);
            for i in 0..n_pixels {
                let j = i * 4;
                color_f32[i * 3] = floats[j];
                color_f32[i * 3 + 1] = floats[j + 1];
                color_f32[i * 3 + 2] = floats[j + 2];
            }
            floats.iter().skip(3).step_by(4).copied().collect()
        }
        DenoiseTextureFormat::Rgba16Float => {
            let u16s: &[u16] = cast_slice(raw);
            for i in 0..n_pixels {
                let j = i * 4;
                color_f32[i * 3] = half::f16::from_bits(u16s[j]).to_f32();
                color_f32[i * 3 + 1] = half::f16::from_bits(u16s[j + 1]).to_f32();
                color_f32[i * 3 + 2] = half::f16::from_bits(u16s[j + 2]).to_f32();
            }
            u16s.iter()
                .skip(3)
                .step_by(4)
                .map(|&b| half::f16::from_bits(b).to_f32())
                .collect()
        }
    };

    drop(mapped);

    // Run OIDN
    let mut filter = RtFilter::new(device)?;
    filter
        .set_dimensions(w, h)
        .set_hdr(options.hdr)
        .set_srgb(options.srgb)
        .set_quality(options.quality);
    if let Some(scale) = options.input_scale {
        filter.set_input_scale(scale);
    }
    filter.execute_in_place(&mut color_f32)?;

    // Convert back to RGBA (preserve alpha from original)
    let mut output_bytes: Vec<u8> = vec![0; n_pixels * bpp as usize];
    match format {
        DenoiseTextureFormat::Rgba32Float => {
            let out_f: &mut [f32] = cast_slice_mut(&mut output_bytes);
            for i in 0..n_pixels {
                let j = i * 4;
                out_f[j] = color_f32[i * 3];
                out_f[j + 1] = color_f32[i * 3 + 1];
                out_f[j + 2] = color_f32[i * 3 + 2];
                out_f[j + 3] = alpha_f32.get(i).copied().unwrap_or(1.0);
            }
        }
        DenoiseTextureFormat::Rgba16Float => {
            let out_u16: &mut [u16] = cast_slice_mut(&mut output_bytes);
            for i in 0..n_pixels {
                out_u16[i * 4] = half::f16::from_f32(color_f32[i * 3]).to_bits();
                out_u16[i * 4 + 1] = half::f16::from_f32(color_f32[i * 3 + 1]).to_bits();
                out_u16[i * 4 + 2] = half::f16::from_f32(color_f32[i * 3 + 2]).to_bits();
                out_u16[i * 4 + 3] = half::f16::from_f32(alpha_f32.get(i).copied().unwrap_or(1.0)).to_bits();
            }
        }
    }

    // Upload: buffer must use padded row stride for WebGPU alignment.
    let mut upload_data = vec![0u8; (padded_bytes_per_row * h) as usize];
    for row in 0..h {
        let src_off = (row * bytes_per_row) as usize;
        let dst_off = (row * padded_bytes_per_row) as usize;
        upload_data[dst_off..dst_off + bytes_per_row as usize]
            .copy_from_slice(&output_bytes[src_off..src_off + bytes_per_row as usize]);
    }
    let write_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("oidn_wgpu upload"),
        contents: &upload_data,
        usage: wgpu::BufferUsages::COPY_SRC,
    });

    let mut enc2 = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc2.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo {
            buffer: &write_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(h),
            },
        },
        wgpu::TexelCopyTextureInfo {
            texture: output,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        size,
    );
    wgpu_queue.submit(Some(enc2.finish()));

    Ok(())
}

/// Denoises a wgpu color texture with optional albedo and normal AOV textures (same size/format as color).
///
/// Higher quality when albedo and normal are provided. Otherwise identical to [`denoise_texture`].
///
/// # Errors
///
/// Same as [`denoise_texture`]; also [`Error::InvalidDimensions`] if any aux texture size does not match.
pub fn denoise_texture_with_aux(
    device: &OidnDevice,
    wgpu_device: &wgpu::Device,
    wgpu_queue: &wgpu::Queue,
    input: &wgpu::Texture,
    output: &wgpu::Texture,
    format: DenoiseTextureFormat,
    options: &DenoiseOptions,
    albedo: Option<&wgpu::Texture>,
    normal: Option<&wgpu::Texture>,
) -> Result<(), Error> {
    let size = input.size();
    if size.depth_or_array_layers != 1 {
        return Err(Error::InvalidDimensions);
    }
    let out_size = output.size();
    if out_size.width != size.width
        || out_size.height != size.height
        || out_size.depth_or_array_layers != 1
    {
        return Err(Error::InvalidDimensions);
    }
    if let Some(tex) = albedo {
        let s = tex.size();
        if s.width != size.width || s.height != size.height || s.depth_or_array_layers != 1 {
            return Err(Error::InvalidDimensions);
        }
    }
    if let Some(tex) = normal {
        let s = tex.size();
        if s.width != size.width || s.height != size.height || s.depth_or_array_layers != 1 {
            return Err(Error::InvalidDimensions);
        }
    }
    let w = size.width;
    let h = size.height;

    let (mut color_rgb, alpha) = read_texture_to_rgba_f32(wgpu_device, wgpu_queue, input, format)?;
    let albedo_rgb = albedo
        .map(|t| read_texture_to_rgba_f32(wgpu_device, wgpu_queue, t, format).map(|(rgb, _)| rgb))
        .transpose()?;
    let normal_rgb = normal
        .map(|t| read_texture_to_rgba_f32(wgpu_device, wgpu_queue, t, format).map(|(rgb, _)| rgb))
        .transpose()?;

    let mut filter = RtFilter::new(device)?;
    filter
        .set_dimensions(w, h)
        .set_hdr(options.hdr)
        .set_srgb(options.srgb)
        .set_quality(options.quality);
    if let Some(scale) = options.input_scale {
        filter.set_input_scale(scale);
    }
    filter.execute_in_place_with_aux(
        &mut color_rgb,
        albedo_rgb.as_deref(),
        normal_rgb.as_deref(),
    )?;

    upload_rgba_to_texture(
        wgpu_device,
        wgpu_queue,
        output,
        format,
        w,
        h,
        &color_rgb,
        &alpha,
    )
}

/// Reads a wgpu texture to CPU as (RGB f32, alpha f32). Blocking.
fn read_texture_to_rgba_f32(
    wgpu_device: &wgpu::Device,
    wgpu_queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    format: DenoiseTextureFormat,
) -> Result<(Vec<f32>, Vec<f32>), Error> {
    let size = texture.size();
    let w = size.width;
    let h = size.height;
    let bpp = format.bytes_per_pixel();
    let bytes_per_row = w * bpp;
    let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = (bytes_per_row + alignment - 1) / alignment * alignment;
    let buffer_size = padded_bytes_per_row as u64 * h as u64;
    let n_pixels = (w * h) as usize;

    let read_buffer = wgpu_device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("oidn_wgpu readback"),
        size: buffer_size,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &read_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(h),
            },
        },
        size,
    );
    wgpu_queue.submit(Some(encoder.finish()));

    let slice = read_buffer.slice(..);
    let (tx, rx) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |r| {
        let _ = tx.send(r);
    });
    loop {
        let _ = wgpu_device.poll(wgpu::PollType::wait_indefinitely());
        match rx.try_recv() {
            Ok(Ok(())) => break,
            Ok(Err(_)) => return Err(Error::BufferMapFailed(wgpu::BufferAsyncError)),
            Err(mpsc::TryRecvError::Disconnected) => return Err(Error::BufferMapFailed(wgpu::BufferAsyncError)),
            Err(mpsc::TryRecvError::Empty) => std::thread::sleep(std::time::Duration::from_micros(100)),
        }
    }
    let mapped = slice.get_mapped_range();
    let raw: &[u8] = &mapped;
    let mut rgb = vec![0.0f32; n_pixels * 3];
    let alpha: Vec<f32> = match format {
        DenoiseTextureFormat::Rgba32Float => {
            let floats: &[f32] = cast_slice(raw);
            for i in 0..n_pixels {
                let j = i * 4;
                rgb[i * 3] = floats[j];
                rgb[i * 3 + 1] = floats[j + 1];
                rgb[i * 3 + 2] = floats[j + 2];
            }
            floats.iter().skip(3).step_by(4).copied().collect()
        }
        DenoiseTextureFormat::Rgba16Float => {
            let u16s: &[u16] = cast_slice(raw);
            for i in 0..n_pixels {
                let j = i * 4;
                rgb[i * 3] = half::f16::from_bits(u16s[j]).to_f32();
                rgb[i * 3 + 1] = half::f16::from_bits(u16s[j + 1]).to_f32();
                rgb[i * 3 + 2] = half::f16::from_bits(u16s[j + 2]).to_f32();
            }
            u16s.iter()
                .skip(3)
                .step_by(4)
                .map(|&b| half::f16::from_bits(b).to_f32())
                .collect()
        }
    };
    drop(mapped);
    Ok((rgb, alpha))
}

/// Uploads denoised RGB + preserved alpha to a wgpu texture (padded row alignment).
fn upload_rgba_to_texture(
    wgpu_device: &wgpu::Device,
    wgpu_queue: &wgpu::Queue,
    output: &wgpu::Texture,
    format: DenoiseTextureFormat,
    w: u32,
    h: u32,
    color_f32: &[f32],
    alpha_f32: &[f32],
) -> Result<(), Error> {
    let n_pixels = (w * h) as usize;
    let bpp = format.bytes_per_pixel();
    let bytes_per_row = w * bpp;
    let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
    let padded_bytes_per_row = (bytes_per_row + alignment - 1) / alignment * alignment;
    let mut output_bytes = vec![0u8; n_pixels * bpp as usize];
    match format {
        DenoiseTextureFormat::Rgba32Float => {
            let out_f: &mut [f32] = cast_slice_mut(&mut output_bytes);
            for i in 0..n_pixels {
                let j = i * 4;
                out_f[j] = color_f32[i * 3];
                out_f[j + 1] = color_f32[i * 3 + 1];
                out_f[j + 2] = color_f32[i * 3 + 2];
                out_f[j + 3] = alpha_f32.get(i).copied().unwrap_or(1.0);
            }
        }
        DenoiseTextureFormat::Rgba16Float => {
            let out_u16: &mut [u16] = cast_slice_mut(&mut output_bytes);
            for i in 0..n_pixels {
                out_u16[i * 4] = half::f16::from_f32(color_f32[i * 3]).to_bits();
                out_u16[i * 4 + 1] = half::f16::from_f32(color_f32[i * 3 + 1]).to_bits();
                out_u16[i * 4 + 2] = half::f16::from_f32(color_f32[i * 3 + 2]).to_bits();
                out_u16[i * 4 + 3] = half::f16::from_f32(alpha_f32.get(i).copied().unwrap_or(1.0)).to_bits();
            }
        }
    }
    let mut upload_data = vec![0u8; (padded_bytes_per_row * h) as usize];
    for row in 0..h {
        let src_off = (row * bytes_per_row) as usize;
        let dst_off = (row * padded_bytes_per_row) as usize;
        upload_data[dst_off..dst_off + bytes_per_row as usize]
            .copy_from_slice(&output_bytes[src_off..src_off + bytes_per_row as usize]);
    }
    let size = output.size();
    let write_buffer = wgpu_device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("oidn_wgpu upload"),
        contents: &upload_data,
        usage: wgpu::BufferUsages::COPY_SRC,
    });
    let mut enc = wgpu_device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
    enc.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo {
            buffer: &write_buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(h),
            },
        },
        wgpu::TexelCopyTextureInfo {
            texture: output,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        size,
    );
    wgpu_queue.submit(Some(enc.finish()));
    Ok(())
}
