//! Example: denoise a wgpu texture (blocking).
//!
//! Run with: cargo run --example wgpu_denoise
//! Requires OIDN built and OIDN_DIR set. Creates a headless wgpu device, a small
//! noisy texture, denoises it, then exits.

use oidn_wgpu::{
    denoise_texture, DenoiseOptions, DenoiseTextureFormat, OidnDevice, Quality,
};
use pollster::block_on;
use wgpu::util::DeviceExt;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    block_on(async {
        let desc = wgpu::InstanceDescriptor::default();
        let instance = wgpu::Instance::new(&desc);
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: None,
            })
            .await
            .map_err(|e| format!("request_adapter: {}", e))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default())
            .await
            .map_err(|e| format!("request_device: {}", e))?;

        let w = 64u32;
        let h = 64u32;
        let format = DenoiseTextureFormat::Rgba16Float;
        let bpp = 8u64;
        let pixel_count = (w * h) as usize;
        let mut cpu_pixels = vec![0u8; pixel_count * (bpp as usize)];
        let u16_view: &mut [u16] = bytemuck::cast_slice_mut(&mut cpu_pixels);
        for i in 0..(pixel_count * 4) {
            let x = (i % (w as usize * 4)) as f32 / (w as f32 * 4.0);
            let y = (i / (w as usize * 4)) as f32 / h as f32;
            let v = (x * y).sin() * 0.5 + 0.5;
            u16_view[i] = half::f16::from_f32(v).to_bits();
        }
        for i in (0..(pixel_count * 4)).step_by(4) {
            u16_view[i + 3] = half::f16::from_f32(1.0).to_bits();
        }

        let alignment = wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let bytes_per_row = w * (bpp as u32);
        let padded = (bytes_per_row + alignment - 1) / alignment * alignment;
        let mut upload = vec![0u8; (padded * h) as usize];
        for row in 0..h {
            let src = (row * bytes_per_row) as usize;
            let dst = (row * padded) as usize;
            upload[dst..dst + bytes_per_row as usize]
                .copy_from_slice(&cpu_pixels[src..src + bytes_per_row as usize]);
        }

        let input_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("oidn_wgpu example input"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let output_tex = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("oidn_wgpu example output"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba16Float,
            usage: wgpu::TextureUsages::COPY_DST | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        let staging = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: &upload,
            usage: wgpu::BufferUsages::COPY_SRC,
        });
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        enc.copy_buffer_to_texture(
            wgpu::TexelCopyBufferInfo {
                buffer: &staging,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded),
                    rows_per_image: Some(h),
                },
            },
            wgpu::TexelCopyTextureInfo {
                texture: &input_tex,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
        );
        queue.submit(Some(enc.finish()));

        // Prefer CUDA (NVIDIA GPU) when available, else default (auto) or CPU
        let oidn = OidnDevice::cuda()
            .or_else(|_| OidnDevice::new())
            .map_err(|e| format!("OIDN device: {}", e))?;
        denoise_texture(
            &oidn,
            &device,
            &queue,
            &input_tex,
            &output_tex,
            format,
            &DenoiseOptions {
                quality: Quality::Balanced,
                hdr: true,
                srgb: false,
                input_scale: None,
            },
        )?;

        println!("Denoised {}x{} wgpu texture successfully.", w, h);
        Ok(())
    })
}
