//! # oidn-wgpu
//!
//! [Intel Open Image Denoise](https://www.openimagedenoise.org) (OIDN) integration for [wgpu](https://docs.rs/wgpu).
//! Denoise path-traced or ray-traced images produced on the GPU by copying to CPU, running OIDN, and copying back.
//!
//! This crate targets **OIDN 2.4.x** and **wgpu 27**. It does not depend on `oidn-rs`.
//!
//! ## Setup
//!
//! Build and install OIDN 2.4.x (e.g. from <https://github.com/OpenImageDenoise/oidn>), then either:
//!
//! - Set **`OIDN_DIR`** to the install directory (containing `include/` and `lib/`), or
//! - Use **pkg-config** (Linux/macOS) with `OpenImageDenoise` installed.
//!
//! ## Example: denoise a wgpu texture
//!
//! ```ignore
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! use oidn_wgpu::{OidnDevice, denoise_texture, DenoiseTextureFormat, DenoiseOptions};
//!
//! let oidn = OidnDevice::new()?;
//! let format = DenoiseTextureFormat::Rgba16Float;
//! denoise_texture(
//!     &oidn,
//!     &wgpu_device,
//!     &wgpu_queue,
//!     &input_texture,
//!     &output_texture,
//!     format,
//!     &DenoiseOptions::default(), // or set quality, hdr, srgb, input_scale
//! )?;
//! # Ok(())
//! # }
//! ```
//!
//! ## Example: denoise CPU buffers (no wgpu)
//!
//! ```ignore
//! # use std::error::Error;
//! # fn main() -> Result<(), Box<dyn Error>> {
//! use oidn_wgpu::{OidnDevice, RtFilter};
//!
//! let device = OidnDevice::new()?;
//! let mut filter = RtFilter::new(&device)?;
//! filter.set_dimensions(width, height).set_hdr(true);
//! filter.execute_in_place(&mut color_rgb_f32)?;
//! # Ok(())
//! # }
//! ```

pub mod buffer;
pub mod device;
pub mod error;
pub mod filter;
mod sys;
pub mod wgpu_integration;

#[cfg(test)]
mod tests;

/// UUID size for physical device (bytes). Use with [`OidnDevice::new_by_uuid`].
pub const OIDN_UUID_SIZE: usize = 16;
/// LUID size for physical device (bytes). Use with [`OidnDevice::new_by_luid`].
pub const OIDN_LUID_SIZE: usize = 8;

pub use buffer::{BufferStorage, ExternalMemoryTypeFlag, OidnBuffer};
pub use device::{
    get_physical_device_bool, get_physical_device_data, get_physical_device_int,
    get_physical_device_string, is_cpu_device_supported, is_cuda_device_supported,
    is_hip_device_supported, is_metal_device_supported, num_physical_devices, OidnDevice,
    OidnDeviceType, take_global_error,
};
pub use error::Error;
pub use filter::{Filter, ImageFormat, OIDNFormat, Quality, RtFilter, RtLightmapFilter};
pub use wgpu_integration::{
    denoise_texture, denoise_texture_with_aux, DenoiseOptions, DenoiseTextureFormat,
};
