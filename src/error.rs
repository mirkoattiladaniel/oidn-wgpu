//! Error types for OIDN and wgpu integration.
//!
//! See [`Error`] for the main error type returned by public APIs.

use std::fmt;

/// Errors from OIDN or oidn-wgpu.
///
/// This type implements [`std::error::Error`], [`Send`], and [`Sync`], so it can be
/// used with `?` and error handling libraries, and across thread boundaries.
#[derive(Debug)]
#[non_exhaustive]
pub enum Error {
    /// OIDN API returned an error.
    OidnError { code: u32, message: String },
    /// Device creation failed (e.g. no supported backend).
    DeviceCreationFailed,
    /// Filter creation failed.
    FilterCreationFailed,
    /// Out of memory.
    OutOfMemory,
    /// Image dimensions do not match buffer size.
    InvalidDimensions,
    /// Unsupported texture format for denoising.
    UnsupportedFormat,
    /// wgpu buffer mapping failed.
    BufferMapFailed(wgpu::BufferAsyncError),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::OidnError { code, message } => write!(f, "OIDN error ({}): {}", code, message),
            Error::DeviceCreationFailed => write!(f, "OIDN device creation failed"),
            Error::FilterCreationFailed => write!(f, "OIDN filter creation failed"),
            Error::OutOfMemory => write!(f, "OIDN out of memory"),
            Error::InvalidDimensions => write!(f, "invalid image dimensions"),
            Error::UnsupportedFormat => write!(f, "unsupported texture format for denoising"),
            Error::BufferMapFailed(e) => write!(f, "wgpu buffer map failed: {:?}", e),
        }
    }
}

impl std::error::Error for Error {}

// Required for use with ? and multithreaded error handling (C-GOOD-ERR).
unsafe impl Send for Error {}
unsafe impl Sync for Error {}
