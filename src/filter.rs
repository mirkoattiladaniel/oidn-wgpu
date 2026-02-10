//! RT and RTLightmap denoising filters, plus generic filter API (full OIDN filter coverage).

use crate::buffer::OidnBuffer;
use crate::device::OidnDevice;
use crate::sys;
use crate::Error;
use std::ffi::CString;

/// Filter quality vs performance trade-off (OIDN 2.x).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Quality {
    /// Default (high quality).
    #[default]
    Default,
    /// Fast — for interactive/real-time preview.
    Fast,
    /// Balanced — interactive/real-time.
    Balanced,
    /// High — for final-frame rendering.
    High,
}

impl Quality {
    fn to_raw(self) -> sys::OIDNQuality {
        match self {
            Quality::Default => sys::OIDNQuality::Default,
            Quality::Fast => sys::OIDNQuality::Fast,
            Quality::Balanced => sys::OIDNQuality::Balanced,
            Quality::High => sys::OIDNQuality::High,
        }
    }
}

/// Ray tracing denoising filter (OIDN "RT" filter).
///
/// Denoises a beauty (color) image, optionally using albedo and normal AOVs.
/// Reuse the same filter for multiple frames when dimensions match.
pub struct RtFilter<'a> {
    device: &'a OidnDevice,
    raw: sys::OIDNFilter,
    width: u32,
    height: u32,
    hdr: bool,
    srgb: bool,
    clean_aux: bool,
    input_scale: f32,
    quality: Quality,
}

impl std::fmt::Debug for RtFilter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RtFilter")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("hdr", &self.hdr)
            .field("srgb", &self.srgb)
            .field("quality", &self.quality)
            .finish_non_exhaustive()
    }
}

impl<'a> RtFilter<'a> {
    /// Creates a new RT filter. Reuse the same filter for multiple frames when dimensions match.
    ///
    /// # Errors
    ///
    /// Returns [`Error::FilterCreationFailed`] if the RT filter type is not available, or the
    /// device's last error (e.g. [`Error::OidnError`]).
    pub fn new(device: &'a OidnDevice) -> Result<Self, Error> {
        let type_name = CString::new("RT").unwrap();
        let raw = unsafe { sys::oidnNewFilter(device.raw(), type_name.as_ptr()) };
        if raw.is_null() {
            return Err(device.take_error().unwrap_or(Error::FilterCreationFailed));
        }
        Ok(Self {
            device,
            raw,
            width: 0,
            height: 0,
            hdr: true,
            srgb: false,
            clean_aux: false,
            input_scale: f32::NAN,
            quality: Quality::Default,
        })
    }

    /// Image dimensions (must be set before execute).
    pub fn set_dimensions(&mut self, width: u32, height: u32) -> &mut Self {
        self.width = width;
        self.height = height;
        self
    }

    /// Whether the input is HDR. Default: true.
    pub fn set_hdr(&mut self, hdr: bool) -> &mut Self {
        self.hdr = hdr;
        self
    }

    /// Whether the input is sRGB (LDR). Default: false.
    pub fn set_srgb(&mut self, srgb: bool) -> &mut Self {
        self.srgb = srgb;
        self
    }

    /// Whether albedo/normal are noise-free (prefiltered). Default: false.
    pub fn set_clean_aux(&mut self, clean: bool) -> &mut Self {
        self.clean_aux = clean;
        self
    }

    /// Input scale (e.g. for HDR). NaN = auto.
    pub fn set_input_scale(&mut self, scale: f32) -> &mut Self {
        self.input_scale = scale;
        self
    }

    /// Filter quality. Default: High.
    pub fn set_quality(&mut self, quality: Quality) -> &mut Self {
        self.quality = quality;
        self
    }

    /// Gets a boolean filter parameter (e.g. `"hdr"`, `"srgb"`).
    pub fn get_bool(&self, name: &str) -> bool {
        let c_name = CString::new(name).unwrap();
        unsafe { sys::oidnGetFilterBool(self.raw, c_name.as_ptr()) }
    }

    /// Gets an integer filter parameter (e.g. `"quality"`).
    pub fn get_int(&self, name: &str) -> i32 {
        let c_name = CString::new(name).unwrap();
        unsafe { sys::oidnGetFilterInt(self.raw, c_name.as_ptr()) }
    }

    /// Gets a float filter parameter (e.g. `"inputScale"`).
    pub fn get_float(&self, name: &str) -> f32 {
        let c_name = CString::new(name).unwrap();
        unsafe { sys::oidnGetFilterFloat(self.raw, c_name.as_ptr()) }
    }

    /// Sets the progress monitor callback. Callback receives progress in [0,1]; return `false` to cancel.
    /// Safe to call with `(None, null)` to clear. Callback must not panic.
    ///
    /// # Safety
    ///
    /// `func` must be a valid progress callback that does not panic. `user_ptr` must remain valid until the filter is dropped or the callback is cleared.
    pub unsafe fn set_progress_monitor_raw(
        &self,
        func: sys::OIDNProgressMonitorFunction,
        user_ptr: *mut std::ffi::c_void,
    ) {
        sys::oidnSetFilterProgressMonitorFunction(self.raw, func, user_ptr);
    }

    /// Denoises color in-place. `color` must be `width * height * 3` floats (RGB).
    ///
    /// # Errors
    ///
    /// Returns [`Error::InvalidDimensions`] if dimensions are unset or buffer sizes do not match,
    /// or an OIDN error from the device.
    pub fn execute_in_place(&self, color: &mut [f32]) -> Result<(), Error> {
        self.execute_with_aux(None, color, None, None)
    }

    /// Denoises color in-place with optional albedo and normal AOVs (each `width * height * 3` floats).
    pub fn execute_in_place_with_aux(
        &self,
        color: &mut [f32],
        albedo: Option<&[f32]>,
        normal: Option<&[f32]>,
    ) -> Result<(), Error> {
        self.execute_with_aux(None, color, albedo, normal)
    }

    /// Denoises color into output. Slices must be `width * height * 3` floats (RGB).
    pub fn execute(&self, color: Option<&[f32]>, output: &mut [f32]) -> Result<(), Error> {
        self.execute_with_aux(color, output, None, None)
    }

    /// Denoises color into output with optional albedo and normal AOVs (each `width * height * 3` floats).
    pub fn execute_with_aux(
        &self,
        color: Option<&[f32]>,
        output: &mut [f32],
        albedo: Option<&[f32]>,
        normal: Option<&[f32]>,
    ) -> Result<(), Error> {
        let w = self.width as usize;
        let h = self.height as usize;
        if w == 0 || h == 0 {
            return Err(Error::InvalidDimensions);
        }
        let n = w * h * 3;
        if output.len() != n {
            return Err(Error::InvalidDimensions);
        }
        if let Some(c) = color {
            if c.len() != n {
                return Err(Error::InvalidDimensions);
            }
        }

        let device = self.device.raw();
        let color_buf = if let Some(c) = color {
            let buf = unsafe { sys::oidnNewBuffer(device, n * std::mem::size_of::<f32>()) };
            if buf.is_null() {
                return Err(self.device.take_error().unwrap_or(Error::OutOfMemory));
            }
            unsafe {
                sys::oidnWriteBuffer(buf, 0, n * std::mem::size_of::<f32>(), c.as_ptr() as *const _);
            }
            Some(buf)
        } else {
            None
        };
        let out_buf = unsafe { sys::oidnNewBuffer(device, n * std::mem::size_of::<f32>()) };
        if out_buf.is_null() {
            if let Some(b) = color_buf {
                unsafe { sys::oidnReleaseBuffer(b) };
            }
            return Err(self.device.take_error().unwrap_or(Error::OutOfMemory));
        }
        if color.is_none() {
            unsafe {
                sys::oidnWriteBuffer(
                    out_buf,
                    0,
                    n * std::mem::size_of::<f32>(),
                    output.as_ptr() as *const _,
                );
            }
        }

        // Validate and create optional albedo/normal buffers
        if let Some(a) = albedo {
            if a.len() != n {
                if let Some(b) = color_buf {
                    unsafe { sys::oidnReleaseBuffer(b) };
                }
                unsafe { sys::oidnReleaseBuffer(out_buf) };
                return Err(Error::InvalidDimensions);
            }
        }
        if let Some(norm) = normal {
            if norm.len() != n {
                if let Some(b) = color_buf {
                    unsafe { sys::oidnReleaseBuffer(b) };
                }
                unsafe { sys::oidnReleaseBuffer(out_buf) };
                return Err(Error::InvalidDimensions);
            }
        }

        let albedo_buf = albedo.map(|a| {
            let buf = unsafe { sys::oidnNewBuffer(device, n * std::mem::size_of::<f32>()) };
            if !buf.is_null() {
                unsafe {
                    sys::oidnWriteBuffer(buf, 0, n * std::mem::size_of::<f32>(), a.as_ptr() as *const _);
                }
            }
            buf
        });
        let normal_buf = normal.map(|norm| {
            let buf = unsafe { sys::oidnNewBuffer(device, n * std::mem::size_of::<f32>()) };
            if !buf.is_null() {
                unsafe {
                    sys::oidnWriteBuffer(buf, 0, n * std::mem::size_of::<f32>(), norm.as_ptr() as *const _);
                }
            }
            buf
        });

        if albedo_buf.is_some_and(|p| p.is_null())
            || normal_buf.is_some_and(|p| p.is_null())
        {
            if let Some(b) = color_buf {
                unsafe { sys::oidnReleaseBuffer(b) };
            }
            unsafe { sys::oidnReleaseBuffer(out_buf) };
            for b in albedo_buf.into_iter().chain(normal_buf) {
                if !b.is_null() {
                    unsafe { sys::oidnReleaseBuffer(b) };
                }
            }
            return Err(self.device.take_error().unwrap_or(Error::OutOfMemory));
        }

        let color_ptr = color_buf.unwrap_or(out_buf);
        unsafe {
            let c_color = CString::new("color").unwrap();
            let c_output = CString::new("output").unwrap();
            let c_albedo = CString::new("albedo").unwrap();
            let c_normal = CString::new("normal").unwrap();
            let c_hdr = CString::new("hdr").unwrap();
            let c_srgb = CString::new("srgb").unwrap();
            let c_clean_aux = CString::new("cleanAux").unwrap();
            let c_input_scale = CString::new("inputScale").unwrap();
            let c_quality = CString::new("quality").unwrap();

            sys::oidnSetFilterImage(
                self.raw,
                c_color.as_ptr(),
                color_ptr,
                sys::OIDNFormat::Float3,
                w,
                h,
                0,
                0,
                0,
            );
            sys::oidnSetFilterImage(
                self.raw,
                c_output.as_ptr(),
                out_buf,
                sys::OIDNFormat::Float3,
                w,
                h,
                0,
                0,
                0,
            );
            if let Some(ab) = albedo_buf {
                if !ab.is_null() {
                    sys::oidnSetFilterImage(
                        self.raw,
                        c_albedo.as_ptr(),
                        ab,
                        sys::OIDNFormat::Float3,
                        w,
                        h,
                        0,
                        0,
                        0,
                    );
                }
            }
            if let Some(nb) = normal_buf {
                if !nb.is_null() {
                    sys::oidnSetFilterImage(
                        self.raw,
                        c_normal.as_ptr(),
                        nb,
                        sys::OIDNFormat::Float3,
                        w,
                        h,
                        0,
                        0,
                        0,
                    );
                }
            }
            sys::oidnSetFilterBool(self.raw, c_hdr.as_ptr(), self.hdr);
            sys::oidnSetFilterBool(self.raw, c_srgb.as_ptr(), self.srgb);
            sys::oidnSetFilterBool(self.raw, c_clean_aux.as_ptr(), self.clean_aux);
            sys::oidnSetFilterFloat(self.raw, c_input_scale.as_ptr(), self.input_scale);
            sys::oidnSetFilterInt(self.raw, c_quality.as_ptr(), self.quality.to_raw() as i32);

            sys::oidnCommitFilter(self.raw);
            sys::oidnExecuteFilter(self.raw);
        }

        // Required for GPU (e.g. CUDA) where execute is async; ensures result is ready before readback.
        self.device.sync();

        if let Some(b) = albedo_buf {
            if !b.is_null() {
                unsafe { sys::oidnReleaseBuffer(b) };
            }
        }
        if let Some(b) = normal_buf {
            if !b.is_null() {
                unsafe { sys::oidnReleaseBuffer(b) };
            }
        }

        unsafe {
            sys::oidnReadBuffer(
                out_buf,
                0,
                n * std::mem::size_of::<f32>(),
                output.as_mut_ptr() as *mut _,
            );
        }

        if let Some(b) = color_buf {
            unsafe { sys::oidnReleaseBuffer(b) };
        }
        unsafe { sys::oidnReleaseBuffer(out_buf) };

        if let Some(e) = self.device.take_error() {
            return Err(e);
        }
        Ok(())
    }
}

impl Drop for RtFilter<'_> {
    fn drop(&mut self) {
        unsafe { sys::oidnReleaseFilter(self.raw) }
    }
}

unsafe impl Send for RtFilter<'_> {}

// ---------------------------------------------------------------------------
// RTLightmap filter (lightmap denoising; requires OIDN built with RTLightmap)
// ---------------------------------------------------------------------------

/// Ray-traced lightmap denoising filter (OIDN "RTLightmap").
///
/// Use for denoising baked lightmaps. Requires OIDN built with `OIDN_FILTER_RTLIGHTMAP`.
pub struct RtLightmapFilter<'a> {
    device: &'a OidnDevice,
    raw: sys::OIDNFilter,
    width: u32,
    height: u32,
    directional: bool,
}

impl std::fmt::Debug for RtLightmapFilter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RtLightmapFilter")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("directional", &self.directional)
            .finish_non_exhaustive()
    }
}

impl<'a> RtLightmapFilter<'a> {
    /// Creates a new RTLightmap filter. Returns an error if OIDN was not built with RTLightmap support.
    pub fn new(device: &'a OidnDevice) -> Result<Self, Error> {
        let type_name = CString::new("RTLightmap").unwrap();
        let raw = unsafe { sys::oidnNewFilter(device.raw(), type_name.as_ptr()) };
        if raw.is_null() {
            return Err(device.take_error().unwrap_or(Error::FilterCreationFailed));
        }
        Ok(Self {
            device,
            raw,
            width: 0,
            height: 0,
            directional: false,
        })
    }

    /// Image dimensions (must be set before execute).
    pub fn set_dimensions(&mut self, width: u32, height: u32) -> &mut Self {
        self.width = width;
        self.height = height;
        self
    }

    /// If true, use directional lightmap model; if false, HDR. Default: false.
    pub fn set_directional(&mut self, directional: bool) -> &mut Self {
        self.directional = directional;
        self
    }

    /// Gets a boolean filter parameter (e.g. `"directional"`).
    pub fn get_bool(&self, name: &str) -> bool {
        let c_name = CString::new(name).unwrap();
        unsafe { sys::oidnGetFilterBool(self.raw, c_name.as_ptr()) }
    }

    /// Gets an integer filter parameter.
    pub fn get_int(&self, name: &str) -> i32 {
        let c_name = CString::new(name).unwrap();
        unsafe { sys::oidnGetFilterInt(self.raw, c_name.as_ptr()) }
    }

    /// Gets a float filter parameter.
    pub fn get_float(&self, name: &str) -> f32 {
        let c_name = CString::new(name).unwrap();
        unsafe { sys::oidnGetFilterFloat(self.raw, c_name.as_ptr()) }
    }

    /// Sets the progress monitor callback. Call with `(None, null)` to clear.
    ///
    /// # Safety
    ///
    /// `func` must be a valid progress callback that does not panic. `user_ptr` must remain valid until the filter is dropped or the callback is cleared.
    pub unsafe fn set_progress_monitor_raw(
        &self,
        func: sys::OIDNProgressMonitorFunction,
        user_ptr: *mut std::ffi::c_void,
    ) {
        sys::oidnSetFilterProgressMonitorFunction(self.raw, func, user_ptr);
    }

    /// Denoises lightmap in-place. `color` must be `width * height * 3` floats (RGB).
    pub fn execute_in_place(&self, color: &mut [f32]) -> Result<(), Error> {
        self.execute(None, color)
    }

    /// Denoises lightmap: reads from `color` (if provided) or uses `output` as input, writes to `output`.
    /// All buffers must be `width * height * 3` floats (RGB).
    pub fn execute(
        &self,
        color: Option<&[f32]>,
        output: &mut [f32],
    ) -> Result<(), Error> {
        let w = self.width as usize;
        let h = self.height as usize;
        if w == 0 || h == 0 {
            return Err(Error::InvalidDimensions);
        }
        let n = w * h * 3;
        if output.len() != n {
            return Err(Error::InvalidDimensions);
        }
        if let Some(c) = color {
            if c.len() != n {
                return Err(Error::InvalidDimensions);
            }
        }

        let device = self.device.raw();
        let color_buf = if let Some(c) = color {
            let buf = unsafe { sys::oidnNewBuffer(device, n * std::mem::size_of::<f32>()) };
            if buf.is_null() {
                return Err(self.device.take_error().unwrap_or(Error::OutOfMemory));
            }
            unsafe {
                sys::oidnWriteBuffer(buf, 0, n * std::mem::size_of::<f32>(), c.as_ptr() as *const _);
            }
            Some(buf)
        } else {
            None
        };
        let out_buf = unsafe { sys::oidnNewBuffer(device, n * std::mem::size_of::<f32>()) };
        if out_buf.is_null() {
            if let Some(b) = color_buf {
                unsafe { sys::oidnReleaseBuffer(b) };
            }
            return Err(self.device.take_error().unwrap_or(Error::OutOfMemory));
        }
        if color.is_none() {
            unsafe {
                sys::oidnWriteBuffer(
                    out_buf,
                    0,
                    n * std::mem::size_of::<f32>(),
                    output.as_ptr() as *const _,
                );
            }
        }

        let color_ptr = color_buf.unwrap_or(out_buf);
        unsafe {
            let c_color = CString::new("color").unwrap();
            let c_output = CString::new("output").unwrap();
            let c_directional = CString::new("directional").unwrap();
            sys::oidnSetFilterImage(
                self.raw,
                c_color.as_ptr(),
                color_ptr,
                sys::OIDNFormat::Float3,
                w,
                h,
                0,
                0,
                0,
            );
            sys::oidnSetFilterImage(
                self.raw,
                c_output.as_ptr(),
                out_buf,
                sys::OIDNFormat::Float3,
                w,
                h,
                0,
                0,
                0,
            );
            sys::oidnSetFilterInt(self.raw, c_directional.as_ptr(), self.directional as i32);
            sys::oidnCommitFilter(self.raw);
            sys::oidnExecuteFilter(self.raw);
        }
        self.device.sync();
        unsafe {
            sys::oidnReadBuffer(
                out_buf,
                0,
                n * std::mem::size_of::<f32>(),
                output.as_mut_ptr() as *mut _,
            );
        }
        if let Some(b) = color_buf {
            unsafe { sys::oidnReleaseBuffer(b) };
        }
        unsafe { sys::oidnReleaseBuffer(out_buf) };
        if let Some(e) = self.device.take_error() {
            return Err(e);
        }
        Ok(())
    }
}

impl Drop for RtLightmapFilter<'_> {
    fn drop(&mut self) {
        unsafe { sys::oidnReleaseFilter(self.raw) }
    }
}

unsafe impl Send for RtLightmapFilter<'_> {}

// ---------------------------------------------------------------------------
// Generic filter (full OIDN filter API: any type, shared images/data, async)
// ---------------------------------------------------------------------------

/// Image format for filter image parameters. Re-exported so variants (e.g. `OIDNFormat::Float3`) are constructible.
pub use crate::sys::OIDNFormat;
/// Type alias for filter image format (same as `OIDNFormat`).
pub type ImageFormat = sys::OIDNFormat;

/// Generic filter created by type name (e.g. `"RT"`, `"RTLightmap"`).
///
/// Exposes the full OIDN filter API: buffer or shared image/data, progress monitor, async execute.
pub struct Filter<'a> {
    device: &'a OidnDevice,
    raw: sys::OIDNFilter,
}

impl std::fmt::Debug for Filter<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Filter").finish_non_exhaustive()
    }
}

impl<'a> Filter<'a> {
    /// Creates a filter of the given type (e.g. `"RT"`, `"RTLightmap"`).
    pub fn new(device: &'a OidnDevice, type_name: &str) -> Result<Self, Error> {
        let c_name = CString::new(type_name).map_err(|_| Error::FilterCreationFailed)?;
        let raw = unsafe { sys::oidnNewFilter(device.raw(), c_name.as_ptr()) };
        if raw.is_null() {
            return Err(device.take_error().unwrap_or(Error::FilterCreationFailed));
        }
        Ok(Self { device, raw })
    }

    /// Sets an image parameter from an OIDN buffer.
    #[allow(clippy::too_many_arguments)]
    pub fn set_image(
        &self,
        name: &str,
        buffer: &OidnBuffer,
        format: ImageFormat,
        width: usize,
        height: usize,
        byte_offset: usize,
        pixel_byte_stride: usize,
        row_byte_stride: usize,
    ) {
        let c_name = CString::new(name).unwrap();
        unsafe {
            sys::oidnSetFilterImage(
                self.raw,
                c_name.as_ptr(),
                buffer.raw(),
                format,
                width,
                height,
                byte_offset,
                pixel_byte_stride,
                row_byte_stride,
            );
        }
    }

    /// Sets an image parameter from a raw device pointer (zero-copy). Caller keeps ownership.
    ///
    /// # Safety
    ///
    /// `dev_ptr` must point to valid device memory of at least `row_byte_stride * height` bytes; the memory must remain valid and unchanged until the filter is executed or the image is unset.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn set_shared_image(
        &self,
        name: &str,
        dev_ptr: *mut std::ffi::c_void,
        format: ImageFormat,
        width: usize,
        height: usize,
        byte_offset: usize,
        pixel_byte_stride: usize,
        row_byte_stride: usize,
    ) {
        let c_name = CString::new(name).unwrap();
        sys::oidnSetSharedFilterImage(
            self.raw,
            c_name.as_ptr(),
            dev_ptr,
            format,
            width,
            height,
            byte_offset,
            pixel_byte_stride,
            row_byte_stride,
        );
    }

    /// Unsets a previously set image parameter.
    pub fn unset_image(&self, name: &str) {
        let c_name = CString::new(name).unwrap();
        unsafe { sys::oidnUnsetFilterImage(self.raw, c_name.as_ptr()) };
    }

    /// Sets an opaque data parameter (host pointer). Caller keeps ownership.
    ///
    /// # Safety
    ///
    /// `host_ptr` must point to valid, readable host memory of at least `byte_size` bytes; it must remain valid until the filter is executed or the data is unset.
    pub unsafe fn set_shared_data(&self, name: &str, host_ptr: *mut std::ffi::c_void, byte_size: usize) {
        let c_name = CString::new(name).unwrap();
        sys::oidnSetSharedFilterData(self.raw, c_name.as_ptr(), host_ptr, byte_size);
    }

    /// Notifies the filter that the contents of an opaque data parameter have changed.
    pub fn update_data(&self, name: &str) {
        let c_name = CString::new(name).unwrap();
        unsafe { sys::oidnUpdateFilterData(self.raw, c_name.as_ptr()) };
    }

    /// Unsets a previously set opaque data parameter.
    pub fn unset_data(&self, name: &str) {
        let c_name = CString::new(name).unwrap();
        unsafe { sys::oidnUnsetFilterData(self.raw, c_name.as_ptr()) };
    }

    /// Sets a boolean parameter.
    pub fn set_bool(&self, name: &str, value: bool) {
        let c_name = CString::new(name).unwrap();
        unsafe { sys::oidnSetFilterBool(self.raw, c_name.as_ptr(), value) };
    }

    /// Gets a boolean parameter.
    pub fn get_bool(&self, name: &str) -> bool {
        let c_name = CString::new(name).unwrap();
        unsafe { sys::oidnGetFilterBool(self.raw, c_name.as_ptr()) }
    }

    /// Sets an integer parameter.
    pub fn set_int(&self, name: &str, value: i32) {
        let c_name = CString::new(name).unwrap();
        unsafe { sys::oidnSetFilterInt(self.raw, c_name.as_ptr(), value) };
    }

    /// Gets an integer parameter.
    pub fn get_int(&self, name: &str) -> i32 {
        let c_name = CString::new(name).unwrap();
        unsafe { sys::oidnGetFilterInt(self.raw, c_name.as_ptr()) }
    }

    /// Sets a float parameter.
    pub fn set_float(&self, name: &str, value: f32) {
        let c_name = CString::new(name).unwrap();
        unsafe { sys::oidnSetFilterFloat(self.raw, c_name.as_ptr(), value) };
    }

    /// Gets a float parameter.
    pub fn get_float(&self, name: &str) -> f32 {
        let c_name = CString::new(name).unwrap();
        unsafe { sys::oidnGetFilterFloat(self.raw, c_name.as_ptr()) }
    }

    /// Sets the progress monitor callback. Call with `(None, null)` to clear.
    ///
    /// # Safety
    ///
    /// `func` must be a valid progress callback that does not panic. `user_ptr` must remain valid until the filter is dropped or the callback is cleared.
    pub unsafe fn set_progress_monitor_raw(
        &self,
        func: sys::OIDNProgressMonitorFunction,
        user_ptr: *mut std::ffi::c_void,
    ) {
        sys::oidnSetFilterProgressMonitorFunction(self.raw, func, user_ptr);
    }

    /// Commits all previous filter parameter changes. Must be called before execute.
    pub fn commit(&self) {
        unsafe { sys::oidnCommitFilter(self.raw) };
    }

    /// Executes the filter (synchronous). Call `device.sync()` after if using a GPU device.
    pub fn execute(&self) {
        unsafe { sys::oidnExecuteFilter(self.raw) };
    }

    /// Executes the filter asynchronously. Call `device.sync()` before reading output.
    pub fn execute_async(&self) {
        unsafe { sys::oidnExecuteFilterAsync(self.raw) };
    }

    /// Underlying device (for sync/error).
    pub fn device(&self) -> &'a OidnDevice {
        self.device
    }

    /// Retains the filter (increments OIDN reference count). For advanced interop when sharing the filter.
    pub fn retain(&self) {
        unsafe { sys::oidnRetainFilter(self.raw) };
    }
}

impl Drop for Filter<'_> {
    fn drop(&mut self) {
        unsafe { sys::oidnReleaseFilter(self.raw) }
    }
}

unsafe impl Send for Filter<'_> {}
