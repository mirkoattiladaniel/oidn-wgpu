//! OIDN logical device (CPU or GPU backend) and physical device queries.
//! Full API: type-based and physical-ID/UUID/LUID/PCI/CUDA/HIP/Metal device creation.

use crate::sys;
use crate::Error;
use std::ffi::{CStr, CString};
use std::ptr;
use std::sync::Arc;

/// Number of physical devices supported by OIDN. Valid IDs are `0 .. num_physical_devices()`.
pub fn num_physical_devices() -> i32 {
    unsafe { sys::oidnGetNumPhysicalDevices() }
}

/// Returns a boolean parameter of the physical device. `name` is e.g. `"type"`, `"name"`.
pub fn get_physical_device_bool(physical_device_id: i32, name: &str) -> bool {
    let c_name = CString::new(name).unwrap();
    unsafe { sys::oidnGetPhysicalDeviceBool(physical_device_id, c_name.as_ptr()) }
}

/// Returns an integer parameter of the physical device.
pub fn get_physical_device_int(physical_device_id: i32, name: &str) -> i32 {
    let c_name = CString::new(name).unwrap();
    unsafe { sys::oidnGetPhysicalDeviceInt(physical_device_id, c_name.as_ptr()) }
}

/// Returns a string parameter of the physical device. Pointer valid until next OIDN call.
pub fn get_physical_device_string(physical_device_id: i32, name: &str) -> Option<String> {
    let c_name = CString::new(name).unwrap();
    let p = unsafe { sys::oidnGetPhysicalDeviceString(physical_device_id, c_name.as_ptr()) };
    if p.is_null() {
        return None;
    }
    Some(unsafe { CStr::from_ptr(p).to_string_lossy().into_owned() })
}

/// Returns opaque data and its size for the physical device. Pointer valid until next OIDN call.
pub fn get_physical_device_data(physical_device_id: i32, name: &str) -> Option<(*const std::ffi::c_void, usize)> {
    let c_name = CString::new(name).unwrap();
    let mut size = 0usize;
    let p = unsafe { sys::oidnGetPhysicalDeviceData(physical_device_id, c_name.as_ptr(), &mut size) };
    if p.is_null() {
        None
    } else {
        Some((p, size))
    }
}

/// Whether the CPU device is supported.
pub fn is_cpu_device_supported() -> bool {
    unsafe { sys::oidnIsCPUDeviceSupported() }
}

/// Whether the given CUDA device ID is supported.
pub fn is_cuda_device_supported(device_id: i32) -> bool {
    unsafe { sys::oidnIsCUDADeviceSupported(device_id) }
}

/// Whether the given HIP device ID is supported.
pub fn is_hip_device_supported(device_id: i32) -> bool {
    unsafe { sys::oidnIsHIPDeviceSupported(device_id) }
}

/// Whether the given Metal device (MTLDevice, passed as raw pointer) is supported.
pub unsafe fn is_metal_device_supported(device: *mut std::ffi::c_void) -> bool {
    sys::oidnIsMetalDeviceSupported(device)
}

/// Returns the first unqueried error for the current thread (e.g. from a failed device creation)
/// and clears it. Can be called without a device to check why `OidnDevice::new()` or similar failed.
pub fn take_global_error() -> Option<Error> {
    let mut msg_ptr: *const std::ffi::c_char = ptr::null();
    let code = unsafe { sys::oidnGetDeviceError(ptr::null_mut(), &mut msg_ptr) };
    if code == sys::OIDNError::None {
        return None;
    }
    let message = if msg_ptr.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(msg_ptr).to_string_lossy().into_owned() }
    };
    Some(Error::OidnError { code: code as u32, message })
}

/// Open Image Denoise logical device.
///
/// Prefer creating one per application and reusing it; filter creation is relatively expensive.
/// See [`Self::new`], [`Self::cpu`], and backend-specific constructors.
#[derive(Clone)]
pub struct OidnDevice {
    pub(crate) raw: sys::OIDNDevice,
    _refcount: Arc<()>,
}

impl std::fmt::Debug for OidnDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OidnDevice").finish_non_exhaustive()
    }
}

impl OidnDevice {
    /// Creates a device using the default backend (auto-selects CPU or GPU when available).
    ///
    /// # Errors
    ///
    /// Returns [`Error::DeviceCreationFailed`] if no backend is available. Use [`take_global_error()`]
    /// to retrieve the underlying OIDN message.
    pub fn new() -> Result<Self, Error> {
        Self::with_type(OidnDeviceType::Default)
    }

    /// Creates a CPU-only device (most portable).
    pub fn cpu() -> Result<Self, Error> {
        Self::with_type(OidnDeviceType::Cpu)
    }

    /// Creates a CUDA device for NVIDIA GPU-accelerated denoising.
    /// Requires OIDN built with CUDA support; returns `DeviceCreationFailed` otherwise.
    pub fn cuda() -> Result<Self, Error> {
        Self::with_type(OidnDeviceType::Cuda)
    }

    /// Creates a SYCL device (Intel GPU/CPU via oneAPI). Requires OIDN built with SYCL.
    pub fn sycl() -> Result<Self, Error> {
        Self::with_type(OidnDeviceType::Sycl)
    }

    /// Creates a HIP device (AMD GPU). Requires OIDN built with HIP.
    pub fn hip() -> Result<Self, Error> {
        Self::with_type(OidnDeviceType::Hip)
    }

    /// Creates a Metal device (Apple GPU). Requires OIDN built with Metal.
    pub fn metal() -> Result<Self, Error> {
        Self::with_type(OidnDeviceType::Metal)
    }

    /// Creates a device of the given type.
    pub fn with_type(device_type: OidnDeviceType) -> Result<Self, Error> {
        let raw = unsafe { sys::oidnNewDevice(device_type.to_raw()) };
        if raw.is_null() {
            return Err(Error::DeviceCreationFailed);
        }
        unsafe { sys::oidnCommitDevice(raw) };
        Ok(Self {
            raw,
            _refcount: Arc::new(()),
        })
    }

    /// Creates a device from a physical device ID (0 to `num_physical_devices()` - 1).
    pub fn new_by_id(physical_device_id: i32) -> Result<Self, Error> {
        let raw = unsafe { sys::oidnNewDeviceByID(physical_device_id) };
        if raw.is_null() {
            return Err(Error::DeviceCreationFailed);
        }
        unsafe { sys::oidnCommitDevice(raw) };
        Ok(Self { raw, _refcount: Arc::new(()) })
    }

    /// Creates a device from a physical device UUID (16 bytes; see [`crate::OIDN_UUID_SIZE`]).
    pub fn new_by_uuid(uuid: &[u8; sys::OIDN_UUID_SIZE]) -> Result<Self, Error> {
        let raw = unsafe { sys::oidnNewDeviceByUUID(uuid.as_ptr() as *const std::ffi::c_void) };
        if raw.is_null() {
            return Err(Error::DeviceCreationFailed);
        }
        unsafe { sys::oidnCommitDevice(raw) };
        Ok(Self { raw, _refcount: Arc::new(()) })
    }

    /// Creates a device from a physical device LUID (8 bytes; see [`crate::OIDN_LUID_SIZE`]).
    pub fn new_by_luid(luid: &[u8; sys::OIDN_LUID_SIZE]) -> Result<Self, Error> {
        let raw = unsafe { sys::oidnNewDeviceByLUID(luid.as_ptr() as *const std::ffi::c_void) };
        if raw.is_null() {
            return Err(Error::DeviceCreationFailed);
        }
        unsafe { sys::oidnCommitDevice(raw) };
        Ok(Self { raw, _refcount: Arc::new(()) })
    }

    /// Creates a device from a PCI address (domain, bus, device, function).
    pub fn new_by_pci_address(
        pci_domain: i32,
        pci_bus: i32,
        pci_device: i32,
        pci_function: i32,
    ) -> Result<Self, Error> {
        let raw = unsafe {
            sys::oidnNewDeviceByPCIAddress(pci_domain, pci_bus, pci_device, pci_function)
        };
        if raw.is_null() {
            return Err(Error::DeviceCreationFailed);
        }
        unsafe { sys::oidnCommitDevice(raw) };
        Ok(Self { raw, _refcount: Arc::new(()) })
    }

    /// Creates a CUDA device for the given device ID and optional stream.
    /// `stream`: `None` = default stream; otherwise a valid `cudaStream_t` (e.g. from rust CUDA bindings).
    /// Currently only one (device_id, stream) pair is supported.
    pub unsafe fn new_cuda_device(
        device_id: i32,
        stream: Option<*mut std::ffi::c_void>,
    ) -> Result<Self, Error> {
        let stream_ptr = stream.unwrap_or(ptr::null_mut());
        let raw = sys::oidnNewCUDADevice(&device_id, &stream_ptr, 1);
        if raw.is_null() {
            return Err(Error::DeviceCreationFailed);
        }
        sys::oidnCommitDevice(raw);
        Ok(Self { raw, _refcount: Arc::new(()) })
    }

    /// Creates a HIP device for the given device ID and optional stream.
    /// `stream`: `None` = default stream. Currently only one pair is supported.
    pub unsafe fn new_hip_device(
        device_id: i32,
        stream: Option<*mut std::ffi::c_void>,
    ) -> Result<Self, Error> {
        let stream_ptr = stream.unwrap_or(ptr::null_mut());
        let raw = sys::oidnNewHIPDevice(&device_id, &stream_ptr, 1);
        if raw.is_null() {
            return Err(Error::DeviceCreationFailed);
        }
        sys::oidnCommitDevice(raw);
        Ok(Self { raw, _refcount: Arc::new(()) })
    }

    /// Creates a Metal device from an array of Metal command queues (MTLCommandQueue).
    /// Currently only one queue is supported. Pass a single pointer.
    pub unsafe fn new_metal_device(command_queues: &[*mut std::ffi::c_void]) -> Result<Self, Error> {
        let raw = sys::oidnNewMetalDevice(command_queues.as_ptr(), command_queues.len() as i32);
        if raw.is_null() {
            return Err(Error::DeviceCreationFailed);
        }
        sys::oidnCommitDevice(raw);
        Ok(Self { raw, _refcount: Arc::new(()) })
    }

    /// Sets a boolean device parameter. Must call `commit()` before first use if you change parameters.
    pub fn set_bool(&self, name: &str, value: bool) {
        let c_name = CString::new(name).unwrap();
        unsafe { sys::oidnSetDeviceBool(self.raw, c_name.as_ptr(), value) };
    }

    /// Sets an integer device parameter.
    pub fn set_int(&self, name: &str, value: i32) {
        let c_name = CString::new(name).unwrap();
        unsafe { sys::oidnSetDeviceInt(self.raw, c_name.as_ptr(), value) };
    }

    /// Gets a boolean device parameter.
    pub fn get_bool(&self, name: &str) -> bool {
        let c_name = CString::new(name).unwrap();
        unsafe { sys::oidnGetDeviceBool(self.raw, c_name.as_ptr()) }
    }

    /// Gets an integer device parameter.
    pub fn get_int(&self, name: &str) -> i32 {
        let c_name = CString::new(name).unwrap();
        unsafe { sys::oidnGetDeviceInt(self.raw, c_name.as_ptr()) }
    }

    /// Gets an unsigned integer device parameter (OIDN exposes this as cast of get_int).
    pub fn get_uint(&self, name: &str) -> u32 {
        self.get_int(name) as u32
    }

    /// Commits all previous device parameter changes. Must be called before first filter creation.
    pub fn commit(&self) {
        unsafe { sys::oidnCommitDevice(self.raw) };
    }

    /// Sets the error callback. The callback is invoked from OIDN; it must not panic.
    /// `user_ptr` is passed to the callback. Must remain valid until device is released or callback is cleared.
    pub unsafe fn set_error_function_raw(
        &self,
        func: sys::OIDNErrorFunction,
        user_ptr: *mut std::ffi::c_void,
    ) {
        sys::oidnSetDeviceErrorFunction(self.raw, func, user_ptr);
    }

    /// Returns the first unqueried error and clears it.
    pub fn take_error(&self) -> Option<Error> {
        let mut msg_ptr: *const std::ffi::c_char = ptr::null();
        let code = unsafe { sys::oidnGetDeviceError(self.raw, &mut msg_ptr) };
        if code == sys::OIDNError::None {
            return None;
        }
        let message = if msg_ptr.is_null() {
            String::new()
        } else {
            unsafe { CStr::from_ptr(msg_ptr).to_string_lossy().into_owned() }
        };
        Some(Error::OidnError { code: code as u32, message })
    }

    /// Waits for all async operations on this device to complete.
    pub fn sync(&self) {
        unsafe { sys::oidnSyncDevice(self.raw) };
    }

    /// Retains the device (increments OIDN reference count). For advanced interop only; our `Clone` uses Arc.
    pub fn retain(&self) {
        unsafe { sys::oidnRetainDevice(self.raw) };
    }

    pub(crate) fn raw(&self) -> sys::OIDNDevice {
        self.raw
    }
}

impl Drop for OidnDevice {
    fn drop(&mut self) {
        unsafe { sys::oidnReleaseDevice(self.raw) }
    }
}

unsafe impl Send for OidnDevice {}
unsafe impl Sync for OidnDevice {}

/// OIDN device type (CPU, GPU backends, or default auto-select).
#[derive(Clone, Copy, Debug, Default)]
pub enum OidnDeviceType {
    /// Auto-select best available (e.g. CUDA if built and available).
    #[default]
    Default,
    /// CPU only (most portable).
    Cpu,
    /// Intel GPU/CPU via SYCL (oneAPI). Requires OIDN built with SYCL.
    Sycl,
    /// NVIDIA GPU via CUDA. Requires OIDN built with CUDA.
    Cuda,
    /// AMD GPU via HIP. Requires OIDN built with HIP.
    Hip,
    /// Apple GPU via Metal. Requires OIDN built with Metal.
    Metal,
}

impl OidnDeviceType {
    fn to_raw(self) -> sys::OIDNDeviceType {
        match self {
            OidnDeviceType::Default => sys::OIDNDeviceType::Default,
            OidnDeviceType::Cpu => sys::OIDNDeviceType::CPU,
            OidnDeviceType::Sycl => sys::OIDNDeviceType::SYCL,
            OidnDeviceType::Cuda => sys::OIDNDeviceType::CUDA,
            OidnDeviceType::Hip => sys::OIDNDeviceType::HIP,
            OidnDeviceType::Metal => sys::OIDNDeviceType::Metal,
        }
    }
}
