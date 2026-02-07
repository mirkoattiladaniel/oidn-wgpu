//! OIDN buffer (host/device memory, shared/imported). Full buffer API.
//!
//! See [`OidnBuffer`] for construction and [`BufferStorage`] for storage modes.

use crate::device::OidnDevice;
use crate::error::Error;
use crate::sys;

/// Buffer storage mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum BufferStorage {
    Undefined = 0,
    /// Host-accessible, device can access.
    Host = 1,
    /// Device-only, host cannot access.
    Device = 2,
    /// Managed (migrated between host and device). Check device param `managedMemorySupported`.
    Managed = 3,
}

impl From<sys::OIDNStorage> for BufferStorage {
    fn from(s: sys::OIDNStorage) -> Self {
        match s {
            sys::OIDNStorage::Undefined => BufferStorage::Undefined,
            sys::OIDNStorage::Host => BufferStorage::Host,
            sys::OIDNStorage::Device => BufferStorage::Device,
            sys::OIDNStorage::Managed => BufferStorage::Managed,
        }
    }
}

impl From<BufferStorage> for sys::OIDNStorage {
    fn from(s: BufferStorage) -> sys::OIDNStorage {
        match s {
            BufferStorage::Undefined => sys::OIDNStorage::Undefined,
            BufferStorage::Host => sys::OIDNStorage::Host,
            BufferStorage::Device => sys::OIDNStorage::Device,
            BufferStorage::Managed => sys::OIDNStorage::Managed,
        }
    }
}

/// External memory type for importing buffers (FD, Win32, etc.).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum ExternalMemoryTypeFlag {
    None = 0,
    OpaqueFD = 1 << 0,
    DmaBuf = 1 << 1,
    OpaqueWin32 = 1 << 2,
    OpaqueWin32KMT = 1 << 3,
    D3D11Texture = 1 << 4,
    D3D11TextureKMT = 1 << 5,
    D3D11Resource = 1 << 6,
    D3D11ResourceKMT = 1 << 7,
    D3D12Heap = 1 << 8,
    D3D12Resource = 1 << 9,
}

impl From<ExternalMemoryTypeFlag> for sys::OIDNExternalMemoryTypeFlag {
    fn from(f: ExternalMemoryTypeFlag) -> sys::OIDNExternalMemoryTypeFlag {
        unsafe { std::mem::transmute(f as u32) }
    }
}

/// OIDN buffer. Owns or wraps device-accessible memory.
pub struct OidnBuffer {
    pub(crate) raw: sys::OIDNBuffer,
    _device: std::marker::PhantomData<OidnDevice>,
}

impl std::fmt::Debug for OidnBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OidnBuffer")
            .field("size", &self.size())
            .field("storage", &self.storage())
            .finish_non_exhaustive()
    }
}

impl OidnBuffer {
    /// Creates a buffer (host and device accessible) of the given size in bytes.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfMemory`] if allocation fails, or an OIDN error from the device.
    pub fn new(device: &OidnDevice, byte_size: usize) -> Result<Self, Error> {
        let raw = unsafe { sys::oidnNewBuffer(device.raw(), byte_size) };
        if raw.is_null() {
            return Err(device.take_error().unwrap_or(Error::OutOfMemory));
        }
        Ok(Self {
            raw,
            _device: std::marker::PhantomData,
        })
    }

    /// Creates a buffer with the specified storage mode.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfMemory`] if allocation fails, or an OIDN error from the device.
    pub fn new_with_storage(
        device: &OidnDevice,
        byte_size: usize,
        storage: BufferStorage,
    ) -> Result<Self, Error> {
        let raw = unsafe {
            sys::oidnNewBufferWithStorage(device.raw(), byte_size, storage.into())
        };
        if raw.is_null() {
            return Err(device.take_error().unwrap_or(Error::OutOfMemory));
        }
        Ok(Self {
            raw,
            _device: std::marker::PhantomData,
        })
    }

    /// Creates a shared buffer from user-owned device memory. OIDN does not take ownership.
    ///
    /// # Safety
    ///
    /// `dev_ptr` must point to valid device memory of at least `byte_size` bytes, and remain
    /// valid for the lifetime of OIDN operations using this buffer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfMemory`] or an OIDN error if creation fails.
    pub unsafe fn new_shared(device: &OidnDevice, dev_ptr: *mut std::ffi::c_void, byte_size: usize) -> Result<Self, Error> {
        let raw = sys::oidnNewSharedBuffer(device.raw(), dev_ptr, byte_size);
        if raw.is_null() {
            return Err(device.take_error().unwrap_or(Error::OutOfMemory));
        }
        Ok(Self {
            raw,
            _device: std::marker::PhantomData,
        })
    }

    /// Creates a shared buffer from a POSIX file descriptor (e.g. DMA-BUF).
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfMemory`] or an OIDN error if creation fails.
    pub fn new_shared_from_fd(
        device: &OidnDevice,
        fd_type: ExternalMemoryTypeFlag,
        fd: i32,
        byte_size: usize,
    ) -> Result<Self, Error> {
        let raw = unsafe {
            sys::oidnNewSharedBufferFromFD(device.raw(), fd_type.into(), fd, byte_size)
        };
        if raw.is_null() {
            return Err(device.take_error().unwrap_or(Error::OutOfMemory));
        }
        Ok(Self {
            raw,
            _device: std::marker::PhantomData,
        })
    }

    /// Creates a shared buffer from a Win32 handle.
    ///
    /// # Safety
    ///
    /// `handle` and `name` must be valid for the given `handle_type` and remain valid for the
    /// lifetime of OIDN operations using this buffer.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfMemory`] or an OIDN error if creation fails.
    pub unsafe fn new_shared_from_win32_handle(
        device: &OidnDevice,
        handle_type: ExternalMemoryTypeFlag,
        handle: *mut std::ffi::c_void,
        name: *const std::ffi::c_void,
        byte_size: usize,
    ) -> Result<Self, Error> {
        let raw = sys::oidnNewSharedBufferFromWin32Handle(
            device.raw(),
            handle_type.into(),
            handle,
            name,
            byte_size,
        );
        if raw.is_null() {
            return Err(device.take_error().unwrap_or(Error::OutOfMemory));
        }
        Ok(Self {
            raw,
            _device: std::marker::PhantomData,
        })
    }

    /// Creates a shared buffer from a Metal buffer (MTLBuffer). Only shared/private with hazard tracking.
    ///
    /// # Safety
    ///
    /// `metal_buffer` must be a valid MTLBuffer and remain valid for the lifetime of OIDN operations.
    ///
    /// # Errors
    ///
    /// Returns [`Error::OutOfMemory`] or an OIDN error if creation fails.
    pub unsafe fn new_shared_from_metal(
        device: &OidnDevice,
        metal_buffer: *mut std::ffi::c_void,
    ) -> Result<Self, Error> {
        let raw = sys::oidnNewSharedBufferFromMetal(device.raw(), metal_buffer);
        if raw.is_null() {
            return Err(device.take_error().unwrap_or(Error::OutOfMemory));
        }
        Ok(Self {
            raw,
            _device: std::marker::PhantomData,
        })
    }

    /// Size in bytes.
    pub fn size(&self) -> usize {
        unsafe { sys::oidnGetBufferSize(self.raw) }
    }

    /// Storage mode.
    pub fn storage(&self) -> BufferStorage {
        unsafe { sys::oidnGetBufferStorage(self.raw) }.into()
    }

    /// Raw pointer to buffer data (device-accessible; may be null for device-only storage).
    pub fn data(&self) -> *mut std::ffi::c_void {
        unsafe { sys::oidnGetBufferData(self.raw) }
    }

    /// Copies from buffer to host memory (synchronous).
    ///
    /// # Safety
    /// `dst` must point to at least `byte_size` bytes of valid, writable memory.
    pub unsafe fn read(&self, byte_offset: usize, byte_size: usize, dst: *mut std::ffi::c_void) {
        sys::oidnReadBuffer(self.raw, byte_offset, byte_size, dst);
    }

    /// Copies from buffer to host memory (asynchronous). Call `device.sync()` before using `dst`.
    ///
    /// # Safety
    /// `dst` must point to at least `byte_size` bytes of valid, writable memory.
    pub unsafe fn read_async(&self, byte_offset: usize, byte_size: usize, dst: *mut std::ffi::c_void) {
        sys::oidnReadBufferAsync(self.raw, byte_offset, byte_size, dst);
    }

    /// Copies from host memory to buffer (synchronous).
    ///
    /// # Safety
    /// `src` must point to at least `byte_size` bytes of valid, readable memory.
    pub unsafe fn write(&self, byte_offset: usize, byte_size: usize, src: *const std::ffi::c_void) {
        sys::oidnWriteBuffer(self.raw, byte_offset, byte_size, src);
    }

    /// Copies from host memory to buffer (asynchronous). Call `device.sync()` before using buffer in a filter.
    ///
    /// # Safety
    /// `src` must point to at least `byte_size` bytes of valid, readable memory.
    pub unsafe fn write_async(&self, byte_offset: usize, byte_size: usize, src: *const std::ffi::c_void) {
        sys::oidnWriteBufferAsync(self.raw, byte_offset, byte_size, src);
    }

    /// Retains the buffer (increments OIDN reference count). For advanced interop when sharing the buffer.
    pub fn retain(&self) {
        unsafe { sys::oidnRetainBuffer(self.raw) };
    }

    pub(crate) fn raw(&self) -> sys::OIDNBuffer {
        self.raw
    }
}

impl Drop for OidnBuffer {
    fn drop(&mut self) {
        unsafe { sys::oidnReleaseBuffer(self.raw) }
    }
}

unsafe impl Send for OidnBuffer {}
unsafe impl Sync for OidnBuffer {}
