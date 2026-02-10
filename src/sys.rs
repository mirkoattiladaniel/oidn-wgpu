//! Raw FFI bindings to Intel Open Image Denoise (OIDN) C API.
//! Targets OIDN 2.4.x. Full API coverage. See <https://www.openimagedenoise.org>.

#![allow(non_camel_case_types, non_upper_case_globals, dead_code)]

use std::os::raw::{c_char, c_int, c_void};

// ---------------------------------------------------------------------------
// Constants (from oidn.h)
// ---------------------------------------------------------------------------

pub const OIDN_UUID_SIZE: usize = 16;
pub const OIDN_LUID_SIZE: usize = 8;

// ---------------------------------------------------------------------------
// Opaque handles
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct OIDNDeviceImpl {
    _private: [u8; 0],
}
#[repr(C)]
pub struct OIDNBufferImpl {
    _private: [u8; 0],
}
#[repr(C)]
pub struct OIDNFilterImpl {
    _private: [u8; 0],
}

pub type OIDNDevice = *mut OIDNDeviceImpl;
pub type OIDNBuffer = *mut OIDNBufferImpl;
pub type OIDNFilter = *mut OIDNFilterImpl;

// ---------------------------------------------------------------------------
// Enums
// ---------------------------------------------------------------------------

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum OIDNDeviceType {
    Default = 0,
    CPU = 1,
    SYCL = 2,
    CUDA = 3,
    HIP = 4,
    Metal = 5,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OIDNError {
    None = 0,
    Unknown = 1,
    InvalidArgument = 2,
    InvalidOperation = 3,
    OutOfMemory = 4,
    UnsupportedHardware = 5,
    Cancelled = 6,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OIDNFormat {
    Undefined = 0,
    Float = 1,
    Float2,
    Float3,
    Float4,
    Half = 257,
    Half2,
    Half3,
    Half4,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OIDNStorage {
    Undefined = 0,
    Host = 1,
    Device = 2,
    Managed = 3,
}

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OIDNExternalMemoryTypeFlag {
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

#[repr(u32)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OIDNQuality {
    Default = 0,
    Fast = 4,
    Balanced = 5,
    High = 6,
}

// Callbacks
pub type OIDNErrorFunction =
    Option<unsafe extern "C" fn(user_ptr: *mut c_void, code: OIDNError, message: *const c_char)>;
pub type OIDNProgressMonitorFunction = Option<unsafe extern "C" fn(user_ptr: *mut c_void, n: f64) -> bool>;

// ---------------------------------------------------------------------------
// Physical device
// ---------------------------------------------------------------------------

extern "C" {
    pub fn oidnGetNumPhysicalDevices() -> c_int;
    pub fn oidnGetPhysicalDeviceBool(physicalDeviceID: c_int, name: *const c_char) -> bool;
    pub fn oidnGetPhysicalDeviceInt(physicalDeviceID: c_int, name: *const c_char) -> c_int;
    pub fn oidnGetPhysicalDeviceString(physicalDeviceID: c_int, name: *const c_char) -> *const c_char;
    pub fn oidnGetPhysicalDeviceData(
        physicalDeviceID: c_int,
        name: *const c_char,
        byteSize: *mut usize,
    ) -> *const c_void;
}

// ---------------------------------------------------------------------------
// Device creation and support
// ---------------------------------------------------------------------------

extern "C" {
    pub fn oidnIsCPUDeviceSupported() -> bool;
    pub fn oidnIsCUDADeviceSupported(deviceID: c_int) -> bool;
    pub fn oidnIsHIPDeviceSupported(deviceID: c_int) -> bool;
    pub fn oidnIsMetalDeviceSupported(device: *mut c_void) -> bool;

    pub fn oidnNewDevice(type_: OIDNDeviceType) -> OIDNDevice;
    pub fn oidnNewDeviceByID(physicalDeviceID: c_int) -> OIDNDevice;
    pub fn oidnNewDeviceByUUID(uuid: *const c_void) -> OIDNDevice;
    pub fn oidnNewDeviceByLUID(luid: *const c_void) -> OIDNDevice;
    pub fn oidnNewDeviceByPCIAddress(
        pciDomain: c_int,
        pciBus: c_int,
        pciDevice: c_int,
        pciFunction: c_int,
    ) -> OIDNDevice;
    /// deviceIDs and streams: arrays of length numPairs. stream null = default stream.
    pub fn oidnNewCUDADevice(
        deviceIDs: *const c_int,
        streams: *const *mut c_void,
        numPairs: c_int,
    ) -> OIDNDevice;
    pub fn oidnNewHIPDevice(
        deviceIDs: *const c_int,
        streams: *const *mut c_void,
        numPairs: c_int,
    ) -> OIDNDevice;
    /// commandQueues: array of Metal MTLCommandQueue (void* when not ObjC).
    pub fn oidnNewMetalDevice(commandQueues: *const *mut c_void, numQueues: c_int) -> OIDNDevice;

    pub fn oidnRetainDevice(device: OIDNDevice);
    pub fn oidnReleaseDevice(device: OIDNDevice);
    pub fn oidnSetDeviceBool(device: OIDNDevice, name: *const c_char, value: bool);
    pub fn oidnSetDeviceInt(device: OIDNDevice, name: *const c_char, value: c_int);
    pub fn oidnGetDeviceBool(device: OIDNDevice, name: *const c_char) -> bool;
    pub fn oidnGetDeviceInt(device: OIDNDevice, name: *const c_char) -> c_int;
    pub fn oidnSetDeviceErrorFunction(
        device: OIDNDevice,
        func: OIDNErrorFunction,
        user_ptr: *mut c_void,
    );
    pub fn oidnGetDeviceError(device: OIDNDevice, out_message: *mut *const c_char) -> OIDNError;
    pub fn oidnCommitDevice(device: OIDNDevice);
    pub fn oidnSyncDevice(device: OIDNDevice);
}

// SYCL device/queue APIs (oidnIsSYCLDeviceSupported, oidnNewSYCLDevice, oidnExecuteSYCLFilterAsync)
// are C++-only in OIDN (sycl::device*, sycl::queue*, sycl::event*). Not exposed from Rust;
// use the type-based oidnNewDevice(OIDN_DEVICE_TYPE_SYCL) instead, or a C++ shim for interop.

// ---------------------------------------------------------------------------
// Buffer
// ---------------------------------------------------------------------------

extern "C" {
    pub fn oidnNewBuffer(device: OIDNDevice, byte_size: usize) -> OIDNBuffer;
    pub fn oidnNewBufferWithStorage(
        device: OIDNDevice,
        byte_size: usize,
        storage: OIDNStorage,
    ) -> OIDNBuffer;
    pub fn oidnNewSharedBuffer(device: OIDNDevice, dev_ptr: *mut c_void, byte_size: usize) -> OIDNBuffer;
    pub fn oidnNewSharedBufferFromFD(
        device: OIDNDevice,
        fd_type: OIDNExternalMemoryTypeFlag,
        fd: c_int,
        byte_size: usize,
    ) -> OIDNBuffer;
    pub fn oidnNewSharedBufferFromWin32Handle(
        device: OIDNDevice,
        handle_type: OIDNExternalMemoryTypeFlag,
        handle: *mut c_void,
        name: *const c_void,
        byte_size: usize,
    ) -> OIDNBuffer;
    pub fn oidnNewSharedBufferFromMetal(device: OIDNDevice, buffer: *mut c_void) -> OIDNBuffer;

    pub fn oidnGetBufferSize(buffer: OIDNBuffer) -> usize;
    pub fn oidnGetBufferStorage(buffer: OIDNBuffer) -> OIDNStorage;
    pub fn oidnGetBufferData(buffer: OIDNBuffer) -> *mut c_void;
    pub fn oidnReadBuffer(
        buffer: OIDNBuffer,
        byte_offset: usize,
        byte_size: usize,
        dst_host_ptr: *mut c_void,
    );
    pub fn oidnReadBufferAsync(
        buffer: OIDNBuffer,
        byte_offset: usize,
        byte_size: usize,
        dst_host_ptr: *mut c_void,
    );
    pub fn oidnWriteBuffer(
        buffer: OIDNBuffer,
        byte_offset: usize,
        byte_size: usize,
        src_host_ptr: *const c_void,
    );
    pub fn oidnWriteBufferAsync(
        buffer: OIDNBuffer,
        byte_offset: usize,
        byte_size: usize,
        src_host_ptr: *const c_void,
    );
    pub fn oidnRetainBuffer(buffer: OIDNBuffer);
    pub fn oidnReleaseBuffer(buffer: OIDNBuffer);
}

// ---------------------------------------------------------------------------
// Filter
// ---------------------------------------------------------------------------

extern "C" {
    pub fn oidnNewFilter(device: OIDNDevice, type_name: *const c_char) -> OIDNFilter;
    pub fn oidnRetainFilter(filter: OIDNFilter);
    pub fn oidnReleaseFilter(filter: OIDNFilter);
    pub fn oidnSetFilterImage(
        filter: OIDNFilter,
        name: *const c_char,
        buffer: OIDNBuffer,
        format: OIDNFormat,
        width: usize,
        height: usize,
        byte_offset: usize,
        pixel_byte_stride: usize,
        row_byte_stride: usize,
    );
    pub fn oidnSetSharedFilterImage(
        filter: OIDNFilter,
        name: *const c_char,
        dev_ptr: *mut c_void,
        format: OIDNFormat,
        width: usize,
        height: usize,
        byte_offset: usize,
        pixel_byte_stride: usize,
        row_byte_stride: usize,
    );
    pub fn oidnUnsetFilterImage(filter: OIDNFilter, name: *const c_char);
    pub fn oidnSetSharedFilterData(
        filter: OIDNFilter,
        name: *const c_char,
        host_ptr: *mut c_void,
        byte_size: usize,
    );
    pub fn oidnUpdateFilterData(filter: OIDNFilter, name: *const c_char);
    pub fn oidnUnsetFilterData(filter: OIDNFilter, name: *const c_char);
    pub fn oidnSetFilterBool(filter: OIDNFilter, name: *const c_char, value: bool);
    pub fn oidnGetFilterBool(filter: OIDNFilter, name: *const c_char) -> bool;
    pub fn oidnSetFilterInt(filter: OIDNFilter, name: *const c_char, value: c_int);
    pub fn oidnGetFilterInt(filter: OIDNFilter, name: *const c_char) -> c_int;
    pub fn oidnSetFilterFloat(filter: OIDNFilter, name: *const c_char, value: f32);
    pub fn oidnGetFilterFloat(filter: OIDNFilter, name: *const c_char) -> f32;
    pub fn oidnSetFilterProgressMonitorFunction(
        filter: OIDNFilter,
        func: OIDNProgressMonitorFunction,
        user_ptr: *mut c_void,
    );
    pub fn oidnCommitFilter(filter: OIDNFilter);
    pub fn oidnExecuteFilter(filter: OIDNFilter);
    pub fn oidnExecuteFilterAsync(filter: OIDNFilter);
}
