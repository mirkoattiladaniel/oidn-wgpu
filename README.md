# oidn-wgpu

[Intel Open Image Denoise](https://www.openimagedenoise.org) (OIDN) integration for [wgpu](https://docs.rs/wgpu). Denoise path-traced or ray-traced images produced on the GPU by copying to CPU, running OIDN, and copying back.

- **OIDN 2.4.x** — uses the latest API (quality modes, etc.).
- **GPU support** — `OidnDevice::cuda()`, `sycl()`, `hip()`, `metal()` for GPU backends (when OIDN is built with them).
- **wgpu 27** — compatible with wgpu 27.

This crate is generic and can be used from any Rust project using wgpu (e.g. game engines, renderers, tools).

## Setup

You need a built **OIDN 2.4.x** library (e.g. from [OpenImageDenoise/oidn](https://github.com/OpenImageDenoise/oidn)).

1. **Option A — `OIDN_DIR`**  
   Set the environment variable to your OIDN **install** directory (containing `include/` and `lib/`), or to the build output directory that contains `OpenImageDenoise.lib` (Windows) / `libOpenImageDenoise.a` or `.so` (Unix).

   ```bash
   # Windows (PowerShell) — set OIDN_DIR then build in the same session
   $env:OIDN_DIR = "C:\path\to\oidn\build"   # or install dir; must contain OpenImageDenoise.lib
   cargo build

   # Linux / macOS
   export OIDN_DIR=/path/to/oidn/install
   ```

2. **Option B — pkg-config**  
   Install OIDN so that `pkg-config --libs OpenImageDenoise` works (common on Linux).

**Windows (dynamic build):** If OIDN was built as a DLL, ensure `OpenImageDenoise.dll` (and any device DLLs, e.g. `OpenImageDenoise_device_cuda.dll`) are on your `PATH` or next to your executable at run time.

Then add to your `Cargo.toml`:

```toml
[dependencies]
oidn-wgpu = "0.1"
```

## Usage

### Denoise a wgpu texture

Use when your path tracer writes to a `wgpu::Texture` (e.g. `Rgba16Float` or `Rgba32Float`). This will read back the texture to CPU, run OIDN, and upload the result.

```rust
use oidn_wgpu::{
    OidnDevice, denoise_texture,
    DenoiseTextureFormat, DenoiseOptions,
};

// One-time setup: create OIDN device (reuse across frames).
let oidn = OidnDevice::new()?;

// When you want to denoise a frame:
let format = DenoiseTextureFormat::Rgba16Float; // or Rgba32Float
denoise_texture(
    &oidn,
    &wgpu_device,
    &wgpu_queue,
    &noisy_texture,
    &output_texture,  // can be the same as input for in-place
    format,
    &DenoiseOptions {
        quality: oidn_wgpu::Quality::Balanced, // Fast | Balanced | High
        hdr: true,
        srgb: false,
        input_scale: None,  // Some(scale) for HDR exposure
    },
)?;
```

**Device types:** `OidnDevice::new()` (auto), `OidnDevice::cpu()`, `OidnDevice::cuda()`, `OidnDevice::sycl()`, `OidnDevice::hip()`, `OidnDevice::metal()`.

Supported texture formats: **`Rgba16Float`**, **`Rgba32Float`**. Alpha is preserved; only RGB is denoised.

### Denoise CPU buffers (no wgpu)

If you already have RGB float data (e.g. from a different backend):

```rust
use oidn_wgpu::{OidnDevice, RtFilter};

let device = OidnDevice::new()?;
let mut filter = RtFilter::new(&device)?;
filter
    .set_dimensions(width, height)
    .set_hdr(true)
    .set_quality(oidn_wgpu::Quality::High);
// color: &mut [f32] with length width * height * 3 (RGB)
filter.execute_in_place(&mut color_rgb_f32)?;
```

### Denoise with albedo and normal (wgpu textures)

For higher quality, pass optional albedo and normal textures (same size/format as color):

```rust
use oidn_wgpu::denoise_texture_with_aux;

denoise_texture_with_aux(
    &oidn,
    &wgpu_device,
    &wgpu_queue,
    &noisy_texture,
    &output_texture,
    format,
    &options,
    Some(&albedo_texture),  // None if not used
    Some(&normal_texture),
)?;
```

### Albedo and normal on CPU (RtFilter)

```rust
filter.execute_in_place_with_aux(&mut color, Some(&albedo[..]), Some(&normal[..]))?;
// or
filter.execute_with_aux(Some(&color), &mut output, Some(&albedo), Some(&normal))?;
```

### Full API (physical devices, buffers, generic filter)

- **Physical devices:** `num_physical_devices()`, `get_physical_device_bool/int/string/data()`, `is_cpu_device_supported()`, `is_cuda_device_supported()`, etc.
- **Device creation:** `OidnDevice::new_by_id()`, `new_by_uuid()`, `new_by_luid()`, `new_by_pci_address()`, `new_cuda_device()`, `new_hip_device()`, `new_metal_device()` (see docs for raw pointer/stream args). Device params: `set_bool()`, `set_int()`, `get_bool()`, `get_int()`, `commit()`, `set_error_function_raw()`.
- **Buffers:** `OidnBuffer::new()`, `new_with_storage()`, `new_shared()`, `new_shared_from_fd()`, `new_shared_from_win32_handle()`, `new_shared_from_metal()` (all return `Result<OidnBuffer, Error>`). Methods: `size()`, `storage()`, `data()`, `read()`/`write()`, `read_async()`/`write_async()`.
- **Generic filter:** `Filter::new(device, "RT")` or `"RTLightmap"` — then `set_image()` or `set_shared_image()`, `set_shared_data()`, `set_progress_monitor_raw()`, `commit()`, `execute()` or `execute_async()`. `RtFilter`/`RtLightmapFilter` also expose `get_bool`, `get_int`, `get_float`, `set_progress_monitor_raw`.

### Lightmap denoising (RTLightmap filter)

For baked lightmaps (requires OIDN built with RTLightmap support):

```rust
use oidn_wgpu::{OidnDevice, RtLightmapFilter};

let device = OidnDevice::new()?;
let mut filter = RtLightmapFilter::new(&device)?;
filter.set_dimensions(w, h).set_directional(false); // false = HDR, true = directional
filter.execute_in_place(&mut lightmap_rgb_f32)?;
```

## Tests and examples

```bash
# Set OIDN_DIR to your OIDN install or build directory (e.g. oidn/build/Release on Windows)
export OIDN_DIR=/path/to/oidn/install   # or on Windows: $env:OIDN_DIR = "C:\path\to\oidn\build\Release"

cargo test
cargo run --example cpu_denoise
cargo run --example wgpu_denoise
```

## Status

- **Implemented:** Full OIDN C API coverage: RT and RTLightmap filters (typed + generic `Filter`), all device creation paths, physical device queries, buffers (host/device/managed/shared/FD/Win32/Metal), async execute and buffer read/write, progress monitor, device/filter get/set parameters, error callback.
- **Limitations:** `OIDN_DIR` required on Windows (no pkg-config). RTLightmap requires OIDN built with `OIDN_FILTER_RTLIGHTMAP`. SYCL queue/device creation and `oidnExecuteSYCLFilterAsync` are C++-only in OIDN (no Rust binding).

### OIDN API coverage (complete)

| Area | API | oidn-wgpu |
|------|-----|-----------|
| **Physical device** | `oidnGetNumPhysicalDevices`, `oidnGetPhysicalDeviceBool/Int/String/Data` | `num_physical_devices()`, `get_physical_device_bool/int/string/data()` |
| **Device support** | `oidnIsCPUDeviceSupported`, `oidnIsCUDADeviceSupported`, `oidnIsHIPDeviceSupported`, `oidnIsMetalDeviceSupported` | `is_cpu_device_supported()`, `is_cuda_device_supported()`, etc. |
| **Device creation** | `oidnNewDevice`, `ByID`, `ByUUID`, `ByLUID`, `ByPCIAddress`, `oidnNewCUDADevice`, `oidnNewHIPDevice`, `oidnNewMetalDevice` | `OidnDevice::new()`, `new_by_id()`, `new_by_uuid()`, `new_by_luid()`, `new_by_pci_address()`, `new_cuda_device()`, `new_hip_device()`, `new_metal_device()` |
| **Device params** | `oidnSetDeviceBool/Int`, `oidnGetDeviceBool/Int`, `oidnCommitDevice`, `oidnSetDeviceErrorFunction`, `oidnSyncDevice` | `set_bool()`, `set_int()`, `get_bool()`, `get_int()`, `commit()`, `set_error_function_raw()`, `sync()` |
| **Buffer** | `oidnNewBuffer`, `oidnNewBufferWithStorage`, `oidnNewSharedBuffer`, `FromFD`, `FromWin32Handle`, `FromMetal`, get size/storage/data, read/write sync and async | `OidnBuffer::new()`, `new_with_storage()`, `new_shared()`, `new_shared_from_fd()`, `new_shared_from_win32_handle()`, `new_shared_from_metal()`, `size()`, `storage()`, `data()`, `read()`/`write()`, `read_async()`/`write_async()` |
| **Filter** | `oidnNewFilter`, set image (buffer or shared), unset image, set/get shared data, update data, unset data, set/get bool/int/float, progress monitor, commit, execute, execute async | `Filter::new()`, `set_image()`/`set_shared_image()`, `unset_image()`, `set_shared_data()`/`update_data()`/`unset_data()`, `set_*`/`get_*`, `set_progress_monitor_raw()`, `commit()`, `execute()`/`execute_async()`. Same on `RtFilter`/`RtLightmapFilter` where applicable. |

**Not bound (C++ only):** `oidnIsSYCLDeviceSupported`, `oidnNewSYCLDevice`, `oidnExecuteSYCLFilterAsync` (SYCL types). Use type-based `OidnDevice::sycl()` instead.

**Coverage audit (vs `oidn.h`):** Every C API function and type is either bound in `sys` and exposed via device/buffer/filter wrappers, or is C++-only (SYCL). Inline helpers in the header (`oidnGetDeviceUInt`, `oidnGetPhysicalDeviceUInt`, `oidnSetDeviceUInt`) are covered by `get_uint()`, `get_physical_device_int()` (cast to u32), and `set_int()` (pass `value as i32`). Refcounting: `oidnRetainDevice/Buffer/Filter` → `retain()` on each type; release in `Drop`. Global error: `take_global_error()`. Format enum: `OIDNFormat` (and alias `ImageFormat`) re-exported so variants (e.g. `OIDNFormat::Float3`) are constructible for `Filter::set_image()`.

**Symbol checklist (every `oidn.h` C symbol):**

| oidn.h symbol | Rust |
|---------------|------|
| `OIDN_UUID_SIZE`, `OIDN_LUID_SIZE` | `OIDN_UUID_SIZE`, `OIDN_LUID_SIZE` (lib) |
| `oidnGetNumPhysicalDevices` | `num_physical_devices()` |
| `oidnGetPhysicalDeviceBool`, `Int`, `String`, `Data` | `get_physical_device_bool/int/string/data()` |
| `oidnGetPhysicalDeviceUInt` (inline) | `get_physical_device_int()` → cast to u32 |
| `OIDNDeviceType` | `OidnDeviceType` |
| `OIDNError` | `sys::OIDNError` (used in `Error::OidnError`; code as u32) |
| `OIDNErrorFunction` | `set_error_function_raw()` |
| `oidnIsCPUDeviceSupported` | `is_cpu_device_supported()` |
| `oidnIsCUDADeviceSupported`, `IsHIPDeviceSupported`, `IsMetalDeviceSupported` | `is_cuda/hip/metal_device_supported()` |
| `oidnNewDevice`, `ByID`, `ByUUID`, `ByLUID`, `ByPCIAddress` | `OidnDevice::new()`, `new_by_id/uuid/luid/pci_address()` |
| `oidnNewCUDADevice`, `oidnNewHIPDevice`, `oidnNewMetalDevice` | `new_cuda_device()`, `new_hip_device()`, `new_metal_device()` |
| `oidnRetainDevice`, `oidnReleaseDevice` | `retain()`, `Drop` |
| `oidnSetDeviceBool`, `oidnSetDeviceInt`, `oidnGetDeviceBool`, `oidnGetDeviceInt` | `set_bool/int()`, `get_bool/int()` |
| `oidnGetDeviceUInt` (inline) | `get_uint()` |
| `oidnSetDeviceErrorFunction`, `oidnGetDeviceError` | `set_error_function_raw()`, `take_error()` / `take_global_error()` |
| `oidnCommitDevice`, `oidnSyncDevice` | `commit()`, `sync()` |
| `OIDNFormat` | `OIDNFormat` / `ImageFormat` (re-exported) |
| `OIDNStorage`, `OIDNExternalMemoryTypeFlag` | `BufferStorage`, `ExternalMemoryTypeFlag` |
| All `oidnNewBuffer*`, `oidnGetBufferSize/Storage/Data`, `oidnRead/WriteBuffer*`, `oidnRetain/ReleaseBuffer` | `OidnBuffer` methods |
| `OIDNQuality` | `Quality` |
| `OIDNProgressMonitorFunction` | `set_progress_monitor_raw()` |
| All `oidnNewFilter`, `oidnSet*FilterImage`, `oidnSetSharedFilterData`, `oidnUpdateFilterData`, `oidnUnset*`, `oidnSet/GetFilterBool/Int/Float`, `oidnSetFilterProgressMonitorFunction`, `oidnCommitFilter`, `oidnExecuteFilter*`, `oidnRetain/ReleaseFilter` | `Filter` (+ `RtFilter` / `RtLightmapFilter`) |
| `oidnIsSYCLDeviceSupported`, `oidnNewSYCLDevice`, `oidnExecuteSYCLFilterAsync` | **C++ only** — use `OidnDevice::sycl()` |

## Building OIDN from source

Clone the repo (requires [Git LFS](https://git-lfs.github.com/)):

```bash
git clone --recursive https://github.com/OpenImageDenoise/oidn.git
cd oidn
```

Then build with CMake (see [oidn documentation](https://github.com/OpenImageDenoise/oidn#compilation)). For a minimal CPU-only build you need CMake, a C++ compiler, and oneTBB. After building, set `OIDN_DIR` to the install prefix or the build directory where the library is produced.

## License

Licensed under either of **Apache-2.0** or **MIT** at your option.  
OIDN itself is under Apache-2.0; see [Intel OIDN](https://www.openimagedenoise.org).
