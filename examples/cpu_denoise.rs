//! Minimal example: denoise a small RGB float image on the CPU (no wgpu).
//!
//! Run with: cargo run --example cpu_denoise
//! Requires OIDN to be built and OIDN_DIR set (or pkg-config).

use oidn_wgpu::{OidnDevice, Quality, RtFilter};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let width = 64u32;
    let height = 64u32;
    let n = (width * height * 3) as usize;

    // Fake noisy RGB f32 image (e.g. from a path tracer)
    let mut color: Vec<f32> = (0..n)
        .map(|i| {
            let x = (i % (width as usize * 3)) as f32 / (width as f32 * 3.0);
            let y = (i / (width as usize * 3)) as f32 / height as f32;
            (x * y).sin() * 0.5 + 0.5 + (rand_simple(i) * 0.1)
        })
        .collect();

    let device = OidnDevice::new()?;
    let mut filter = RtFilter::new(&device)?;
    filter
        .set_dimensions(width, height)
        .set_hdr(true)
        .set_quality(Quality::High);

    filter.execute_in_place(&mut color)?;

    if let Some(e) = device.take_error() {
        return Err(e.into());
    }

    println!("Denoised {}x{} image successfully.", width, height);
    Ok(())
}

fn rand_simple(seed: usize) -> f32 {
    let x = (seed as u64).wrapping_mul(0x9e3779b97f4a7c15);
    ((x >> 32) as f32) / (u32::MAX as f32)
}
