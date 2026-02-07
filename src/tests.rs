//! Unit tests (require OIDN to be built and linked).

use crate::{OidnDevice, Quality, RtFilter, RtLightmapFilter};

#[test]
fn test_rt_filter_dimensions_and_execute_in_place() {
    let device = OidnDevice::new().expect("OIDN device creation");
    let mut filter = RtFilter::new(&device).expect("RT filter creation");
    let w = 8u32;
    let h = 8u32;
    let n = (w * h * 3) as usize;
    filter.set_dimensions(w, h).set_hdr(true).set_quality(Quality::High);

    let mut color = vec![0.1f32; n];
    for i in (0..n).step_by(3) {
        color[i] = 0.5;
        color[i + 1] = 0.5;
        color[i + 2] = 0.5;
    }

    filter.execute_in_place(&mut color).expect("execute_in_place");
    assert!(device.take_error().is_none());
}

#[test]
fn test_rt_filter_invalid_dimensions() {
    let device = OidnDevice::new().expect("OIDN device");
    let mut filter = RtFilter::new(&device).expect("RT filter");
    filter.set_dimensions(4, 4);

    let mut output = vec![0.0f32; 10]; // wrong size
    let err = filter.execute(None, &mut output).expect_err("should error");
    assert!(matches!(err, crate::Error::InvalidDimensions));
}

#[test]
fn test_rt_filter_zero_dimensions() {
    let device = OidnDevice::new().expect("OIDN device");
    let filter = RtFilter::new(&device).expect("RT filter");
    // dimensions left at default 0,0
    let mut output = vec![0.0f32; 3];
    let err = filter.execute(None, &mut output).expect_err("should error");
    assert!(matches!(err, crate::Error::InvalidDimensions));
}

#[test]
fn test_rt_lightmap_filter_new_or_unsupported() {
    let device = OidnDevice::new().expect("OIDN device");
    let result = RtLightmapFilter::new(&device);
    match result {
        Ok(mut filter) => {
            filter.set_dimensions(4, 4);
            let mut buf = vec![0.0f32; 4 * 4 * 3];
            let _ = filter.execute_in_place(&mut buf);
        }
        Err(e) => assert!(matches!(e, crate::Error::FilterCreationFailed)),
    }
}
