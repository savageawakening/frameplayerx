//! The autofocus measurement on a real (software) Vulkan device: synthetic
//! stereo pictures with a known disparity.

use fp_core::autofocus::{self, Autofocus, GRADIENT_MIN_SCORE, Match, Reading};
use fp_core::format::{Projection, StereoLayout, VideoFormat};
use fp_core::view::ViewSettings;
use fp_media::{PixelLayout, VideoFrame};
use fp_render::capture::OffscreenEyes;
use fp_render::disparity::Measure;
use fp_render::math::{Fov, projection, view};
use fp_render::{EyeView, Gpu, Renderer, VideoParams};
use glam::{Quat, Vec3};
use std::sync::Arc;

const SIZE: u32 = 128;
const EYE: u32 = 1024;

fn setup() -> Option<(Renderer, OffscreenEyes)> {
    let gpu = match Gpu::headless() {
        Ok(g) => Arc::new(g),
        Err(e) => {
            eprintln!("skipping: no Vulkan device ({e})");
            return None;
        }
    };
    let r = Renderer::new(gpu, ash::vk::Format::R8G8B8A8_SRGB).expect("renderer");
    let eyes = OffscreenEyes::new(&r, SIZE, SIZE).expect("eyes");
    Some((r, eyes))
}

/// Grey value noise over longitude and latitude, degrees, in features of
/// about `size` degrees: something to correlate.
fn noise_of(lon: f32, lat: f32, size: f32) -> f32 {
    let lattice = |x: i32, y: i32| {
        let h = (x as u32).wrapping_mul(73856093) ^ (y as u32).wrapping_mul(19349663);
        let h = (h ^ (h >> 13)).wrapping_mul(1274126177);
        ((h ^ (h >> 16)) & 255) as f32 / 255.0
    };
    let (x, y) = (lon / size, lat / size);
    let (fx, fy) = (x - x.floor(), y - y.floor());
    let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let (x, y) = (x.floor() as i32, y.floor() as i32);
    let top = lattice(x, y) * (1.0 - sx) + lattice(x + 1, y) * sx;
    let bottom = lattice(x, y + 1) * (1.0 - sx) + lattice(x + 1, y + 1) * sx;
    top * (1.0 - sy) + bottom * sy
}

fn noise(lon: f32, lat: f32) -> f32 {
    noise_of(lon, lat, 1.5)
}

/// Noise at three scales that don't divide into each other: [`noise`]
/// repeats the shape of its lattice every 1.5°, which a shift of a whole
/// cell matches up and down as well as across.
fn rich(lon: f32, lat: f32) -> f32 {
    0.5 * noise_of(lon, lat, 1.5)
        + 0.3 * noise_of(lon + 17.3, lat + 5.1, 0.93)
        + 0.2 * noise_of(lon - 7.7, lat + 11.9, 2.7)
}

/// A 180° side by side frame of `picture`. A subject at longitude `lon`
/// (degrees, right positive) of the left image is `disparity(lon)` degrees
/// further left in the right image, as something near the cameras is, and
/// `down` degrees lower.
fn frame_of(
    layout: PixelLayout,
    picture: impl Fn(f32, f32) -> f32,
    disparity: impl Fn(f32) -> f32,
    down: f32,
) -> Arc<VideoFrame> {
    let (w, h) = (EYE * 2, EYE);
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    for y in 0..h {
        let lat = 90.0 - (y as f32 + 0.5) / h as f32 * 180.0;
        for x in 0..w {
            let lon = ((x % EYE) as f32 + 0.5) / EYE as f32 * 180.0 - 90.0;
            let v = if x >= EYE {
                picture(lon + disparity(lon), lat + down)
            } else {
                picture(lon, lat)
            };
            let g = (v.clamp(0.0, 1.0) * 255.0) as u8;
            rgba.extend_from_slice(&[g, g, g, 255]);
        }
    }
    Arc::new(VideoFrame::from_rgba_as(w, h, &rgba, layout, true).expect("frame"))
}

fn stereo_frame(layout: PixelLayout, disparity: impl Fn(f32) -> f32) -> Arc<VideoFrame> {
    frame_of(layout, noise, disparity, 0.0)
}

fn eyes() -> [EyeView; 2] {
    let ev = |x: f32| EyeView {
        view: view(Vec3::new(x, 1.6, 0.0), Quat::IDENTITY),
        proj: projection(Fov::symmetric(30.0), 0.05, 100.0),
    };
    [ev(-0.032), ev(0.032)]
}

fn at(ray: Vec3) -> Measure {
    Measure {
        ray,
        vertical: 0.0,
        prefilter: true,
        centre: 0.0,
        tag: 4,
    }
}

fn video(m: Measure) -> VideoParams {
    VideoParams {
        format: VideoFormat::new(Projection::EQUIRECT_180, StereoLayout::SideBySide),
        settings: ViewSettings::default(),
        measure: Some(m),
        ..Default::default()
    }
}

/// Draws one frame that measures, then what it read back: its slot comes
/// round again two frames on.
fn measure(
    r: &mut Renderer,
    e: &mut OffscreenEyes,
    f: &Arc<VideoFrame>,
    p: &VideoParams,
) -> Option<Reading> {
    r.begin_frame().unwrap();
    r.set_video(Some(f)).unwrap();
    r.draw(&e.targets(), &eyes(), p, &[]).unwrap();
    r.end_frame().unwrap();
    assert!(r.disparity_reading().is_none(), "not read back yet");
    r.begin_frame().unwrap();
    r.end_frame().unwrap();
    assert!(r.disparity_reading().is_none(), "not read back yet");
    r.begin_frame().unwrap();
    r.end_frame().unwrap();
    r.disparity_reading()
}

fn turned(yaw: f32, pitch: f32) -> Vec3 {
    Quat::from_euler(
        glam::EulerRot::YXZ,
        yaw.to_radians(),
        pitch.to_radians(),
        0.0,
    ) * Vec3::NEG_Z
}

fn judged(reading: &Reading) -> Option<f32> {
    reading.judge(Match::Gradient, GRADIENT_MIN_SCORE).ok()
}

fn near(reading: Option<Reading>, want: f32) {
    let reading = reading.expect("a measurement");
    let got = judged(&reading);
    assert!(
        got.is_some_and(|d| (d - want).abs() < 0.1),
        "{got:?}, wanted {want}"
    );
}

#[test]
fn measures_the_disparity_where_the_eyes_look() {
    let Some((mut r, mut e)) = setup() else {
        return;
    };
    // Far on the viewer's left, near on the right.
    for layout in [PixelLayout::Nv12, PixelLayout::I420 { bits: 8 }] {
        let f = stereo_frame(layout, |lon| if lon < 0.0 { 1.5 } else { 8.0 });
        near(
            measure(&mut r, &mut e, &f, &video(at(turned(30.0, 0.0)))),
            1.5,
        );
        near(
            measure(&mut r, &mut e, &f, &video(at(turned(-30.0, 0.0)))),
            8.0,
        );
    }
    e.destroy(&r);
}

#[test]
fn ten_bit_is_measured_like_eight_bit() {
    let Some((mut r, mut e)) = setup() else {
        return;
    };
    let p = video(at(turned(0.0, 0.0)));
    let f = stereo_frame(PixelLayout::Nv12, |_| 6.0);
    let eight = measure(&mut r, &mut e, &f, &p).expect("a measurement");
    for layout in [PixelLayout::P010, PixelLayout::I420 { bits: 10 }] {
        let f = stereo_frame(layout, |_| 6.0);
        let reading = measure(&mut r, &mut e, &f, &p);
        let variance = reading.expect("a measurement").gradient_variance;
        // The thresholds are the same only if the luma's scale is.
        assert!(
            (variance / eight.gradient_variance - 1.0).abs() < 0.05,
            "{layout:?}: {variance}, 8-bit {}",
            eight.gradient_variance
        );
        near(reading, 6.0);
    }
    e.destroy(&r);
}

#[test]
fn the_search_reaches_a_tenth_of_a_metre() {
    let Some((mut r, mut e)) = setup() else {
        return;
    };
    let p = video(at(turned(-20.0, 0.0)));
    // 35° is 0.1 m at a 64 mm baseline.
    for want in [17.0, 30.0, 35.0] {
        let f = stereo_frame(PixelLayout::Nv12, move |_| want);
        near(measure(&mut r, &mut e, &f, &p), want);
    }
    e.destroy(&r);
}

#[test]
fn up_and_down_a_disparity_is_still_measured_as_a_shift_in_longitude() {
    let Some((mut r, mut e)) = setup() else {
        return;
    };
    let f = stereo_frame(PixelLayout::Nv12, |_| 5.0);
    for pitch in [-40.0, 30.0, 50.0] {
        near(
            measure(&mut r, &mut e, &f, &video(at(turned(10.0, pitch)))),
            5.0,
        );
    }
    e.destroy(&r);
}

#[test]
fn a_vertical_offset_between_the_eyes_is_found_and_searched_at() {
    let Some((mut r, mut e)) = setup() else {
        return;
    };
    // The right image 0.8° lower, as from a rig a little out of line.
    // Searched near it, the shift is found, and the vertical search finds
    // the rest of the offset.
    let f = frame_of(PixelLayout::Nv12, rich, |_| 6.0, 0.8);
    let ahead = at(turned(-10.0, 5.0));
    let level = measure(&mut r, &mut e, &f, &video(ahead)).expect("a measurement");
    let close = Measure {
        vertical: 0.5,
        ..ahead
    };
    let close = measure(&mut r, &mut e, &f, &video(close)).expect("a measurement");
    near(Some(close), 6.0);
    let found = close.vertical_found().expect("a clear vertical match");
    assert!((found - 0.8).abs() < 0.1, "{found}");
    let searched = measure(
        &mut r,
        &mut e,
        &f,
        &video(Measure {
            vertical: found,
            ..ahead
        }),
    )
    .expect("a measurement");
    near(Some(searched), 6.0);
    // Searching at the offset scores better than level.
    println!(
        "0.8° down: level {:?} best score {:.2}; at {found:.2}° {:.2}",
        judged(&level),
        level.score(Match::Gradient),
        searched.score(Match::Gradient)
    );
    assert!(searched.score(Match::Gradient) > level.score(Match::Gradient) + 0.1);
    // And the offset found is what's left to find.
    let rest = searched.vertical_found().expect("still clear");
    assert!((rest - 0.8).abs() < 0.1, "{rest}");
    e.destroy(&r);
}

#[test]
fn autofocus_learns_the_offset_and_measures_through_it() {
    let Some((mut r, mut e)) = setup() else {
        return;
    };
    // 1.6° out of line on fine detail: level, nothing matches.
    let f = frame_of(PixelLayout::Nv12, rich, |_| 4.0, -1.6);
    let mut a = Autofocus::default();
    for _ in 0..20 {
        let m = Measure {
            vertical: a.vertical_offset(),
            ..at(turned(0.0, 0.0))
        };
        let reading = measure(&mut r, &mut e, &f, &video(m)).expect("a measurement");
        a.measured(&reading);
    }
    let learned = a.learned_vertical().expect("learned");
    assert!((learned + 1.6).abs() < 0.1, "{learned}");
    let d = a.focus_distance().expect("a focus");
    let want = autofocus::distance(4.0, autofocus::BASELINE);
    assert!((d - want).abs() < want * 0.03, "{d} {want}");
    e.destroy(&r);
}

/// Fine detail, finer than the samples are apart: not a limit but a report,
/// `cargo test prefilter -- --nocapture`.
#[test]
fn prefiltering_is_reported() {
    let Some((mut r, mut e)) = setup() else {
        return;
    };
    let fine = |lon: f32, lat: f32| 0.5 * noise_of(lon, lat, 0.12) + 0.5 * noise(lon, lat);
    let f = frame_of(PixelLayout::Nv12, fine, |_| 5.0, 0.0);
    for prefilter in [false, true] {
        let mut trusted = 0;
        let mut right = 0;
        let mut score = 0.0;
        for yaw in (-40..=40).step_by(10) {
            let m = Measure {
                prefilter,
                ..at(turned(yaw as f32, 0.0))
            };
            let reading = measure(&mut r, &mut e, &f, &video(m)).expect("a measurement");
            score += reading.score(Match::Gradient);
            if let Some(d) = judged(&reading) {
                trusted += 1;
                right += u32::from((d - 5.0).abs() < 0.2);
            }
        }
        println!(
            "prefilter {prefilter}: 9 windows, {trusted} trusted, {right} right, mean score {:.2}",
            score / 9.0
        );
        if prefilter {
            assert!(right >= 7, "{right}");
        }
    }
    e.destroy(&r);
}

#[test]
fn a_cut_shows_in_the_coarse_picture() {
    let Some((mut r, mut e)) = setup() else {
        return;
    };
    let p = video(at(turned(0.0, 0.0)));
    let one = stereo_frame(PixelLayout::Nv12, |_| 3.0);
    let other = frame_of(
        PixelLayout::Nv12,
        |lon, lat| 0.2 + 0.6 * noise_of(lon + 400.0, lat, 20.0),
        |_| 3.0,
        0.0,
    );
    let mut a = Autofocus::default();
    for f in [&one, &one, &other] {
        a.measured(&measure(&mut r, &mut e, f, &p).expect("a measurement"));
    }
    assert_eq!(a.take_stats().cuts, 1);
    e.destroy(&r);
}

#[test]
fn a_blank_picture_has_no_disparity() {
    let Some((mut r, mut e)) = setup() else {
        return;
    };
    let f = frame_of(PixelLayout::Nv12, |_, _| 0.5, |_| 0.0, 0.0);
    let reading = measure(&mut r, &mut e, &f, &video(at(turned(0.0, 0.0)))).expect("a measurement");
    assert_eq!(judged(&reading), None);
    assert_eq!(reading.vertical_found(), None);
    e.destroy(&r);
}

#[test]
fn what_cannot_be_measured_is_not() {
    let Some((mut r, mut e)) = setup() else {
        return;
    };
    let f = stereo_frame(PixelLayout::Nv12, |_| 3.0);
    let mut top_bottom = video(at(turned(0.0, 0.0)));
    top_bottom.format.stereo = StereoLayout::TopBottom;
    assert!(measure(&mut r, &mut e, &f, &top_bottom).is_none());
    let behind = video(at(Vec3::Z));
    assert!(measure(&mut r, &mut e, &f, &behind).is_none());
    e.destroy(&r);
}

#[test]
fn a_measurement_is_timed_as_its_own_part() {
    let Some((mut r, mut e)) = setup() else {
        return;
    };
    let f = stereo_frame(PixelLayout::Nv12, |_| 3.0);
    let p = video(at(turned(0.0, 0.0)));
    measure(&mut r, &mut e, &f, &p).expect("a measurement");
    // Read back with it, as its slot came round.
    let Some(times) = r.gpu_times().filter(|t| t.total.is_some()) else {
        eprintln!("skipping: this device's frames aren't timed");
        e.destroy(&r);
        return;
    };
    assert!(times.autofocus.is_some(), "{times:?}");
    // Not a limit: it depends on the device. `cargo test timed -- --nocapture`.
    println!("autofocus: {:?} ms of {:?}", times.autofocus, times.total);
    e.destroy(&r);
}
