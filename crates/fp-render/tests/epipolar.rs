//! Autofocus's two ways of measuring compared on a real (software or
//! hardware) Vulkan device: a scene at a known distance, as two cameras
//! would film it, measured where the eyes look by each. Not only limits: a
//! report too, `cargo test --test epipolar -- --nocapture`.

use fp_core::autofocus::{self, BASELINE};
use fp_core::epipolar::{self, Aim, Layout, VERTICAL_SEARCH};
use fp_core::format::{Projection, StereoLayout, VideoFormat};
use fp_media::{PixelLayout, VideoFrame};
use fp_render::capture::OffscreenEyes;
use fp_render::math::{Fov, projection, view};
use fp_render::patch::PatchAsk;
use fp_render::{EyeView, Gpu, Renderer, VideoParams};
use glam::{Quat, Vec3};
use std::sync::Arc;
use std::time::Instant;

const EYE: u32 = 1536;

fn setup() -> Option<(Renderer, OffscreenEyes)> {
    let gpu = match Gpu::headless() {
        Ok(g) => Arc::new(g),
        Err(e) => {
            eprintln!("skipping: no Vulkan device ({e})");
            return None;
        }
    };
    let r = Renderer::new(gpu, ash::vk::Format::R8G8B8A8_SRGB).expect("renderer");
    let eyes = OffscreenEyes::new(&r, 64, 64).expect("eyes");
    Some((r, eyes))
}

fn lattice(x: i32, y: i32, seed: u32) -> f32 {
    // Mixed well enough that no two places look alike.
    let mut h = (x as u32)
        .wrapping_mul(0x9E37_79B1)
        .wrapping_add((y as u32).wrapping_mul(0x85EB_CA77))
        .wrapping_add(seed.wrapping_mul(0xC2B2_AE3D));
    h ^= h >> 16;
    h = h.wrapping_mul(0x7FEB_352D);
    h ^= h >> 15;
    h = h.wrapping_mul(0x846C_A68B);
    h ^= h >> 16;
    (h & 1023) as f32 / 1023.0
}

fn value(lon: f32, lat: f32, size: f32, seed: u32) -> f32 {
    let (x, y) = (lon / size, lat / size);
    let (fx, fy) = (x - x.floor(), y - y.floor());
    let (sx, sy) = (fx * fx * (3.0 - 2.0 * fx), fy * fy * (3.0 - 2.0 * fy));
    let (x, y) = (x.floor() as i32, y.floor() as i32);
    let top = lattice(x, y, seed) * (1.0 - sx) + lattice(x + 1, y, seed) * sx;
    let bottom = lattice(x, y + 1, seed) * (1.0 - sx) + lattice(x + 1, y + 1, seed) * sx;
    top * (1.0 - sy) + bottom * sy
}

/// Detail at three scales over directions from the cameras' middle.
fn texture(d: Vec3) -> f32 {
    let lon = d.x.atan2(-d.z).to_degrees();
    let lat = d.y.clamp(-1.0, 1.0).asin().to_degrees();
    0.5 * value(lon, lat, 1.5, 1) + 0.3 * value(lon, lat, 0.93, 2) + 0.2 * value(lon, lat, 2.7, 3)
}

/// A 180° side by side frame of the inside of a textured sphere `distance`
/// metres round the cameras' middle, filmed by cameras `BASELINE` apart, the
/// right one turned `pitch` degrees down.
fn scene(layout: PixelLayout, distance: f32, pitch: f32) -> Arc<VideoFrame> {
    scene_turned(layout, distance, pitch, 0.0)
}

/// The same with the right camera also rolled `roll` degrees about its
/// forward axis.
fn scene_turned(layout: PixelLayout, distance: f32, pitch: f32, roll: f32) -> Arc<VideoFrame> {
    let (w, h) = (EYE * 2, EYE);
    let mut rgba = Vec::with_capacity((w * h * 4) as usize);
    let down =
        Quat::from_rotation_x(-pitch.to_radians()) * Quat::from_rotation_z(roll.to_radians());
    for y in 0..h {
        let lat = (90.0 - (y as f32 + 0.5) / h as f32 * 180.0).to_radians();
        for x in 0..w {
            let right = x >= EYE;
            let lon = (((x % EYE) as f32 + 0.5) / EYE as f32 * 180.0 - 90.0).to_radians();
            let mut d = Vec3::new(lat.cos() * lon.sin(), lat.sin(), -lat.cos() * lon.cos());
            if right {
                d = down * d;
            }
            let c = Vec3::new(if right { 0.5 } else { -0.5 } * BASELINE, 0.0, 0.0);
            // Where the ray from the camera meets the sphere.
            let b = c.dot(d);
            let t = -b + (b * b - c.length_squared() + distance * distance).sqrt();
            let g = (texture((c + t * d).normalize()) * 255.0) as u8;
            rgba.extend_from_slice(&[g, g, g, 255]);
        }
    }
    Arc::new(VideoFrame::from_rgba_as(w, h, &rgba, layout, true).expect("frame"))
}

fn eyes() -> [EyeView; 2] {
    let ev = |x: f32| EyeView {
        view: view(Vec3::new(x, 1.6, 0.0), Quat::IDENTITY),
        proj: projection(Fov::symmetric(30.0), 0.05, 100.0),
    };
    [ev(-0.032), ev(0.032)]
}

fn video() -> VideoParams {
    VideoParams {
        format: VideoFormat::new(Projection::EQUIRECT_180, StereoLayout::SideBySide),
        ..Default::default()
    }
}

/// Draws one frame with `p`, then two more for its slot to come round:
/// the GPU times of the frame that measured.
fn frame(
    r: &mut Renderer,
    e: &mut OffscreenEyes,
    f: &Arc<VideoFrame>,
    p: &VideoParams,
) -> fp_render::GpuTimes {
    r.begin_frame().unwrap();
    r.set_video(Some(f)).unwrap();
    r.draw(&e.targets(), &eyes(), p, &[]).unwrap();
    r.end_frame().unwrap();
    for _ in 0..2 {
        r.begin_frame().unwrap();
        r.end_frame().unwrap();
    }
    r.gpu_times().unwrap_or_default()
}

fn turned(yaw: f32, pitch: f32) -> Vec3 {
    Quat::from_euler(
        glam::EulerRot::YXZ,
        yaw.to_radians(),
        pitch.to_radians(),
        0.0,
    ) * Vec3::NEG_Z
}

/// Epipolar: a coarse pass and a fine one, its distance, and the GPU and
/// CPU times of both.
fn epipolar(
    r: &mut Renderer,
    e: &mut OffscreenEyes,
    f: &Arc<VideoFrame>,
    ray: Vec3,
    fit: Option<(f32, f32)>,
) -> (Option<f32>, Option<epipolar::Found>, f32, f32) {
    let (d, f, g, c, _) = epipolar_at(r, e, f, ray, fit);
    (d, f, g, c)
}

/// The same, and where in the picture it measured: `α`, `β`.
fn epipolar_at(
    r: &mut Renderer,
    e: &mut OffscreenEyes,
    f: &Arc<VideoFrame>,
    ray: Vec3,
    fit: Option<(f32, f32)>,
) -> (Option<f32>, Option<epipolar::Found>, f32, f32, [f32; 2]) {
    let (vertical, reach) = match fit {
        Some(_) => (f32::NAN, epipolar::VERTICAL_KNOWN),
        None => (0.0, VERTICAL_SEARCH),
    };
    let mut ask = PatchAsk {
        ray,
        aim: Aim::Ray([0.0, 0.0]),
        layout: Layout::coarse(256, vertical, reach),
        vertical: fit,
        tag: 0,
    };
    let (mut gpu, mut cpu) = (0.0, 0.0);
    let mut pass = |ask: &PatchAsk, r: &mut Renderer| {
        let t = frame(
            r,
            e,
            f,
            &VideoParams {
                patch: Some(*ask),
                ..video()
            },
        );
        gpu += t.autofocus.unwrap_or(f32::NAN);
        let patch = r.patch_result().expect("a patch");
        let start = Instant::now();
        let found = epipolar::search(&patch.layout, &patch.left, &patch.right);
        cpu += start.elapsed().as_secs_f32() * 1e6;
        (patch, found)
    };
    let (patch, coarse) = pass(&ask, r);
    if coarse.judge().is_err() {
        return (None, Some(coarse), gpu, cpu, patch.at);
    }
    COARSE_ONLY.with(|c| {
        c.set(Some(autofocus::distance(
            epipolar::frontal_angle(coarse.shift, patch.at[0]),
            BASELINE,
        )));
    });
    let fine_reach = if fit.is_some() {
        epipolar::VERTICAL_KNOWN
    } else {
        patch.layout.pitch
    };
    ask.aim = Aim::At(patch.at);
    ask.layout = Layout::fine(
        coarse.shift,
        patch.layout.pitch,
        coarse.vertical,
        fine_reach,
    );
    let (patch, fine) = pass(&ask, r);
    let d = fine
        .judge()
        .ok()
        .map(|s| autofocus::distance(epipolar::frontal_angle(s, patch.at[0]), BASELINE));
    (d, Some(fine), gpu, cpu, patch.at)
}

thread_local! {
    /// The last coarse pass's own distance, before a fine pass refines it.
    static COARSE_ONLY: std::cell::Cell<Option<f32>> = const { std::cell::Cell::new(None) };
}

const PLACES: [(f32, f32); 6] = [
    (0.0, 0.0),
    (-25.0, 0.0),
    (0.0, 30.0),
    (30.0, -25.0),
    (-45.0, 35.0),
    (50.0, 15.0),
];

#[test]
fn a_known_distance_is_found_anywhere_and_what_it_costs() {
    let Some((mut r, mut e)) = setup() else {
        return;
    };
    let mut report = String::new();
    for (layout, distance, pitch) in [
        (PixelLayout::Nv12, 0.6, 0.0),
        (PixelLayout::P010, 1.5, 0.0),
        (PixelLayout::Nv12, 0.6, 1.2),
    ] {
        let f = scene(layout, distance, pitch);
        let (mut ep_ok, mut coarse_ok, mut coarse_err) = (0, 0, 0.0f32);
        let (mut gpu_e, mut cpu_e) = (0.0, 0.0);
        for (yaw, up) in PLACES {
            let ray = turned(-yaw, up);
            let near = |d: Option<f32>| d.is_some_and(|d| (d - distance).abs() < distance * 0.05);
            let (de, found, g, c) = epipolar(&mut r, &mut e, &f, ray, None);
            gpu_e += g;
            cpu_e += c;
            ep_ok += u32::from(near(de));
            let dc = COARSE_ONLY.with(std::cell::Cell::take);
            coarse_ok += u32::from(near(dc));
            if let Some(dc) = dc {
                coarse_err = coarse_err.max((dc - distance).abs() / distance);
            }
            report += &format!(
                "\n{layout:?} {distance} m, pitch {pitch}°, at {yaw}° {up}°: {de:?} \
                 (vertical {:?}, correlation {:?}, {:?})",
                found.map(|f| (f.vertical * 100.0).round() / 100.0),
                found.map(|f| (f.correlation * 100.0).round() / 100.0),
                found.map(|f| (f.judge(), f.second)),
            );
        }
        let n = PLACES.len() as f32;
        report += &format!(
            "\n  right within 5%: {ep_ok} of {}, the coarse pass alone {coarse_ok} (worst \
             {:.1}% off); GPU ms a reading {:.3} (coarse and fine); CPU {:.0} µs",
            PLACES.len(),
            coarse_err * 100.0,
            gpu_e / n,
            cpu_e / n,
        );
        assert!(ep_ok as f32 >= n - 1.0, "{report}");
    }
    println!("{report}");
    e.destroy(&r);
}

#[test]
fn epipolar_works_through_a_known_vertical_offset() {
    let Some((mut r, mut e)) = setup() else {
        return;
    };
    let f = scene(PixelLayout::Nv12, 0.8, -1.4);
    let (d, found, _, _) = epipolar(&mut r, &mut e, &f, turned(10.0, 5.0), Some((1.4, 0.0)));
    let found = found.unwrap();
    // Turned up, it shows the picture lower.
    assert!((found.vertical - 1.4).abs() < 0.1, "{found:?}");
    assert!(d.is_some_and(|d| (d - 0.8).abs() < 0.04), "{d:?}");
    e.destroy(&r);
}

#[test]
fn a_rolled_camera_is_fitted_and_its_bias_taken_off_the_distance() {
    let Some((mut r, mut e)) = setup() else {
        return;
    };
    let distance = 0.8;
    let f = scene_turned(PixelLayout::Nv12, distance, 0.0, 1.5);
    let mut fit = epipolar::VerticalFit::default();
    let mut found = Vec::new();
    for (yaw, up) in PLACES {
        let (_, fd, _, _, at) = epipolar_at(&mut r, &mut e, &f, turned(-yaw, up), None);
        if let Some(fd) = fd.filter(|fd| fd.judge().is_ok()) {
            fit.add(at[0], at[1], fd.vertical);
            found.push((at, fd));
        }
    }
    assert!(found.len() >= 5, "{found:?}");
    let (pitch, roll) = fit.fit().unwrap();
    let mut report = format!("rolled 1.5°: fitted pitch {pitch:+.2}° roll {roll:+.2}°");
    for (at, fd) in &found {
        let predicted = fit.predict(at[0], at[1]).unwrap();
        let raw = autofocus::distance(epipolar::frontal_angle(fd.shift, at[0]), BASELINE);
        let fixed = autofocus::distance(
            epipolar::frontal_angle(fd.shift - fit.shift_bias(at[1]), at[0]),
            BASELINE,
        );
        report += &format!(
            "\n  at α {:.0}° β {:.0}°: vertical {:+.2}° predicted {predicted:+.2}°; \
             {raw:.3} m as measured, {fixed:.3} m without the roll's bias",
            at[0], at[1], fd.vertical
        );
        assert!((predicted - fd.vertical).abs() < 0.1, "{report}");
        assert!((fixed - distance).abs() < distance * 0.03, "{report}");
    }
    println!("{report}");
    e.destroy(&r);
}
