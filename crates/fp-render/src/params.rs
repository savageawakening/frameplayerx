//! Builds the projection shader's uniform block.

use fp_core::format::{Projection, StereoLayout, VideoFormat};
use fp_core::view::ViewSettings;
use fp_media::PixelLayout;
use fp_media::frame::{ColorInfo, Matrix};
use fp_media::info::Transfer;
use glam::{EulerRot, Mat4, Quat, Vec3, Vec4, Vec4Swizzles};

/// Matches `struct Params` in shaders/scene.wgsl (std140-compatible: every
/// member is 16-byte aligned).
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct SceneParams {
    pub inv_view_proj: [Mat4; 2],
    pub correction: Mat4,
    pub anchor_inv: Mat4,
    pub screen_inv: Mat4,
    pub mode: [u32; 4],
    pub proj: Vec4,
    pub lens: Vec4,
    pub circle: Vec4,
    pub centres: Vec4,
    pub screen: Vec4,
    pub yuv_r: Vec4,
    pub yuv_g: Vec4,
    pub yuv_b: Vec4,
    pub yuv_offset: Vec4,
    pub picture: Vec4,
    pub extra: Vec4,
    pub tex_size: Vec4,
    pub bg_color: Vec4,
    pub key: Vec4,
    pub key2: Vec4,
    pub viewer: Vec4,
    pub display: [u32; 4],
    pub hdr: Vec4,
    pub tp10: Vec4,
    pub tp10_uv: Vec4,
    pub focus: Vec4,
    pub eyes: [Vec4; 2],
}

/// Sets `p` up for a Q10C picture of `height` rows sampled where it lies
/// (`planes_tp10.wgsl`), read in bands `luma` and `chroma`.
pub(crate) fn sample_tp10(
    p: &mut SceneParams,
    height: u32,
    luma: crate::q10c::SampleBands,
    chroma: crate::q10c::SampleBands,
) {
    let bands = |b: crate::q10c::SampleBands, rows: u32| {
        Vec4::new(b.step as f32, b.rows as f32, rows as f32, b.count as f32)
    };
    // TP10 reads 10-bit values as they are; P010 keeps them at the top of 16.
    p.yuv_offset.w = 1.0;
    p.tp10 = bands(luma, height);
    p.tp10_uv = bands(chroma, height.div_ceil(2));
}

/// The shader's `hdr`: an HDR scene peaking at `peak` nits on the screen's
/// white, both PQ-coded.
pub(crate) fn hdr_levels(peak: f32) -> Vec4 {
    Vec4::new(
        crate::hdr::pq_from_nits(peak),
        crate::hdr::pq_from_nits(crate::hdr::SDR_WHITE),
        0.0,
        0.0,
    )
}

/// Everything about the video that the projection pass needs, other than
/// the eye matrices.
#[derive(Clone, Copy, Debug)]
pub struct VideoParams {
    pub format: VideoFormat,
    pub settings: ViewSettings,
    /// Flat screens: where the screen's centre sits in the world (the screen
    /// faces +Z of this transform, i.e. towards a viewer looking down -Z).
    pub screen_pose: Mat4,
    /// Background around/behind the video, linear premultiplied RGBA.
    /// Alpha 0 lets passthrough show through.
    pub background: [f32; 4],
    /// The room frame the video was centred in (recentering's anchor): a
    /// 180°/360° picture, its per-eye corrections and the viewer's height
    /// and forward turn with it.
    pub anchor: Quat,
    /// Clockwise turn that shows the video upright: 0, 90, 180 or 270
    /// degrees.
    pub rotation: u32,
    /// Width of a video pixel relative to its height: anamorphic video is
    /// shown this much wider than it is stored.
    pub pixel_aspect: f32,
    /// Autofocus: how far away the picture is taken to be, for a 180°/360°
    /// video. `None` draws it as it always has been.
    pub focus: Option<Focus>,
    /// Autofocus: where to measure the picture's depth this frame, read back
    /// with [`crate::Renderer::disparity_reading`].
    pub measure: Option<crate::disparity::Measure>,
}

/// The depth autofocus draws a 180°/360° picture at.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Focus {
    /// From the cameras to what the eyes are on, metres; no nearer than
    /// [`MIN_FOCUS_DISTANCE`].
    pub distance: f32,
    /// How far apart the cameras were, metres.
    pub baseline: f32,
}

/// The nearest the picture is drawn, metres: nearer, the sphere is smaller
/// than the viewer's eyes can be kept well inside.
pub const MIN_FOCUS_DISTANCE: f32 = 0.1;

impl Default for VideoParams {
    fn default() -> Self {
        VideoParams {
            format: VideoFormat::default(),
            settings: ViewSettings::default(),
            screen_pose: Mat4::from_translation(glam::vec3(0.0, 0.0, -4.0)),
            background: [0.008, 0.008, 0.012, 1.0],
            anchor: Quat::IDENTITY,
            rotation: 0,
            pixel_aspect: 1.0,
            focus: None,
            measure: None,
        }
    }
}

/// Description of the uploaded frame the parameters refer to.
#[derive(Clone, Copy, Debug)]
pub(crate) struct FrameDesc {
    pub width: u32,
    pub height: u32,
    pub layout: PixelLayout,
    pub color: ColorInfo,
}

/// YUV → RGB matrix with range expansion folded in, plus offsets.
pub(crate) fn yuv_matrix(color: &ColorInfo, layout: PixelLayout) -> ([Vec4; 3], Vec4) {
    let (kr, kb) = match color.matrix {
        Matrix::Bt601 => (0.299, 0.114),
        Matrix::Bt709 => (0.2126, 0.0722),
        Matrix::Bt2020 => (0.2627, 0.0593),
    };
    let kg = 1.0 - kr - kb;
    let bits = match layout {
        PixelLayout::I420 { bits } => bits,
        PixelLayout::Nv12 => 8,
        PixelLayout::P010 | PixelLayout::DrmPrime | PixelLayout::Q10c => 10,
    };
    let max = ((1u32 << bits) - 1) as f32;
    // Sample scale: values are normalised by the texture format's range.
    // Q10C frames are converted into P010 planes.
    let scale = match layout {
        PixelLayout::I420 { bits: 10 } => 65535.0 / 1023.0,
        PixelLayout::P010 | PixelLayout::Q10c => 65535.0 / 65472.0,
        _ => 1.0,
    };
    let (y_off, y_mul, c_off, c_mul) = if color.full_range {
        (0.0, 1.0, (1u32 << (bits - 1)) as f32 / max, 1.0)
    } else {
        let s = (1u32 << (bits - 8)) as f32;
        (
            16.0 * s / max,
            max / (219.0 * s),
            128.0 * s / max,
            max / (224.0 * s),
        )
    };
    // rgb = M * [Y', U, V] with U,V in [-0.5, 0.5].
    let m = [
        [1.0, 0.0, 2.0 * (1.0 - kr)],
        [
            1.0,
            -2.0 * kb * (1.0 - kb) / kg,
            -2.0 * kr * (1.0 - kr) / kg,
        ],
        [1.0, 2.0 * (1.0 - kb), 0.0],
    ];
    let rows = m.map(|r| Vec4::new(r[0] * y_mul, r[1] * c_mul, r[2] * c_mul, 0.0));
    (rows, Vec4::new(y_off, c_off, c_off, scale))
}

/// Radius, metres, of the sphere a 180°/360° video is drawn on when the viewer
/// moves off its centre (height, forward). Rays from the centre ignore it.
const SPHERE_RADIUS: f32 = 3.0;

pub(crate) fn build(
    video: &VideoParams,
    frame: Option<&FrameDesc>,
    inv_view_proj: [Mat4; 2],
) -> SceneParams {
    let s = &video.settings;
    let deg = |d: f32| d.to_radians();
    let (kind, h_fov, v_fov) = match video.format.projection {
        Projection::Flat => (0u32, 0.0, 0.0),
        Projection::Equirect { h_fov, v_fov } => (1, deg(h_fov), deg(v_fov)),
        Projection::Fisheye { fov } => (2, deg(fov), deg(fov)),
        Projection::Eac { h_fov } => (3, deg(h_fov), deg(180.0)),
    };
    let stereo = match video.format.stereo {
        StereoLayout::Mono => 0u32,
        StereoLayout::SideBySide | StereoLayout::HalfSideBySide | StereoLayout::FrameSequential => {
            1
        }
        StereoLayout::TopBottom | StereoLayout::HalfTopBottom => 2,
    };
    let swap = (video.format.eyes_swapped ^ s.swap_eyes) as u32;
    let (rows, offset, layout_code, tex) = match frame {
        Some(f) => {
            let (rows, off) = yuv_matrix(&f.color, f.layout);
            let lc = match f.layout {
                PixelLayout::I420 { .. } => 0u32,
                _ => 1,
            };
            (rows, off, lc, (f.width.max(1), f.height.max(1)))
        }
        None => (
            [Vec4::X, Vec4::Y, Vec4::Z],
            Vec4::new(0.0, 0.0, 0.0, 1.0),
            0,
            (1, 1),
        ),
    };
    // The picture as shown: unsqueezed, then turned upright. Stereo halves
    // are halves of that.
    let pixel_aspect = video
        .format
        .stereo
        .eye_pixel_aspect(video.pixel_aspect as f64) as f32;
    let turns = video.rotation / 90 % 4;
    let (pw, ph) = (tex.0 as f32 * pixel_aspect.max(0.01), tex.1 as f32);
    let (pw, ph) = if turns % 2 == 1 { (ph, pw) } else { (pw, ph) };
    // Per-eye aspect for flat screens.
    let (ew, eh) = match video.format.stereo {
        StereoLayout::SideBySide | StereoLayout::FrameSequential => (pw / 2.0, ph),
        StereoLayout::TopBottom => (pw, ph / 2.0),
        StereoLayout::HalfSideBySide | StereoLayout::HalfTopBottom | StereoLayout::Mono => (pw, ph),
    };
    // A fisheye's image circle fills the eye image's shorter side, and stays
    // round when the image is not square.
    let circle = Vec4::new((eh / ew).min(1.0), (ew / eh).min(1.0), 0.0, 0.0);
    let [c0, c1] = s.lens_centre;
    // The screen's width setting is its longer side, so a portrait video
    // stands as tall as a landscape one is wide.
    let long = s.screen_width;
    let (screen_w, screen_h) = if ew >= eh {
        (long, long * eh / ew.max(1e-3))
    } else {
        (long * ew / eh.max(1e-3), long)
    };
    let arc = s.screen_curvature.clamp(0.0, 1.0) * std::f32::consts::TAU / 3.0;
    let correction =
        Mat4::from_euler(EulerRot::YXZ, deg(s.yaw), deg(s.pitch), deg(s.roll)).inverse();
    let kc = chroma(s.key_color);
    let (transfer, wide) = match frame {
        Some(f) => (
            match f.color.transfer {
                Transfer::Sdr => 0.0,
                Transfer::Pq => 1.0,
                Transfer::Hlg => 2.0,
            },
            f.color.wide_gamut as u32 as f32,
        ),
        None => (0.0, 0.0),
    };
    SceneParams {
        inv_view_proj,
        correction,
        anchor_inv: Mat4::from_quat(video.anchor.inverse()),
        screen_inv: video.screen_pose.inverse(),
        mode: [kind, stereo, swap, layout_code],
        proj: Vec4::new(h_fov, v_fov, s.zoom.max(0.05), deg(s.ipd_offset)),
        lens: Vec4::new(
            s.lens_k1,
            s.lens_k2,
            deg(s.vertical_align),
            deg(s.rotation_align),
        ),
        circle,
        centres: Vec4::new(c0[0], -c0[1], c1[0], -c1[1]),
        screen: Vec4::new(screen_w, screen_h, arc, frame.is_some() as u32 as f32),
        yuv_r: rows[0],
        yuv_g: rows[1],
        yuv_b: rows[2],
        yuv_offset: offset,
        picture: Vec4::new(s.brightness, s.contrast, s.saturation, s.gamma),
        extra: Vec4::new(s.sharpen, transfer, wide, video.background[3]),
        tex_size: Vec4::new(
            tex.0 as f32,
            tex.1 as f32,
            1.0 / tex.0 as f32,
            1.0 / tex.1 as f32,
        ),
        bg_color: Vec4::from(video.background),
        key: Vec4::new(kc.x, kc.y, s.key_similarity, s.key_smoothness),
        key2: Vec4::new(
            s.key_spill,
            s.chroma_key as u32 as f32,
            video.format.alpha_pack_scale().unwrap_or(0.0),
            0.0,
        ),
        viewer: glam::vec3(0.0, s.height, -s.forward).extend(SPHERE_RADIUS),
        display: [turns, 0, 0, 0],
        hdr: hdr_levels(crate::hdr::DEFAULT_PEAK),
        tp10: Vec4::ZERO,
        tp10_uv: Vec4::ZERO,
        focus: match video.focus {
            Some(f) if kind != 0 => Vec4::new(
                f.distance.max(MIN_FOCUS_DISTANCE),
                f.baseline.max(0.0),
                0.0,
                0.0,
            ),
            _ => Vec4::ZERO,
        },
        eyes: [Vec4::ZERO; 2],
    }
}

/// Sets where each eye is from the middle of the head, from the eyes' world
/// positions, in the room frame of `anchor`: what autofocus draws from.
pub(crate) fn place_eyes(p: &mut SceneParams, anchor: Quat, eyes: [Vec3; 2]) {
    let middle = 0.5 * (eyes[0] + eyes[1]);
    p.eyes = eyes.map(|e| (anchor.inverse() * (e - middle)).extend(0.0));
}

/// The world-space position of the eye a view is from.
pub(crate) fn eye_position(view: Mat4) -> Vec3 {
    view.inverse().w_axis.xyz()
}

/// Mirrors `off_centre` in shaders/scene.wgsl.
pub(crate) fn off_centre(d: Vec3, p: Vec3, r: f32) -> Vec3 {
    if p.length_squared() < 1e-8 {
        return d;
    }
    let q = p * (0.9 * r / p.length()).min(1.0);
    let b = q.dot(d);
    let t = -b + (b * b - q.length_squared() + r * r).sqrt();
    (q + t * d).normalize()
}

/// Mirrors `zoom_dir` in shaders/scene.wgsl.
pub(crate) fn zoom_dir(d: Vec3, zoom: f32) -> Vec3 {
    let r = d.truncate().length();
    if (zoom - 1.0).abs() < 1e-4 || r < 1e-6 {
        return d;
    }
    let theta = (-d.z).clamp(-1.0, 1.0).acos();
    let t = (theta / zoom).min(std::f32::consts::PI);
    (d.truncate() / r * t.sin()).extend(-t.cos())
}

/// Mirrors the spherical path of `fs_main` in shaders/scene.wgsl: the
/// direction in the video, from its camera, that eye `eye` sees along the
/// world-space `ray`.
#[cfg(test)]
pub(crate) fn video_dir(p: &SceneParams, ray: Vec3, eye: usize) -> Vec3 {
    let s = if eye == 1 { 0.5 } else { -0.5 };
    let mut d = p.anchor_inv.transform_vector3(ray);
    d = Mat4::from_rotation_y(p.proj.w * s).transform_vector3(d);
    d = Mat4::from_rotation_x(p.lens.z * s).transform_vector3(d);
    d = Mat4::from_rotation_z(p.lens.w * s).transform_vector3(d);
    d = p.correction.transform_vector3(d).normalize();
    d = if p.focus.x > 0.0 {
        let at = p.viewer.xyz() + p.eyes[eye].xyz();
        let camera = Vec3::new(s * p.focus.y, 0.0, 0.0);
        off_centre(d, p.correction.transform_vector3(at) - camera, p.focus.x)
    } else {
        off_centre(
            d,
            p.correction.transform_vector3(p.viewer.xyz()),
            p.viewer.w,
        )
    };
    zoom_dir(d, p.proj.z)
}

/// Chroma (BT.709 Cb, Cr) of an sRGB-encoded colour, as the shader computes it
/// from the video's samples.
pub(crate) fn chroma(rgb: [f32; 3]) -> glam::Vec2 {
    let [r, g, b] = rgb;
    glam::Vec2::new(
        -0.1146 * r - 0.3854 * g + 0.5 * b,
        0.5 * r - 0.4542 * g - 0.0458 * b,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn apply(rows: &[Vec4; 3], off: Vec4, yuv: [f32; 3]) -> [f32; 3] {
        let v = Vec3::new(yuv[0], yuv[1], yuv[2]) * off.w - off.truncate();
        [
            rows[0].truncate().dot(v),
            rows[1].truncate().dot(v),
            rows[2].truncate().dot(v),
        ]
    }

    #[test]
    fn limited_range_bt709_white_black_red() {
        let c = ColorInfo {
            matrix: Matrix::Bt709,
            full_range: false,
            transfer: Transfer::Sdr,
            wide_gamut: false,
        };
        let (rows, off) = yuv_matrix(&c, PixelLayout::I420 { bits: 8 });
        let near = |a: [f32; 3], b: [f32; 3]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 0.01);
        assert!(near(
            apply(&rows, off, [16.0 / 255.0, 0.5019, 0.5019]),
            [0.0, 0.0, 0.0]
        ));
        assert!(near(
            apply(&rows, off, [235.0 / 255.0, 0.5019, 0.5019]),
            [1.0, 1.0, 1.0]
        ));
        // BT.709 limited red: Y 63, Cb 102, Cr 240.
        assert!(near(
            apply(&rows, off, [63.0 / 255.0, 102.0 / 255.0, 240.0 / 255.0]),
            [1.0, 0.0, 0.0]
        ));
    }

    #[test]
    fn ten_bit_low_bits_scale() {
        let c = ColorInfo {
            matrix: Matrix::Bt2020,
            full_range: false,
            transfer: Transfer::Pq,
            wide_gamut: true,
        };
        let (rows, off) = yuv_matrix(&c, PixelLayout::I420 { bits: 10 });
        // 10-bit white (940) stored in the low bits of a 16-bit texel.
        let s = 1.0 / 65535.0;
        let out = apply(&rows, off, [940.0 * s, 512.0 * s, 512.0 * s]);
        assert!(out.iter().all(|v| (v - 1.0).abs() < 0.01), "{out:?}");
    }

    #[test]
    fn flat_screen_aspect_ratios_full_vs_half() {
        let s = ViewSettings::default();
        let make_video = |stereo| VideoParams {
            format: VideoFormat::new(Projection::Flat, stereo),
            settings: s,
            ..Default::default()
        };
        let frame = FrameDesc {
            width: 1920,
            height: 1080,
            layout: PixelLayout::Nv12,
            color: ColorInfo {
                matrix: Matrix::Bt709,
                full_range: false,
                transfer: Transfer::Sdr,
                wide_gamut: false,
            },
        };
        let sp_fsbs = build(
            &make_video(StereoLayout::SideBySide),
            Some(&frame),
            [Mat4::IDENTITY; 2],
        );
        let sp_hsbs = build(
            &make_video(StereoLayout::HalfSideBySide),
            Some(&frame),
            [Mat4::IDENTITY; 2],
        );
        let sp_ftb = build(
            &make_video(StereoLayout::TopBottom),
            Some(&frame),
            [Mat4::IDENTITY; 2],
        );
        let sp_htb = build(
            &make_video(StereoLayout::HalfTopBottom),
            Some(&frame),
            [Mat4::IDENTITY; 2],
        );

        // Full SBS expects 960x1080 per eye (8:9), so screen is narrow/tall:
        let ar_fsbs = sp_fsbs.screen.x / sp_fsbs.screen.y;
        assert!((ar_fsbs - 960.0 / 1080.0).abs() < 1e-4);

        // Half SBS restores 16:9 widescreen per eye:
        let ar_hsbs = sp_hsbs.screen.x / sp_hsbs.screen.y;
        assert!((ar_hsbs - 1920.0 / 1080.0).abs() < 1e-4);

        // Full TB expects 1920x540 per eye (32:9), so screen is ultra-wide:
        let ar_ftb = sp_ftb.screen.x / sp_ftb.screen.y;
        assert!((ar_ftb - 1920.0 / 540.0).abs() < 1e-4);

        // Half TB restores 16:9 widescreen per eye:
        let ar_htb = sp_htb.screen.x / sp_htb.screen.y;
        assert!((ar_htb - 1920.0 / 1080.0).abs() < 1e-4);
    }

    #[test]
    fn portrait_and_anamorphic_screens() {
        let frame = FrameDesc {
            width: 1920,
            height: 1080,
            layout: PixelLayout::Nv12,
            color: ColorInfo {
                matrix: Matrix::Bt709,
                full_range: false,
                transfer: Transfer::Sdr,
                wide_gamut: false,
            },
        };
        let screen = |rotation, pixel_aspect| {
            let video = VideoParams {
                rotation,
                pixel_aspect,
                ..Default::default()
            };
            let p = build(&video, Some(&frame), [Mat4::IDENTITY; 2]);
            (p.screen.x, p.screen.y, p.display[0])
        };
        let near = |(w, h, t): (f32, f32, u32), want: (f32, f32, u32)| {
            (w - want.0).abs() < 1e-3 && (h - want.1).abs() < 1e-3 && t == want.2
        };
        // The default 6 m width: a phone clip stored sideways stands 6 m tall.
        assert!(near(screen(0, 1.0), (6.0, 3.375, 0)));
        assert!(near(screen(90, 1.0), (3.375, 6.0, 1)));
        assert!(near(screen(270, 1.0), (3.375, 6.0, 3)));
        assert!(near(screen(180, 1.0), (6.0, 3.375, 2)));
        // 4:3 pixels widen a 16:9 frame to 2.37:1.
        assert!(near(screen(0, 4.0 / 3.0), (6.0, 6.0 * 1080.0 / 2560.0, 0)));
    }

    /// A 180° video with `focus`, seen by eyes `ipd` apart, the picture
    /// turned by `yaw` degrees and recentred to `anchor`.
    fn focused_params(focus: Option<Focus>, ipd: f32, yaw: f32, anchor: Quat) -> SceneParams {
        let video = VideoParams {
            format: VideoFormat::new(Projection::EQUIRECT_180, StereoLayout::SideBySide),
            settings: ViewSettings {
                yaw,
                ..Default::default()
            },
            anchor,
            focus,
            ..Default::default()
        };
        let mut p = build(&video, None, [Mat4::IDENTITY; 2]);
        // The eyes' world positions: across the video's own x axis, wherever
        // the head is.
        let to_world = anchor * Quat::from_mat4(&p.correction).inverse();
        let head = Vec3::new(0.3, 1.6, -0.2);
        let eyes = [-0.5, 0.5].map(|s| head + to_world * Vec3::new(s * ipd, 0.0, 0.0));
        place_eyes(&mut p, anchor, eyes);
        p
    }

    /// Within a hundredth of a millimetre at a metre.
    fn close(a: Vec3, b: Vec3) -> bool {
        (a.normalize() - b.normalize()).length() < 1e-5
    }

    #[test]
    fn a_point_at_the_focus_distance_is_seen_where_it_really_is() {
        let anchor = Quat::from_rotation_y(0.7);
        for (ipd, baseline) in [(0.064, 0.064), (0.058, 0.064), (0.072, 0.060), (0.064, 0.0)] {
            for distance in [0.1, 0.35, 1.0, 4.0, 40.0] {
                for yaw in [0.0, 25.0] {
                    let focus = Focus { distance, baseline };
                    let p = focused_params(Some(focus), ipd, yaw, anchor);
                    let to_world = anchor * Quat::from_mat4(&p.correction).inverse();
                    for eye in 0..2 {
                        let s = if eye == 1 { 0.5 } else { -0.5 };
                        let camera = Vec3::new(s * baseline, 0.0, 0.0);
                        let viewer = Vec3::new(s * ipd, 0.0, 0.0);
                        for (lon, lat) in
                            [(0.0f32, 0.0f32), (30.0, 10.0), (-60.0, -35.0), (5.0, 70.0)]
                        {
                            let (lon, lat) = (lon.to_radians(), lat.to_radians());
                            let towards =
                                Vec3::new(lat.cos() * lon.sin(), lat.sin(), -lat.cos() * lon.cos());
                            // On the sphere around this eye's camera.
                            let point = camera + distance * towards;
                            let ray = to_world * (point - viewer).normalize();
                            let seen = video_dir(&p, ray, eye);
                            assert!(
                                close(seen, towards),
                                "ipd {ipd} baseline {baseline} at {distance} m, eye {eye}: \
                                 {seen} for {towards}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn eyes_where_the_cameras_were_see_what_they_always_did() {
        let anchor = Quat::from_rotation_y(-0.4);
        let off = focused_params(None, 0.064, 10.0, anchor);
        for distance in [0.1, 1.0, 10.0] {
            let focus = Focus {
                distance,
                baseline: 0.064,
            };
            let on = focused_params(Some(focus), 0.064, 10.0, anchor);
            for eye in 0..2 {
                for ray in [Vec3::NEG_Z, Vec3::new(0.5, 0.3, -1.0).normalize()] {
                    assert!(close(video_dir(&on, ray, eye), video_dir(&off, ray, eye)));
                }
            }
        }
    }

    #[test]
    fn without_focus_nothing_changes() {
        let video = VideoParams {
            format: VideoFormat::new(Projection::EQUIRECT_180, StereoLayout::SideBySide),
            ..Default::default()
        };
        let p = build(&video, None, [Mat4::IDENTITY; 2]);
        assert_eq!(p.focus, Vec4::ZERO);
        // Off means the shader takes upstream's call as it was: the viewer's
        // offset from the sphere's centre, on the 3 m sphere.
        let mut q = focused_params(None, 0.07, 0.0, Quat::IDENTITY);
        q.viewer = Vec3::new(0.0, 0.2, -0.5).extend(SPHERE_RADIUS);
        let ray = Vec3::new(0.2, -0.1, -1.0).normalize();
        let upstream = zoom_dir(
            off_centre(
                q.correction.transform_vector3(ray).normalize(),
                q.correction.transform_vector3(q.viewer.xyz()),
                q.viewer.w,
            ),
            q.proj.z,
        );
        for eye in 0..2 {
            assert_eq!(video_dir(&q, ray, eye), upstream);
        }
        // A flat screen has no sphere to put a focus on.
        let flat = VideoParams {
            focus: Some(Focus {
                distance: 1.0,
                baseline: 0.064,
            }),
            ..Default::default()
        };
        assert_eq!(build(&flat, None, [Mat4::IDENTITY; 2]).focus, Vec4::ZERO);
    }

    #[test]
    fn the_focus_is_never_nearer_than_a_tenth_of_a_metre() {
        let p = focused_params(
            Some(Focus {
                distance: 0.01,
                baseline: 0.064,
            }),
            0.064,
            0.0,
            Quat::IDENTITY,
        );
        assert_eq!(p.focus.x, MIN_FOCUS_DISTANCE);
    }

    #[test]
    fn the_viewer_stays_well_inside_a_near_sphere() {
        let mut p = focused_params(
            Some(Focus {
                distance: 0.1,
                baseline: 0.064,
            }),
            0.064,
            0.0,
            Quat::IDENTITY,
        );
        // Leaning a metre forward with the picture only 10 cm away.
        p.viewer = Vec3::new(0.0, 0.0, -1.0).extend(SPHERE_RADIUS);
        for ray in [
            Vec3::NEG_Z,
            Vec3::Z,
            Vec3::X,
            Vec3::new(0.3, 0.9, 0.1).normalize(),
        ] {
            let d = video_dir(&p, ray, 0);
            assert!(d.is_finite() && (d.length() - 1.0).abs() < 1e-4, "{d}");
        }
    }

    #[test]
    fn eyes_are_placed_from_the_middle_of_the_head_in_the_room_frame() {
        let mut p = build(&VideoParams::default(), None, [Mat4::IDENTITY; 2]);
        let anchor = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        // Eyes 64 mm apart along world -z: the room's x after a quarter turn.
        let head = Vec3::new(1.0, 1.7, 2.0);
        place_eyes(
            &mut p,
            anchor,
            [
                head + Vec3::new(0.0, 0.0, 0.032),
                head - Vec3::new(0.0, 0.0, 0.032),
            ],
        );
        assert!((p.eyes[0].xyz() - Vec3::new(-0.032, 0.0, 0.0)).length() < 1e-6);
        assert!((p.eyes[1].xyz() - Vec3::new(0.032, 0.0, 0.0)).length() < 1e-6);
        let view = Mat4::from_rotation_y(0.3).inverse() * Mat4::from_translation(-head);
        assert!((eye_position(view) - head).length() < 1e-5);
    }

    #[test]
    fn full_side_by_side_marked_1_2_shows_eyes_at_their_shape() {
        let eye = |width, height, stereo, pixel_aspect| {
            let video = VideoParams {
                format: VideoFormat::new(Projection::Flat, stereo),
                pixel_aspect,
                ..Default::default()
            };
            let frame = FrameDesc {
                width,
                height,
                layout: PixelLayout::Nv12,
                color: ColorInfo {
                    matrix: Matrix::Bt709,
                    full_range: false,
                    transfer: Transfer::Sdr,
                    wide_gamut: false,
                },
            };
            let p = build(&video, Some(&frame), [Mat4::IDENTITY; 2]);
            p.screen.x / p.screen.y
        };
        let near = |a: f32, want: f32| (a - want).abs() < 1e-3;
        // A 2.40:1 film as full side by side, marked 1:2 so the frame shows
        // at 2.40:1 in 2D: each eye is 2.40:1, whether picked full or half.
        assert!(near(eye(3840, 800, StereoLayout::SideBySide, 0.5), 2.4));
        assert!(near(eye(3840, 800, StereoLayout::HalfSideBySide, 0.5), 2.4));
        // 2x anamorphic top/bottom is unsqueezed: a 1920x3200 clip of
        // 2.40:1 eyes.
        assert!(near(eye(1920, 3200, StereoLayout::TopBottom, 2.0), 2.4));
        // Other pixel aspects still widen each eye; mono keeps 1:2.
        assert!(near(
            eye(1920, 480, StereoLayout::SideBySide, 32.0 / 27.0),
            960.0 * 32.0 / 27.0 / 480.0
        ));
        assert!(near(eye(1920, 800, StereoLayout::Mono, 0.5), 1.2));
    }
}
