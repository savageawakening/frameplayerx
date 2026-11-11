//! Measures stereo disparity for autofocus: where a ray from the viewer falls
//! in both eye images, and the compute pass that compares them there
//! (`shaders/disparity.wgsl`). The pass reads the frame as the eye passes do,
//! uploaded or converted planes through the scene's descriptor set, or a Q10C
//! picture where it lies through the TP10 descriptors ([`crate::tp10_eyes`]).
//! Results come back without waiting: a slot's measurement is read when the
//! slot comes round again.

use crate::gpu::Gpu;
use crate::mem::{self, Buffer};
use crate::params::{SceneParams, off_centre, zoom_dir};
use crate::{Result, VkContext};
use ash::vk;
use fp_core::autofocus::{
    CANDIDATES, ROW_STEP_DEGREES, Reading, SIGNATURE, STEP_DEGREES, VERTICAL_CANDIDATES,
};
use fp_core::format::{Projection, StereoLayout, VideoFormat};
use fp_media::PixelLayout;
use glam::{Mat4, Vec2, Vec3, Vec4Swizzles};
use gpu_allocator::MemoryLocation;
use std::f32::consts::TAU;

/// No closer to the picture's left or right edge than this, degrees: past it
/// the window itself runs out of picture.
const EDGE_DEGREES: f32 = 3.0;

/// What to measure this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Measure {
    /// A world-space direction from the viewer.
    pub ray: Vec3,
    /// How far the right image's picture sits below the left's, degrees: the
    /// search is made at this offset.
    pub vertical: f32,
    /// Each sample the mean of four taps over its share of the window, rather
    /// than one point.
    pub prefilter: bool,
    /// How much more the middle of the window counts than its edge: 0 is
    /// evenly.
    pub centre: f32,
    /// Handed back with the reading.
    pub tag: usize,
}

/// Whether a video can be measured: 180° equirectangular, side by side, in a
/// layout with a luma plane the eye passes sample.
pub fn measurable(format: &VideoFormat, rotation: u32, layout: PixelLayout) -> bool {
    let projection =
        matches!(format.projection, Projection::Equirect { h_fov, .. } if h_fov < 360.0);
    let stereo = matches!(
        format.stereo,
        StereoLayout::SideBySide | StereoLayout::HalfSideBySide
    );
    let depth = matches!(
        layout,
        PixelLayout::Nv12
            | PixelLayout::I420 { bits: 8 | 10 }
            | PixelLayout::P010
            | PixelLayout::Q10c
    );
    projection && stereo && depth && rotation.is_multiple_of(360)
}

/// Where to measure, in the stored frame; the shader's `Window`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Window {
    left: Vec2,
    right: Vec2,
    step: Vec2,
    left_bounds: Vec2,
    right_bounds: Vec2,
    left_v: Vec2,
    prefilter: f32,
    centre: f32,
}

/// Where the world-space ray of `m` falls in each eye image of the video that
/// `p` draws, for a video [`measurable`] accepts, and where that is in the
/// picture, degrees: longitude and latitude. `None` when it misses the
/// picture or is too near its edge.
///
/// The ray goes through the projection shader's steps without the IPD offset
/// or the autofocus: what's measured is the video's own shift between the
/// eyes. The subject the eyes are on is taken to be between the images, half
/// its disparity to either side of the ray.
pub(crate) fn window(p: &SceneParams, m: &Measure) -> Option<(Window, [f32; 2])> {
    let ray = m.ray.try_normalize()?;
    let (h_fov, v_fov) = (p.proj.x, p.proj.y);
    if h_fov <= 0.0 || v_fov <= 0.0 || h_fov >= TAU - 1e-3 {
        return None;
    }
    let room = p.anchor_inv.transform_vector3(ray);
    let in_video = |eye: f32| {
        let align = Mat4::from_rotation_z(p.lens.w * eye) * Mat4::from_rotation_x(p.lens.z * eye);
        let d = p
            .correction
            .transform_vector3(align.transform_vector3(room))
            .normalize();
        let viewer = p.correction.transform_vector3(p.viewer.xyz());
        let d = zoom_dir(off_centre(d, viewer, p.viewer.w), p.proj.z);
        Vec2::new(
            0.5 + d.x.atan2(-d.z) / h_fov,
            0.5 - d.y.clamp(-1.0, 1.0).asin() / v_fov,
        )
    };
    let half_vertical = 0.5 * m.vertical.to_radians() / v_fov;
    let left = in_video(-0.5) - Vec2::new(0.0, half_vertical);
    let right = in_video(0.5) + Vec2::new(0.0, half_vertical);
    let edge = EDGE_DEGREES.to_radians() / h_fov;
    if (left.x - 0.5).abs() > 0.5 - edge
        || !(0.0..=1.0).contains(&left.y)
        || !(0.0..=1.0).contains(&right.y)
    {
        return None;
    }
    // Which half of the frame each is in, with the eyes perhaps swapped.
    let left_slot = p.mode[2] as f32 % 2.0;
    let right_slot = 1.0 - left_slot;
    let stored = |uv: Vec2, slot: f32| Vec2::new(uv.x * 0.5 + slot * 0.5, uv.y);
    // Half a texel in from the edges, so that nothing of the other eye blends in.
    let texel = 0.5 * p.tex_size.z;
    let bounds = |slot: f32| Vec2::new(slot * 0.5 + texel, slot * 0.5 + 0.5 - texel);
    Some((
        Window {
            left: stored(left, left_slot),
            right: stored(right, right_slot),
            step: Vec2::new(
                STEP_DEGREES.to_radians() / h_fov * 0.5,
                ROW_STEP_DEGREES.to_radians() / v_fov,
            ),
            left_bounds: bounds(left_slot),
            right_bounds: bounds(right_slot),
            left_v: Vec2::new(0.0, 1.0),
            prefilter: if m.prefilter { 1.0 } else { 0.0 },
            centre: m.centre,
        },
        [
            (left.x - 0.5) * h_fov.to_degrees(),
            (0.5 - left.y - half_vertical) * v_fov.to_degrees(),
        ],
    ))
}

/// The result's floats: luma and gradient scores, variances, the best shift
/// and padding, the vertical scores and the coarse picture.
pub(crate) const RESULT_FLOATS: usize = CANDIDATES * 2 + 4 + VERTICAL_CANDIDATES + SIGNATURE;
pub(crate) const RESULT_BYTES: u64 = RESULT_FLOATS as u64 * 4;
/// What the shader's workgroup memory needs: two strips of eight rows.
pub(crate) const SHARED_BYTES: u32 = (2 * (2 * 24 + CANDIDATES as u32 - 2) * 8 + 1) * 4;

/// The reading in a result buffer's bytes.
pub(crate) fn reading(bytes: &[u8], place: [f32; 2], vertical: f32, tag: usize) -> Reading {
    let f: Vec<f32> = bytes[..RESULT_BYTES as usize]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_ne_bytes(*b))
        .collect();
    let take = |from: usize, into: &mut [f32]| into.copy_from_slice(&f[from..from + into.len()]);
    let mut r = Reading {
        place,
        vertical_used: vertical,
        tag,
        ..Reading::default()
    };
    take(0, &mut r.luma);
    take(CANDIDATES, &mut r.gradient);
    r.variance = f[2 * CANDIDATES];
    r.gradient_variance = f[2 * CANDIDATES + 1];
    take(2 * CANDIDATES + 4, &mut r.vertical);
    take(2 * CANDIDATES + 4 + VERTICAL_CANDIDATES, &mut r.signature);
    r
}

/// Whether the device has what the shader needs.
pub(crate) fn supported(gpu: &Gpu) -> Result<()> {
    // SAFETY: valid physical device.
    let limits = unsafe { gpu.instance.get_physical_device_properties(gpu.pdev) }.limits;
    let invocations = CANDIDATES as u32;
    if limits.max_compute_shared_memory_size < SHARED_BYTES
        || limits.max_compute_work_group_invocations < invocations
        || limits.max_compute_work_group_size[0] < invocations
    {
        return Err(crate::Error::Unsupported(format!(
            "needs {SHARED_BYTES} bytes of workgroup memory and {invocations} invocations",
        )));
    }
    Ok(())
}

/// A compute pipeline of `spv` laid out as `layout`, with the window pushed.
pub(crate) fn pipeline(
    gpu: &Gpu,
    spv: &[u8],
    sets: &[vk::DescriptorSetLayout],
    descriptor_buffer: bool,
) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
    let d = &gpu.device;
    let ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .size(size_of::<Window>() as u32)];
    // SAFETY: plain object creation on a valid device; the caller destroys
    // what's returned.
    unsafe {
        let layout = d
            .create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default()
                    .set_layouts(sets)
                    .push_constant_ranges(&ranges),
                None,
            )
            .ctx("disparity pipeline layout")?;
        let module = match gpu.shader(spv) {
            Ok(m) => m,
            Err(e) => {
                d.destroy_pipeline_layout(layout, None);
                return Err(e);
            }
        };
        let flags = if descriptor_buffer {
            vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT
        } else {
            vk::PipelineCreateFlags::empty()
        };
        let info = vk::ComputePipelineCreateInfo::default()
            .flags(flags)
            .stage(
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::COMPUTE)
                    .module(module)
                    .name(c"main"),
            )
            .layout(layout);
        let made = d.create_compute_pipelines(vk::PipelineCache::null(), &[info], None);
        d.destroy_shader_module(module, None);
        match made {
            Ok(p) => Ok((layout, p[0])),
            Err((_, e)) => {
                d.destroy_pipeline_layout(layout, None);
                Err(e).ctx("disparity pipeline")
            }
        }
    }
}

/// Orders the dispatch after the eye passes' reads of the frame (and so
/// after whatever wrote or acquired it), and records it with `window`
/// pushed. The pipeline and its descriptors are bound.
pub(crate) fn dispatch(gpu: &Gpu, cmd: vk::CommandBuffer, layout: vk::PipelineLayout, w: &Window) {
    let before = (
        vk::PipelineStageFlags2::TRANSFER
            | vk::PipelineStageFlags2::COMPUTE_SHADER
            | vk::PipelineStageFlags2::FRAGMENT_SHADER,
        vk::AccessFlags2::TRANSFER_WRITE | vk::AccessFlags2::SHADER_STORAGE_WRITE,
    );
    let read = (
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_SAMPLED_READ | vk::AccessFlags2::SHADER_STORAGE_WRITE,
    );
    let written = (
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_STORAGE_WRITE,
    );
    let host = (vk::PipelineStageFlags2::HOST, vk::AccessFlags2::HOST_READ);
    mem::memory_barrier(gpu, cmd, before, read);
    // SAFETY: recording outside a render pass, with the pipeline bound.
    unsafe {
        gpu.device.cmd_push_constants(
            cmd,
            layout,
            vk::ShaderStageFlags::COMPUTE,
            0,
            bytemuck::bytes_of(w),
        );
        gpu.device.cmd_dispatch(cmd, 1, 1, 1);
    }
    mem::memory_barrier(gpu, cmd, written, host);
}

/// What a slot measured, until it's read.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Pending {
    pub place: [f32; 2],
    pub vertical: f32,
    pub tag: usize,
    /// Through the TP10 pipeline, whose result is in its own buffer.
    pub tp10: bool,
}

/// The pipeline for frames in planes, sampled through the scene's
/// descriptor set; its result buffers; and what each slot measured.
pub(crate) struct Disparity {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    results: Vec<Buffer>,
    pub pending: Vec<Option<Pending>>,
}

impl Disparity {
    /// `None` (and a log line) where it can't be made: the video still plays,
    /// without autofocus.
    pub fn new(
        gpu: &Gpu,
        scene_layout: vk::DescriptorSetLayout,
        slots: usize,
    ) -> Option<Disparity> {
        let mut d = Disparity {
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            pool: vk::DescriptorPool::null(),
            sets: Vec::new(),
            results: Vec::new(),
            pending: vec![None; slots],
        };
        match d.create(gpu, scene_layout, slots) {
            Ok(()) => Some(d),
            Err(e) => {
                log::warn!("autofocus unavailable: {e}");
                d.destroy(gpu);
                None
            }
        }
    }

    fn create(
        &mut self,
        gpu: &Gpu,
        scene_layout: vk::DescriptorSetLayout,
        slots: usize,
    ) -> Result<()> {
        supported(gpu)?;
        let d = &gpu.device;
        let binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        // SAFETY: plain object creation on a valid device; everything made
        // here is destroyed by `destroy`.
        unsafe {
            self.set_layout = d
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding),
                    None,
                )
                .ctx("disparity descriptor set layout")?;
            (self.layout, self.pipeline) = pipeline(
                gpu,
                crate::DISPARITY_SPV,
                &[scene_layout, self.set_layout],
                false,
            )?;
            let sizes = [vk::DescriptorPoolSize {
                ty: vk::DescriptorType::STORAGE_BUFFER,
                descriptor_count: slots as u32,
            }];
            self.pool = d
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo::default()
                        .max_sets(slots as u32)
                        .pool_sizes(&sizes),
                    None,
                )
                .ctx("disparity descriptor pool")?;
            for _ in 0..slots {
                let layouts = [self.set_layout];
                let set = d
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(self.pool)
                            .set_layouts(&layouts),
                    )
                    .ctx("disparity descriptor set")?[0];
                let result = Buffer::new(
                    gpu,
                    RESULT_BYTES,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    MemoryLocation::GpuToCpu,
                    "disparity",
                )?;
                let info = [vk::DescriptorBufferInfo::default()
                    .buffer(result.buffer)
                    .range(vk::WHOLE_SIZE)];
                d.update_descriptor_sets(
                    &[vk::WriteDescriptorSet::default()
                        .dst_set(set)
                        .dst_binding(0)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .buffer_info(&info)],
                    &[],
                );
                self.sets.push(set);
                self.results.push(result);
            }
        }
        Ok(())
    }

    /// Records a measurement of the frame `scene_set` points at, for slot
    /// `slot`, outside a render pass.
    pub fn record(
        &mut self,
        gpu: &Gpu,
        cmd: vk::CommandBuffer,
        slot: usize,
        scene_set: vk::DescriptorSet,
        w: &Window,
    ) {
        // SAFETY: recording; the sets are valid and not in use by a pending
        // submission (the slot's fence has signalled).
        unsafe {
            gpu.device
                .cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            gpu.device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.layout,
                0,
                &[scene_set, self.sets[slot]],
                &[],
            );
        }
        dispatch(gpu, cmd, self.layout, w);
    }

    /// The bytes slot `slot`'s planes measurement wrote.
    pub fn result(&mut self, slot: usize) -> &[u8] {
        self.results[slot].bytes()
    }

    pub fn destroy(&mut self, gpu: &Gpu) {
        for mut b in self.results.drain(..) {
            b.destroy(gpu);
        }
        // SAFETY: the device is idle; destroying null handles is allowed.
        unsafe {
            let d = &gpu.device;
            d.destroy_descriptor_pool(std::mem::take(&mut self.pool), None);
            d.destroy_pipeline(std::mem::take(&mut self.pipeline), None);
            d.destroy_pipeline_layout(std::mem::take(&mut self.layout), None);
            d.destroy_descriptor_set_layout(std::mem::take(&mut self.set_layout), None);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::params::{self, VideoParams};
    use fp_core::view::ViewSettings;
    use glam::Quat;

    const FORWARD: Vec3 = Vec3::NEG_Z;

    fn video(settings: ViewSettings) -> VideoParams {
        VideoParams {
            format: VideoFormat::new(Projection::EQUIRECT_180, StereoLayout::SideBySide),
            settings,
            ..Default::default()
        }
    }

    fn measure(ray: Vec3) -> Measure {
        Measure {
            ray,
            vertical: 0.0,
            prefilter: true,
            centre: 0.0,
            tag: 0,
        }
    }

    fn at(video: &VideoParams, ray: Vec3) -> Option<Window> {
        window(
            &params::build(video, None, [Mat4::IDENTITY; 2]),
            &measure(ray),
        )
        .map(|w| w.0)
    }

    fn near(a: Vec2, b: Vec2) -> bool {
        (a - b).length() < 1e-4
    }

    fn turned(yaw: f32, pitch: f32) -> Vec3 {
        Quat::from_euler(
            glam::EulerRot::YXZ,
            yaw.to_radians(),
            pitch.to_radians(),
            0.0,
        ) * FORWARD
    }

    #[test]
    fn straight_ahead_is_the_middle_of_each_eye() {
        let w = at(&video(ViewSettings::default()), FORWARD).unwrap();
        assert!(near(w.left, Vec2::new(0.25, 0.5)));
        assert!(near(w.right, Vec2::new(0.75, 0.5)));
        // 0.15° of 180°, in half the frame's width.
        assert!((w.step.x - 0.15 / 180.0 * 0.5).abs() < 1e-6);
        assert!((w.step.y - 0.1875 / 180.0).abs() < 1e-6);
        assert!(w.right_bounds.x >= 0.5 && w.right_bounds.y <= 1.0);
        assert!(w.left_bounds.x >= 0.0 && w.left_bounds.y <= 0.5);
    }

    #[test]
    fn the_window_follows_the_head() {
        let v = video(ViewSettings::default());
        let w = at(&v, turned(-45.0, 0.0)).unwrap();
        assert!(near(w.left, Vec2::new(0.375, 0.5)), "{w:?}");
        assert!(near(w.right, Vec2::new(0.875, 0.5)), "{w:?}");
        let w = at(&v, turned(0.0, 45.0)).unwrap();
        assert!(near(w.left, Vec2::new(0.25, 0.25)), "{w:?}");
    }

    #[test]
    fn the_picture_turning_turns_the_window() {
        let v = video(ViewSettings {
            yaw: 30.0,
            ..Default::default()
        });
        let shown = at(&v, turned(30.0, 0.0)).unwrap();
        assert!(near(shown.left, Vec2::new(0.25, 0.5)), "{shown:?}");
        let anchored = VideoParams {
            anchor: Quat::from_rotation_y(90f32.to_radians()),
            ..video(ViewSettings::default())
        };
        let w = at(&anchored, turned(90.0, 0.0)).unwrap();
        assert!(near(w.left, Vec2::new(0.25, 0.5)), "{w:?}");
    }

    #[test]
    fn the_ipd_offset_and_the_focus_do_not_change_where_it_measures() {
        let plain = at(&video(ViewSettings::default()), turned(-20.0, 10.0)).unwrap();
        let mut v = video(ViewSettings {
            ipd_offset: 9.0,
            ..Default::default()
        });
        v.focus = Some(params::Focus {
            distance: 0.3,
            baseline: 0.05,
        });
        let mut p = params::build(&v, None, [Mat4::IDENTITY; 2]);
        params::place_eyes(
            &mut p,
            Quat::IDENTITY,
            [Vec3::new(-0.04, 0.0, 0.0), Vec3::new(0.04, 0.0, 0.0)],
        );
        let w = window(&p, &measure(turned(-20.0, 10.0))).unwrap().0;
        assert_eq!(w, plain);
    }

    #[test]
    fn vertical_alignment_moves_the_eyes_apart() {
        let w = at(
            &video(ViewSettings {
                vertical_align: 2.0,
                ..Default::default()
            }),
            FORWARD,
        )
        .unwrap();
        assert!((w.left.y - 0.5).abs() > 1e-3);
        assert!((w.left.y - 0.5 + (w.right.y - 0.5)).abs() < 1e-4, "{w:?}");
    }

    #[test]
    fn the_learned_vertical_offset_is_split_between_the_eyes() {
        let p = params::build(&video(ViewSettings::default()), None, [Mat4::IDENTITY; 2]);
        let m = Measure {
            vertical: 1.8,
            ..measure(FORWARD)
        };
        let (w, place) = window(&p, &m).unwrap();
        let half = 0.9 / 180.0;
        assert!(near(w.left, Vec2::new(0.25, 0.5 - half)), "{w:?}");
        assert!(near(w.right, Vec2::new(0.75, 0.5 + half)), "{w:?}");
        // Still where the ray is, in the picture.
        assert!(place[1].abs() < 1e-3, "{place:?}");
    }

    #[test]
    fn a_window_knows_where_it_is_in_the_picture() {
        let of = |v: &VideoParams, ray| {
            let p = params::build(v, None, [Mat4::IDENTITY; 2]);
            window(&p, &measure(ray)).unwrap().1
        };
        let near =
            |a: [f32; 2], b: [f32; 2]| (a[0] - b[0]).abs() < 0.01 && (a[1] - b[1]).abs() < 0.01;
        let v = video(ViewSettings::default());
        assert!(near(of(&v, turned(-30.0, 0.0)), [30.0, 0.0]));
        assert!(near(of(&v, turned(0.0, 45.0)), [0.0, 45.0]));
        let swapped = video(ViewSettings {
            swap_eyes: true,
            ..Default::default()
        });
        assert!(near(of(&swapped, turned(-30.0, -20.0)), [30.0, -20.0]));
    }

    #[test]
    fn swapped_eyes_swap_the_images() {
        let w = at(
            &video(ViewSettings {
                swap_eyes: true,
                ..Default::default()
            }),
            FORWARD,
        )
        .unwrap();
        assert!(near(w.left, Vec2::new(0.75, 0.5)));
        assert!(near(w.right, Vec2::new(0.25, 0.5)));
        assert!(w.right_bounds.y <= 0.5);
    }

    #[test]
    fn zoom_narrows_what_a_turn_of_the_head_covers() {
        let v = video(ViewSettings {
            zoom: 2.0,
            ..Default::default()
        });
        let w = at(&v, turned(-40.0, 0.0)).unwrap();
        assert!(
            near(w.left, Vec2::new(0.25 + 20.0 / 180.0 * 0.5, 0.5)),
            "{w:?}"
        );
    }

    #[test]
    fn off_the_picture_is_nothing_to_measure() {
        let v = video(ViewSettings::default());
        assert!(at(&v, Vec3::Z).is_none(), "behind");
        assert!(at(&v, turned(-88.0, 0.0)).is_none(), "at the edge");
        assert!(at(&v, Vec3::ZERO).is_none(), "no ray");
        assert!(at(&v, turned(-85.0, 0.0)).is_some());
    }

    #[test]
    fn a_result_is_read_back_into_its_parts() {
        let mut floats = vec![0.0f32; RESULT_FLOATS];
        floats[CANDIDATES - 1] = 0.5;
        floats[CANDIDATES] = 0.25;
        floats[2 * CANDIDATES] = 0.01;
        floats[2 * CANDIDATES + 1] = 0.002;
        floats[2 * CANDIDATES + 4] = 0.75;
        floats[RESULT_FLOATS - 1] = 0.125;
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_ne_bytes()).collect();
        let r = reading(&bytes, [1.0, 2.0], 0.5, 7);
        assert_eq!(r.luma[CANDIDATES - 1], 0.5);
        assert_eq!(r.gradient[0], 0.25);
        assert_eq!((r.variance, r.gradient_variance), (0.01, 0.002));
        assert_eq!(r.vertical[0], 0.75);
        assert_eq!(r.signature[SIGNATURE - 1], 0.125);
        assert_eq!((r.place, r.vertical_used, r.tag), ([1.0, 2.0], 0.5, 7));
    }

    #[test]
    fn only_180_side_by_side_is_measured() {
        let f = |p, s| VideoFormat::new(p, s);
        let sbs = f(Projection::EQUIRECT_180, StereoLayout::SideBySide);
        let eight = PixelLayout::Nv12;
        for layout in [
            eight,
            PixelLayout::I420 { bits: 8 },
            PixelLayout::I420 { bits: 10 },
            PixelLayout::P010,
            PixelLayout::Q10c,
        ] {
            assert!(measurable(&sbs, 0, layout), "{layout:?}");
        }
        assert!(measurable(
            &f(Projection::EQUIRECT_180, StereoLayout::HalfSideBySide),
            0,
            eight
        ));
        assert!(!measurable(&sbs, 0, PixelLayout::DrmPrime));
        assert!(!measurable(&sbs, 0, PixelLayout::I420 { bits: 12 }));
        assert!(!measurable(&sbs, 90, eight));
        for other in [
            f(Projection::EQUIRECT_180, StereoLayout::TopBottom),
            f(Projection::EQUIRECT_180, StereoLayout::Mono),
            f(Projection::EQUIRECT_360, StereoLayout::SideBySide),
            f(Projection::Flat, StereoLayout::SideBySide),
        ] {
            assert!(!measurable(&other, 0, eight), "{other:?}");
        }
    }
}
