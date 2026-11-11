//! Autofocus's measuring (`fp_core::epipolar`) on the GPU: each eye's
//! picture around a ray drawn into strips along epipolar lines
//! (`shaders/patch.wgsl`), read back when the frame's slot comes round, and
//! matched by the caller off the render thread. It reads the frame as the
//! eye passes do: planes through the scene's set, or a Q10C picture in place
//! through the TP10 descriptors ([`crate::tp10_eyes`]).

use crate::gpu::Gpu;
use crate::mem::{self, Buffer};
use crate::params::{SceneParams, off_centre, zoom_dir};
use crate::{Result, VkContext};
use ash::vk;
use fp_core::epipolar::SIGNATURE;
use fp_core::epipolar::{Aim, Layout};
use fp_core::format::{Projection, StereoLayout, VideoFormat};
use fp_media::PixelLayout;
use glam::{Vec2, Vec3, Vec4Swizzles};
use gpu_allocator::MemoryLocation;
use std::f32::consts::TAU;

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
pub(crate) fn pipeline_pushing(
    gpu: &Gpu,
    spv: &[u8],
    sets: &[vk::DescriptorSetLayout],
    descriptor_buffer: bool,
    push: u32,
) -> Result<(vk::PipelineLayout, vk::Pipeline)> {
    let d = &gpu.device;
    let ranges = [vk::PushConstantRange::default()
        .stage_flags(vk::ShaderStageFlags::COMPUTE)
        .size(push)];
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
pub(crate) fn dispatch_bytes(
    gpu: &Gpu,
    cmd: vk::CommandBuffer,
    layout: vk::PipelineLayout,
    push: &[u8],
    groups: u32,
) {
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
        gpu.device
            .cmd_push_constants(cmd, layout, vk::ShaderStageFlags::COMPUTE, 0, push);
        gpu.device.cmd_dispatch(cmd, groups, 1, 1);
    }
    mem::memory_barrier(gpu, cmd, written, host);
}

/// The most floats a patch and its coarse picture take.
pub(crate) const PATCH_FLOATS: usize = 24_576;
pub(crate) const PATCH_BYTES: u64 = PATCH_FLOATS as u64 * 4;

/// What to draw this frame.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PatchAsk {
    /// A world-space direction from the viewer, and the probe's turn from it
    /// (`Aim::Ray`), or a place in the picture (`Aim::At`).
    pub ray: Vec3,
    pub aim: Aim,
    pub layout: Layout,
    /// The rig's pitch and roll, degrees, for a layout whose vertical offset
    /// is left to be predicted where it lands (NaN).
    pub vertical: Option<(f32, f32)>,
}

/// A drawn patch.
#[derive(Clone, Debug)]
pub struct PatchResult {
    /// Its middle's `α`, `β`, degrees, and the longitude it's at.
    pub at: [f32; 2],
    pub lon: f32,
    pub layout: Layout,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub signature: [f32; SIGNATURE],
}

/// The shader's `Strips`.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub(crate) struct Strips {
    centre: Vec2,
    pitch: f32,
    left_columns: u32,
    right_columns: u32,
    left_rows: u32,
    right_rows: u32,
    padding: u32,
    left_start: Vec2,
    right_start: Vec2,
    fov: Vec2,
    slots: Vec2,
    texel: f32,
    padding2: f32,
}

/// What a slot drew, until it's read.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Drawn {
    pub at: [f32; 2],
    pub lon: f32,
    pub layout: Layout,
    pub tp10: bool,
}

/// Turns `d` by `right` and `up` degrees, about the world's up and then the
/// axis across.
fn turned(d: Vec3, right: f32, up: f32) -> Vec3 {
    let across = d.cross(Vec3::Y).try_normalize().unwrap_or(Vec3::X);
    let d = glam::Quat::from_rotation_y(-right.to_radians()) * d;
    glam::Quat::from_axis_angle(across, up.to_radians()) * d
}

/// Where a patch asked for goes, for a 180° side by side picture drawn by
/// `p`: the shader's push constants and what to note. `None` when it misses
/// the picture or doesn't fit.
pub(crate) fn place(p: &SceneParams, ask: &PatchAsk) -> Option<(Strips, Drawn)> {
    let (h_fov, v_fov) = (p.proj.x, p.proj.y);
    if h_fov <= 0.0 || v_fov <= 0.0 || h_fov >= TAU - 1e-3 {
        return None;
    }
    let (alpha, beta) = match ask.aim {
        Aim::At([a, b]) => (a.to_radians(), b.to_radians()),
        Aim::Ray([right, up]) => {
            let ray = ask.ray.try_normalize()?;
            let room = p.anchor_inv.transform_vector3(turned(ray, right, up));
            let d = p.correction.transform_vector3(room).normalize();
            let viewer = p.correction.transform_vector3(p.viewer.xyz());
            let d = zoom_dir(off_centre(d, viewer, p.viewer.w), p.proj.z);
            if d.z >= 0.0 {
                return None;
            }
            (d.x.clamp(-1.0, 1.0).asin(), (-d.y).atan2(-d.z))
        }
    };
    let mut l = ask.layout;
    if l.vertical_base.is_nan() {
        let fit = ask.vertical.unwrap_or((0.0, 0.0));
        l = l.at_vertical(fp_core::epipolar::vertical_of(
            fit,
            alpha.to_degrees(),
            beta.to_degrees(),
        ));
    }
    if l.floats() + SIGNATURE > PATCH_FLOATS {
        return None;
    }
    let d = Vec3::new(
        alpha.sin(),
        -alpha.cos() * beta.sin(),
        -alpha.cos() * beta.cos(),
    );
    let lon = d.x.atan2(-d.z).to_degrees();
    let left_slot = p.mode[2] as f32 % 2.0;
    let r = |deg: f32| deg.to_radians();
    Some((
        Strips {
            centre: Vec2::new(alpha, beta),
            pitch: r(l.pitch),
            left_columns: l.left_columns,
            right_columns: l.right_columns,
            left_rows: l.left_rows,
            right_rows: l.right_rows,
            padding: 0,
            left_start: Vec2::new(r(l.left_alpha), r(l.left_beta)),
            right_start: Vec2::new(r(l.right_alpha), r(l.right_beta)),
            fov: Vec2::new(h_fov, v_fov),
            slots: Vec2::new(left_slot, 1.0 - left_slot),
            texel: 0.5 * p.tex_size.z,
            padding2: 0.0,
        },
        Drawn {
            at: [alpha.to_degrees(), beta.to_degrees()],
            lon,
            layout: l,
            tp10: false,
        },
    ))
}

/// Workgroups for a patch.
pub(crate) fn groups(s: &Strips) -> u32 {
    let n = s.left_columns * s.left_rows + s.right_columns * s.right_rows + SIGNATURE as u32;
    n.div_ceil(64)
}

pub(crate) fn push_size() -> u32 {
    size_of::<Strips>() as u32
}

/// The patch in a result buffer's bytes.
pub(crate) fn result(bytes: &[u8], d: &Drawn) -> PatchResult {
    let f: Vec<f32> = bytes[..(d.layout.floats() + SIGNATURE) * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_ne_bytes(*b))
        .collect();
    let split = (d.layout.left_columns * d.layout.left_rows) as usize;
    let end = d.layout.floats();
    let mut signature = [0.0; SIGNATURE];
    signature.copy_from_slice(&f[end..end + SIGNATURE]);
    PatchResult {
        at: d.at,
        lon: d.lon,
        layout: d.layout,
        left: f[..split].to_vec(),
        right: f[split..end].to_vec(),
        signature,
    }
}

/// The pipeline for frames in planes, through the scene's descriptor set,
/// and its result buffers.
pub(crate) struct Patcher {
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    pool: vk::DescriptorPool,
    sets: Vec<vk::DescriptorSet>,
    results: Vec<Buffer>,
    pub drawn: Vec<Option<Drawn>>,
}

impl Patcher {
    pub fn new(gpu: &Gpu, scene_layout: vk::DescriptorSetLayout, slots: usize) -> Option<Patcher> {
        let mut p = Patcher {
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            pool: vk::DescriptorPool::null(),
            sets: Vec::new(),
            results: Vec::new(),
            drawn: vec![None; slots],
        };
        match p.create(gpu, scene_layout, slots) {
            Ok(()) => Some(p),
            Err(e) => {
                log::warn!("epipolar measuring unavailable: {e}");
                p.destroy(gpu);
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
        let d = &gpu.device;
        let binding = [vk::DescriptorSetLayoutBinding::default()
            .binding(0)
            .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(1)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)];
        // SAFETY: plain object creation on a valid device; destroyed by
        // `destroy`.
        unsafe {
            self.set_layout = d
                .create_descriptor_set_layout(
                    &vk::DescriptorSetLayoutCreateInfo::default().bindings(&binding),
                    None,
                )
                .ctx("patch descriptor set layout")?;
            (self.layout, self.pipeline) = pipeline_pushing(
                gpu,
                crate::PATCH_SPV,
                &[scene_layout, self.set_layout],
                false,
                push_size(),
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
                .ctx("patch descriptor pool")?;
            for _ in 0..slots {
                let layouts = [self.set_layout];
                let set = d
                    .allocate_descriptor_sets(
                        &vk::DescriptorSetAllocateInfo::default()
                            .descriptor_pool(self.pool)
                            .set_layouts(&layouts),
                    )
                    .ctx("patch descriptor set")?[0];
                let result = Buffer::new(
                    gpu,
                    PATCH_BYTES,
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                    MemoryLocation::GpuToCpu,
                    "patch",
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

    pub fn record(
        &mut self,
        gpu: &Gpu,
        cmd: vk::CommandBuffer,
        slot: usize,
        scene_set: vk::DescriptorSet,
        s: &Strips,
    ) {
        // SAFETY: recording; the sets aren't in use by a pending submission.
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
        dispatch_bytes(gpu, cmd, self.layout, bytemuck::bytes_of(s), groups(s));
    }

    pub fn result(&mut self, slot: usize) -> &[u8] {
        self.results[slot].bytes()
    }

    pub fn destroy(&mut self, gpu: &Gpu) {
        for mut b in self.results.drain(..) {
            b.destroy(gpu);
        }
        // SAFETY: the device is idle; null handles are allowed.
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
    use fp_core::format::{Projection, StereoLayout, VideoFormat};
    use glam::Mat4;

    fn params() -> SceneParams {
        params::build(
            &VideoParams {
                format: VideoFormat::new(Projection::EQUIRECT_180, StereoLayout::SideBySide),
                ..Default::default()
            },
            None,
            [Mat4::IDENTITY; 2],
        )
    }

    fn ask(ray: Vec3, aim: Aim) -> PatchAsk {
        PatchAsk {
            ray,
            aim,
            layout: Layout::coarse(256, 0.0, 2.0),
            vertical: None,
        }
    }

    #[test]
    fn straight_ahead_is_the_middle_of_both_angles() {
        let (s, d) = place(&params(), &ask(Vec3::NEG_Z, Aim::Ray([0.0, 0.0]))).unwrap();
        assert!(s.centre.length() < 1e-6);
        assert!(d.lon.abs() < 1e-4);
        assert_eq!(s.slots, Vec2::new(0.0, 1.0));
        assert!(groups(&s) * 64 >= (d.layout.floats() + SIGNATURE) as u32);
    }

    #[test]
    fn up_and_to_the_side_turn_the_plane_and_move_along_it() {
        // 30° up: the plane turns up (β negative), α stays 0.
        let up = glam::Quat::from_rotation_x(30f32.to_radians()) * Vec3::NEG_Z;
        let (_, d) = place(&params(), &ask(up, Aim::Ray([0.0, 0.0]))).unwrap();
        assert!(
            d.at[0].abs() < 1e-3 && (d.at[1] + 30.0).abs() < 1e-3,
            "{:?}",
            d.at
        );
        // 20° right: along the plane.
        let (_, d) = place(&params(), &ask(Vec3::NEG_Z, Aim::Ray([20.0, 0.0]))).unwrap();
        assert!(
            (d.at[0] - 20.0).abs() < 1e-3 && d.at[1].abs() < 1e-3,
            "{:?}",
            d.at
        );
        assert!((d.lon - 20.0).abs() < 1e-3);
    }

    #[test]
    fn a_place_is_kept_and_an_unknown_vertical_is_predicted_there() {
        let mut a = ask(Vec3::NEG_Z, Aim::At([10.0, -5.0]));
        a.layout = Layout::coarse(256, f32::NAN, 0.5);
        a.vertical = Some((1.0, 2.0));
        let (s, d) = place(&params(), &a).unwrap();
        assert!((s.centre - Vec2::new(10f32.to_radians(), -5f32.to_radians())).length() < 1e-6);
        let want = 1.0 + 2.0 * 10f32.to_radians().tan() * (-5f32).to_radians().cos();
        assert!((d.layout.vertical_base - want).abs() < 1e-4);
    }

    #[test]
    fn behind_is_nothing_to_draw() {
        assert!(place(&params(), &ask(Vec3::Z, Aim::Ray([0.0, 0.0]))).is_none());
    }
}
