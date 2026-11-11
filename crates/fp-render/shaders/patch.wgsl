// Autofocus's epipolar measuring (fp_core::epipolar): each eye's picture
// around the ray drawn into a strip along epipolar lines, a row to each plane
// through both cameras, so that a point's two images differ only along the
// row. A sample is the mean of 4 by 4 taps over its share of the strip, so
// a fine picture (8K) doesn't alias at the coarse pitch. Then a coarse 16 by
// 8 picture of the left image, each cell the mean of 16 by 16 taps over its
// area, to notice cuts.
//
// Built after scene.wgsl and planes.wgsl or planes_tp10.wgsl (build.rs) for
// `luma_at`, and after patch_planes.wgsl or patch_tp10.wgsl, which say where
// the strips go.

struct Strips {
    // The strips' middle: α, the angle within the plane from its middle,
    // and β, the plane's turn about the cameras' axis (down positive),
    // radians.
    centre: vec2<f32>,
    pitch: f32,
    left_columns: u32,
    right_columns: u32,
    left_rows: u32,
    right_rows: u32,
    padding: u32,
    // Each strip's first sample from the middle, radians.
    left_start: vec2<f32>,
    right_start: vec2<f32>,
    // The picture's fields of view, radians.
    fov: vec2<f32>,
    // Which half of the frame each eye's image is in.
    slots: vec2<f32>,
    // Half a texel of the frame's width, to keep each eye to its half.
    texel: f32,
    padding2: f32,
};
var<immediate> strips: Strips;

const SIGNATURE: u32 = 128u;
// Taps across a sample's square, and across a signature cell.
const SAMPLE_TAPS: u32 = 4u;
const CELL_TAPS: u32 = 16u;

fn epipolar_luma(alpha: f32, beta: f32, slot: f32) -> f32 {
    let d = vec3<f32>(sin(alpha), -cos(alpha) * sin(beta), -cos(alpha) * cos(beta));
    let lon = atan2(d.x, -d.z);
    let lat = asin(clamp(d.y, -1.0, 1.0));
    // Off the eye's picture: no sample, which drops the window it's in.
    let x = 0.5 + lon / strips.fov.x;
    let v = 0.5 - lat / strips.fov.y;
    if (x < 0.0 || x > 1.0 || v < 0.0 || v > 1.0) {
        return -1.0;
    }
    let u = x * 0.5 + 0.5 * slot;
    let lo = 0.5 * slot + strips.texel;
    let hi = 0.5 * slot + 0.5 - strips.texel;
    return luma_at(vec2<f32>(clamp(u, lo, hi), v));
}

fn sample_at(at: vec2<f32>, slot: f32) -> f32 {
    var sum = 0.0;
    for (var t = 0u; t < SAMPLE_TAPS * SAMPLE_TAPS; t++) {
        let f = (vec2<f32>(f32(t % SAMPLE_TAPS), f32(t / SAMPLE_TAPS)) + 0.5) / f32(SAMPLE_TAPS) - 0.5;
        let v = epipolar_luma(at.x + f.x * strips.pitch, at.y + f.y * strips.pitch, slot);
        if (v < 0.0) {
            return -1.0;
        }
        sum += v;
    }
    return sum / f32(SAMPLE_TAPS * SAMPLE_TAPS);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let i = id.x;
    let left = strips.left_columns * strips.left_rows;
    let right = strips.right_columns * strips.right_rows;
    if (i < left) {
        let cr = vec2<f32>(f32(i % strips.left_columns), f32(i / strips.left_columns));
        out[i] = sample_at(strips.centre + strips.left_start + cr * strips.pitch, strips.slots.x);
    } else if (i < left + right) {
        let j = i - left;
        let cr = vec2<f32>(f32(j % strips.right_columns), f32(j / strips.right_columns));
        out[i] = sample_at(strips.centre + strips.right_start + cr * strips.pitch, strips.slots.y);
    } else if (i < left + right + SIGNATURE * CELL_TAPS) {
        // A row of a cell's taps to a thread, so no thread has many to do;
        // the rows are summed on the CPU (patch.rs).
        let k = i - left - right;
        let c = k / CELL_TAPS;
        let row = k % CELL_TAPS;
        let cell = vec2<f32>(f32(c % 16u), f32(c / 16u));
        let lo = 0.5 * strips.slots.x;
        var sum = 0.0;
        for (var t = 0u; t < CELL_TAPS; t++) {
            let tap = (vec2<f32>(f32(t), f32(row)) + 0.5) / f32(CELL_TAPS);
            let f = (cell + tap) / vec2<f32>(16.0, 8.0);
            sum += luma_at(vec2<f32>(lo + 0.5 * f.x, f.y));
        }
        out[i] = sum / f32(CELL_TAPS * CELL_TAPS);
    }
}
