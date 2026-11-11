// Where the patch shader built with planes.wgsl writes: its own set, after
// the scene's.
@group(1) @binding(0) var<storage, read_write> out: array<f32>;
