// Where the patch shader built with planes_tp10.wgsl writes: a binding after
// the TP10 bands (src/tp10_eyes.rs).
@group(0) @binding(14) var<storage, read_write> out: array<f32>;
