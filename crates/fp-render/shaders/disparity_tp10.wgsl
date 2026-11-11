// Where the disparity shader built with planes_tp10.wgsl writes: a pipeline
// reading descriptor buffers reads them all from there, so it is a binding
// after the TP10 bands (src/tp10_eyes.rs).
@group(0) @binding(14) var<storage, read_write> result: Result;
