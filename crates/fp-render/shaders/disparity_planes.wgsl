// Where the disparity shader built with planes.wgsl writes: its own set, as
// the scene's set comes first.
@group(1) @binding(0) var<storage, read_write> result: Result;
