// Autofocus measurement: how far the picture around where the eyes look is
// shifted between the left and the right eye image. A window of each image is
// compared at every candidate shift, half of it either side of the ray in each
// (the subject the eyes are on is between them); the score is the normalized
// cross-correlation of the luma, and of the differences between neighbouring
// samples across the window, which a smooth change in brightness doesn't fool.
// Then, at the best shift, how far the right image is shifted up or down; and
// a coarse picture of the whole left image, to notice a cut.
//
// Built after scene.wgsl and planes.wgsl or planes_tp10.wgsl (build.rs), whose
// `luma_at` reads either kind of frame where it lies, and after
// disparity_planes.wgsl or disparity_tp10.wgsl, which say where the result
// goes.
//
// Adapted from Stereonaut's tools/openxr-playback/stereo-disparity.comp,
// Copyright (c) 2026 Erdnusschokolade, MIT License
// (packaging/licenses/Stereonaut-MIT.txt).

struct Window {
    // The window's centre in each image, stored-frame uv, at no disparity and
    // with the learned vertical offset already split between them.
    left: vec2<f32>,
    right: vec2<f32>,
    // Between columns and between rows, uv. A shift moves each window half a
    // column.
    step: vec2<f32>,
    // The u range of each image.
    left_bounds: vec2<f32>,
    right_bounds: vec2<f32>,
    // The u range and v range of the left image, for the coarse picture.
    left_v: vec2<f32>,
    // How far apart a sample's taps are, as a share of a column and a row: 0
    // samples a point.
    prefilter: f32,
    // How much more the middle of the window counts than its edge.
    centre: f32,
};
var<immediate> win: Window;

// The layout is fp_core::autofocus's.
const CANDIDATES: u32 = 256u;
const ZERO_CANDIDATE: u32 = 16u;
const COLUMNS: u32 = 24u;
const ROWS: u32 = 16u;
const VERTICAL: u32 = 23u;
const SIGNATURE: u32 = 32u;
// Half-column positions across the window and every shift.
const STRIP: u32 = 2u * COLUMNS + CANDIDATES - 2u;
// Rows held at a time: two strips of eight rows is what the workgroup memory
// devices give allows.
const CHUNK: u32 = 8u;
// The right window's rows for the vertical search, beyond the window's.
const REACH: u32 = (VERTICAL - 1u) / 2u;
const TALL: u32 = ROWS + 2u * REACH;

struct Result {
    score: array<f32, CANDIDATES>,
    gradient: array<f32, CANDIDATES>,
    variance: f32,
    gradient_variance: f32,
    best: f32,
    padding: f32,
    vertical: array<f32, VERTICAL>,
    signature: array<f32, SIGNATURE>,
};

var<workgroup> strip_l: array<f32, STRIP * CHUNK>;
var<workgroup> strip_r: array<f32, STRIP * CHUNK>;
var<workgroup> best_shift: u32;

fn from_middle(i: u32, count: u32) -> f32 {
    return f32(i) - 0.5 * f32(count - 1u);
}

// Luma around `uv`: the mean of four taps a share of a column and a row
// apart, so that detail finer than the samples' spacing averages out rather
// than aliasing.
fn luma_near(uv: vec2<f32>, lo: f32, hi: f32) -> f32 {
    if (win.prefilter <= 0.0) {
        return luma_at(vec2<f32>(clamp(uv.x, lo, hi), uv.y));
    }
    let o = 0.25 * win.prefilter * win.step;
    var sum = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let at = uv + vec2<f32>(select(-o.x, o.x, (i & 1u) == 1u), select(-o.y, o.y, i >= 2u));
        sum += luma_at(vec2<f32>(clamp(at.x, lo, hi), at.y));
    }
    return 0.25 * sum;
}

struct Sums {
    a: f32,
    b: f32,
    aa: f32,
    bb: f32,
    ab: f32,
    n: f32,
};

fn added(s: Sums, a: f32, b: f32, w: f32) -> Sums {
    return Sums(s.a + w * a, s.b + w * b, s.aa + w * a * a, s.bb + w * b * b, s.ab + w * a * b, s.n + w);
}

fn weight(x: u32, y: u32) -> f32 {
    if (win.centre <= 0.0) { return 1.0; }
    let dx = from_middle(x, COLUMNS) / (0.5 * f32(COLUMNS));
    let dy = from_middle(y, ROWS) / (0.5 * f32(ROWS));
    return exp(-win.centre * (dx * dx + dy * dy));
}

fn variance_a(s: Sums) -> f32 {
    return max(s.aa / max(s.n, 1e-6) - (s.a / max(s.n, 1e-6)) * (s.a / max(s.n, 1e-6)), 0.0);
}

fn correlation(s: Sums) -> f32 {
    let n = max(s.n, 1e-6);
    let variance_b = s.bb / n - (s.b / n) * (s.b / n);
    let covariance = s.ab / n - (s.a / n) * (s.b / n);
    return covariance / sqrt(max(variance_a(s) * variance_b, 1e-12));
}

fn row_v(centre: f32, y: f32) -> f32 {
    return centre + y * win.step.y;
}

@compute @workgroup_size(CANDIDATES)
fn main(@builtin(local_invocation_index) c: u32) {
    // Candidate c moves the left window right and the right window left by
    // half a column each: column x of the left window at half-column 2x + c,
    // of the right at 2x - c + CANDIDATES - 1.
    let half = 0.5 * win.step.x;
    let left_start = win.left.x - (f32(COLUMNS - 1u) + f32(ZERO_CANDIDATE)) * half;
    let right_start = win.right.x - (f32(COLUMNS - 1u) + f32(CANDIDATES - 1u - ZERO_CANDIDATE)) * half;
    var sums = Sums(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    var differences = sums;
    for (var chunk = 0u; chunk < ROWS; chunk += CHUNK) {
        for (var i = c; i < STRIP * CHUNK; i += CANDIDATES) {
            let k = f32(i % STRIP);
            let y = from_middle(chunk + i / STRIP, ROWS);
            strip_l[i] = luma_near(vec2<f32>(left_start + k * half, row_v(win.left.y, y)),
                                   win.left_bounds.x, win.left_bounds.y);
            strip_r[i] = luma_near(vec2<f32>(right_start + k * half, row_v(win.right.y, y)),
                                   win.right_bounds.x, win.right_bounds.y);
        }
        workgroupBarrier();
        for (var y = 0u; y < CHUNK; y++) {
            for (var x = 0u; x < COLUMNS; x++) {
                let at_l = y * STRIP + 2u * x + c;
                let at_r = y * STRIP + 2u * x + CANDIDATES - 1u - c;
                let l = strip_l[at_l];
                let r = strip_r[at_r];
                let w = weight(x, chunk + y);
                sums = added(sums, l, r, w);
                if (x > 0u) {
                    differences = added(differences, l - strip_l[at_l - 2u], r - strip_r[at_r - 2u], w);
                }
            }
        }
        workgroupBarrier();
    }
    let gradient = correlation(differences);
    result.score[c] = correlation(sums);
    result.gradient[c] = gradient;
    if (c == ZERO_CANDIDATE) {
        result.variance = variance_a(sums);
        result.gradient_variance = variance_a(differences);
    }

    // The best shift by the differences, away from the search's ends.
    strip_l[c] = select(-2.0, gradient, c > 0u && c < CANDIDATES - 1u);
    workgroupBarrier();
    if (c == 0u) {
        var best = 1u;
        for (var i = 2u; i < CANDIDATES - 1u; i++) {
            if (strip_l[i] > strip_l[best]) { best = i; }
        }
        best_shift = best;
        result.best = f32(best);
    }
    workgroupBarrier();

    // At that shift, the left window and the right window over a taller
    // range of rows; then each vertical candidate compares the differences
    // down the window, which a vertical shift changes most.
    let shift = (f32(best_shift) - f32(ZERO_CANDIDATE)) * half;
    for (var i = c; i < COLUMNS * (ROWS + TALL); i += CANDIDATES) {
        let x = from_middle(i % COLUMNS, COLUMNS);
        if (i < COLUMNS * ROWS) {
            let y = from_middle(i / COLUMNS, ROWS);
            strip_l[i] = luma_near(vec2<f32>(win.left.x + x * win.step.x + shift, row_v(win.left.y, y)),
                                   win.left_bounds.x, win.left_bounds.y);
        } else {
            let y = from_middle(i / COLUMNS - ROWS, TALL);
            strip_r[i - COLUMNS * ROWS] = luma_near(
                vec2<f32>(win.right.x + x * win.step.x - shift, row_v(win.right.y, y)),
                win.right_bounds.x, win.right_bounds.y);
        }
    }
    workgroupBarrier();
    if (c < VERTICAL) {
        var down = Sums(0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
        for (var y = 1u; y < ROWS; y++) {
            for (var x = 0u; x < COLUMNS; x++) {
                let l = strip_l[y * COLUMNS + x] - strip_l[(y - 1u) * COLUMNS + x];
                let at = (y + c) * COLUMNS + x;
                let r = strip_r[at] - strip_r[at - COLUMNS];
                down = added(down, l, r, weight(x, y));
            }
        }
        result.vertical[c] = correlation(down);
    } else if (c >= 64u && c < 64u + SIGNATURE) {
        // The coarse picture: 8 across, 4 down, each the mean of four taps
        // spread over its cell.
        let i = c - 64u;
        let cell = vec2<f32>(f32(i % 8u), f32(i / 8u));
        let size = vec2<f32>((win.left_bounds.y - win.left_bounds.x) / 8.0, (win.left_v.y - win.left_v.x) / 4.0);
        let corner = vec2<f32>(win.left_bounds.x, win.left_v.x) + cell * size;
        var sum = 0.0;
        for (var t = 0u; t < 4u; t++) {
            let at = corner + size * vec2<f32>(0.25 + 0.5 * f32(t & 1u), 0.25 + 0.5 * f32(t / 2u));
            sum += luma_at(at);
        }
        result.signature[i] = 0.25 * sum;
    }
}
