//! How autofocus measures depth where the eyes look, for
//! [`crate::autofocus`] to draw the picture at.
//!
//! Each eye's picture around the ray is drawn into a small patch along
//! epipolar lines: each row lies in a plane through both cameras, so a
//! point's two images differ only along the row, at any elevation. Columns
//! are the angle `α` within the plane, from the plane's middle; rows the
//! plane's turn `β` about the cameras' axis. Candidates are depths, as half
//! the cameras' convergence on the subject, from just beyond infinity to
//! [`crate::autofocus::MIN_DISTANCE`]. A coarse pass over all of them is
//! followed by a fine one around the best at full resolution, each matched
//! by zero-mean normalized cross-correlation on the CPU, off the render
//! thread. The rig's vertical misalignment is searched for ±2° until a few
//! very clear matches fit a pitch and a roll, then ±0.5° about what they
//! predict.

use crate::autofocus::{BASELINE, MIN_DISTANCE, angle};

/// The coarse picture of the left image each patch carries, 8 by 4, to
/// notice cuts. `shaders/patch.wgsl` has the same.
pub const SIGNATURE: usize = 32;

/// How wide and tall the matched window is, degrees.
pub const WINDOW_DEGREES: f32 = 2.9;
/// The finest pitch, degrees, and the coarsest the coarse pass uses.
pub const FINE_PITCH: f32 = 0.07;
const COARSEST_PITCH: f32 = 0.32;
/// Below this correlation a match is more likely chance.
pub const MIN_CORRELATION: f32 = 0.55;
/// A separate second peak this close to the best makes it ambiguous, and
/// how far (degrees of shift) it must be to count as separate.
pub const RIVAL_MARGIN: f32 = 0.08;
const RIVAL_APART: f32 = 0.6;
/// Below this standard deviation (luma 0..1) the patch is blank.
pub const MIN_SPREAD: f32 = 0.012;
/// Dim or low-contrast detail below that, down to this, is used when it
/// matches this well: its correlation, normalized, is as good a sign as
/// any; only quantization and compression noise, uncorrelated between the
/// eyes, has to be kept from passing for it.
pub const LOW_SPREAD: f32 = 0.006;
const LOW_SPREAD_CORRELATION: f32 = 0.75;
/// How far beyond infinity the search starts, degrees of half-convergence:
/// cameras turned a little towards or away from each other, or a calibration
/// that leaves them so, put far things beyond infinity, and real rigs do.
const BEYOND_INFINITY: f32 = 1.0;
/// How much better a vertical shift at the search's end must match than the
/// one expected for it to mean the offset is beyond it: on detail that
/// only changes across (trunks, poles, edges up and down) every vertical
/// shift matches about as well, and the best is anywhere.
const VERTICAL_UNRESOLVED: f32 = 0.03;
/// How far the best depth must stand above the mean of them all; and the
/// best vertical shift above the mean of those, to learn the offset from.
const MIN_SHARPNESS: f32 = 0.1;
const VERTICAL_SHARPNESS: f32 = 0.05;
/// Vertical search either way while the offset is unknown, and once known.
pub const VERTICAL_SEARCH: f32 = 2.0;
pub const VERTICAL_KNOWN: f32 = 0.5;
/// No nearer the picture's left or right edge, nor the cameras' axis, than
/// this, degrees.
const EDGE_DEGREES: f32 = 3.0;
const AXIS_DEGREES: f32 = 15.0;
/// What a match must score to teach the vertical offset, how many must, and
/// how much each older one counts for less with each new one.
const VERTICAL_CLEAR: f32 = 0.85;
const VERTICAL_NEEDED: usize = 3;
const VERTICAL_DECAY: f32 = 0.9;
const VERTICAL_SAMPLES: usize = 16;
/// Readings kept to agree, how many must, and how close, degrees of vergence.
const RECENT: usize = 5;
const AGREEING: usize = 3;
pub const AGREEMENT_DEGREES: f32 = 0.3;
/// The probe on the target and the one just before it agree on a depth if
/// this close, degrees of vergence: two places, two readings, enough.
pub const NEIGHBOUR_AGREEMENT: f32 = 0.5;
/// How often a fine pass whose best is at its range's edge moves its range
/// there and looks again.
const RECENTRES: u32 = 2;
/// Probes about the target, degrees right and up: on it every other time,
/// and 1.2° off on the diagonals between.
pub const PROBES: [[f32; 2]; 8] = [
    [0.0, 0.0],
    [1.2, 1.2],
    [0.0, 0.0],
    [-1.2, -1.2],
    [0.0, 0.0],
    [-1.2, 1.2],
    [0.0, 0.0],
    [1.2, -1.2],
];
/// How much the coarse picture's luma must change on average for a cut.
const CUT_CHANGE: f32 = 0.08;

/// When readings agree on a depth.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Agreement {
    /// Three of the last five within 0.3° of vergence.
    Strict,
    /// That, or the probe on the target and the one just before or after it
    /// within 0.5°, at their mean.
    #[default]
    Fast,
}

impl Agreement {
    pub const ALL: [Agreement; 2] = [Agreement::Strict, Agreement::Fast];

    pub fn name(self) -> &'static str {
        match self {
            Agreement::Strict => "Strict",
            Agreement::Fast => "Fast",
        }
    }
}

/// Its own tuning, beside autofocus's.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
#[serde(default)]
pub struct EpiTuning {
    /// Depths tried in the coarse pass: its pitch follows.
    pub candidates: u32,
    /// The window matched, degrees: where the head points, and where the
    /// eyes do, whose aim wanders more.
    pub window: f32,
    pub gaze_window: f32,
    /// When readings agree, and whether a fine pass follows the coarse one
    /// (without, a reading is the coarse pass's, refined between its
    /// samples: half the patches to a depth).
    pub agreement: Agreement,
    pub fine_pass: bool,
}

impl Default for EpiTuning {
    fn default() -> Self {
        EpiTuning {
            candidates: 256,
            window: WINDOW_DEGREES,
            gaze_window: 4.0,
            agreement: Agreement::Fast,
            fine_pass: false,
        }
    }
}

/// The patch to draw: two strips, each `columns` by its rows, starting at an
/// `α` and a `β` from the centre, degrees, `pitch` apart.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Layout {
    pub pitch: f32,
    pub left_columns: u32,
    pub right_columns: u32,
    pub left_rows: u32,
    pub right_rows: u32,
    pub left_alpha: f32,
    pub right_alpha: f32,
    pub left_beta: f32,
    pub right_beta: f32,
    /// Samples across the window.
    pub window: u32,
    /// The shifts tried, in samples: shift `m` puts the left window
    /// `ceil(m/2)` right of the centre and the right one `floor(m/2)` left.
    pub first_shift: i32,
    pub shifts: u32,
    /// The vertical shifts tried, in samples either way, and the offset
    /// the right strip already has, degrees.
    pub vertical_reach: u32,
    pub vertical_base: f32,
    /// The window's width, degrees, and whether this is a coarse pass over
    /// every depth (a fine one only looks about the coarse one's).
    pub window_degrees: f32,
    pub coarse: bool,
}

/// `m / 2` rounded up and down.
fn halves(m: i32) -> (i32, i32) {
    (m.div_euclid(2) + m.rem_euclid(2), m.div_euclid(2))
}

impl Layout {
    /// Shifts `first..first + count` of `pitch`, the right strip `vertical`
    /// degrees below the left, searched `reach` degrees either way.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        pitch: f32,
        first: i32,
        count: u32,
        vertical: f32,
        reach: f32,
        window_degrees: f32,
        coarse: bool,
    ) -> Layout {
        let window = ((window_degrees / pitch).round() as u32).max(3) | 1;
        let half_w = (window as i32 - 1) / 2;
        let last = first + count as i32 - 1;
        let (l0, l1) = (halves(first).0, halves(last).0);
        let (r0, r1) = (-halves(last).1, -halves(first).1);
        let vertical_reach = (reach / pitch).ceil() as u32;
        Layout {
            pitch,
            left_columns: window + (l1 - l0) as u32,
            right_columns: window + (r1 - r0) as u32,
            left_rows: window,
            right_rows: window + 2 * vertical_reach,
            left_alpha: (l0 - half_w) as f32 * pitch,
            right_alpha: (r0 - half_w) as f32 * pitch,
            left_beta: -(half_w as f32) * pitch - 0.5 * vertical,
            right_beta: -((half_w + vertical_reach as i32) as f32) * pitch + 0.5 * vertical,
            window,
            first_shift: first,
            shifts: count,
            vertical_reach,
            vertical_base: vertical,
            window_degrees,
            coarse,
        }
    }

    /// The coarse pass over every depth, from beyond infinity to the nearest,
    /// in about `candidates` steps.
    pub fn coarse(candidates: u32, vertical: f32, reach: f32) -> Layout {
        Layout::coarse_in(candidates, vertical, reach, WINDOW_DEGREES)
    }

    /// The same with a window `window` degrees wide.
    pub fn coarse_in(candidates: u32, vertical: f32, reach: f32, window: f32) -> Layout {
        let nearest = angle(MIN_DISTANCE, BASELINE) / 2.0;
        let range = 2.0 * (nearest + BEYOND_INFINITY);
        let pitch = (range / candidates.max(8) as f32).clamp(FINE_PITCH, COARSEST_PITCH);
        let first = (-2.0 * BEYOND_INFINITY / pitch).floor() as i32;
        let count = ((2.0 * nearest) / pitch - first as f32).ceil() as u32 + 1;
        Layout::new(pitch, first, count, vertical, reach, window, true)
    }

    /// The fine pass about `shift` degrees with the right strip `vertical`
    /// below, from a coarse pass of `coarse` pitch.
    pub fn fine(shift: f32, coarse: f32, vertical: f32, reach: f32) -> Layout {
        Layout::fine_in(shift, coarse, vertical, reach, WINDOW_DEGREES)
    }

    /// The same with a window `window` degrees wide.
    pub fn fine_in(shift: f32, coarse: f32, vertical: f32, reach: f32, window: f32) -> Layout {
        let from = ((shift - 1.5 * coarse) / FINE_PITCH).floor() as i32;
        let to = ((shift + 1.5 * coarse) / FINE_PITCH).ceil() as i32;
        Layout::new(
            FINE_PITCH,
            from,
            (to - from + 1) as u32,
            vertical,
            reach,
            window,
            false,
        )
    }

    /// The same with the right strip `vertical` degrees below the left.
    pub fn at_vertical(&self, vertical: f32) -> Layout {
        let reach = self.vertical_reach as f32 * self.pitch;
        Layout::new(
            self.pitch,
            self.first_shift,
            self.shifts,
            vertical,
            reach - 1e-4,
            self.window_degrees,
            self.coarse,
        )
    }

    pub fn floats(&self) -> usize {
        (self.left_columns * self.left_rows + self.right_columns * self.right_rows) as usize
    }
}

/// What a search found: shift (degrees of `α` between the eyes' images,
/// twice the half-convergence) and how far the right image is below the
/// left (degrees), the correlation, and what can't be trusted about it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Found {
    pub shift: f32,
    pub vertical: f32,
    pub correlation: f32,
    pub spread: f32,
    pub rival: bool,
    /// The best separate second peak: its shift, degrees, and correlation.
    pub second: Option<(f32, f32)>,
    /// How far the best depth stands above the mean of them all.
    pub sharpness: f32,
    /// At an end of the search, if it is.
    pub end: Option<End>,
    /// The vertical shift was clearly found, rather than taken as expected
    /// for want of detail that tells.
    pub vertical_clear: bool,
    /// A fine pass's best at its range's edge: the shift to look about again.
    pub recentre: Option<f32>,
}

/// Which end of the search a match is at.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum End {
    /// Beyond infinity, as far as the search goes.
    Far,
    /// As near as it goes.
    Near,
    /// As far up or down as it goes, and clearly better there.
    Vertical,
}

/// Why a measurement isn't used.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejected {
    Blank,
    /// No depth stands out: detail only up and down, or none to speak of.
    Flat,
    LowCorrelation,
    Rival,
    SearchEnd(End),
    Edge,
    Axis,
}

impl Rejected {
    /// Its place in [`EpiStats::rejected`] and the log's name for it.
    pub const NAMES: [&str; 9] = [
        "blank",
        "flat",
        "low correlation",
        "rival",
        "far end",
        "near end",
        "vertical end",
        "edge",
        "axis",
    ];

    pub fn index(self) -> usize {
        match self {
            Rejected::Blank => 0,
            Rejected::Flat => 1,
            Rejected::LowCorrelation => 2,
            Rejected::Rival => 3,
            Rejected::SearchEnd(End::Far) => 4,
            Rejected::SearchEnd(End::Near) => 5,
            Rejected::SearchEnd(End::Vertical) => 6,
            Rejected::Edge => 7,
            Rejected::Axis => 8,
        }
    }
}

impl Found {
    pub fn judge(&self) -> Result<f32, Rejected> {
        // What a curve of chance is, before where on it its best fell.
        if self.spread.is_nan()
            || self.spread < LOW_SPREAD
            || (self.spread < MIN_SPREAD
                && (self.correlation.is_nan() || self.correlation < LOW_SPREAD_CORRELATION))
        {
            return Err(Rejected::Blank);
        }
        if self.correlation.is_nan() || self.correlation < MIN_CORRELATION {
            return Err(Rejected::LowCorrelation);
        }
        if self.sharpness.is_nan() || self.sharpness < MIN_SHARPNESS {
            return Err(Rejected::Flat);
        }
        if let Some(end) = self.end {
            return Err(Rejected::SearchEnd(end));
        }
        if self.rival {
            return Err(Rejected::Rival);
        }
        Ok(self.shift)
    }
}

/// Mean and standard deviation of a window, and its values less the mean.
fn zero_mean(values: impl Iterator<Item = f32>, n: usize, out: &mut Vec<f32>) -> f32 {
    out.clear();
    out.extend(values);
    let mean = out.iter().sum::<f32>() / n as f32;
    let mut sq = 0.0;
    for v in out.iter_mut() {
        *v -= mean;
        sq += *v * *v;
    }
    (sq / n as f32).sqrt()
}

fn parabola(a: f32, b: f32, c: f32) -> f32 {
    let curvature = a - 2.0 * b + c;
    if curvature < 0.0 {
        (0.5 * (a - c) / curvature).clamp(-0.5, 0.5)
    } else {
        0.0
    }
}

/// Matches the strips of a drawn patch: every shift and vertical shift, by
/// zero-mean normalized cross-correlation. Samples off the picture are
/// negative, and a window with any is no match.
pub fn search(l: &Layout, left: &[f32], right: &[f32]) -> Found {
    let w = l.window as usize;
    let n = w * w;
    let reach = l.vertical_reach as i32;
    let rows = 2 * reach + 1;
    let l0 = halves(l.first_shift).0;
    let r0 = -halves(l.first_shift + l.shifts as i32 - 1).1;
    let lc = l.left_columns as usize;
    let rc = l.right_columns as usize;
    let mut a = Vec::with_capacity(n);
    let mut b = Vec::with_capacity(n);
    // The best over vertical shifts at each shift, and where.
    let mut by_shift = vec![(f32::MIN, 0i32); l.shifts as usize];
    let mut grid = vec![f32::MIN; l.shifts as usize * rows as usize];
    let mut spread_zero = 0.0;
    // Most patches are all on the picture: only look for samples off it in
    // the strips that have any.
    let left_has_off = left.iter().any(|x| *x < 0.0);
    let right_has_off = right.iter().any(|x| *x < 0.0);
    for (k, best) in by_shift.iter_mut().enumerate() {
        let m = l.first_shift + k as i32;
        let (up, down) = halves(m);
        let lx = (up - l0) as usize;
        let rx = (-down - r0) as usize;
        let sa = zero_mean((0..n).map(|i| left[(i / w) * lc + lx + i % w]), n, &mut a);
        let left_off = left_has_off && (0..n).any(|i| left[(i / w) * lc + lx + i % w] < 0.0);
        if k == 0 {
            spread_zero = sa;
        }
        for v in 0..rows {
            let sb = zero_mean(
                (0..n).map(|i| right[(i / w + v as usize) * rc + rx + i % w]),
                n,
                &mut b,
            );
            let right_off = right_has_off
                && (0..n).any(|i| right[(i / w + v as usize) * rc + rx + i % w] < 0.0);
            let dot: f32 = a.iter().zip(&b).map(|(x, y)| x * y).sum();
            let c = if left_off || right_off {
                -1.0
            } else {
                dot / (n as f32 * (sa * sb).max(1e-9))
            };
            grid[k * rows as usize + v as usize] = c;
            if c > best.0 {
                *best = (c, v);
            }
        }
    }
    let (k, &(best, best_v)) = by_shift
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.0.total_cmp(&y.1.0))
        .unwrap_or((0, &(f32::MIN, 0)));
    let g = |kk: usize, vv: i32| grid[kk * rows as usize + vv as usize];
    // At a vertical end: beyond the search if clearly better there, else
    // nothing tells up from down here, and the offset expected stands.
    let (mut v, mut corr, mut end) = (best_v, best, None);
    if reach > 0 && (v == 0 || v == rows - 1) {
        if best - g(k, reach) < VERTICAL_UNRESOLVED {
            (v, corr) = (reach, g(k, reach));
        } else {
            end = Some(End::Vertical);
        }
    }
    let vertical_mean = (0..rows).map(|vv| g(k, vv)).sum::<f32>() / rows as f32;
    let vertical_clear =
        reach > 0 && end.is_none() && v == best_v && best - vertical_mean >= VERTICAL_SHARPNESS;
    // Only the coarse pass's ends are the search's; a fine pass's best at
    // the edge of its narrow range says where to look again.
    let at_edge = k == 0 || k + 1 == by_shift.len();
    let recentre = (at_edge && !l.coarse).then(|| (l.first_shift + k as i32) as f32 * l.pitch);
    if l.coarse && k == 0 {
        end = Some(End::Far);
    } else if l.coarse && k + 1 == by_shift.len() {
        end = Some(End::Near);
    }
    // Only a search over the depths says whether one stands out; the fine
    // pass looks only about the coarse one's.
    let sharpness = if l.coarse {
        best - by_shift.iter().map(|b| b.0).sum::<f32>() / by_shift.len() as f32
    } else {
        f32::INFINITY
    };
    let along = if k > 0 && k + 1 < by_shift.len() {
        parabola(by_shift[k - 1].0, corr, by_shift[k + 1].0)
    } else {
        0.0
    };
    let down = if v > 0 && v < rows - 1 {
        parabola(g(k, v - 1), corr, g(k, v + 1))
    } else {
        0.0
    };
    let apart = (RIVAL_APART / l.pitch).ceil() as usize;
    // A second peak, separate from the best: far enough, and with a valley
    // between them, not the best one's shoulder.
    let separate = |i: usize| {
        let peak = by_shift[i].0;
        let valley = by_shift[i.min(k)..=i.max(k)]
            .iter()
            .map(|b| b.0)
            .fold(f32::MAX, f32::min);
        i.abs_diff(k) >= apart
            && peak >= by_shift[i - 1].0
            && peak >= by_shift[i + 1].0
            && valley < peak - RIVAL_MARGIN
    };
    let second = (1..by_shift.len().saturating_sub(1))
        .filter(|&i| separate(i))
        .max_by(|a, b| by_shift[*a].0.total_cmp(&by_shift[*b].0));
    let rival = second.is_some_and(|i| by_shift[i].0 >= corr - RIVAL_MARGIN);
    let second = second.map(|i| ((l.first_shift + i as i32) as f32 * l.pitch, by_shift[i].0));
    // The left window at the best shift says how much detail there is.
    let (up, _) = halves(l.first_shift + k as i32);
    let spread = zero_mean(
        (0..n).map(|i| left[(i / w) * lc + (up - l0) as usize + i % w]),
        n,
        &mut a,
    );
    let _ = spread_zero;
    Found {
        shift: (l.first_shift as f32 + k as f32 + along) * l.pitch,
        vertical: l.vertical_base + (v - reach) as f32 * l.pitch + down * l.pitch,
        correlation: corr,
        spread,
        rival,
        second,
        sharpness,
        end,
        vertical_clear,
        recentre,
    }
}

/// The angle straight ahead at the same distance, degrees, from a shift
/// (twice the half-convergence) at `alpha` from the middle of the plane:
/// the cameras' separation looks shorter by its cosine there.
pub fn frontal_angle(shift: f32, alpha: f32) -> f32 {
    let half = (0.5 * shift.to_radians()).tan() / alpha.to_radians().cos().max(0.17);
    2.0 * half.atan().to_degrees()
}

/// The rig's vertical misalignment, fitted as a pitch and a roll from clear
/// matches across the picture: the right image `pitch + roll · tan α · cos β`
/// degrees below the left at `α`, `β`.
///
/// A camera turned about the cameras' axis turns every plane through them
/// alike: a constant shift in `β`. Turned about its own forward axis by a
/// small angle, a direction's `x` and `y` trade by that angle, and
/// `β = atan2(-y, -z)` moves by `-angle · tan α · cos β`; `α` moves by
/// `angle · sin β`, which the depth would take for disparity, so
/// [`VerticalFit::shift_bias`] says how much to take off.
#[derive(Clone, Copy, Debug, Default)]
pub struct VerticalFit {
    samples: [(f32, f32, f32); VERTICAL_SAMPLES],
    count: usize,
}

impl VerticalFit {
    /// How much a roll moves `β` at `α`, `β`, per degree of it.
    fn lever(alpha: f32, beta: f32) -> f32 {
        alpha.to_radians().tan().clamp(-4.0, 4.0) * beta.to_radians().cos()
    }

    pub fn add(&mut self, alpha: f32, beta: f32, vertical: f32) {
        for s in &mut self.samples[..self.count.min(VERTICAL_SAMPLES)] {
            s.2 *= VERTICAL_DECAY;
        }
        self.samples[self.count % VERTICAL_SAMPLES] = (Self::lever(alpha, beta), vertical, 1.0);
        self.count += 1;
    }

    /// Pitch and roll, once enough clear matches are in.
    pub fn fit(&self) -> Option<(f32, f32)> {
        let s = &self.samples[..self.count.min(VERTICAL_SAMPLES)];
        if s.len() < VERTICAL_NEEDED {
            return None;
        }
        let (w, x, y) = s.iter().fold((0.0, 0.0, 0.0), |(w, x, y), &(sx, sy, sw)| {
            (w + sw, x + sw * sx, y + sw * sy)
        });
        let (mx, my) = (x / w, y / w);
        let (sxx, sxy) = s.iter().fold((0.0, 0.0), |(a, b), &(sx, sy, sw)| {
            (
                a + sw * (sx - mx) * (sx - mx),
                b + sw * (sx - mx) * (sy - my),
            )
        });
        // Too little spread across the picture to tell a roll: the pitch only.
        let roll = if sxx / w > 0.01 { sxy / sxx } else { 0.0 };
        Some((my - roll * mx, roll))
    }

    pub fn predict(&self, alpha: f32, beta: f32) -> Option<f32> {
        self.fit().map(|fit| vertical_of(fit, alpha, beta))
    }

    /// How much of a shift measured at `β` the fitted roll put there,
    /// degrees.
    pub fn shift_bias(&self, beta: f32) -> f32 {
        self.fit()
            .map_or(0.0, |(_, roll)| roll * beta.to_radians().sin())
    }
}

/// The vertical offset a fitted pitch and roll give at `α`, `β`, degrees.
pub fn vertical_of((pitch, roll): (f32, f32), alpha: f32, beta: f32) -> f32 {
    pitch + roll * VerticalFit::lever(alpha, beta)
}

/// What to draw next.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Aim {
    /// Along the ray, turned by this much right and up, degrees.
    Ray([f32; 2]),
    /// At this `α`, `β` in the picture, degrees: a fine pass where the
    /// coarse one was.
    At([f32; 2]),
}

/// How measuring went since it was last asked, for the log.
#[derive(Clone, Copy, Debug, Default)]
pub struct EpiStats {
    pub coarse: u32,
    pub fine: u32,
    pub trusted: u32,
    /// Rejected by why, as [`Rejected::NAMES`].
    pub rejected: [u32; 9],
    /// Trusted fine readings beyond infinity, and their mean shift, degrees:
    /// what would say the cameras are turned towards each other.
    pub beyond_infinity: u32,
    pub beyond_sum: f32,
    /// Coarse readings at the far end, taken as infinity.
    pub far_as_infinity: u32,
    /// Fine passes looked again about the edge of their range.
    pub recentred: u32,
    /// Locks by three of five agreeing, and by the target and its neighbour.
    pub locks_by_neighbour: u32,
    pub locks: u32,
    pub cuts: u32,
    pub clears: u32,
    /// Matching time on the worker, microseconds.
    pub cpu_sum: f32,
    pub cpu_max: f32,
    pub cpu_count: u32,
    pub last_distance: Option<f32>,
    pub correlation_sum: f32,
}

impl std::fmt::Display for EpiStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let why: Vec<String> = Rejected::NAMES
            .iter()
            .zip(self.rejected)
            .filter(|(_, n)| *n > 0)
            .map(|(w, n)| format!("{w} {n}"))
            .collect();
        write!(
            f,
            "{} coarse, {} fine, {} trusted (mean correlation {:.2}); rejected {}; {} locks, \
             {} by neighbours, {} cuts, {} cleared; {} beyond infinity (mean {:+.2}°), {} far end \
             as infinity, {} fine recentred; CPU {:.0} µs a match, at most {:.0}",
            self.coarse,
            self.fine,
            self.trusted,
            self.correlation_sum / self.trusted.max(1) as f32,
            if why.is_empty() {
                "none".into()
            } else {
                why.join(", ")
            },
            self.locks,
            self.locks_by_neighbour,
            self.cuts,
            self.clears,
            self.beyond_infinity,
            self.beyond_sum / self.beyond_infinity.max(1) as f32,
            self.far_as_infinity,
            self.recentred,
            self.cpu_sum / self.cpu_count.max(1) as f32,
            self.cpu_max,
        )
    }
}

/// The measuring's state for one open video.
#[derive(Clone, Debug, Default)]
pub struct Epipolar {
    pub tuning: EpiTuning,
    probe: usize,
    /// A patch is being drawn or matched.
    in_flight: bool,
    /// The coarse pass's answer, for the fine pass to follow: where, shift,
    /// pitch, vertical.
    coarse: Option<([f32; 2], f32, f32, f32)>,
    /// Fine passes looked again about this coarse reading's.
    recentres: u32,
    /// The patch in flight is of the probe on the target, and the window
    /// it's matched in.
    centre: bool,
    window: f32,
    vertical: VerticalFit,
    /// Recent readings: frontal angle, and whether on the target.
    recent: Vec<(f32, bool)>,
    signature: Option<[f32; SIGNATURE]>,
    stats: EpiStats,
}

impl Epipolar {
    /// What to draw next and how, unless a patch is still in flight.
    /// What to draw next and how, unless a patch is still in flight; `gaze`
    /// when the eyes rather than the head aim it.
    pub fn next_patch(&mut self, gaze: bool) -> Option<(Aim, Layout)> {
        if self.in_flight {
            return None;
        }
        self.in_flight = true;
        if let Some((at, shift, pitch, vertical)) = self.coarse.take() {
            let reach = if self.vertical.fit().is_some() {
                VERTICAL_KNOWN
            } else {
                pitch
            };
            return Some((
                Aim::At(at),
                Layout::fine_in(shift, pitch, vertical, reach, self.window),
            ));
        }
        self.recentres = 0;
        let offset = PROBES[self.probe % PROBES.len()];
        self.centre = offset == [0.0, 0.0];
        self.probe += 1;
        self.window = if gaze {
            self.tuning.gaze_window
        } else {
            self.tuning.window
        };
        let (vertical, reach) = match self.vertical.fit() {
            Some(_) => (f32::NAN, VERTICAL_KNOWN),
            None => (0.0, VERTICAL_SEARCH),
        };
        Some((
            Aim::Ray(offset),
            Layout::coarse_in(self.tuning.candidates, vertical, reach, self.window),
        ))
    }

    /// The vertical offset predicted at `alpha`, degrees: what the renderer
    /// puts in a coarse layout asked for with an unknown (NaN) one.
    pub fn vertical_at(&self, alpha: f32, beta: f32) -> f32 {
        self.vertical.predict(alpha, beta).unwrap_or(0.0)
    }

    pub fn vertical_fit(&self) -> Option<(f32, f32)> {
        self.vertical.fit()
    }

    /// The patch asked for wasn't drawn (nothing to measure there).
    pub fn missed(&mut self) {
        self.in_flight = false;
        self.coarse = None;
    }

    /// Takes in a matched patch drawn at `at` (`α`, `β`, degrees) with
    /// `layout`, how long matching took, and the picture's coarse
    /// signature; returns a frontal angle (degrees) once readings agree.
    pub fn matched(
        &mut self,
        at: [f32; 2],
        lon: f32,
        layout: &Layout,
        found: &Found,
        micros: f32,
        signature: &[f32; SIGNATURE],
    ) -> Option<f32> {
        self.in_flight = false;
        let s = &mut self.stats;
        s.cpu_sum += micros;
        s.cpu_max = s.cpu_max.max(micros);
        s.cpu_count += 1;
        if let Some(last) = self.signature {
            let change: f32 = last.iter().zip(signature).map(|(a, b)| (a - b).abs()).sum();
            if change / SIGNATURE as f32 > CUT_CHANGE {
                self.stats.cuts += 1;
                self.clear();
            }
        }
        self.signature = Some(*signature);
        let fine = !layout.coarse;
        if fine {
            self.stats.fine += 1;
        } else {
            self.stats.coarse += 1;
        }
        let why = if lon.abs() > 90.0 - EDGE_DEGREES {
            Err(Rejected::Edge)
        } else if at[0].abs() > 90.0 - AXIS_DEGREES {
            Err(Rejected::Axis)
        } else {
            found.judge()
        };
        let shift = match why {
            Ok(shift) => shift,
            // A good match as far as the search goes beyond infinity is far
            // away; and on detail that only moves sideways the farthest
            // shift is as good as any, which that's as near as says.
            Err(Rejected::SearchEnd(End::Far)) => {
                self.stats.far_as_infinity += 1;
                return self.agree(0.0);
            }
            Err(r) => {
                self.stats.rejected[r.index()] += 1;
                return None;
            }
        };
        if !fine && self.tuning.fine_pass {
            self.coarse = Some((at, shift, layout.pitch, found.vertical));
            return None;
        }
        if let Some(again) = found.recentre {
            if self.recentres < RECENTRES {
                self.recentres += 1;
                self.stats.recentred += 1;
                let coarse = 3.0 * FINE_PITCH.max(layout.shifts as f32 * layout.pitch / 3.0);
                self.coarse = Some((at, again, coarse / 3.0, found.vertical));
            }
            return None;
        }
        self.stats.trusted += 1;
        self.stats.correlation_sum += found.correlation;
        if shift < 0.0 {
            self.stats.beyond_infinity += 1;
            self.stats.beyond_sum += shift;
        }
        let bias = self.vertical.shift_bias(at[1]);
        if found.correlation >= VERTICAL_CLEAR && found.vertical_clear {
            self.vertical.add(at[0], at[1], found.vertical);
        }
        let frontal = frontal_angle(shift - bias, at[0]);
        self.stats.last_distance = Some(crate::autofocus::distance(frontal, BASELINE));
        self.agree(frontal)
    }

    /// One more reading; a depth to lock on, once readings agree: three of
    /// the last five, or the target and the probe just before or after it.
    fn agree(&mut self, frontal: f32) -> Option<f32> {
        if !frontal.is_finite() {
            return None;
        }
        self.recent.push((frontal, self.centre));
        if self.recent.len() > RECENT {
            self.recent.remove(0);
        }
        let mut sorted: Vec<f32> = self.recent.iter().map(|r| r.0).collect();
        sorted.sort_by(f32::total_cmp);
        let median = sorted[sorted.len() / 2];
        let close = sorted
            .iter()
            .filter(|a| (*a - median).abs() <= AGREEMENT_DEGREES)
            .count();
        if close >= AGREEING {
            self.stats.locks += 1;
            return Some(median);
        }
        let n = self.recent.len();
        if self.tuning.agreement == Agreement::Fast && n >= 2 {
            let (a, b) = (self.recent[n - 1], self.recent[n - 2]);
            if a.1 != b.1 && (a.0 - b.0).abs() <= NEIGHBOUR_AGREEMENT {
                self.stats.locks += 1;
                self.stats.locks_by_neighbour += 1;
                // Their mean: two readings' noise, halved in variance.
                return Some(0.5 * (a.0 + b.0));
            }
        }
        None
    }

    /// The eyes moved on: what was read where they were doesn't count.
    pub fn moved(&mut self) {
        self.recent.clear();
        self.coarse = None;
    }

    /// Forgets pending readings, for a seek or a cut.
    pub fn clear(&mut self) {
        self.recent.clear();
        self.coarse = None;
        self.signature = None;
        self.stats.clears += 1;
    }

    pub fn take_stats(&mut self) -> EpiStats {
        std::mem::take(&mut self.stats)
    }
}

/// Smooths a gaze direction as it rests and follows it at once as it moves:
/// a low-pass whose cut-off rises with the speed, so the eyes' tremor is
/// taken out without lagging a look somewhere else.
#[derive(Clone, Copy, Debug, Default)]
pub struct GazeFilter {
    at: Option<[f32; 3]>,
    speed: f32,
}

/// The cut-off at rest, Hz, how much it rises per degree a second, and the
/// cut-off of the speed it goes by.
const GAZE_MIN_CUTOFF: f32 = 1.0;
const GAZE_SPEED_GAIN: f32 = 0.05;
const GAZE_SPEED_CUTOFF: f32 = 1.0;

fn low_pass(cutoff: f32, dt: f32) -> f32 {
    let tau = 1.0 / (std::f32::consts::TAU * cutoff);
    1.0 / (1.0 + tau / dt.max(1e-4))
}

impl GazeFilter {
    /// Takes this frame's gaze, a unit direction, `dt` seconds on; returns
    /// the smoothed one. `None` (the eyes lost) starts again.
    pub fn take(&mut self, gaze: Option<[f32; 3]>, dt: f32) -> Option<[f32; 3]> {
        let Some(now) = gaze else {
            *self = GazeFilter::default();
            return None;
        };
        let Some(at) = self.at else {
            self.at = Some(now);
            return Some(now);
        };
        let dot: f32 = (0..3).map(|i| at[i] * now[i]).sum();
        let degrees = dot.clamp(-1.0, 1.0).acos().to_degrees();
        let speed = degrees / dt.max(1e-4);
        self.speed += low_pass(GAZE_SPEED_CUTOFF, dt) * (speed - self.speed);
        let k = low_pass(GAZE_MIN_CUTOFF + GAZE_SPEED_GAIN * self.speed, dt);
        let mixed: [f32; 3] = std::array::from_fn(|i| at[i] + k * (now[i] - at[i]));
        let len = mixed.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-6);
        let out = mixed.map(|v| v / len);
        self.at = Some(out);
        Some(out)
    }
}

/// A drawn patch to match, and what to hand back with the match.
#[derive(Clone, Debug)]
pub struct Job {
    pub at: [f32; 2],
    pub lon: f32,
    pub layout: Layout,
    pub left: Vec<f32>,
    pub right: Vec<f32>,
    pub signature: [f32; SIGNATURE],
}

/// A matched patch: the job's, what was found, and how long it took, µs.
#[derive(Clone, Debug)]
pub struct Done {
    pub at: [f32; 2],
    pub lon: f32,
    pub layout: Layout,
    pub signature: [f32; SIGNATURE],
    pub found: Found,
    pub micros: f32,
}

/// Matches patches on a thread of its own, so the render thread never
/// waits; it ends when this is dropped.
pub struct Worker {
    jobs: std::sync::mpsc::Sender<Job>,
    done: std::sync::mpsc::Receiver<Done>,
}

impl Worker {
    pub fn start() -> Option<Worker> {
        let (jobs, take) = std::sync::mpsc::channel::<Job>();
        let (give, done) = std::sync::mpsc::channel();
        std::thread::Builder::new()
            .name("fp-epipolar".into())
            .spawn(move || {
                for job in take {
                    let start = std::time::Instant::now();
                    let found = search(&job.layout, &job.left, &job.right);
                    let micros = start.elapsed().as_secs_f32() * 1e6;
                    let done = Done {
                        at: job.at,
                        lon: job.lon,
                        layout: job.layout,
                        signature: job.signature,
                        found,
                        micros,
                    };
                    if give.send(done).is_err() {
                        break;
                    }
                }
            })
            .ok()?;
        Some(Worker { jobs, done })
    }

    pub fn send(&self, job: Job) {
        let _ = self.jobs.send(job);
    }

    pub fn take(&self) -> Option<Done> {
        self.done.try_recv().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Grey noise with no repeat, over `α`, `β` degrees.
    fn noise(a: f32, b: f32) -> f32 {
        let h = |x: f32, y: f32| {
            let v = (x * 12.9898 + y * 78.233).sin() * 43758.547;
            v - v.floor()
        };
        let smooth = |a: f32, b: f32, s: f32| {
            let (x, y) = (a / s, b / s);
            let (fx, fy) = (x - x.floor(), y - y.floor());
            let (x, y) = (x.floor(), y.floor());
            let top = h(x, y) * (1.0 - fx) + h(x + 1.0, y) * fx;
            let bottom = h(x, y + 1.0) * (1.0 - fx) + h(x + 1.0, y + 1.0) * fx;
            top * (1.0 - fy) + bottom * fy
        };
        0.5 * smooth(a, b, 0.7) + 0.3 * smooth(a + 3.1, b - 1.7, 0.31) + 0.2 * smooth(a, b, 1.9)
    }

    /// Draws `layout`'s strips of a scene whose subject is `shift` degrees
    /// apart between the eyes (half each way) and `down` degrees lower in
    /// the right image.
    fn draw(l: &Layout, shift: f32, down: f32) -> (Vec<f32>, Vec<f32>) {
        draw_of(l, shift, down, noise)
    }

    fn draw_of(
        l: &Layout,
        shift: f32,
        down: f32,
        noise: impl Fn(f32, f32) -> f32,
    ) -> (Vec<f32>, Vec<f32>) {
        let strip = |cols: u32, rows: u32, a0: f32, b0: f32, sa: f32, sb: f32| {
            let mut v = Vec::new();
            for r in 0..rows {
                for c in 0..cols {
                    let a = a0 + c as f32 * l.pitch;
                    let b = b0 + r as f32 * l.pitch;
                    v.push(noise(a + sa, b + sb));
                }
            }
            v
        };
        (
            strip(
                l.left_columns,
                l.left_rows,
                l.left_alpha,
                l.left_beta,
                -shift / 2.0,
                0.0,
            ),
            strip(
                l.right_columns,
                l.right_rows,
                l.right_alpha,
                l.right_beta,
                shift / 2.0,
                -down,
            ),
        )
    }

    #[test]
    fn a_shift_is_found_coarse_then_fine() {
        for (shift, down) in [(0.0, 0.0), (3.7, 0.0), (11.3, 1.3), (30.2, -1.6)] {
            let coarse = Layout::coarse(256, 0.0, VERTICAL_SEARCH);
            let (l, r) = draw(&coarse, shift, down);
            let c = search(&coarse, &l, &r);
            assert!(c.judge().is_ok(), "{shift} {down}: {c:?}");
            assert!((c.shift - shift).abs() < coarse.pitch, "{shift}: {c:?}");
            assert!((c.vertical - down).abs() < coarse.pitch, "{down}: {c:?}");
            let fine = Layout::fine(c.shift, coarse.pitch, c.vertical, coarse.pitch);
            let (l, r) = draw(&fine, shift, down);
            let f = search(&fine, &l, &r);
            assert!(f.judge().is_ok(), "{f:?}");
            assert!((f.shift - shift).abs() < 0.02, "{shift}: {f:?}");
            assert!((f.vertical - down).abs() < 0.02, "{down}: {f:?}");
        }
    }

    #[test]
    fn the_worker_matches_off_the_calling_thread() {
        let w = Worker::start().unwrap();
        let l = Layout::fine(4.0, 0.14, 0.0, 0.2);
        let (left, right) = draw(&l, 4.0, 0.0);
        w.send(Job {
            at: [1.0, 2.0],
            lon: 1.0,
            layout: l,
            left,
            right,
            signature: [0.0; SIGNATURE],
        });
        let done = (0..500)
            .find_map(|_| {
                std::thread::sleep(std::time::Duration::from_millis(5));
                w.take()
            })
            .expect("matched");
        assert_eq!(done.at, [1.0, 2.0]);
        assert!((done.found.shift - 4.0).abs() < 0.02 && done.micros > 0.0);
    }

    /// Coarse, then fine about what it found, as the measuring does.
    fn measure_of(
        shift: f32,
        down: f32,
        expected: f32,
        reach: f32,
        tex: impl Fn(f32, f32) -> f32 + Copy,
    ) -> Found {
        let coarse = Layout::coarse(256, expected, reach);
        let (l, r) = draw_of(&coarse, shift, down, tex);
        let c = search(&coarse, &l, &r);
        if c.judge().is_err() {
            return c;
        }
        let fine = Layout::fine(c.shift, coarse.pitch, c.vertical, reach.min(coarse.pitch));
        let (l, r) = draw_of(&fine, shift, down, tex);
        search(&fine, &l, &r)
    }

    #[test]
    fn detail_only_across_is_measured_with_the_vertical_offset_expected() {
        // Trunks, poles, edges up and down: every vertical shift matches as
        // well as any other. This was a "search end" at a vertical end.
        let across = |a: f32, _b: f32| 0.5 * noise(a, 0.0) + 0.5 * noise(0.37 * a + 11.0, 7.0);
        for (expected, reach) in [(0.0, VERTICAL_SEARCH), (0.4, VERTICAL_KNOWN)] {
            let f = measure_of(7.3, expected, expected, reach, across);
            // Found where it is; a window of such detail can also match
            // elsewhere, and is then rejected as a rival, but never as an end.
            assert!((f.shift - 7.3).abs() < 0.05, "{f:?}");
            assert!(!matches!(f.judge(), Err(Rejected::SearchEnd(_))), "{f:?}");
            assert!(!f.vertical_clear, "nothing to learn the offset from: {f:?}");
            assert!((f.vertical - expected).abs() < 0.05, "{f:?}");
        }
    }

    #[test]
    fn detail_only_up_and_down_is_flat_not_an_end() {
        let up_and_down = |_a: f32, b: f32| noise(0.0, b);
        let f = measure_of(7.3, 0.0, 0.0, VERTICAL_SEARCH, up_and_down);
        assert_eq!(f.judge(), Err(Rejected::Flat), "{f:?}");
    }

    #[test]
    fn cameras_turned_towards_each_other_put_far_things_beyond_infinity() {
        // 1.5° of toe-in: the far end used to be 0.5° beyond infinity.
        let f = measure_of(-1.5, 0.0, 0.0, VERTICAL_SEARCH, noise);
        assert_eq!(f.judge().map(|s| (s + 1.5).abs() < 0.05), Ok(true), "{f:?}");
        let c = Layout::coarse(256, 0.0, VERTICAL_SEARCH);
        let first = c.first_shift as f32 * c.pitch;
        assert!(first <= -2.0, "{first}");
        let f = measure_of(first - 0.1, 0.0, 0.0, VERTICAL_SEARCH, noise);
        assert_eq!(f.judge(), Err(Rejected::SearchEnd(End::Far)), "{f:?}");
    }

    #[test]
    fn each_end_is_told_apart() {
        let c = Layout::coarse(256, 0.0, VERTICAL_SEARCH);
        let last = (c.first_shift + c.shifts as i32 - 1) as f32 * c.pitch;
        let f = measure_of(last + 0.1, 0.0, 0.0, VERTICAL_SEARCH, noise);
        assert_eq!(f.judge(), Err(Rejected::SearchEnd(End::Near)), "{f:?}");
        // Clearly better beyond the vertical search: the offset is past it.
        let coarse = Layout::coarse(256, 0.0, 0.5);
        let beyond = coarse.vertical_reach as f32 * coarse.pitch + 0.08;
        let (l, r) = draw(&coarse, 6.0, beyond);
        let f = search(&coarse, &l, &r);
        assert_eq!(f.judge(), Err(Rejected::SearchEnd(End::Vertical)), "{f:?}");
    }

    #[test]
    fn a_fine_pass_at_its_edge_looks_again_rather_than_rejecting() {
        // The fine pass about 5° finds the subject at its range's edge.
        let fine = Layout::fine(5.0, 0.14, 0.0, 0.14);
        let edge = (fine.first_shift + fine.shifts as i32 - 1) as f32 * fine.pitch + 0.03;
        let (l, r) = draw(&fine, edge, 0.0);
        let f = search(&fine, &l, &r);
        assert_eq!(f.end, None, "{f:?}");
        assert!(f.recentre.is_some_and(|s| (s - edge).abs() < 0.1), "{f:?}");
        // The controller looks again about it, at most twice.
        let mut e = Epipolar::default();
        e.tuning.fine_pass = true;
        e.next_patch(false);
        // A fine pass's sharpness doesn't count: it looks only about one.
        let found = Found {
            correlation: 0.9,
            spread: 0.1,
            shift: 5.0,
            sharpness: f32::INFINITY,
            ..Default::default()
        };
        let coarse = Layout::coarse(256, 0.0, VERTICAL_SEARCH);
        e.matched(
            [0.0; 2],
            0.0,
            &coarse,
            &Found {
                sharpness: 0.5,
                ..found
            },
            1.0,
            &[0.4; SIGNATURE],
        );
        let (aim, mut l) = e.next_patch(false).unwrap();
        assert_eq!(aim, Aim::At([0.0; 2]));
        for again in 0..3 {
            assert!(!l.coarse);
            let edge = Found {
                recentre: Some(5.3),
                ..found
            };
            e.matched([0.0; 2], 0.0, &l, &edge, 1.0, &[0.4; SIGNATURE]);
            let (aim, next) = e.next_patch(false).unwrap();
            // Twice about the edge, then on to the next probe.
            assert_eq!(matches!(aim, Aim::At(_)), again < 2, "{again}");
            l = next;
        }
        assert_eq!(e.take_stats().recentred, 2);
    }

    #[test]
    fn a_curve_of_chance_is_low_correlation_wherever_its_best_falls() {
        let junk = Found {
            correlation: 0.3,
            spread: 0.1,
            sharpness: 0.2,
            end: Some(End::Near),
            ..Default::default()
        };
        assert_eq!(junk.judge(), Err(Rejected::LowCorrelation));
    }

    #[test]
    fn a_good_match_at_the_far_end_is_infinity() {
        let mut e = Epipolar::default();
        let far = Found {
            correlation: 0.9,
            spread: 0.1,
            sharpness: 0.5,
            end: Some(End::Far),
            ..Default::default()
        };
        let coarse = Layout::coarse(256, 0.0, VERTICAL_SEARCH);
        let mut locked = None;
        for _ in 0..3 {
            e.next_patch(false);
            locked = locked.or(e.matched([0.0; 2], 0.0, &coarse, &far, 1.0, &[0.4; SIGNATURE]));
        }
        assert_eq!(locked, Some(0.0), "infinity");
        assert_eq!(e.take_stats().far_as_infinity, 3);
    }

    #[test]
    fn a_window_off_the_picture_is_no_match() {
        let l = Layout::fine(4.0, 0.14, 0.0, 0.14);
        let (mut left, right) = draw(&l, 4.0, 0.0);
        // The left strip's middle columns off the picture: the true shift's
        // window covers them, so it isn't the match.
        let c0 = l.left_columns as usize / 2;
        for row in 0..l.left_rows as usize {
            for c in c0 - 2..c0 + 2 {
                left[row * l.left_columns as usize + c] = -1.0;
            }
        }
        let f = search(&l, &left, &right);
        assert!((f.shift - 4.0).abs() > 0.05 || f.correlation < 0.0, "{f:?}");
    }

    #[test]
    fn a_roll_moves_the_offset_by_tan_alpha_cos_beta_and_biases_the_shift() {
        // Right camera rolled by ρ: a direction's x and y trade by ρ.
        let rho = 1.5f32.to_radians();
        let at = |a: f32, b: f32| {
            let (a, b) = (a.to_radians(), b.to_radians());
            [a.sin(), -a.cos() * b.sin(), -a.cos() * b.cos()]
        };
        let angles = |d: [f32; 3]| {
            (
                d[0].clamp(-1.0, 1.0).asin().to_degrees(),
                (-d[1]).atan2(-d[2]).to_degrees(),
            )
        };
        let mut fit = VerticalFit::default();
        let mut seen = Vec::new();
        for (a, b) in [
            (-50.0f32, 20.0f32),
            (0.0, -30.0),
            (30.0, 10.0),
            (60.0, 30.0),
            (20.0, -40.0),
        ] {
            let d = at(a, b);
            let rolled = [d[0] - rho * d[1], d[1] + rho * d[0], d[2]];
            let (ra, rb) = angles(rolled);
            seen.push((a, b, ra - a, rb - b));
            fit.add(a, b, rb - b);
        }
        let (pitch, roll) = fit.fit().unwrap();
        assert!(pitch.abs() < 0.02, "{pitch}");
        for (a, b, da, db) in seen {
            let predicted = fit.predict(a, b).unwrap();
            assert!(
                (predicted - db).abs() < 0.05,
                "at {a} {b}: {predicted} for {db}"
            );
            // The shift the roll adds, as the left image's α less the right's.
            assert!(
                (fit.shift_bias(b) + da).abs() < 0.03,
                "at {a} {b}: {} for {da}",
                fit.shift_bias(b)
            );
        }
        let _ = roll;
        // At 60° across and 30° up, sin α would have said about half.
        let half = roll * 60f32.to_radians().sin();
        assert!((fit.predict(60.0, 30.0).unwrap() - half).abs() > 0.25 * half.abs());
    }

    #[test]
    fn the_gaze_is_steady_at_rest_and_quick_on_the_move() {
        let mut f = GazeFilter::default();
        let dir = |deg: f32| {
            let r = deg.to_radians();
            [r.sin(), 0.0, -r.cos()]
        };
        let angle = |d: [f32; 3]| d[0].atan2(-d[2]).to_degrees();
        f.take(Some(dir(0.0)), 1.0 / 90.0);
        // Tremor of ±0.5° at rest: mostly taken out.
        let mut worst = 0.0f32;
        for i in 0..90 {
            let jitter = if i % 2 == 0 { 0.5 } else { -0.5 };
            let out = f.take(Some(dir(jitter)), 1.0 / 90.0).unwrap();
            if i > 10 {
                worst = worst.max(angle(out).abs());
            }
        }
        assert!(worst < 0.2, "{worst}");
        // A look 20° away: most of the way there within 0.1 s.
        let mut out = dir(0.0);
        for _ in 0..9 {
            out = f.take(Some(dir(20.0)), 1.0 / 90.0).unwrap();
        }
        assert!(angle(out) > 15.0, "{}", angle(out));
        assert_eq!(f.take(None, 1.0 / 90.0), None);
    }

    /// How each agreement rule does on a stream of readings after the eyes
    /// move to a subject: readings to the first lock, and how often that
    /// lock is wrong (more than 0.5° from the truth). Not only a limit: a
    /// report, `cargo test agreement_rules -- --nocapture`.
    fn agreement_trials(rule: Agreement, sigma: f32, outliers: f32) -> (f32, f32, f32) {
        let mut seed = 12345u32;
        let mut rand = || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let (mut counts, mut wrong, mut none) = (Vec::new(), 0, 0);
        let trials = 2000;
        for _ in 0..trials {
            let mut e = Epipolar::default();
            e.tuning.agreement = rule;
            let truth = 1.0 + 10.0 * rand();
            let mut locked = None;
            for n in 1..=20 {
                e.next_patch(false);
                // Box-Muller noise, and now and then a wrong peak anywhere.
                let g =
                    (-2.0 * rand().max(1e-6).ln()).sqrt() * (std::f32::consts::TAU * rand()).cos();
                let shift = if rand() < outliers {
                    20.0 * rand()
                } else {
                    truth + sigma * g
                };
                let (l, mut f) = fine_found(shift);
                f.shift = shift;
                if let Some(a) = e.matched([0.0; 2], 0.0, &l, &f, 1.0, &[0.4; SIGNATURE]) {
                    locked = Some((n, a));
                    break;
                }
            }
            match locked {
                Some((n, a)) => {
                    counts.push(n as f32);
                    wrong += u32::from((a - truth).abs() > 0.5);
                }
                None => none += 1,
            }
        }
        counts.sort_by(f32::total_cmp);
        let median = counts.get(counts.len() / 2).copied().unwrap_or(f32::NAN);
        (
            median,
            100.0 * wrong as f32 / trials as f32,
            100.0 * none as f32 / trials as f32,
        )
    }

    #[test]
    fn agreement_rules_are_compared() {
        let mut report = String::new();
        for (name, sigma, outliers) in [
            ("clean", 0.05, 0.0),
            ("noisy", 0.25, 0.0),
            ("ambiguous", 0.1, 0.25),
            ("noisy and ambiguous", 0.25, 0.25),
        ] {
            let s = agreement_trials(Agreement::Strict, sigma, outliers);
            let f = agreement_trials(Agreement::Fast, sigma, outliers);
            report += &format!(
                "\n{name} (σ {sigma}°, {:.0}% wrong peaks): strict median {} readings, {:.1}% \
                 wrong, {:.1}% never; fast {} readings, {:.1}% wrong, {:.1}% never",
                outliers * 100.0,
                s.0,
                s.1,
                s.2,
                f.0,
                f.1,
                f.2
            );
            // Fast is never slower.
            assert!(f.0 <= s.0, "{report}");
        }
        println!("{report}");
    }

    #[test]
    fn without_a_fine_pass_each_coarse_reading_counts() {
        let mut e = Epipolar::default();
        assert!(!e.tuning.fine_pass);
        let coarse = Layout::coarse(256, 0.0, VERTICAL_SEARCH);
        let found = Found {
            shift: 5.0,
            correlation: 0.9,
            spread: 0.1,
            sharpness: 0.5,
            ..Default::default()
        };
        let mut locked = None;
        for _ in 0..2 {
            let (aim, _) = e.next_patch(false).unwrap();
            assert!(matches!(aim, Aim::Ray(_)), "no fine pass between");
            locked = e.matched([0.0; 2], 0.0, &coarse, &found, 1.0, &[0.4; SIGNATURE]);
        }
        assert_eq!(
            locked,
            Some(5.0),
            "the target and its neighbour, two patches"
        );
    }

    #[test]
    fn dim_detail_counts_when_it_matches_well() {
        let dim = Found {
            shift: 4.0,
            correlation: 0.9,
            spread: 0.008,
            sharpness: 0.5,
            ..Default::default()
        };
        assert_eq!(dim.judge(), Ok(4.0));
        let noisy = Found {
            correlation: 0.6,
            ..dim
        };
        assert_eq!(noisy.judge(), Err(Rejected::Blank));
        let flat = Found {
            spread: 0.004,
            ..dim
        };
        assert_eq!(flat.judge(), Err(Rejected::Blank));
    }

    #[test]
    fn a_blank_patch_is_rejected() {
        let l = Layout::coarse(256, 0.0, VERTICAL_SEARCH);
        let blank = vec![0.5; l.floats()];
        let (left, right) = blank.split_at((l.left_columns * l.left_rows) as usize);
        assert_eq!(search(&l, left, right).judge(), Err(Rejected::Blank));
    }

    #[test]
    fn a_repeating_pattern_is_ambiguous() {
        let l = Layout::coarse(256, 0.0, 0.0);
        let stripes = |cols: u32, rows: u32, a0: f32| {
            (0..rows)
                .flat_map(|_| {
                    (0..cols).map(move |c| {
                        let a = a0 + c as f32 * l.pitch;
                        0.5 + 0.4 * (a * std::f32::consts::TAU / 2.0).sin()
                    })
                })
                .collect::<Vec<f32>>()
        };
        let left = stripes(l.left_columns, l.left_rows, l.left_alpha);
        let right = stripes(l.right_columns, l.right_rows, l.right_alpha);
        let f = search(&l, &left, &right);
        assert!(
            f.rival || f.end.is_some() || f.sharpness < MIN_SHARPNESS,
            "{f:?}"
        );
        assert!(f.judge().is_err());
    }

    #[test]
    fn the_layouts_reach_from_beyond_infinity_to_a_tenth_of_a_metre() {
        let l = Layout::coarse(256, 0.0, VERTICAL_SEARCH);
        let first = l.first_shift as f32 * l.pitch;
        let last = (l.first_shift + l.shifts as i32 - 1) as f32 * l.pitch;
        assert!(
            first < 0.0 && first >= -2.0 * BEYOND_INFINITY - l.pitch,
            "{first}"
        );
        assert!(last >= angle(MIN_DISTANCE, BASELINE), "{last}");
        assert!(l.pitch > 0.13 && l.pitch < 0.15, "{}", l.pitch);
        // Fewer candidates, a coarser pitch; never finer than the fine pass.
        assert!(Layout::coarse(64, 0.0, 0.0).pitch > 0.5 * COARSEST_PITCH);
        assert_eq!(Layout::coarse(100_000, 0.0, 0.0).pitch, FINE_PITCH);
        let f = Layout::fine(5.0, l.pitch, 0.0, VERTICAL_KNOWN);
        assert_eq!(f.window, 41);
        assert_eq!(f.vertical_reach, 8);
    }

    #[test]
    fn the_frontal_angle_holds_off_to_the_side() {
        // 1 m away, 40° to the side: the cameras' separation looks shorter.
        let (d, a) = (1.0f32, 40.0f32);
        let shift = 2.0
            * (0.5 * BASELINE * a.to_radians().cos() / d)
                .atan()
                .to_degrees();
        let got = crate::autofocus::distance(frontal_angle(shift, a), BASELINE);
        assert!((got - d).abs() < 0.01, "{got}");
    }

    #[test]
    fn the_vertical_offset_is_fitted_as_a_pitch_and_a_roll() {
        let mut v = VerticalFit::default();
        assert_eq!(v.fit(), None);
        for a in [-30.0f32, 0.0, 20.0, 40.0] {
            v.add(a, 0.0, 1.2 + 0.5 * a.to_radians().tan());
        }
        let (pitch, roll) = v.fit().unwrap();
        assert!(
            (pitch - 1.2).abs() < 1e-3 && (roll - 0.5).abs() < 1e-3,
            "{pitch} {roll}"
        );
        // All in one place: the pitch only.
        let mut one = VerticalFit::default();
        for _ in 0..3 {
            one.add(10.0, 0.0, -0.8);
        }
        let (pitch, roll) = one.fit().unwrap();
        assert!((pitch + 0.8).abs() < 1e-3 && roll == 0.0);
        // Older samples count for less.
        for _ in 0..6 {
            one.add(10.0, 0.0, 0.4);
        }
        assert!(one.predict(10.0, 0.0).unwrap() > 0.0);
    }

    fn fine_found(shift: f32) -> (Layout, Found) {
        let l = Layout::fine(shift, 0.14, 0.0, VERTICAL_KNOWN);
        (
            l,
            Found {
                shift,
                vertical: 0.3,
                correlation: 0.9,
                spread: 0.1,
                rival: false,
                second: None,
                sharpness: 0.5,
                end: None,
                vertical_clear: true,
                recentre: None,
            },
        )
    }

    #[test]
    fn coarse_then_fine_and_three_agreeing_readings_lock() {
        let mut e = Epipolar::default();
        e.tuning.fine_pass = true;
        let sig = [0.4; SIGNATURE];
        let mut locked = None;
        for _ in 0..3 {
            let (aim, layout) = e.next_patch(false).unwrap();
            assert!(matches!(aim, Aim::Ray(_)));
            assert!(e.next_patch(false).is_none(), "one in flight");
            let found = Found {
                shift: 5.0,
                vertical: 0.0,
                correlation: 0.9,
                spread: 0.1,
                rival: false,
                second: None,
                sharpness: 0.5,
                end: None,
                vertical_clear: true,
                recentre: None,
            };
            assert_eq!(
                e.matched([0.0, 0.0], 0.0, &layout, &found, 100.0, &sig),
                None
            );
            let (aim, layout) = e.next_patch(false).unwrap();
            assert_eq!(aim, Aim::At([0.0, 0.0]));
            assert_eq!(layout.pitch, FINE_PITCH);
            let (_, f) = fine_found(5.02);
            locked = e.matched([0.0, 0.0], 0.0, &layout, &f, 400.0, &sig);
        }
        let got = locked.expect("locked");
        assert!((got - 5.02).abs() < 1e-3);
        let s = e.take_stats();
        // The target and its neighbour agreed at the second; three of
        // five at the third.
        assert_eq!((s.coarse, s.fine, s.trusted, s.locks), (3, 3, 3, 2));
        assert_eq!(s.locks_by_neighbour, 1);
        assert!((s.cpu_sum / s.cpu_count as f32 - 250.0).abs() < 1e-3);
    }

    #[test]
    fn readings_that_disagree_do_not_lock() {
        let mut e = Epipolar::default();
        let sig = [0.4; SIGNATURE];
        for shift in [5.0, 7.0, 9.0, 11.0] {
            let (l, f) = fine_found(shift);
            e.next_patch(false);
            assert_eq!(e.matched([0.0, 0.0], 0.0, &l, &f, 1.0, &sig), None);
        }
    }

    #[test]
    fn a_cut_clears_the_readings() {
        let mut e = Epipolar::default();
        for _ in 0..2 {
            let (l, f) = fine_found(5.0);
            e.next_patch(false);
            e.matched([0.0, 0.0], 0.0, &l, &f, 1.0, &[0.4; SIGNATURE]);
        }
        let (l, f) = fine_found(5.0);
        e.next_patch(false);
        assert_eq!(
            e.matched([0.0, 0.0], 0.0, &l, &f, 1.0, &[0.9; SIGNATURE]),
            None
        );
        assert_eq!(e.take_stats().cuts, 1);
    }

    #[test]
    fn near_the_edge_or_the_axis_nothing_is_used() {
        let mut e = Epipolar::default();
        let (l, f) = fine_found(5.0);
        e.next_patch(false);
        e.matched([10.0, 0.0], 88.0, &l, &f, 1.0, &[0.4; SIGNATURE]);
        e.next_patch(false);
        e.matched([80.0, 0.0], 60.0, &l, &f, 1.0, &[0.4; SIGNATURE]);
        let s = e.take_stats();
        assert_eq!((s.rejected[7], s.rejected[8]), (1, 1));
    }

    #[test]
    fn very_clear_matches_teach_the_vertical_offset_and_narrow_the_search() {
        let mut e = Epipolar::default();
        assert_eq!(
            e.next_patch(false).unwrap().1.vertical_reach,
            Layout::coarse(256, 0.0, VERTICAL_SEARCH).vertical_reach
        );
        e.missed();
        for _ in 0..VERTICAL_NEEDED {
            let (l, f) = fine_found(5.0);
            e.next_patch(false);
            e.matched([0.0, 0.0], 0.0, &l, &f, 1.0, &[0.4; SIGNATURE]);
        }
        assert!((e.vertical_at(0.0, 0.0) - 0.3).abs() < 1e-3);
        let (_, l) = e.next_patch(false).unwrap();
        assert!(
            l.vertical_base.is_nan(),
            "the renderer fills in the prediction"
        );
        assert_eq!(l.vertical_reach, (VERTICAL_KNOWN / l.pitch).ceil() as u32);
    }
}
