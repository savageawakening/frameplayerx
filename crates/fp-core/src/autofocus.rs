//! Autofocus: from the stereo disparity measured where the viewer looks to
//! the distance the picture is drawn at (fp_render's `Focus`), so that what
//! the eyes are on is seen at its own depth whatever the viewer's IPD.
//!
//! The first decisions are adapted from Stereonaut's
//! `src/player/autofocus.c`, Copyright (c) 2026 Erdnusschokolade, MIT License
//! (`packaging/licenses/Stereonaut-MIT.txt`).

use serde::{Deserialize, Serialize};

/// Horizontal shifts tried per measurement. `shaders/disparity.wgsl` has the
/// same layout.
pub const CANDIDATES: usize = 256;
/// The candidate that means no disparity; those below it mean a picture
/// behind infinity.
pub const ZERO_CANDIDATE: usize = 16;
/// Degrees of video between candidates, and between the window's columns.
pub const STEP_DEGREES: f32 = 0.15;
/// Degrees of video between the window's rows, and between vertical shifts.
pub const ROW_STEP_DEGREES: f32 = 0.1875;
/// Vertical shifts tried at the best horizontal one: 11 rows either way,
/// about 2°.
pub const VERTICAL_CANDIDATES: usize = 23;
/// The coarse picture of the left image each measurement takes, 8 by 4.
pub const SIGNATURE: usize = 32;

/// A measurement every this many frames: at 90 Hz, 30 a second.
const MEASURE_EVERY: u32 = 3;
/// Below this the luma's correlation is more likely chance than the same
/// subject.
pub const MIN_SCORE: f32 = 0.6;
/// The same for the differences between samples, which score far lower on the
/// same picture: they keep the fine detail that differs between the lenses.
pub const GRADIENT_MIN_SCORE: f32 = 0.4;
/// Below this luma variance (luma scaled to 0..1) there's too little detail.
const MIN_VARIANCE: f32 = 2e-4;
/// The least variance the differences must have: above the steps between
/// 8-bit levels on a smooth ramp, and below real detail.
const MIN_GRADIENT_VARIANCE: f32 = 2e-6;
/// A second peak this close to the best one's score, and this many candidates
/// (1.5°) from it, makes a reading ambiguous.
const AMBIGUITY_MARGIN: f32 = 0.05;
const AMBIGUITY_APART: usize = 10;
/// For the differences, whose scores run lower: no more than this share of the
/// score.
const GRADIENT_AMBIGUITY_SHARE: f32 = 0.1;
/// How far the best score must stand above the mean of them all: on something
/// smooth the whole curve is high and flat, and its peak means nothing.
const MIN_SHARPNESS: f32 = 0.15;
/// The cameras' separation assumed until set, metres.
pub const BASELINE: f32 = 0.064;
/// The nearest the picture is drawn, metres (fp_render's own limit too).
pub const MIN_DISTANCE: f32 = 0.1;
/// How far, as a share of the distance, the subject must move for a refocus,
/// so that noise in the measurement doesn't move the picture.
const HYSTERESIS: f32 = 0.15;
/// Degrees of vergence a second the drawn distance moves at, and degrees a
/// second per second it speeds up and slows down at.
pub const RATE: f32 = 5.0;
pub const ACCELERATION: f32 = 62.5;
/// How many recent trusted readings vote, how many must agree, and how close
/// to their median counts as agreeing, degrees.
const VOTES: usize = 5;
const AGREEING: usize = 3;
const AGREEMENT_DEGREES: f32 = 0.6;
/// Once the focus has gone stale, how close counts as agreeing, degrees, and
/// after how many seconds.
const STALE_AGREEMENT_DEGREES: f32 = 1.5;
const STALE_AFTER: f32 = 5.0;
/// The eyes have settled once they've stayed within this many degrees for this
/// long; moving further starts again, and forgets the votes.
pub const SETTLE_DEGREES: f32 = 2.0;
const SETTLE_SECONDS: f32 = 0.1;
/// After the eyes are lost, seconds before the head takes over: a blink
/// shouldn't refocus on whatever the head happens to point at.
const EYES_LOST_SECONDS: f32 = 0.3;
/// The least a cosine is taken to be: near a pole, or straight to the side,
/// the cameras' separation is next to nothing and dividing by it means nothing.
const MIN_COS: f32 = 0.17;
/// How many windows around the gaze, how far apart, degrees, how much one
/// beside the gaze counts against the one on it, and seconds before one is
/// too old to count.
const NEAR_WINDOWS: usize = 9;
const NEAR_SPACING: f32 = 3.0;
const NEAR_SIDE_WEIGHT: f32 = 0.6;
const NEAR_MAX_AGE: f32 = 1.0;
/// How close depths must be to support each other, degrees.
const SUPPORT_DEGREES: f32 = 1.0;
/// What a vertical search must score to be learned from: correlation of the
/// differences down the window, and how far above their mean.
const VERTICAL_MIN_SCORE: f32 = 0.5;
const VERTICAL_MIN_SHARPNESS: f32 = 0.1;
/// The latest clear matches the vertical offset is found among, how many
/// must be this close together, degrees, for their mean to be it: a false
/// horizontal match can still look clear vertically, but false ones scatter.
const VERTICAL_SAMPLES: usize = 9;
const VERTICAL_AGREEING: usize = 3;
const VERTICAL_AGREEMENT: f32 = 0.3;
/// The furthest the vertical offset is taken to be, degrees.
pub const VERTICAL_LIMIT: f32 = 2.0;
/// The vertical offsets tried in turn until one gives a clear match: a rig
/// far enough out of line matches nowhere level, so there'd be nothing to
/// learn from.
const VERTICAL_TRIES: [f32; 5] = [0.0, 1.0, -1.0, 2.0, -2.0];
/// How much the coarse picture's luma must change on average for a cut.
const CUT_CHANGE: f32 = 0.08;

/// What a measurement goes by.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Match {
    /// The brightness. A smooth change in brightness matches itself at any shift.
    Luma,
    /// The differences between neighbouring samples across the window.
    #[default]
    Gradient,
}

impl Match {
    pub const ALL: [Match; 2] = [Match::Luma, Match::Gradient];

    pub fn name(self) -> &'static str {
        match self {
            Match::Luma => "Luma",
            Match::Gradient => "Gradient",
        }
    }
}

/// How autofocus measures and moves; the defaults are the feature's.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Tuning {
    /// Degrees of vergence a second the drawn distance moves at, however
    /// large the change, and degrees a second per second it gets there.
    pub rate: f32,
    pub acceleration: f32,
    /// How far apart the cameras were, metres.
    pub baseline: f32,
    /// What a reading goes by, and the score it must reach.
    pub method: Match,
    pub gradient_min_score: f32,
    /// Each sample the mean of four taps rather than one point.
    pub prefilter: bool,
    /// Learns the picture's vertical offset between the eyes, and searches
    /// at it.
    pub vertical_search: bool,
    /// How far the eyes can wander and still count as settled, degrees.
    pub settle_degrees: f32,
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning {
            rate: RATE,
            acceleration: ACCELERATION,
            baseline: BASELINE,
            method: Match::Gradient,
            gradient_min_score: GRADIENT_MIN_SCORE,
            prefilter: true,
            vertical_search: true,
            settle_degrees: SETTLE_DEGREES,
        }
    }
}

/// Where a window around the gaze goes from it, degrees: right and up.
pub fn near_offset(index: usize) -> [f32; 2] {
    let (column, row) = ((index % 3) as f32 - 1.0, (index / 3) as f32 - 1.0);
    [column * NEAR_SPACING, row * NEAR_SPACING]
}

/// The depth with the most support from (angle, weight) pairs: each is backed
/// by the weights of those near it, and the answer is their weighted mean.
fn best_supported(spots: &[(f32, f32)]) -> Option<f32> {
    let support = |at: f32| {
        spots
            .iter()
            .filter(|(angle, _)| (angle - at).abs() <= SUPPORT_DEGREES)
            .fold((0.0, 0.0), |(sum, weight), (angle, w)| {
                (sum + angle * w, weight + w)
            })
    };
    let (sum, weight) = spots
        .iter()
        .map(|(angle, _)| support(*angle))
        .max_by(|a, b| a.1.total_cmp(&b.1))?;
    (weight > 0.0).then(|| sum / weight)
}

/// One measurement.
#[derive(Clone, Copy, Debug)]
pub struct Reading {
    /// Normalized cross-correlation of the luma for each candidate shift,
    /// -1..1, and of the differences between neighbouring samples.
    pub luma: [f32; CANDIDATES],
    pub gradient: [f32; CANDIDATES],
    /// Variance in the window of the luma (scaled to 0..1) and of the
    /// differences.
    pub variance: f32,
    pub gradient_variance: f32,
    /// At the best shift by the differences, the correlation of the
    /// differences down the window for each vertical shift of the right
    /// image, from [`VERTICAL_CANDIDATES`]` / 2` rows up to as many down.
    pub vertical: [f32; VERTICAL_CANDIDATES],
    /// The vertical offset the search was made at, degrees.
    pub vertical_used: f32,
    /// The coarse picture of the left image.
    pub signature: [f32; SIGNATURE],
    /// Where the window was in the picture, degrees: longitude (right
    /// positive) and latitude (up positive).
    pub place: [f32; 2],
    /// Which window around the gaze it was.
    pub tag: usize,
}

impl Default for Reading {
    fn default() -> Self {
        Reading {
            luma: [0.0; CANDIDATES],
            gradient: [0.0; CANDIDATES],
            variance: 0.0,
            gradient_variance: 0.0,
            vertical: [0.0; VERTICAL_CANDIDATES],
            vertical_used: 0.0,
            signature: [0.0; SIGNATURE],
            place: [0.0; 2],
            tag: 0,
        }
    }
}

/// Why a reading can't be trusted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Rejected {
    LowDetail,
    SearchEnd,
    LowScore,
    Flat,
    Ambiguous,
}

/// One correlation curve and what it must reach.
struct Curve<'a> {
    score: &'a [f32; CANDIDATES],
    variance: f32,
    min_variance: f32,
    min_score: f32,
    gradient: bool,
}

impl Curve<'_> {
    fn best(&self) -> usize {
        let mut best = 0;
        for (i, &s) in self.score.iter().enumerate() {
            if s > self.score[best] {
                best = i;
            }
        }
        best
    }

    fn sharpness(&self) -> f32 {
        self.score[self.best()] - self.score.iter().sum::<f32>() / CANDIDATES as f32
    }

    fn judge(&self) -> Result<f32, Rejected> {
        let best = self.best();
        let score = self.score[best];
        let margin = if self.gradient {
            (score.max(0.0) * GRADIENT_AMBIGUITY_SHARE).min(AMBIGUITY_MARGIN)
        } else {
            AMBIGUITY_MARGIN
        };
        if self.variance.is_nan() || self.variance < self.min_variance {
            return Err(Rejected::LowDetail);
        }
        if best == 0 || best == CANDIDATES - 1 {
            return Err(Rejected::SearchEnd);
        }
        if score.is_nan() || score < self.min_score {
            return Err(Rejected::LowScore);
        }
        if self.sharpness().is_nan() || self.sharpness() < MIN_SHARPNESS {
            return Err(Rejected::Flat);
        }
        let rival = (1..CANDIDATES - 1).any(|i| {
            i.abs_diff(best) >= AMBIGUITY_APART
                && self.score[i] >= score - margin
                && self.score[i] >= self.score[i - 1]
                && self.score[i] >= self.score[i + 1]
        });
        if rival {
            return Err(Rejected::Ambiguous);
        }
        Ok(self.refined(best))
    }

    /// The disparity at a candidate, from a parabola through it and its
    /// neighbours.
    fn refined(&self, i: usize) -> f32 {
        (i as f32 + parabola(self.score[i - 1], self.score[i], self.score[i + 1])
            - ZERO_CANDIDATE as f32)
            * STEP_DEGREES
    }

    fn peaks(&self, count: usize) -> Vec<(f32, f32)> {
        let mut all: Vec<usize> = (1..CANDIDATES - 1)
            .filter(|&i| self.score[i] >= self.score[i - 1] && self.score[i] >= self.score[i + 1])
            .collect();
        all.sort_by(|a, b| self.score[*b].total_cmp(&self.score[*a]));
        let mut kept: Vec<usize> = Vec::new();
        for i in all {
            if kept.len() < count && kept.iter().all(|k| k.abs_diff(i) >= AMBIGUITY_APART) {
                kept.push(i);
            }
        }
        kept.iter()
            .map(|&i| (self.refined(i), self.score[i]))
            .collect()
    }
}

/// Where the top of a parabola through three evenly spaced values is, from
/// the middle one, in their spacing; 0 when they don't peak.
fn parabola(a: f32, b: f32, c: f32) -> f32 {
    let curvature = a - 2.0 * b + c;
    if curvature < 0.0 {
        (0.5 * (a - c) / curvature).clamp(-0.5, 0.5)
    } else {
        0.0
    }
}

impl Reading {
    fn curve(&self, method: Match, gradient_min_score: f32) -> Curve<'_> {
        match method {
            Match::Luma => Curve {
                score: &self.luma,
                variance: self.variance,
                min_variance: MIN_VARIANCE,
                min_score: MIN_SCORE,
                gradient: false,
            },
            Match::Gradient => Curve {
                score: &self.gradient,
                variance: self.gradient_variance,
                min_variance: MIN_GRADIENT_VARIANCE,
                min_score: gradient_min_score,
                gradient: true,
            },
        }
    }

    /// The disparity in degrees of longitude, positive when nearer than
    /// infinity, or why there's nothing reliable to go on.
    pub fn judge(&self, method: Match, gradient_min_score: f32) -> Result<f32, Rejected> {
        self.curve(method, gradient_min_score).judge()
    }

    /// The best score going by `method`.
    pub fn score(&self, method: Match) -> f32 {
        let c = self.curve(method, GRADIENT_MIN_SCORE);
        c.score[c.best()]
    }

    /// How far below the left image's the right image's picture is here,
    /// degrees, when the vertical search found it clearly.
    pub fn vertical_found(&self) -> Option<f32> {
        let v = &self.vertical;
        let mut best = 1;
        for i in 1..VERTICAL_CANDIDATES - 1 {
            if v[i] > v[best] {
                best = i;
            }
        }
        let mean = v.iter().sum::<f32>() / VERTICAL_CANDIDATES as f32;
        if !(v[best] >= VERTICAL_MIN_SCORE && v[best] - mean >= VERTICAL_MIN_SHARPNESS) {
            return None;
        }
        let rows = best as f32 + parabola(v[best - 1], v[best], v[best + 1])
            - (VERTICAL_CANDIDATES / 2) as f32;
        Some(self.vertical_used + rows * ROW_STEP_DEGREES)
    }

    /// A line for the log, to tell apart why a reading came out as it did.
    pub fn probe(&self, tuning: &Tuning) -> String {
        let [lon, lat] = self.place;
        let curve = self.curve(tuning.method, tuning.gradient_min_score);
        let verdict = curve.judge();
        let peaks: Vec<String> = curve
            .peaks(3)
            .iter()
            .map(|(d, s)| format!("{d:.2}° {s:.2}"))
            .collect();
        let why = match verdict {
            Ok(d) => format!(
                "trusted: {:.2} m from the cameras",
                distance(frontal_angle(d, self.place), tuning.baseline)
            ),
            Err(why) => format!("rejected: {why:?}"),
        };
        let vertical = match self.vertical_found() {
            Some(v) => format!("{v:+.2}°"),
            None => "unclear".into(),
        };
        format!(
            "window lon {lon:.1}° lat {lat:.1}°; {} peaks {}; sharp {:.2}; \
             vertical {vertical} (searched at {:+.2}°); {why}",
            tuning.method.name(),
            peaks.join(", "),
            curve.sharpness(),
            self.vertical_used,
        )
    }
}

/// The angle cameras would see a subject through straight ahead at the same
/// distance, degrees, from its disparity in longitude and where it is in the
/// picture (degrees of longitude and latitude). Longitude is measured around
/// the vertical, in which the cameras' separation lies: seen from above, they
/// see the subject through the disparity, at its horizontal distance, with
/// their separation shortened by the cosine of the longitude. Its distance is
/// that over the cosine of the latitude. None of this depends on the
/// separation, so it holds whatever the baseline is set to.
pub fn frontal_angle(disparity: f32, place: [f32; 2]) -> f32 {
    if !(disparity.is_finite() && place[0].is_finite() && place[1].is_finite()) {
        return f32::NAN;
    }
    let (lon, lat) = (place[0].to_radians(), place[1].to_radians());
    let half = (0.5 * disparity.to_radians()).tan() * lat.cos() / lon.cos().max(MIN_COS);
    2.0 * half.atan().to_degrees()
}

/// How far away, metres, a subject is that cameras `baseline` apart see
/// through `angle` degrees straight ahead. Infinite at no angle.
pub fn distance(angle: f32, baseline: f32) -> f32 {
    if angle <= 0.0 {
        return f32::INFINITY;
    }
    0.5 * baseline / (0.5 * angle.to_radians()).tan()
}

/// The angle straight ahead for a distance: [`distance`]'s inverse.
pub fn angle(distance: f32, baseline: f32) -> f32 {
    2.0 * (0.5 * baseline / distance).atan().to_degrees()
}

/// The median of `votes`, if enough of them are close to it.
fn agreement(votes: &[f32], tolerance: f32) -> Option<f32> {
    let mut sorted = [0.0; VOTES];
    let sorted = &mut sorted[..votes.len()];
    sorted.copy_from_slice(votes);
    sorted.sort_by(f32::total_cmp);
    let median = sorted[sorted.len() / 2];
    let close = votes
        .iter()
        .filter(|v| (**v - median).abs() <= tolerance)
        .count();
    (close >= AGREEING).then_some(median)
}

fn degrees_between(a: [f32; 3], b: [f32; 3]) -> f32 {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    dot.clamp(-1.0, 1.0).acos().to_degrees()
}

/// A value moving towards a target at a rate, speeding up and slowing down
/// softly: as fast as the rate allows while still able to stop on the target.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Easing {
    pub value: f32,
    pub speed: f32,
}

impl Easing {
    /// Moves `dt` seconds towards `target` and returns where it is.
    pub fn step(&mut self, target: f32, dt: f32, rate: f32, acceleration: f32) -> f32 {
        if dt <= 0.0 {
            return self.value;
        }
        let rate = rate.max(0.01);
        let accel = acceleration.max(0.01);
        let away = target - self.value;
        // The fastest it can go and still stop on the target, slowing by a
        // frame's worth at a time: the discrete form of v = sqrt(2ad). A new
        // target acts at once, from wherever it is and however fast it's
        // going, so one behind it means slowing through nothing first, v²/2a
        // further on.
        let frame = 0.5 * accel * dt;
        let stopping = (2.0 * accel * away.abs() + frame * frame).sqrt() - frame;
        let wanted = away.signum() * rate.min(stopping);
        self.speed += (wanted - self.speed).clamp(-accel * dt, accel * dt);
        self.speed = self.speed.clamp(-rate, rate);
        let step = self.speed * dt;
        // It only stops on the target if it's going slowly enough to: too fast
        // for one that's just appeared ahead, it carries on past and comes back.
        let stoppable = self.speed.abs() <= 4.0 * accel * dt;
        if step * away >= 0.0 && step.abs() >= away.abs() && stoppable {
            self.value = target;
            self.speed = 0.0;
        } else {
            self.value += step;
        }
        self.value
    }
}

/// What can stop a reading reaching the focus, as the log names them.
pub const GATES: [&str; 7] = [
    "eyes moving",
    "low detail",
    "search end",
    "low score",
    "flat",
    "ambiguous",
    "disagreed",
];

/// How measuring has gone since it was last asked, for the log.
#[derive(Clone, Copy, Debug, Default)]
pub struct Stats {
    pub measured: u32,
    pub trusted: u32,
    /// Readings stopped, by [`GATES`].
    pub blocked: [u32; GATES.len()],
    pub focus_changes: u32,
    pub cuts: u32,
    pub clears: u32,
    /// Clear vertical matches, and the offset learned.
    pub vertical_found: u32,
    pub vertical: Option<f32>,
    /// Seconds in all, seconds the eyes were on a panel rather than the
    /// picture, and how often they went there.
    pub time: f32,
    pub ui_time: f32,
    pub ui_holds: u32,
    /// The longest time without readings agreeing, seconds, and what stopped
    /// them most in it.
    pub longest_stuck: f32,
    pub stuck_by: &'static str,
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let gates: Vec<String> = GATES
            .iter()
            .zip(self.blocked)
            .filter(|(_, n)| *n > 0)
            .map(|(g, n)| format!("{g} {n}"))
            .collect();
        write!(
            f,
            "{} of {} trusted; stopped by {}; {} refocus, {} cuts, {} cleared; \
             vertical offset {} from {} clear; stuck {:.1} s at most ({}); eyes on a panel \
             {:.0}% of the time, {} times",
            self.trusted,
            self.measured,
            if gates.is_empty() {
                "nothing".into()
            } else {
                gates.join(", ")
            },
            self.focus_changes,
            self.cuts,
            self.clears,
            self.vertical
                .map_or("not yet learned".into(), |v| format!("{v:+.2}°")),
            self.vertical_found,
            self.longest_stuck,
            if self.stuck_by.is_empty() {
                "nothing measured"
            } else {
                self.stuck_by
            },
            100.0 * self.ui_time / self.time.max(1e-6),
            self.ui_holds,
        )
    }
}

/// What a window around the gaze last found.
#[derive(Clone, Copy, Debug, Default)]
struct Spot {
    /// [`frontal_angle`], degrees.
    angle: f32,
    /// Its score, 0 if it has found nothing, and its age in seconds.
    score: f32,
    age: f32,
}

/// The state of one open video.
#[derive(Clone, Copy, Debug)]
pub struct Autofocus {
    pub tuning: Tuning,
    /// What the readings last agreed on: [`frontal_angle`], degrees.
    focus: Option<f32>,
    /// The angle drawn, moving towards the focus.
    shown: Easing,
    frames: u32,
    votes: [f32; VOTES],
    voted: usize,
    near: [Spot; NEAR_WINDOWS],
    near_next: usize,
    /// Where the eyes have stayed, and for how long.
    anchor: Option<[f32; 3]>,
    held: f32,
    /// Seconds since the eyes were lost, until the head takes over.
    lost: Option<f32>,
    /// The eyes are on their way somewhere.
    unsettled: bool,
    /// Clear vertical matches, newest overwriting the oldest.
    vertical: [f32; VERTICAL_SAMPLES],
    vertical_count: usize,
    /// Measurements made without one, for trying offsets in turn.
    vertical_tries: usize,
    /// The last measurement's coarse picture.
    signature: Option<[f32; SIGNATURE]>,
    since_agreed: f32,
    /// Cleared since the readings last agreed: the next agreement moves the
    /// focus however close it is.
    cleared: bool,
    /// The eyes are on a panel rather than the picture.
    on_ui: bool,
    blocked: [u32; GATES.len()],
    stats: Stats,
}

impl Default for Autofocus {
    fn default() -> Self {
        Autofocus {
            tuning: Tuning::default(),
            focus: None,
            shown: Easing::default(),
            frames: 0,
            votes: [0.0; VOTES],
            voted: 0,
            near: [Spot::default(); NEAR_WINDOWS],
            near_next: 0,
            anchor: None,
            held: 0.0,
            lost: None,
            unsettled: false,
            vertical: [0.0; VERTICAL_SAMPLES],
            vertical_count: 0,
            vertical_tries: 0,
            signature: None,
            since_agreed: 0.0,
            cleared: false,
            on_ui: false,
            blocked: [0; GATES.len()],
            stats: Stats::default(),
        }
    }
}

impl Autofocus {
    /// Whether this frame should measure. Call once per frame.
    pub fn due(&mut self) -> bool {
        self.frames = self.frames.wrapping_add(1);
        self.frames.is_multiple_of(MEASURE_EVERY)
    }

    /// Takes this frame's gaze, a unit direction, if the eyes are tracked;
    /// whether it may be measured along. Without eye tracking, always.
    pub fn look(&mut self, gaze: Option<[f32; 3]>, dt: f32) -> bool {
        self.on_ui = false;
        let settled = match gaze {
            Some(now) => {
                self.lost = None;
                if self
                    .anchor
                    .is_some_and(|at| degrees_between(at, now) <= self.tuning.settle_degrees)
                {
                    self.held += dt;
                } else {
                    self.anchor = Some(now);
                    self.held = 0.0;
                    self.forget_readings();
                }
                self.held >= SETTLE_SECONDS
            }
            None => {
                if self.anchor.take().is_some() {
                    self.lost = Some(0.0);
                    self.voted = 0;
                }
                self.lost = self.lost.map(|t| t + dt).filter(|t| *t < EYES_LOST_SECONDS);
                self.lost.is_none()
            }
        };
        self.unsettled = !settled;
        settled
    }

    /// The eyes are on a panel this frame, `dt` seconds long: nothing is
    /// measured and nothing counts against the eyes' settling; the focus holds
    /// where it is. Off the panel, they settle afresh, as after any look
    /// elsewhere. Call instead of [`Autofocus::look`].
    pub fn on_ui(&mut self, dt: f32) {
        if !self.on_ui {
            self.stats.ui_holds += 1;
        }
        self.on_ui = true;
        self.stats.ui_time += dt;
        self.anchor = None;
        self.lost = None;
        self.held = 0.0;
        self.unsettled = true;
        self.forget_readings();
    }

    /// Notes a measurement not asked for because the eyes were moving.
    pub fn eyes_moving(&mut self) {
        self.block(0);
    }

    /// Which window around the gaze to measure next, and where it is from the
    /// gaze: right and up, degrees.
    pub fn next_near(&mut self) -> (usize, [f32; 2]) {
        let index = self.near_next;
        self.near_next = (index + 1) % NEAR_WINDOWS;
        (index, near_offset(index))
    }

    /// The vertical offset to search at, degrees: how far the right image's
    /// picture sits below the left's. Until clear matches agree on it, each of
    /// a few in turn.
    pub fn vertical_offset(&self) -> f32 {
        if !self.tuning.vertical_search {
            return 0.0;
        }
        self.learned_vertical()
            .unwrap_or(VERTICAL_TRIES[self.vertical_tries % VERTICAL_TRIES.len()])
    }

    /// The vertical offset clear matches agree on, degrees: the mean of the
    /// largest group of them close together, if it's large enough.
    pub fn learned_vertical(&self) -> Option<f32> {
        let samples = &self.vertical[..self.vertical_count.min(VERTICAL_SAMPLES)];
        let group = |at: f32| {
            samples
                .iter()
                .filter(move |v| (*v - at).abs() <= VERTICAL_AGREEMENT)
        };
        let at = samples
            .iter()
            .copied()
            .max_by_key(|&at| group(at).count())?;
        let n = group(at).count();
        (n >= VERTICAL_AGREEING).then(|| group(at).sum::<f32>() / n as f32)
    }

    /// Takes in a measurement of a window around the gaze. One that can't be
    /// trusted changes nothing: the focus stays until something else can be
    /// measured. A trusted one only moves it once most of the recent ones
    /// agree, so one wrong peak can't.
    pub fn measured(&mut self, reading: &Reading) {
        if self.cut(reading) {
            self.stats.cuts += 1;
            self.clear();
        }
        self.signature = Some(reading.signature);
        self.stats.measured += 1;
        self.vertical_tries += 1;
        let curve = reading.curve(self.tuning.method, self.tuning.gradient_min_score);
        let disparity = match curve.judge() {
            Ok(d) => d,
            Err(why) => {
                self.block(match why {
                    Rejected::LowDetail => 1,
                    Rejected::SearchEnd => 2,
                    Rejected::LowScore => 3,
                    Rejected::Flat => 4,
                    Rejected::Ambiguous => 5,
                });
                if let Some(spot) = self.near.get_mut(reading.tag) {
                    *spot = Spot::default();
                }
                return;
            }
        };
        self.stats.trusted += 1;
        if self.tuning.vertical_search
            && let Some(v) = reading.vertical_found()
        {
            self.vertical[self.vertical_count % VERTICAL_SAMPLES] =
                v.clamp(-VERTICAL_LIMIT, VERTICAL_LIMIT);
            self.vertical_count += 1;
            self.stats.vertical_found += 1;
        }
        if self.unsettled {
            // Measured before the eyes moved away.
            self.block(0);
            return;
        }
        let Some(spot) = self.near.get_mut(reading.tag) else {
            return;
        };
        *spot = Spot {
            angle: frontal_angle(disparity, reading.place),
            score: curve.score[curve.best()],
            age: 0.0,
        };
        let middle = NEAR_WINDOWS / 2;
        let spots: Vec<(f32, f32)> = self
            .near
            .iter()
            .enumerate()
            .filter(|(_, s)| s.score > 0.0 && s.age < NEAR_MAX_AGE)
            .map(|(i, s)| {
                let side = if i == middle { 1.0 } else { NEAR_SIDE_WEIGHT };
                (s.angle, s.score * side)
            })
            .collect();
        if let Some(angle) = best_supported(&spots) {
            self.vote(angle);
        }
    }

    /// Whether the picture changed so much since the last measurement that
    /// it's another scene.
    fn cut(&self, reading: &Reading) -> bool {
        self.signature.is_some_and(|last| {
            let change: f32 = last
                .iter()
                .zip(reading.signature)
                .map(|(a, b)| (a - b).abs())
                .sum();
            change / SIGNATURE as f32 > CUT_CHANGE
        })
    }

    fn block(&mut self, gate: usize) {
        self.blocked[gate] += 1;
        self.stats.blocked[gate] += 1;
    }

    fn forget_readings(&mut self) {
        self.voted = 0;
        self.near = [Spot::default(); NEAR_WINDOWS];
    }

    /// Forgets what was measured, for a seek or a cut: the picture now may be
    /// anywhere. The drawn distance stays until new readings agree, and they
    /// needn't be far from it to move it.
    pub fn clear(&mut self) {
        self.forget_readings();
        self.signature = None;
        self.since_agreed = STALE_AFTER;
        self.cleared = true;
        self.stats.clears += 1;
    }

    /// One more trusted depth.
    fn vote(&mut self, angle: f32) {
        if !angle.is_finite() {
            return;
        }
        self.votes[self.voted % VOTES] = angle;
        self.voted += 1;
        // A focus can't be held for long against readings it's trusting: they
        // only have to roughly agree once it has gone stale. Something
        // featureless gives no trusted readings, so there it holds.
        let stale = self.since_agreed >= STALE_AFTER;
        let tolerance = if stale {
            STALE_AGREEMENT_DEGREES
        } else {
            AGREEMENT_DEGREES
        };
        let Some(agreed) = agreement(&self.votes[..self.voted.min(VOTES)], tolerance) else {
            if self.voted >= AGREEING {
                self.block(6);
            }
            return;
        };
        self.since_agreed = 0.0;
        self.blocked = [0; GATES.len()];
        let b = self.tuning.baseline;
        let moved = std::mem::take(&mut self.cleared)
            || self.focus.is_none_or(|focus| {
                let at = distance(focus, b);
                (distance(agreed, b) - at).abs() > HYSTERESIS * at
            });
        if moved {
            self.focus = Some(agreed);
            self.stats.focus_changes += 1;
        }
    }

    /// How far away the focused subject is, metres, as the readings agreed.
    pub fn focus_distance(&self) -> Option<f32> {
        self.focus
            .map(|a| distance(a, self.tuning.baseline).max(MIN_DISTANCE))
    }

    /// What has happened since the last call.
    pub fn take_stats(&mut self) -> Stats {
        let vertical = self.learned_vertical();
        let mut s = std::mem::take(&mut self.stats);
        s.vertical = vertical;
        s
    }

    /// Back to no autofocus at once: the picture is drawn as without it.
    pub fn off(&mut self) {
        self.focus = None;
        self.shown = Easing::default();
        self.forget_readings();
    }

    /// Moves the drawn distance towards the focus by `dt` seconds, and returns
    /// it, metres: `None` while it's at infinity, where the picture is drawn
    /// as without autofocus.
    pub fn advance(&mut self, dt: f32) -> Option<f32> {
        self.stats.time += dt;
        self.since_agreed += dt;
        for spot in &mut self.near {
            spot.age += dt;
        }
        if self.since_agreed > self.stats.longest_stuck {
            self.stats.longest_stuck = self.since_agreed;
            let most = (0..GATES.len()).max_by_key(|&i| self.blocked[i]);
            self.stats.stuck_by = match most {
                Some(i) if self.blocked[i] > 0 => GATES[i],
                _ => "nothing measured",
            };
        }
        let t = self.tuning;
        let nearest = angle(MIN_DISTANCE, t.baseline);
        let target = self.focus.map_or(0.0, |a| a.clamp(0.0, nearest));
        if !(self.shown.value.is_finite() && self.shown.speed.is_finite()) {
            self.shown = Easing::default();
        }
        let shown = self.shown.step(target, dt, t.rate, t.acceleration);
        (shown > 1e-4).then(|| distance(shown, t.baseline).max(MIN_DISTANCE))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A clean peak at `disparity` degrees in both curves, straight ahead,
    /// with a clear vertical match `vertical` degrees down.
    fn reading_at(disparity: f32, place: [f32; 2], vertical: f32) -> Reading {
        let peak = disparity / STEP_DEGREES + ZERO_CANDIDATE as f32;
        let mut score = [0.0; CANDIDATES];
        for (i, s) in score.iter_mut().enumerate() {
            *s = 0.95 - 0.02 * (i as f32 - peak).powi(2);
        }
        let row = vertical / ROW_STEP_DEGREES + (VERTICAL_CANDIDATES / 2) as f32;
        let mut v = [0.0; VERTICAL_CANDIDATES];
        for (i, s) in v.iter_mut().enumerate() {
            *s = 0.9 - 0.05 * (i as f32 - row).powi(2);
        }
        Reading {
            luma: score,
            gradient: score,
            variance: 0.01,
            gradient_variance: 0.01,
            vertical: v,
            place,
            ..Reading::default()
        }
    }

    fn reading(disparity: f32) -> Reading {
        reading_at(disparity, [0.0; 2], 0.0)
    }

    /// The middle window, so that one reading is all the support there is.
    fn on_gaze(r: Reading) -> Reading {
        Reading {
            tag: NEAR_WINDOWS / 2,
            ..r
        }
    }

    /// Enough good readings in a row at `disparity` to move the focus.
    fn agree(a: &mut Autofocus, disparity: f32) {
        for _ in 0..AGREEING {
            a.measured(&on_gaze(reading(disparity)));
        }
    }

    fn settle(a: &mut Autofocus) -> Option<f32> {
        for _ in 0..1500 {
            a.advance(1.0 / 90.0);
        }
        a.advance(0.0)
    }

    fn judged(r: &Reading) -> Option<f32> {
        r.judge(Match::Gradient, GRADIENT_MIN_SCORE).ok()
    }

    #[test]
    fn the_peak_is_found_between_candidates() {
        for d in [-1.0, 0.0, 2.0, 6.07, 12.33] {
            let got = judged(&reading(d)).unwrap();
            assert!((got - d).abs() < 1e-3, "{d}: {got}");
        }
    }

    #[test]
    fn nothing_reliable_is_no_disparity() {
        let flat = Reading {
            gradient_variance: MIN_GRADIENT_VARIANCE / 2.0,
            ..reading(6.0)
        };
        assert_eq!(
            flat.judge(Match::Gradient, GRADIENT_MIN_SCORE),
            Err(Rejected::LowDetail)
        );
        let mut poor = reading(6.0);
        poor.gradient.iter_mut().for_each(|s| *s -= 0.6);
        assert_eq!(
            poor.judge(Match::Gradient, GRADIENT_MIN_SCORE),
            Err(Rejected::LowScore)
        );
        let mut edge = reading(6.0);
        edge.gradient[CANDIDATES - 1] = 1.0;
        assert_eq!(
            edge.judge(Match::Gradient, GRADIENT_MIN_SCORE),
            Err(Rejected::SearchEnd)
        );
        let nan = Reading {
            gradient: [f32::NAN; CANDIDATES],
            ..reading(6.0)
        };
        assert_eq!(judged(&nan), None);
    }

    #[test]
    fn a_flat_curve_is_no_measurement() {
        let mut r = reading(4.0);
        for s in &mut r.gradient {
            *s = 0.9 + 0.08 * (*s - 0.9).max(-1.0) / 10.0;
        }
        assert_eq!(
            r.judge(Match::Gradient, GRADIENT_MIN_SCORE),
            Err(Rejected::Flat)
        );
    }

    #[test]
    fn a_rival_peak_makes_a_reading_ambiguous() {
        let mut r = reading(4.0);
        let best = r.gradient.iter().cloned().fold(f32::MIN, f32::max);
        // A second peak 3° away, nearly as good.
        let at = (4.0 / STEP_DEGREES) as usize + ZERO_CANDIDATE + 20;
        r.gradient[at] = best - 0.01;
        assert_eq!(
            r.judge(Match::Gradient, GRADIENT_MIN_SCORE),
            Err(Rejected::Ambiguous)
        );
        // Further below, by more than a tenth of the score, it isn't.
        r.gradient[at] = best - 0.2;
        assert!(r.judge(Match::Gradient, GRADIENT_MIN_SCORE).is_ok());
    }

    #[test]
    fn each_method_goes_by_its_own_curve() {
        let mut r = reading(4.0);
        r.luma = reading(9.0).luma;
        assert!((r.judge(Match::Luma, GRADIENT_MIN_SCORE).unwrap() - 9.0).abs() < 1e-3);
        assert!((r.judge(Match::Gradient, GRADIENT_MIN_SCORE).unwrap() - 4.0).abs() < 1e-3);
        // The differences' own threshold.
        assert!(r.judge(Match::Gradient, 0.99).is_err());
    }

    #[test]
    fn the_search_reaches_a_tenth_of_a_metre() {
        let farthest = (CANDIDATES - 2 - ZERO_CANDIDATE) as f32 * STEP_DEGREES;
        assert!(distance(farthest, BASELINE) <= MIN_DISTANCE);
        let got = judged(&reading(35.0)).unwrap();
        assert!((got - 35.0).abs() < 1e-3, "{got}");
    }

    #[test]
    fn distance_and_angle_are_each_others_inverse() {
        for b in [0.05, 0.064, 0.08] {
            for d in [0.1, 0.5, 2.0, 30.0] {
                assert!((distance(angle(d, b), b) - d).abs() < d * 1e-4);
            }
        }
        assert_eq!(distance(0.0, BASELINE), f32::INFINITY);
    }

    /// Where a point `r` metres from the middle of cameras `b` apart, at
    /// longitude and latitude `lon`, `lat` degrees, shows in each image:
    /// the window's place (between them) and the disparity in longitude.
    fn seen(r: f32, b: f32, lon: f32, lat: f32) -> ([f32; 2], f32) {
        let (lon, lat) = (lon.to_radians(), lat.to_radians());
        let p = [
            r * lat.cos() * lon.sin(),
            r * lat.sin(),
            -r * lat.cos() * lon.cos(),
        ];
        let from = |cx: f32| {
            let v = [p[0] - cx, p[1], p[2]];
            (v[0].atan2(-v[2]).to_degrees(), {
                let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                (v[1] / len).asin().to_degrees()
            })
        };
        let (l, r) = (from(-b / 2.0), from(b / 2.0));
        ([(l.0 + r.0) / 2.0, (l.1 + r.1) / 2.0], l.0 - r.0)
    }

    #[test]
    fn a_subject_is_measured_at_its_distance_anywhere_in_the_picture() {
        for b in [0.06, 0.064] {
            for r in [0.3, 1.0, 3.0] {
                for (lon, lat) in [
                    (0.0, 0.0),
                    (40.0, 0.0),
                    (0.0, 45.0),
                    (-50.0, 30.0),
                    (60.0, -40.0),
                ] {
                    let (place, disparity) = seen(r, b, lon, lat);
                    let got = distance(frontal_angle(disparity, place), b);
                    assert!(
                        (got - r).abs() < r * 0.02,
                        "{r} m at {lon}°, {lat}° with {b}: {got}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_baseline_scales_the_distance_without_measuring_again() {
        let mut a = Autofocus::default();
        agree(&mut a, angle(0.5, BASELINE));
        let at = a.focus_distance().unwrap();
        assert!((at - 0.5).abs() < 1e-3, "{at}");
        a.tuning.baseline = 0.032;
        assert!((a.focus_distance().unwrap() - 0.25).abs() < 1e-3);
    }

    #[test]
    fn with_nothing_agreed_the_picture_is_drawn_as_without_autofocus() {
        let mut a = Autofocus::default();
        assert_eq!(settle(&mut a), None);
        a.measured(&on_gaze(reading(4.0)));
        assert_eq!(settle(&mut a), None, "one reading isn't enough");
    }

    #[test]
    fn agreed_readings_draw_the_picture_at_their_distance() {
        let mut a = Autofocus::default();
        agree(&mut a, angle(0.6, BASELINE));
        let got = settle(&mut a).unwrap();
        assert!((got - 0.6).abs() < 1e-3, "{got}");
    }

    #[test]
    fn never_nearer_than_a_tenth_of_a_metre() {
        let mut a = Autofocus::default();
        // Cameras 5 cm apart see 0.1 m through 28°.
        a.tuning.baseline = 0.05;
        agree(&mut a, 34.0);
        let got = settle(&mut a).unwrap();
        assert!((got - MIN_DISTANCE).abs() < 1e-4, "{got}");
    }

    #[test]
    fn small_movements_keep_the_focus() {
        let mut a = Autofocus::default();
        agree(&mut a, angle(0.45, BASELINE));
        let held = settle(&mut a).unwrap();
        // 42 cm is within 15% of 45.
        agree(&mut a, angle(0.42, BASELINE));
        assert_eq!(settle(&mut a), Some(held));
        // 33 cm is beyond it.
        agree(&mut a, angle(0.33, BASELINE));
        assert!((settle(&mut a).unwrap() - 0.33).abs() < 1e-3);
    }

    #[test]
    fn an_unreliable_measurement_keeps_the_focus() {
        let mut a = Autofocus::default();
        agree(&mut a, 8.0);
        let held = settle(&mut a);
        a.measured(&on_gaze(Reading {
            gradient_variance: 0.0,
            ..reading(1.0)
        }));
        assert_eq!(settle(&mut a), held);
    }

    #[test]
    fn one_outlier_among_good_readings_does_not_move_the_focus() {
        let mut a = Autofocus::default();
        agree(&mut a, 8.0);
        let held = settle(&mut a);
        a.measured(&on_gaze(reading(2.0)));
        a.measured(&on_gaze(reading(8.0)));
        assert_eq!(settle(&mut a), held);
    }

    #[test]
    fn readings_that_disagree_hold_the_focus_and_are_counted() {
        let mut a = Autofocus::default();
        agree(&mut a, 8.0);
        let held = settle(&mut a);
        for d in [2.0, 12.0, 5.0, 16.0, 3.0] {
            a.measured(&on_gaze(reading(d)));
        }
        assert_eq!(settle(&mut a), held);
        assert!(a.take_stats().blocked[6] > 0);
    }

    #[test]
    fn eyes_on_the_move_are_not_measured_until_they_settle() {
        let mut a = Autofocus::default();
        let ahead = [0.0, 0.0, -1.0];
        assert!(!a.look(Some(ahead), 1.0 / 90.0));
        agree(&mut a, 8.0);
        assert_eq!(settle(&mut a), None);
        for _ in 0..12 {
            a.look(Some(ahead), 1.0 / 90.0);
        }
        assert!(a.look(Some(ahead), 1.0 / 90.0));
        agree(&mut a, 8.0);
        assert!(settle(&mut a).is_some());
    }

    #[test]
    fn a_jump_forgets_the_votes_from_the_last_place() {
        let mut a = Autofocus::default();
        let ahead = [0.0, 0.0, -1.0];
        for _ in 0..12 {
            a.look(Some(ahead), 1.0 / 90.0);
        }
        a.measured(&on_gaze(reading(8.0)));
        a.measured(&on_gaze(reading(8.0)));
        // 10° away: the third vote would have agreed, but it's elsewhere now.
        let away = [10f32.to_radians().sin(), 0.0, -10f32.to_radians().cos()];
        a.look(Some(away), 1.0 / 90.0);
        for _ in 0..12 {
            a.look(Some(away), 1.0 / 90.0);
        }
        a.measured(&on_gaze(reading(8.0)));
        assert_eq!(settle(&mut a), None);
    }

    #[test]
    fn a_blink_does_not_hand_over_to_the_head_at_once() {
        let mut a = Autofocus::default();
        let ahead = [0.0, 0.0, -1.0];
        for _ in 0..12 {
            a.look(Some(ahead), 1.0 / 90.0);
        }
        assert!(!a.look(None, 0.1), "just lost");
        assert!(!a.look(None, 0.1));
        assert!(a.look(None, 0.15), "lost long enough for the head to lead");
    }

    #[test]
    fn near_the_gaze_a_wrong_window_is_outvoted() {
        let mut a = Autofocus::default();
        for _ in 0..AGREEING {
            for tag in 0..NEAR_WINDOWS {
                let d = if tag == 0 { 20.0 } else { 8.0 };
                a.measured(&Reading { tag, ..reading(d) });
            }
        }
        let got = settle(&mut a).unwrap();
        let want = distance(8.0, BASELINE);
        assert!((got - want).abs() < want * 0.01, "{got} {want}");
    }

    #[test]
    fn windows_around_the_gaze_are_three_degrees_apart_with_one_on_it() {
        assert_eq!(near_offset(NEAR_WINDOWS / 2), [0.0, 0.0]);
        assert_eq!(near_offset(0), [-3.0, -3.0]);
        assert_eq!(near_offset(8), [3.0, 3.0]);
        let mut a = Autofocus::default();
        let tags: Vec<usize> = (0..10).map(|_| a.next_near().0).collect();
        assert_eq!(tags, [0, 1, 2, 3, 4, 5, 6, 7, 8, 0]);
    }

    #[test]
    fn the_depth_with_the_most_support_wins() {
        let got = best_supported(&[(8.0, 1.0), (8.4, 0.6), (3.0, 1.0)]).unwrap();
        assert!((got - 8.15).abs() < 1e-3, "{got}");
        assert_eq!(best_supported(&[]), None);
    }

    #[test]
    fn until_one_is_found_vertical_offsets_are_tried_in_turn() {
        let mut a = Autofocus::default();
        let mut tried = Vec::new();
        for _ in 0..6 {
            tried.push(a.vertical_offset());
            a.measured(&on_gaze(Reading {
                gradient_variance: 0.0,
                ..reading(6.0)
            }));
        }
        assert_eq!(tried, [0.0, 1.0, -1.0, 2.0, -2.0, 0.0]);
        // One clear match, even two, could be a false one that looked clear.
        for _ in 0..2 {
            a.measured(&on_gaze(Reading {
                vertical_used: 1.0,
                ..reading_at(6.0, [0.0; 2], 0.4)
            }));
        }
        assert_eq!(a.learned_vertical(), None);
        a.measured(&on_gaze(reading_at(6.0, [0.0; 2], 1.45)));
        assert!(
            (a.vertical_offset() - 1.417).abs() < 0.01,
            "{}",
            a.vertical_offset()
        );
    }

    #[test]
    fn the_vertical_offset_is_learned_from_clear_matches() {
        let mut a = Autofocus::default();
        for v in [1.6, -2.2, 1.55, 0.7, 1.65, 1.6, -0.9] {
            a.measured(&on_gaze(reading_at(6.0, [0.0; 2], v)));
        }
        assert!(
            (a.vertical_offset() - 1.6).abs() < 0.02,
            "{}",
            a.vertical_offset()
        );
        // The search was made at some offset: what it found is on top.
        let r = Reading {
            vertical_used: 1.0,
            ..reading_at(6.0, [0.0; 2], 0.5)
        };
        assert!((r.vertical_found().unwrap() - 1.5).abs() < 0.02);
        // Unclear vertical curves teach nothing.
        let mut unclear = reading(6.0);
        unclear.vertical = [0.3; VERTICAL_CANDIDATES];
        assert_eq!(unclear.vertical_found(), None);
        // Readings that can't be trusted teach nothing either.
        let mut b = Autofocus::default();
        b.measured(&Reading {
            gradient_variance: 0.0,
            ..reading_at(6.0, [0.0; 2], 1.0)
        });
        assert_eq!(b.learned_vertical(), None);
        // And it can be switched off.
        a.tuning.vertical_search = false;
        assert_eq!(a.vertical_offset(), 0.0);
    }

    #[test]
    fn a_cut_clears_the_readings_and_the_new_scene_takes_over() {
        let mut a = Autofocus::default();
        agree(&mut a, 8.0);
        settle(&mut a);
        // Two readings at a new depth before the cut...
        a.measured(&on_gaze(reading(3.0)));
        a.measured(&on_gaze(reading(3.0)));
        // ...then a different picture: they're forgotten, so one more isn't
        // enough...
        let mut cut = on_gaze(reading(3.0));
        cut.signature = [0.5; SIGNATURE];
        a.measured(&cut);
        assert_eq!(a.take_stats().cuts, 1);
        assert!((a.focus_distance().unwrap() - distance(8.0, BASELINE)).abs() < 1e-3);
        // ...but after it the new scene needn't be far from the old focus.
        cut.place = [0.0; 2];
        a.measured(&cut);
        a.measured(&cut);
        assert!((a.focus_distance().unwrap() - distance(3.0, BASELINE)).abs() < 1e-3);
    }

    #[test]
    fn a_seek_clears_the_readings() {
        let mut a = Autofocus::default();
        a.measured(&on_gaze(reading(3.0)));
        a.measured(&on_gaze(reading(3.0)));
        a.clear();
        a.measured(&on_gaze(reading(3.0)));
        assert_eq!(a.focus_distance(), None);
        assert_eq!(a.take_stats().clears, 1);
    }

    #[test]
    fn nothing_unknown_moves_the_focus() {
        let mut a = Autofocus::default();
        let mut r = on_gaze(reading(6.0));
        r.place = [f32::NAN, 0.0];
        for _ in 0..AGREEING {
            a.measured(&r);
        }
        assert_eq!(a.focus_distance(), None);
        a.shown.value = f32::NAN;
        assert_eq!(a.advance(1.0 / 90.0), None);
        agree(&mut a, 8.0);
        assert!(settle(&mut a).is_some_and(f32::is_finite));
    }

    #[test]
    fn eyes_on_a_panel_hold_the_focus_and_settle_afresh_after() {
        let mut a = Autofocus::default();
        let ahead = [0.0, 0.0, -1.0];
        for _ in 0..12 {
            a.look(Some(ahead), 1.0 / 90.0);
        }
        agree(&mut a, 8.0);
        let held = settle(&mut a);
        // Two votes in, then a look at a panel: they're forgotten, and
        // nothing counts as the eyes moving.
        a.measured(&on_gaze(reading(3.0)));
        a.measured(&on_gaze(reading(3.0)));
        for _ in 0..45 {
            a.on_ui(1.0 / 90.0);
            a.advance(1.0 / 90.0);
        }
        assert_eq!(settle(&mut a), held);
        // Back on the picture: settling again before anything counts.
        assert!(!a.look(Some(ahead), 1.0 / 90.0));
        a.measured(&on_gaze(reading(3.0)));
        assert_eq!(settle(&mut a), held);
        let s = a.take_stats();
        assert_eq!(s.ui_holds, 1);
        assert!((s.ui_time - 0.5).abs() < 0.01, "{}", s.ui_time);
        assert_eq!(s.blocked[0], 1, "only the reading while settling again");
        assert!(s.to_string().contains("eyes on a panel"), "{s}");
    }

    #[test]
    fn off_is_at_once() {
        let mut a = Autofocus::default();
        agree(&mut a, 8.0);
        assert!(settle(&mut a).is_some());
        a.off();
        assert_eq!(a.advance(1.0 / 90.0), None);
    }

    #[test]
    fn the_log_says_how_long_it_stuck_and_why() {
        let mut a = Autofocus::default();
        for _ in 0..4 {
            a.measured(&on_gaze(Reading {
                gradient_variance: 0.0,
                ..reading(3.0)
            }));
        }
        for _ in 0..90 {
            a.advance(1.0 / 90.0);
        }
        let s = a.take_stats();
        assert_eq!(s.stuck_by, "low detail");
        assert!(s.longest_stuck > 0.9);
        let line = s.to_string();
        assert!(line.contains("low detail 4"), "{line}");
    }

    #[test]
    fn the_focus_eases_in_vergence_not_metres() {
        // From 2 m to 0.5 m: the same rate of vergence throughout, so the
        // metres change slowly far off and quickly close up.
        let mut a = Autofocus::default();
        agree(&mut a, angle(2.0, BASELINE));
        settle(&mut a);
        agree(&mut a, angle(0.5, BASELINE));
        let dt = 1.0 / 90.0;
        let mut angles = Vec::new();
        for _ in 0..90 {
            let d = a.advance(dt).unwrap();
            angles.push(angle(d, BASELINE));
        }
        let steps: Vec<f32> = angles.windows(2).map(|w| (w[1] - w[0]) / dt).collect();
        let cruising = &steps[20..60];
        for s in cruising {
            assert!((s - RATE).abs() < 0.05, "{s}");
        }
    }

    /// Moves `change` degrees at `hz`: how long it took, and the fastest it
    /// went, degrees a second.
    fn move_with(change: f32, rate: f32, acceleration: f32, hz: f32) -> (f32, f32) {
        let mut e = Easing::default();
        let dt = 1.0 / hz;
        let (mut t, mut last, mut fastest) = (0.0, 0.0f32, 0.0f32);
        while (change - last).abs() > 1e-4 {
            let now = e.step(change, dt, rate, acceleration);
            assert!(now <= change + 1e-3, "overshoots: {now}");
            fastest = fastest.max((now - last) / dt);
            last = now;
            t += dt;
            assert!(t < 60.0, "doesn't arrive: {now}");
        }
        (t, fastest)
    }

    /// How long a change takes with constant acceleration up to the rate and
    /// back down: never reaching the rate if it's too short to.
    fn expected_time(change: f32, rate: f32, acceleration: f32) -> f32 {
        if change > rate * rate / acceleration {
            change / rate + rate / acceleration
        } else {
            2.0 * (change / acceleration).sqrt()
        }
    }

    #[test]
    fn rate_and_acceleration_set_how_long_a_change_takes() {
        for (rate, acceleration) in [
            (5.0, 62.5),
            (5.0, 20.0),
            (5.0, 5.0),
            (15.0, 62.5),
            (30.0, 20.0),
            (60.0, 100.0),
        ] {
            for change in [2.0, 10.0] {
                for hz in [72.0, 90.0, 120.0] {
                    let (took, fastest) = move_with(change, rate, acceleration, hz);
                    let want = expected_time(change, rate, acceleration);
                    assert!(
                        (took - want).abs() < 0.05,
                        "{rate} {acceleration} {change} at {hz}: {took}, {want}"
                    );
                    assert!(fastest <= rate * 1.01, "{rate} {acceleration}: {fastest}");
                }
            }
        }
    }

    #[test]
    fn it_starts_and_stops_softly() {
        let mut e = Easing::default();
        let dt = 1.0 / 120.0;
        let (mut last, mut speed, mut biggest_change) = (0.0, 0.0f32, 0.0f32);
        for _ in 0..200 {
            let now = e.step(4.0, dt, 5.0, ACCELERATION);
            let new_speed = (now - last) / dt;
            biggest_change = biggest_change.max((new_speed - speed).abs());
            speed = new_speed;
            last = now;
        }
        assert!((last - 4.0).abs() < 1e-3, "{last}");
        assert!(biggest_change < ACCELERATION * dt * 3.0, "{biggest_change}");
    }

    #[test]
    fn turning_round_goes_on_only_as_far_as_it_has_to() {
        for (speed, acceleration) in [(5.0f32, 62.5f32), (5.0, 10.0), (20.0, 20.0), (60.0, 100.0)] {
            let mut e = Easing::default();
            let dt = 1.0 / 120.0;
            while e.speed < speed - 1e-3 {
                e.step(1000.0, dt, speed, acceleration);
            }
            let from = e.value;
            let (mut furthest, mut was, mut jolt) = (0.0f32, e.speed, 0.0f32);
            for _ in 0..4000 {
                e.step(from - 5.0, dt, speed, acceleration);
                furthest = furthest.max(e.value - from);
                jolt = jolt.max((e.speed - was).abs());
                was = e.speed;
            }
            let needed = speed * speed / (2.0 * acceleration);
            assert!((e.value - (from - 5.0)).abs() < 1e-3, "arrives");
            assert!(
                (furthest - needed).abs() < needed * 0.1 + 0.05,
                "{furthest}, {needed}"
            );
            assert!(jolt <= acceleration / 120.0 * 4.01, "{jolt}");
        }
    }

    #[test]
    fn a_probe_line_says_why() {
        let t = Tuning::default();
        let line = reading_at(6.0, [10.0, 5.0], 1.2).probe(&t);
        assert!(
            line.contains("trusted") && line.contains("+1.20°"),
            "{line}"
        );
        let line = Reading {
            gradient_variance: 0.0,
            ..reading(6.0)
        }
        .probe(&t);
        assert!(line.contains("LowDetail"), "{line}");
    }
}
