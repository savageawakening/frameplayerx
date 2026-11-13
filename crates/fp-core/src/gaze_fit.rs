//! Gaze calibration: where the eye tracker says the eyes look,
//! against targets at known places, fitted as a correction autofocus then
//! applies to the raw gaze.
//!
//! Angles are degrees in the head's frame: yaw right, pitch up. The error at
//! a target is where it is less where the eyes were said to look; it's
//! fitted as an offset, and, with enough targets, slopes in yaw and pitch.

use serde::{Deserialize, Serialize};

/// The targets, degrees right and up from where the head faces: a 3×3 grid,
/// 20° either side, and rows at +14°, −3° and −20°. The Steam Frame's
/// lenses show less above than below, and a target near the top edge can't
/// be held in view; the eyes also rest a little below where the head
/// faces. So the grid sits lower than the head's forward, and still spans
/// 34° up and down and 40° across: enough for the slopes to mean something.
pub const TARGETS: [[f32; 2]; 9] = [
    [-20.0, 14.0],
    [0.0, 14.0],
    [20.0, 14.0],
    [-20.0, -3.0],
    [0.0, -3.0],
    [20.0, -3.0],
    [-20.0, -20.0],
    [0.0, -20.0],
    [20.0, -20.0],
];
/// Targets for checking a fit: places it wasn't made at, inside the same
/// comfortable range.
pub const CHECKS: [[f32; 2]; 4] = [[12.0, 6.0], [-12.0, 6.0], [12.0, -12.0], [-12.0, -12.0]];
/// Seconds after a target shows before the gaze counts (the eyes get there
/// and settle), and seconds of gaze collected.
pub const SETTLE: f32 = 0.8;
pub const COLLECT: f32 = 0.6;
/// How many times the targets are shown: each showing is a point of the fit.
pub const ROUNDS: usize = 2;
/// Samples further than this from a target's median are a blink or a
/// glance away, degrees, and at least this share must be left.
const SAMPLE_REACH: f32 = 1.0;
const SAMPLES_KEPT: f32 = 0.6;
/// The spread of a target's samples past which it's retried (a blink, a
/// look away), degrees.
pub const MAX_SPREAD: f32 = 1.0;
/// Targets needed for slopes as well as an offset.
const FOR_SLOPES: usize = 6;
/// Slopes are only taken if they cut the residual by this much, and none is
/// steeper than this per degree: with a noisy fit, slopes fit the noise.
const SLOPES_GAIN: f32 = 0.7;
const MAX_SLOPE: f32 = 0.08;
/// A residual above this suggests calibrating again, degrees.
pub const GOOD_RMS: f32 = 1.0;
/// A target whose error is this much worse than the fit's typical one, and
/// worse than this, is left out and the fit made again, degrees.
const OUTLIER_TIMES: f32 = 3.0;
const OUTLIER_FLOOR: f32 = 1.0;

/// A direction in the head's frame as yaw and pitch, degrees.
pub fn angles(d: [f32; 3]) -> [f32; 2] {
    [
        d[0].atan2(-d[2]).to_degrees(),
        d[1].clamp(-1.0, 1.0).asin().to_degrees(),
    ]
}

/// The direction of a yaw and pitch, degrees.
pub fn direction([yaw, pitch]: [f32; 2]) -> [f32; 3] {
    let (y, p) = (yaw.to_radians(), pitch.to_radians());
    [p.cos() * y.sin(), p.sin(), -p.cos() * y.cos()]
}

/// One target's gaze samples, as yaw and pitch.
#[derive(Clone, Debug, Default)]
pub struct Samples(pub Vec<[f32; 2]>);

fn median(v: &mut [f32]) -> f32 {
    v.sort_by(f32::total_cmp);
    v[v.len() / 2]
}

impl Samples {
    /// Their median, without those further than [`SAMPLE_REACH`] from it,
    /// and the spread of those left; `None` if too few are left.
    pub fn robust(&self) -> Option<([f32; 2], f32)> {
        let n = self.0.len();
        if n == 0 {
            return None;
        }
        let mid = [0, 1].map(|k| median(&mut self.0.iter().map(|s| s[k]).collect::<Vec<_>>()));
        let kept = Samples(
            self.0
                .iter()
                .copied()
                .filter(|s| (s[0] - mid[0]).hypot(s[1] - mid[1]) <= SAMPLE_REACH)
                .collect(),
        );
        if (kept.0.len() as f32) < SAMPLES_KEPT * n as f32 {
            return None;
        }
        let (_, spread) = kept.mean_spread()?;
        let at = [0, 1].map(|k| median(&mut kept.0.iter().map(|s| s[k]).collect::<Vec<_>>()));
        Some((at, spread))
    }

    /// Their mean and their spread (root mean square distance from it).
    pub fn mean_spread(&self) -> Option<([f32; 2], f32)> {
        let n = self.0.len();
        if n == 0 {
            return None;
        }
        let mean = [0, 1].map(|k| self.0.iter().map(|s| s[k]).sum::<f32>() / n as f32);
        let sq: f32 = self
            .0
            .iter()
            .map(|s| (s[0] - mean[0]).powi(2) + (s[1] - mean[1]).powi(2))
            .sum();
        Some((mean, (sq / n as f32).sqrt()))
    }
}

/// The fitted correction: added to where the eyes are said to look.
#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct GazeFit {
    /// The error at the middle, degrees: yaw and pitch.
    pub offset: [f32; 2],
    /// How the yaw and pitch error change per degree of yaw, and per degree
    /// of pitch.
    pub per_yaw: [f32; 2],
    pub per_pitch: [f32; 2],
    /// The residual left, degrees, as fitted.
    pub rms: f32,
}

impl GazeFit {
    /// The correction at a gaze of `at` (yaw, pitch), degrees.
    pub fn error_at(&self, at: [f32; 2]) -> [f32; 2] {
        [0, 1].map(|k| self.offset[k] + self.per_yaw[k] * at[0] + self.per_pitch[k] * at[1])
    }

    /// A raw gaze direction in the head's frame, corrected.
    pub fn correct(&self, d: [f32; 3]) -> [f32; 3] {
        let at = angles(d);
        let e = self.error_at(at);
        direction([at[0] + e[0], at[1] + e[1]])
    }
}

/// One target in a fit: where it is, the error before and after the fit,
/// and whether it was used.
pub type TargetFit = ([f32; 2], [f32; 2], [f32; 2], bool);

/// How a fit came out.
#[derive(Clone, Debug, Default)]
pub struct FitReport {
    pub fit: GazeFit,
    /// Root mean square of the errors left, degrees, over the targets used.
    pub rms: f32,
    /// Each target: where it is, the error before and after the fit, and
    /// whether it was used.
    pub targets: Vec<TargetFit>,
    pub slopes: bool,
}

impl std::fmt::Display for FitReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let g = &self.fit;
        write!(
            f,
            "offset yaw {:+.2}° pitch {:+.2}°; {}; residual RMS {:.2}° over {} targets{}",
            g.offset[0],
            g.offset[1],
            if self.slopes {
                format!(
                    "per degree of yaw {:+.3} {:+.3}, of pitch {:+.3} {:+.3}",
                    g.per_yaw[0], g.per_yaw[1], g.per_pitch[0], g.per_pitch[1]
                )
            } else {
                "offset only (slopes didn't earn their place)".into()
            },
            self.rms,
            self.targets.iter().filter(|t| t.3).count(),
            if self.rms > GOOD_RMS {
                "; worth calibrating again"
            } else {
                ""
            },
        )?;
        for (at, before, after, used) in &self.targets {
            write!(
                f,
                "; at {:+.0}° {:+.0}°: {:+.2}° {:+.2}° → {:+.2}° {:+.2}°{}",
                at[0],
                at[1],
                before[0],
                before[1],
                after[0],
                after[1],
                if *used { "" } else { " (left out)" }
            )?;
        }
        Ok(())
    }
}

/// Least squares of `y = a + b·u + c·v` over `(u, v, y)`, or `y = a`.
fn solve(points: &[(f32, f32, f32)], slopes: bool) -> [f32; 3] {
    let n = points.len() as f64;
    if !slopes || points.len() < FOR_SLOPES {
        let a = points.iter().map(|p| p.2 as f64).sum::<f64>() / n.max(1.0);
        return [a as f32, 0.0, 0.0];
    }
    // Normal equations, in double precision.
    let mut m = [[0.0f64; 4]; 3];
    for &(u, v, y) in points {
        let x = [1.0, u as f64, v as f64];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] += x[i] * x[j];
            }
            m[i][3] += x[i] * y as f64;
        }
    }
    for c in 0..3 {
        let pivot = (c..3)
            .max_by(|a, b| m[*a][c].abs().total_cmp(&m[*b][c].abs()))
            .unwrap_or(c);
        m.swap(c, pivot);
        if m[c][c].abs() < 1e-9 {
            return solve(points, false);
        }
        for r in 0..3 {
            if r != c {
                let k = m[r][c] / m[c][c];
                let pivot_row = m[c];
                for (x, p) in m[r].iter_mut().zip(pivot_row).skip(c) {
                    *x -= k * p;
                }
            }
        }
    }
    [0, 1, 2].map(|i| (m[i][3] / m[i][i]) as f32)
}

/// Fits the correction from each target (where it is) and where the eyes
/// were said to look at it; leaves out targets far worse than the rest and
/// fits again. Slopes are taken only where they clearly cut the residual
/// and none is steeper than [`MAX_SLOPE`]; otherwise the offset alone.
pub fn fit(seen: &[([f32; 2], [f32; 2])]) -> Option<FitReport> {
    if seen.is_empty() {
        return None;
    }
    let residual = |g: &GazeFit, (target, gaze): &([f32; 2], [f32; 2])| {
        let e = g.error_at(*gaze);
        [target[0] - gaze[0] - e[0], target[1] - gaze[1] - e[1]]
    };
    let size = |r: [f32; 2]| (r[0] * r[0] + r[1] * r[1]).sqrt();
    let rms_of = |g: &GazeFit, used: &[bool]| {
        let (sum, n) = seen
            .iter()
            .zip(used)
            .filter(|(_, u)| **u)
            .fold((0.0, 0), |(s, n), (t, _)| {
                (s + size(residual(g, t)).powi(2), n + 1)
            });
        (sum / n.max(1) as f32).sqrt()
    };
    let make_with = |used: &[bool], slopes: bool| {
        let pts: Vec<_> = seen
            .iter()
            .zip(used)
            .filter(|(_, u)| **u)
            .map(|(s, _)| *s)
            .collect();
        let slopes = slopes && pts.len() >= FOR_SLOPES;
        let k = |axis: usize| {
            let p: Vec<(f32, f32, f32)> = pts
                .iter()
                .map(|(target, gaze)| (gaze[0], gaze[1], target[axis] - gaze[axis]))
                .collect();
            solve(&p, slopes)
        };
        let (y, p) = (k(0), k(1));
        (
            GazeFit {
                offset: [y[0], p[0]],
                per_yaw: [y[1], p[1]],
                per_pitch: [y[2], p[2]],
                rms: 0.0,
            },
            slopes,
        )
    };
    let make = |used: &[bool]| {
        let (offset, _) = make_with(used, false);
        let (sloped, slopes) = make_with(used, true);
        let sane = sloped
            .per_yaw
            .iter()
            .chain(&sloped.per_pitch)
            .all(|s| s.abs() <= MAX_SLOPE);
        if slopes && sane && rms_of(&sloped, used) < SLOPES_GAIN * rms_of(&offset, used) {
            (sloped, true)
        } else {
            (offset, false)
        }
    };
    let mut used = vec![true; seen.len()];
    let (mut g, mut slopes) = make(&used);
    let sizes: Vec<f32> = seen.iter().map(|s| size(residual(&g, s))).collect();
    let mut sorted = sizes.clone();
    sorted.sort_by(f32::total_cmp);
    let typical = sorted[sorted.len() / 2];
    let mut dropped = false;
    for (u, s) in used.iter_mut().zip(&sizes) {
        if *s > OUTLIER_FLOOR && *s > OUTLIER_TIMES * typical {
            *u = false;
            dropped = true;
        }
    }
    if dropped && used.iter().any(|u| *u) {
        (g, slopes) = make(&used);
    }
    g.rms = rms_of(&g, &used);
    let targets = seen
        .iter()
        .zip(&used)
        .map(|(s, u)| (s.0, [s.0[0] - s.1[0], s.0[1] - s.1[1]], residual(&g, s), *u))
        .collect();
    Some(FitReport {
        fit: g,
        rms: g.rms,
        targets,
        slopes,
    })
}

/// Where calibrating or checking is, frame by frame.
#[derive(Clone, Debug)]
pub struct Routine {
    /// The targets, and which is showing.
    pub targets: Vec<[f32; 2]>,
    pub index: usize,
    /// Seconds since it showed, and whether it has been retried.
    pub time: f32,
    pub retried: bool,
    pub samples: Samples,
    /// Each target done: where it is, and the mean gaze at it.
    pub seen: Vec<([f32; 2], [f32; 2])>,
    /// Targets given up on.
    pub skipped: u32,
}

/// What a frame of the routine came to.
#[derive(Clone, Debug, PartialEq)]
pub enum Step {
    /// Keep showing the target at this place.
    Show([f32; 2]),
    /// Every target done.
    Done,
}

impl Routine {
    pub fn new(targets: &[[f32; 2]]) -> Routine {
        Routine {
            targets: targets.to_vec(),
            index: 0,
            time: 0.0,
            retried: false,
            samples: Samples::default(),
            seen: Vec::new(),
            skipped: 0,
        }
    }

    /// A frame `dt` long with the raw gaze, in the head's frame, if tracked.
    pub fn step(&mut self, gaze: Option<[f32; 3]>, dt: f32) -> Step {
        let Some(&target) = self.targets.get(self.index) else {
            return Step::Done;
        };
        self.time += dt;
        if self.time > SETTLE
            && let Some(g) = gaze
        {
            self.samples.0.push(angles(g));
        }
        if self.time >= SETTLE + COLLECT {
            match self.samples.robust() {
                Some((mean, spread)) if spread <= MAX_SPREAD && self.samples.0.len() >= 10 => {
                    self.seen.push((target, mean));
                    self.next();
                }
                _ if !self.retried => {
                    self.retried = true;
                    self.time = 0.0;
                    self.samples = Samples::default();
                }
                _ => {
                    self.skipped += 1;
                    self.next();
                }
            }
        }
        match self.targets.get(self.index) {
            Some(&t) => Step::Show(t),
            None => Step::Done,
        }
    }

    fn next(&mut self) {
        self.index += 1;
        self.time = 0.0;
        self.retried = false;
        self.samples = Samples::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rand(seed: &mut u32) -> f32 {
        *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        (*seed >> 8) as f32 / (1u32 << 24) as f32 - 0.5
    }

    /// Where a tracker with error `truth` says the eyes look at `target`:
    /// the fit's model run backwards, by iteration, and some noise.
    fn tracked(truth: &GazeFit, target: [f32; 2], noise: f32, seed: &mut u32) -> [f32; 2] {
        let mut g = target;
        for _ in 0..20 {
            let e = truth.error_at(g);
            g = [target[0] - e[0], target[1] - e[1]];
        }
        [g[0] + noise * rand(seed), g[1] + noise * rand(seed)]
    }

    #[test]
    fn an_offset_and_slopes_are_recovered_from_noisy_targets() {
        let truth = GazeFit {
            offset: [-1.5, 3.0],
            per_yaw: [0.04, -0.02],
            per_pitch: [0.01, 0.05],
            rms: 0.0,
        };
        let mut seed = 7;
        let seen: Vec<_> = TARGETS
            .iter()
            .map(|&t| (t, tracked(&truth, t, 0.3, &mut seed)))
            .collect();
        let r = fit(&seen).unwrap();
        assert!(r.slopes);
        for k in 0..2 {
            assert!((r.fit.offset[k] - truth.offset[k]).abs() < 0.2, "{r}");
            assert!((r.fit.per_yaw[k] - truth.per_yaw[k]).abs() < 0.01, "{r}");
            assert!(
                (r.fit.per_pitch[k] - truth.per_pitch[k]).abs() < 0.01,
                "{r}"
            );
        }
        assert!(r.rms < 0.3, "{r}");
        // Corrected, the gaze at a target lands on it.
        let at = tracked(&truth, [18.0, -10.0], 0.0, &mut seed);
        let fixed = angles(r.fit.correct(direction(at)));
        assert!(
            (fixed[0] - 18.0).abs() < 0.3 && (fixed[1] + 10.0).abs() < 0.3,
            "{fixed:?}"
        );
    }

    #[test]
    fn slopes_that_only_fit_the_noise_are_left_out() {
        // An offset and noise, nothing more: slopes would fit the noise.
        let truth = GazeFit {
            offset: [-2.0, 1.5],
            ..Default::default()
        };
        let mut seed = 11;
        let seen: Vec<_> = TARGETS
            .iter()
            .map(|&t| (t, tracked(&truth, t, 2.0, &mut seed)))
            .collect();
        let r = fit(&seen).unwrap();
        assert!(!r.slopes, "{r}");
        assert_eq!(r.fit.per_yaw, [0.0, 0.0]);
        assert!(r.to_string().contains("offset only"), "{r}");
        // Slopes steeper than any sane tracker's are refused too.
        let steep = GazeFit {
            per_yaw: [0.15, 0.0],
            ..Default::default()
        };
        let seen: Vec<_> = TARGETS
            .iter()
            .map(|&t| (t, tracked(&steep, t, 0.05, &mut seed)))
            .collect();
        assert!(!fit(&seen).unwrap().slopes);
    }

    #[test]
    fn a_blink_in_a_targets_samples_is_left_out_of_it() {
        let mut s = Samples(
            (0..40)
                .map(|i| [1.0 + 0.01 * (i % 5) as f32, -2.0])
                .collect(),
        );
        for _ in 0..8 {
            s.0.push([15.0, 9.0]);
        }
        let (at, spread) = s.robust().unwrap();
        assert!(
            (at[0] - 1.02).abs() < 0.02 && (at[1] + 2.0).abs() < 1e-4,
            "{at:?}"
        );
        assert!(spread < 0.05, "{spread}");
        // Mostly away: nothing to go on.
        let away = Samples((0..10).map(|i| [i as f32 * 3.0, 0.0]).collect());
        assert!(away.robust().is_none());
    }

    #[test]
    fn a_target_looked_away_from_is_left_out() {
        let truth = GazeFit {
            offset: [1.0, -2.0],
            ..Default::default()
        };
        let mut seed = 3;
        let mut seen: Vec<_> = TARGETS
            .iter()
            .map(|&t| (t, tracked(&truth, t, 0.2, &mut seed)))
            .collect();
        seen[6].1[0] += 9.0;
        let r = fit(&seen).unwrap();
        assert!(!r.targets[6].3, "{r}");
        assert!(
            (r.fit.offset[0] - 1.0).abs() < 0.2 && (r.fit.offset[1] + 2.0).abs() < 0.2,
            "{r}"
        );
        assert!(r.to_string().contains("(left out)"));
    }

    #[test]
    fn few_targets_fit_an_offset_only() {
        let seen = [([0.0, 0.0], [1.0, -1.0]), ([12.0, 0.0], [13.0, -1.0])];
        let r = fit(&seen).unwrap();
        assert!(!r.slopes);
        assert_eq!(r.fit.offset, [-1.0, 1.0]);
        assert_eq!(r.fit.per_yaw, [0.0, 0.0]);
    }

    #[test]
    fn the_routine_waits_collects_and_retries_a_blink_once() {
        let mut r = Routine::new(&TARGETS[..2]);
        let dt = 1.0 / 90.0;
        let at = |t: [f32; 2]| Some(direction([t[0] + 1.0, t[1] - 2.0]));
        // The first target: the eyes there throughout.
        let mut frames = 0;
        while r.index == 0 {
            let step = r.step(at(TARGETS[0]), dt);
            let showing = TARGETS[r.index.min(1)];
            assert_eq!(step, Step::Show(showing));
            frames += 1;
        }
        let want = ((SETTLE + COLLECT) / dt) as i32;
        assert!((frames - want).abs() <= 2, "{frames}");
        // The second: a look away the first time, then steady.
        let mut first = true;
        loop {
            let g = if first && r.time > SETTLE + 0.5 {
                Some(direction([40.0, 30.0]))
            } else {
                at(TARGETS[1])
            };
            if r.retried {
                first = false;
            }
            if r.step(g, dt) == Step::Done {
                break;
            }
        }
        assert_eq!(r.seen.len(), 2);
        assert_eq!(r.skipped, 0);
        for (t, mean) in &r.seen {
            assert!((mean[0] - t[0] - 1.0).abs() < 0.01 && (mean[1] - t[1] + 2.0).abs() < 0.01);
        }
        // Lost every time: skipped after one retry.
        let mut r = Routine::new(&TARGETS[..1]);
        while r.step(None, dt) != Step::Done {}
        assert_eq!((r.seen.len(), r.skipped), (0, 1));
    }

    #[test]
    fn every_target_is_comfortably_in_view_and_the_grid_spans_enough() {
        for t in TARGETS.iter().chain(&CHECKS) {
            assert!(t[0].abs() <= 20.0, "{t:?}");
            assert!(t[1] <= 14.0 && t[1] >= -20.0, "{t:?}");
        }
        let pitch: Vec<f32> = TARGETS.iter().map(|t| t[1]).collect();
        let span = pitch.iter().cloned().fold(f32::MIN, f32::max)
            - pitch.iter().cloned().fold(f32::MAX, f32::min);
        assert!(span >= 30.0, "{span}");
        // Checks are where calibrating didn't look.
        assert!(CHECKS.iter().all(|c| !TARGETS.contains(c)));
    }

    #[test]
    fn angles_and_directions_are_each_others_inverse() {
        for a in [[0.0, 0.0], [24.0, -24.0], [-12.0, 30.0]] {
            let b = angles(direction(a));
            assert!((a[0] - b[0]).abs() < 1e-4 && (a[1] - b[1]).abs() < 1e-4);
        }
    }
}
