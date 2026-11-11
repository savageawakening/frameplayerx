//! Autofocus: the distance a 3D 180° picture is drawn at (fp_render's
//! `Focus`), so that what the eyes are on is seen at its own depth whatever
//! the viewer's IPD. The depth is measured by [`crate::epipolar`]; this
//! keeps the eyes' settling, takes the depths readings agree on, and eases
//! the drawn distance towards them.
//!
//! Refocusing only past a share of the distance is adapted from Stereonaut's
//! `src/player/autofocus.c`, Copyright (c) 2026 Erdnusschokolade, MIT License
//! (`packaging/licenses/Stereonaut-MIT.txt`).

use serde::{Deserialize, Serialize};

/// The cameras' separation assumed until set, metres.
pub const BASELINE: f32 = 0.064;
/// The nearest the picture is drawn, metres (fp_render's own limit too).
pub const MIN_DISTANCE: f32 = 0.1;
/// How far, as a share of the distance, a new depth must be from the focus to
/// refocus, so that noise in the measurement doesn't move the picture; it's
/// reckoned in vergence, where infinity is no different, and never less
/// than a small step.
const HYSTERESIS: f32 = 0.15;
const MIN_STEP: f32 = 0.15;
/// Beyond the rate, the drawn distance glides at this share of what's left
/// a second, up to this many times the rate: a big change doesn't crawl,
/// and a small one stays gentle.
const GLIDE: f32 = 2.0;
const MAX_GLIDE: f32 = 6.0;
/// The eyes are on the way somewhere (a saccade) when their speed, from the
/// gaze as tracked (not smoothed), passes this, degrees a second; slower and
/// steady, they're following something (pursuit), and that's measured along.
const SACCADE_SPEED: f32 = 60.0;
/// Seconds the speed is smoothed over: a sample or two, not the 1 Hz the
/// gaze smoothing has, which would spread a saccade into a slow glide.
const SPEED_SMOOTHING: f32 = 0.02;
/// The farthest a depth is taken to be, metres: past it (sky, far scenery,
/// the far end of the search) the focus settles here rather than at
/// infinity, so looking up at the sky and back doesn't ease all the way out
/// and in again. Here the IPD offset is nearly all back (fp_render).
pub const FAR_LIMIT: f32 = 25.0;
/// Readings in a row within a step of each other before a held focus creeps
/// towards them, and the share of the way it creeps with each one after:
/// the hysteresis then can't hold the picture off the measured depth for
/// good.
const CREEP_AFTER: u32 = 3;
const CREEP_SHARE: f32 = 0.2;
/// Degrees of vergence a second the drawn distance moves at, and degrees a
/// second per second it speeds up and slows down at.
pub const RATE: f32 = 5.0;
pub const ACCELERATION: f32 = 62.5;
/// The eyes have settled once they've moved no faster than a pursuit for this
/// long; a jump of this many degrees from one frame to the next starts
/// again whatever the speed.
pub const SETTLE_DEGREES: f32 = 2.0;
const SETTLE_SECONDS: f32 = 0.1;
/// After the eyes are lost, seconds before the head takes over: a blink
/// shouldn't refocus on whatever the head happens to point at.
const EYES_LOST_SECONDS: f32 = 0.3;
/// Times to lock kept for the log.
const LOCK_TIMES: usize = 64;

/// How autofocus moves; the defaults are the feature's.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Tuning {
    /// Degrees of vergence a second the drawn distance moves at, however
    /// large the change, and degrees a second per second it gets there.
    pub rate: f32,
    pub acceleration: f32,
    /// How far apart the cameras were, metres.
    pub baseline: f32,
    /// A jump between frames that starts settling again, degrees.
    pub settle_degrees: f32,
}

impl Tuning {
    /// Within what the feature can run with: settings.json isn't checked
    /// when it's read, and a negative rate or baseline would stop the
    /// easing from making sense (or panic a clamp).
    pub fn sanitized(self) -> Tuning {
        let d = Tuning::default();
        let within = |v: f32, lo: f32, hi: f32, or: f32| {
            if v.is_finite() { v.max(lo).min(hi) } else { or }
        };
        Tuning {
            rate: within(self.rate, 0.1, 90.0, d.rate),
            acceleration: within(self.acceleration, 0.1, 2000.0, d.acceleration),
            baseline: within(self.baseline, 0.01, 0.5, d.baseline),
            settle_degrees: within(self.settle_degrees, 0.1, 30.0, d.settle_degrees),
        }
    }
}

/// Whether a depth `angle` is far enough from the focus `focus` (both
/// vergence, degrees) to move it: past a share of the larger, and never
/// less than a small step.
pub fn moves_focus(focus: f32, angle: f32) -> bool {
    (angle - focus).abs() > (HYSTERESIS * angle.max(focus)).max(MIN_STEP)
}

impl Default for Tuning {
    fn default() -> Self {
        Tuning {
            rate: RATE,
            acceleration: ACCELERATION,
            baseline: BASELINE,
            settle_degrees: SETTLE_DEGREES,
        }
    }
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
        // Towards what it wants within the acceleration, slowing down too: a
        // lower rate (a new, nearer target mid-glide) is reached by
        // decelerating, not at once.
        self.speed += (wanted - self.speed).clamp(-accel * dt, accel * dt);
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

/// How autofocus has gone since it was last asked, for the log.
#[derive(Clone, Copy, Debug)]
pub struct Stats {
    /// Depths locked, and how many moved the focus.
    pub locks: u32,
    pub focus_changes: u32,
    pub clears: u32,
    /// Frames the eyes were moving, unsettled.
    pub eyes_moving: u32,
    /// Seconds in all, seconds the eyes were on a panel rather than the
    /// picture, and how often they went there.
    pub time: f32,
    pub ui_time: f32,
    pub ui_holds: u32,
    /// Seconds from the eyes moving to the next lock, the latest of them.
    pub lock_times: [f32; LOCK_TIMES],
    pub lock_count: usize,
}

impl Default for Stats {
    fn default() -> Self {
        Stats {
            locks: 0,
            focus_changes: 0,
            clears: 0,
            eyes_moving: 0,
            time: 0.0,
            ui_time: 0.0,
            ui_holds: 0,
            lock_times: [0.0; LOCK_TIMES],
            lock_count: 0,
        }
    }
}

impl Stats {
    /// The median and the 90th percentile of the times to lock, seconds.
    pub fn time_to_lock(&self) -> Option<(f32, f32)> {
        let n = self.lock_count.min(LOCK_TIMES);
        if n == 0 {
            return None;
        }
        let mut t = self.lock_times;
        let t = &mut t[..n];
        t.sort_by(f32::total_cmp);
        let at = |q: f32| t[((n - 1) as f32 * q).round() as usize];
        Some((at(0.5), at(0.9)))
    }
}

impl std::fmt::Display for Stats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} locks, {} refocus, {} cleared, {} frames with the eyes moving; ",
            self.locks, self.focus_changes, self.clears, self.eyes_moving
        )?;
        match self.time_to_lock() {
            Some((median, p90)) => write!(
                f,
                "time to lock after a move median {median:.2} s, 90% {p90:.2} s ({} moves)",
                self.lock_count
            )?,
            None => write!(f, "no locks after a move")?,
        }
        write!(
            f,
            "; eyes on a panel {:.0}% of the time, {} times",
            100.0 * self.ui_time / self.time.max(1e-6),
            self.ui_holds
        )
    }
}

/// The state of one open video.
#[derive(Clone, Copy, Debug, Default)]
pub struct Autofocus {
    pub tuning: Tuning,
    /// The depth readings last agreed on: an angle straight ahead, degrees.
    focus: Option<f32>,
    /// The angle drawn, moving towards the focus.
    shown: Easing,
    /// Where the eyes were last frame, how fast they're going (degrees a
    /// second, smoothed), and how long they've been no faster than a pursuit.
    anchor: Option<[f32; 3]>,
    speed: f32,
    held: f32,
    /// Seconds since the gaze last changed: a tracker slower than the
    /// display repeats a sample, and the next step is over all of them.
    since_sample: f32,
    /// The last reading that didn't move the focus, and how many in a row
    /// were within a step of each other.
    last_reading: Option<f32>,
    steady: u32,
    /// Seconds since the eyes were lost, until the head takes over.
    lost: Option<f32>,
    /// The eyes are on a panel rather than the picture.
    on_ui: bool,
    /// Cleared since the last lock: the next moves the focus however close.
    cleared: bool,
    /// Seconds since the eyes moved, until the next lock.
    since_move: Option<f32>,
    stats: Stats,
}

impl Autofocus {
    /// Takes this frame's gaze, a unit direction, if the eyes are tracked;
    /// whether it may be measured along. Without eye tracking, always.
    pub fn look(&mut self, gaze: Option<[f32; 3]>, dt: f32) -> bool {
        self.on_ui = false;
        let settled = match gaze {
            Some(now) => {
                self.lost = None;
                let step = self.anchor.map(|at| degrees_between(at, now));
                match step {
                    // The same sample again: the speed waits for the next.
                    Some(step) if step < 1e-4 => self.since_sample += dt,
                    Some(step) => {
                        let over = (self.since_sample + dt).max(1e-4);
                        let k = 1.0 - (-over / SPEED_SMOOTHING).exp();
                        self.speed += k * (step / over - self.speed);
                        self.since_sample = 0.0;
                    }
                    None => self.since_sample = 0.0,
                }
                self.anchor = Some(now);
                let settle_degrees = self.tuning.sanitized().settle_degrees;
                let saccade =
                    step.is_none_or(|step| step > settle_degrees || self.speed > SACCADE_SPEED);
                if saccade {
                    if self.held > 0.0 || step.is_none() {
                        self.since_move = Some(0.0);
                    }
                    self.held = 0.0;
                } else {
                    self.held += dt;
                }
                self.held >= SETTLE_SECONDS
            }
            None => {
                if self.anchor.take().is_some() {
                    self.lost = Some(0.0);
                    self.speed = 0.0;
                }
                self.lost = self.lost.map(|t| t + dt).filter(|t| *t < EYES_LOST_SECONDS);
                self.lost.is_none()
            }
        };
        if !settled {
            self.stats.eyes_moving += 1;
        }
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
        self.speed = 0.0;
        self.lost = None;
        self.held = 0.0;
        self.since_move = None;
    }

    /// A depth readings agreed on, as an angle straight ahead, degrees. It
    /// moves the focus if it's far enough from it, or after a clear.
    pub fn lock(&mut self, angle: f32) {
        if !angle.is_finite() {
            return;
        }
        let t = self.tuning.sanitized();
        let angle = angle.max(self::angle(FAR_LIMIT, t.baseline));
        self.stats.locks += 1;
        if let Some(t) = self.since_move.take() {
            self.stats.lock_times[self.stats.lock_count % LOCK_TIMES] = t;
            self.stats.lock_count += 1;
        }
        let moved = std::mem::take(&mut self.cleared)
            || self.focus.is_none_or(|focus| moves_focus(focus, angle));
        if moved {
            self.focus = Some(angle);
            self.stats.focus_changes += 1;
            self.last_reading = None;
            self.steady = 0;
            return;
        }
        // Held: readings steady somewhere else draw it there slowly.
        self.steady = match self.last_reading {
            Some(last) if (angle - last).abs() <= MIN_STEP => self.steady + 1,
            _ => 1,
        };
        self.last_reading = Some(angle);
        if self.steady >= CREEP_AFTER
            && let Some(focus) = self.focus.as_mut()
        {
            *focus += CREEP_SHARE * (angle - *focus);
        }
    }

    /// The focus as an angle, degrees, as the readings agreed.
    pub fn focus_angle(&self) -> Option<f32> {
        self.focus
    }

    /// Autofocus turned off (or the video can't be measured any more): the
    /// drawn distance eases out to infinity, where the picture is as
    /// without it, rather than snapping there.
    pub fn release(&mut self) {
        self.focus = None;
        self.cleared = false;
        self.last_reading = None;
        self.steady = 0;
    }

    /// For a seek or a cut: the picture now may be anywhere. The drawn
    /// distance stays until readings agree again, and they needn't be far
    /// from it to move it.
    pub fn clear(&mut self) {
        self.cleared = true;
        self.stats.clears += 1;
    }

    /// How far away the focused subject is, metres, as the readings agreed.
    pub fn focus_distance(&self) -> Option<f32> {
        self.focus
            .map(|a| distance(a, self.tuning.sanitized().baseline).max(MIN_DISTANCE))
    }

    /// What has happened since the last call.
    pub fn take_stats(&mut self) -> Stats {
        std::mem::take(&mut self.stats)
    }

    /// Back to no autofocus at once: the picture is drawn as without it.
    pub fn off(&mut self) {
        self.focus = None;
        self.shown = Easing::default();
        self.last_reading = None;
        self.steady = 0;
    }

    /// Moves the drawn distance towards the focus by `dt` seconds, and returns
    /// it, metres: `None` while it's at infinity, where the picture is drawn
    /// as without autofocus.
    pub fn advance(&mut self, dt: f32) -> Option<f32> {
        self.stats.time += dt;
        if let Some(t) = &mut self.since_move {
            *t += dt;
            if *t > 10.0 {
                self.since_move = None;
            }
        }
        let t = self.tuning.sanitized();
        if !(self.shown.value.is_finite() && self.shown.speed.is_finite()) {
            self.shown = Easing::default();
        }
        let nearest = angle(MIN_DISTANCE, t.baseline);
        let target = self.focus.map_or(0.0, |a| a.clamp(0.0, nearest));
        let gap = (target - self.shown.value).abs();
        // Never a clamp that panics: the tuning is sanitized, and max/min
        // don't mind their order.
        let rate = (GLIDE * gap).max(t.rate).min(MAX_GLIDE * t.rate);
        let shown = self.shown.step(target, dt, rate, t.acceleration);
        (shown > 1e-4).then(|| distance(shown, t.baseline).max(MIN_DISTANCE))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn settle(a: &mut Autofocus) -> Option<f32> {
        for _ in 0..1500 {
            a.advance(1.0 / 90.0);
        }
        a.advance(0.0)
    }

    const AHEAD: [f32; 3] = [0.0, 0.0, -1.0];

    fn settled(a: &mut Autofocus) {
        for _ in 0..12 {
            a.look(Some(AHEAD), 1.0 / 90.0);
        }
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

    #[test]
    fn with_nothing_locked_the_picture_is_drawn_as_without_autofocus() {
        let mut a = Autofocus::default();
        assert_eq!(settle(&mut a), None);
    }

    #[test]
    fn a_lock_draws_the_picture_at_its_distance() {
        let mut a = Autofocus::default();
        a.lock(angle(0.6, BASELINE));
        let got = settle(&mut a).unwrap();
        assert!((got - 0.6).abs() < 1e-3, "{got}");
    }

    #[test]
    fn the_baseline_scales_the_distance() {
        let mut a = Autofocus::default();
        a.lock(angle(0.5, BASELINE));
        a.tuning.baseline = 0.032;
        assert!((a.focus_distance().unwrap() - 0.25).abs() < 1e-3);
    }

    #[test]
    fn never_nearer_than_a_tenth_of_a_metre() {
        let mut a = Autofocus::default();
        a.tuning.baseline = 0.05;
        a.lock(34.0);
        let got = settle(&mut a).unwrap();
        assert!((got - MIN_DISTANCE).abs() < 1e-4, "{got}");
    }

    #[test]
    fn small_movements_keep_the_focus_and_a_clear_lets_them_through() {
        let mut a = Autofocus::default();
        a.lock(angle(0.45, BASELINE));
        let held = settle(&mut a).unwrap();
        // 42 cm is within 15% of 45.
        a.lock(angle(0.42, BASELINE));
        assert_eq!(settle(&mut a), Some(held));
        a.lock(angle(0.33, BASELINE));
        assert!((settle(&mut a).unwrap() - 0.33).abs() < 1e-3);
        a.clear();
        a.lock(angle(0.35, BASELINE));
        assert!((settle(&mut a).unwrap() - 0.35).abs() < 1e-3);
        assert_eq!(a.take_stats().focus_changes, 3);
    }

    #[test]
    fn nothing_unknown_moves_the_focus() {
        let mut a = Autofocus::default();
        a.lock(f32::NAN);
        assert_eq!(a.focus_distance(), None);
        a.shown.value = f32::NAN;
        assert_eq!(a.advance(1.0 / 90.0), None);
    }

    #[test]
    fn eyes_on_the_move_settle_before_they_count() {
        let mut a = Autofocus::default();
        assert!(!a.look(Some(AHEAD), 1.0 / 90.0));
        settled(&mut a);
        assert!(a.look(Some(AHEAD), 1.0 / 90.0));
        let away = [10f32.to_radians().sin(), 0.0, -10f32.to_radians().cos()];
        assert!(!a.look(Some(away), 1.0 / 90.0), "a jump starts again");
    }

    #[test]
    fn a_blink_does_not_hand_over_to_the_head_at_once() {
        let mut a = Autofocus::default();
        settled(&mut a);
        assert!(!a.look(None, 0.1), "just lost");
        assert!(!a.look(None, 0.1));
        assert!(a.look(None, 0.15), "lost long enough for the head to lead");
    }

    #[test]
    fn eyes_on_a_panel_hold_and_settle_afresh_after() {
        let mut a = Autofocus::default();
        settled(&mut a);
        a.lock(8.0);
        let held = settle(&mut a);
        a.take_stats();
        for _ in 0..45 {
            a.on_ui(1.0 / 90.0);
            a.advance(1.0 / 90.0);
        }
        assert_eq!(settle(&mut a), held);
        assert!(!a.look(Some(AHEAD), 1.0 / 90.0));
        let s = a.take_stats();
        assert_eq!(s.ui_holds, 1);
        assert!((s.ui_time - 0.5).abs() < 0.01, "{}", s.ui_time);
        assert_eq!(s.eyes_moving, 1, "only settling again afterwards");
        assert!(s.to_string().contains("eyes on a panel"), "{s}");
    }

    #[test]
    fn the_time_from_a_move_to_a_lock_is_logged() {
        let mut a = Autofocus::default();
        for (i, wait) in [0.2f32, 0.5, 1.5].into_iter().enumerate() {
            let r = (10.0 * (i + 1) as f32).to_radians();
            let away = [r.sin(), 0.0, -r.cos()];
            a.look(Some(away), 1.0 / 90.0);
            let frames = (wait * 90.0).round() as usize;
            for _ in 0..frames {
                a.look(Some(away), 1.0 / 90.0);
                a.advance(1.0 / 90.0);
            }
            a.lock(5.0);
        }
        let s = a.take_stats();
        let (median, p90) = s.time_to_lock().unwrap();
        assert!(
            (median - 0.5).abs() < 0.05 && (p90 - 1.5).abs() < 0.05,
            "{median} {p90}"
        );
        assert!(
            s.to_string()
                .contains("time to lock after a move median 0.5"),
            "{s}"
        );
    }

    #[test]
    fn from_far_a_near_reading_moves_the_focus() {
        let mut a = Autofocus::default();
        // Infinity (the far end of the search, the sky) settles at the far
        // limit rather than easing out of autofocus.
        a.lock(0.0);
        let far = settle(&mut a).unwrap();
        assert!((far - FAR_LIMIT).abs() < 0.01, "{far}");
        a.lock(angle(1.5, BASELINE));
        let got = settle(&mut a).expect("moved off far");
        assert!((got - 1.5).abs() < 0.01, "{got}");
        // And back out, past the hysteresis.
        a.lock(0.0);
        assert!((settle(&mut a).unwrap() - FAR_LIMIT).abs() < 0.01);
        // 40 m is past the limit too: it holds there.
        a.lock(angle(40.0, BASELINE));
        assert!((settle(&mut a).unwrap() - FAR_LIMIT).abs() < 0.01);
    }

    #[test]
    fn turned_off_it_eases_out_rather_than_snapping() {
        let mut a = Autofocus::default();
        a.lock(angle(0.4, BASELINE));
        settle(&mut a);
        a.release();
        let dt = 1.0 / 120.0;
        let mut last = angle(0.4, BASELINE);
        let mut frames = 0;
        while let Some(d) = a.advance(dt) {
            let now = angle(d, BASELINE);
            // No more than the fastest glide a frame.
            assert!(
                (last - now).abs() <= MAX_GLIDE * RATE * dt + 1e-3,
                "{last} {now}"
            );
            last = now;
            frames += 1;
            assert!(frames < 1000);
        }
        assert!(last < 0.01, "let go only at almost no vergence: {last}");
        assert!(frames > 10, "not at once: {frames}");
    }

    #[test]
    fn settings_that_make_no_sense_dont_panic() {
        for t in [
            Tuning {
                rate: -5.0,
                ..Default::default()
            },
            Tuning {
                baseline: -0.064,
                ..Default::default()
            },
            Tuning {
                acceleration: f32::NAN,
                rate: f32::INFINITY,
                ..Default::default()
            },
            Tuning {
                settle_degrees: -1.0,
                baseline: 0.0,
                ..Default::default()
            },
        ] {
            let mut a = Autofocus {
                tuning: t,
                ..Default::default()
            };
            a.lock(3.0);
            for _ in 0..100 {
                a.look(Some(AHEAD), 1.0 / 90.0);
                let d = a.advance(1.0 / 90.0);
                assert!(d.is_none_or(|d| d.is_finite() && d > 0.0), "{t:?}: {d:?}");
            }
            assert!(a.focus_distance().is_none_or(|d| d > 0.0));
        }
    }

    #[test]
    fn a_new_target_mid_glide_slows_within_the_acceleration() {
        let mut a = Autofocus::default();
        a.lock(angle(3.0, BASELINE));
        settle(&mut a);
        a.lock(angle(0.25, BASELINE));
        let dt = 1.0 / 90.0;
        for _ in 0..20 {
            a.advance(dt);
        }
        let fast = a.shown.speed;
        assert!(fast > 2.0 * RATE, "gliding fast: {fast}");
        // A target just ahead: the rate drops, but the speed only as fast as
        // it may slow.
        a.cleared = true;
        a.lock(a.shown.value + 1.0);
        a.advance(dt);
        assert!(
            fast - a.shown.speed <= ACCELERATION * dt + 1e-3,
            "{fast} {}",
            a.shown.speed
        );
    }

    #[test]
    fn a_small_saccade_is_seen_on_the_raw_gaze_even_at_a_lower_rate() {
        let at = |deg: f32| {
            let r = deg.to_radians();
            [r.sin(), 0.0, -r.cos()]
        };
        // A 3° saccade over 3 frames at 120 Hz: no step over the 2° jump,
        // but 120°/s.
        let mut a = Autofocus::default();
        for _ in 0..30 {
            a.look(Some(at(0.0)), 1.0 / 120.0);
        }
        let mut settled = true;
        for i in 1..=3 {
            settled &= a.look(Some(at(i as f32)), 1.0 / 120.0);
        }
        assert!(!settled, "a saccade");
        // The tracker at 60 Hz under a 120 Hz display: every sample twice.
        // A 20°/s pursuit is no saccade, though each new sample steps twice
        // as far as a frame would.
        let mut a = Autofocus::default();
        let mut states = Vec::new();
        for i in 0..240 {
            let deg = (i / 2) as f32 * 20.0 / 60.0;
            states.push(a.look(Some(at(deg)), 1.0 / 120.0));
        }
        assert!(states[30..].iter().all(|s| *s), "pursuit stays settled");
    }

    #[test]
    fn a_held_focus_creeps_to_steady_readings_within_the_hysteresis() {
        let mut a = Autofocus::default();
        a.lock(angle(1.0, BASELINE));
        let start = a.focus_angle().unwrap();
        // 10% nearer: within the hysteresis, but steady.
        let near = angle(0.9, BASELINE);
        a.lock(near);
        a.lock(near);
        assert_eq!(a.focus_angle(), Some(start), "two aren't steady yet");
        for _ in 0..30 {
            a.lock(near);
        }
        assert!((a.focus_angle().unwrap() - near).abs() < 0.01);
        // Readings that wander don't.
        let mut b = Autofocus::default();
        b.lock(angle(1.0, BASELINE));
        for i in 0..30 {
            b.lock(if i % 2 == 0 {
                angle(0.9, BASELINE)
            } else {
                angle(1.1, BASELINE)
            });
        }
        assert_eq!(b.focus_angle(), Some(start));
    }

    #[test]
    fn a_big_change_glides_faster_than_the_rate_and_a_small_one_doesnt() {
        let time_to = |from: f32, to: f32| {
            let mut a = Autofocus::default();
            a.lock(angle(from, BASELINE));
            settle(&mut a);
            a.lock(angle(to, BASELINE));
            let target = angle(to, BASELINE);
            let mut t = 0.0;
            while (angle(a.advance(1.0 / 90.0).unwrap(), BASELINE) - target).abs() > 0.01 {
                t += 1.0 / 90.0;
                assert!(t < 10.0);
            }
            t
        };
        // 3 m to 0.25 m is about 13.4° of vergence: at 5°/s alone 2.8 s.
        let big = time_to(3.0, 0.25);
        assert!(big < 1.8, "{big}");
        // 1 m to 0.8 m is under 1°: no faster than the rate.
        let small = time_to(1.0, 0.8);
        let at_rate = (angle(0.8, BASELINE) - angle(1.0, BASELINE)) / RATE;
        assert!(small >= at_rate * 0.9, "{small} {at_rate}");
    }

    #[test]
    fn following_something_moving_is_measured_along() {
        let mut a = Autofocus::default();
        let at = |deg: f32| {
            let r = deg.to_radians();
            [r.sin(), 0.0, -r.cos()]
        };
        // A pursuit at 20°/s: settled after the settle time, and stays so.
        let mut settled = Vec::new();
        for i in 0..90 {
            settled.push(a.look(Some(at(i as f32 * 20.0 / 90.0)), 1.0 / 90.0));
        }
        assert!(settled[20..].iter().all(|s| *s), "{settled:?}");
        // A saccade: unsettled again.
        assert!(!a.look(Some(at(40.0)), 1.0 / 90.0));
    }

    #[test]
    fn off_is_at_once() {
        let mut a = Autofocus::default();
        a.lock(8.0);
        assert!(settle(&mut a).is_some());
        a.off();
        assert_eq!(a.advance(1.0 / 90.0), None);
    }

    #[test]
    fn the_focus_eases_in_vergence_not_metres() {
        let mut a = Autofocus::default();
        a.lock(angle(2.0, BASELINE));
        settle(&mut a);
        a.lock(angle(0.5, BASELINE));
        let dt = 1.0 / 90.0;
        let mut angles = Vec::new();
        for _ in 0..180 {
            let d = a.advance(dt).unwrap();
            angles.push(angle(d, BASELINE));
        }
        // Rising in vergence throughout, never past the target, and there.
        assert!(angles.windows(2).all(|w| w[1] >= w[0] - 1e-5));
        let target = angle(0.5, BASELINE);
        assert!(angles.iter().all(|a| *a <= target + 1e-3));
        assert!((angles[179] - target).abs() < 1e-3);
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

    fn expected_time(change: f32, rate: f32, acceleration: f32) -> f32 {
        if change > rate * rate / acceleration {
            change / rate + rate / acceleration
        } else {
            2.0 * (change / acceleration).sqrt()
        }
    }

    #[test]
    fn rate_and_acceleration_set_how_long_a_change_takes() {
        for (rate, acceleration) in [(5.0, 62.5), (5.0, 5.0), (15.0, 62.5), (60.0, 100.0)] {
            for change in [2.0, 10.0] {
                for hz in [72.0, 90.0, 120.0] {
                    let (took, fastest) = move_with(change, rate, acceleration, hz);
                    let want = expected_time(change, rate, acceleration);
                    assert!(
                        (took - want).abs() < 0.05,
                        "{rate} {acceleration} {change}: {took}"
                    );
                    assert!(fastest <= rate * 1.01);
                }
            }
        }
    }

    #[test]
    fn turning_round_goes_on_only_as_far_as_it_has_to() {
        for (speed, acceleration) in [(5.0f32, 62.5f32), (20.0, 20.0), (60.0, 100.0)] {
            let mut e = Easing::default();
            let dt = 1.0 / 120.0;
            while e.speed < speed - 1e-3 {
                e.step(1000.0, dt, speed, acceleration);
            }
            let from = e.value;
            let mut furthest = 0.0f32;
            for _ in 0..4000 {
                e.step(from - 5.0, dt, speed, acceleration);
                furthest = furthest.max(e.value - from);
            }
            let needed = speed * speed / (2.0 * acceleration);
            assert!((e.value - (from - 5.0)).abs() < 1e-3);
            assert!(
                (furthest - needed).abs() < needed * 0.1 + 0.05,
                "{furthest}, {needed}"
            );
        }
    }
}
