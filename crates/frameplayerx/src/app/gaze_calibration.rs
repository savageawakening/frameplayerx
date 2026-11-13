//! Gaze calibration: targets shown one at a time, fixed to
//! the head, the raw gaze collected at each, and the fit
//! (`fp_core::gaze_fit`) saved as the correction autofocus applies. A check
//! afterwards shows new targets with a small dot at the corrected gaze.
//!
//! The targets are 2 m away: about where a picture's near subjects are, so
//! the eyes converge comfortably, and far enough that the eyes being a few
//! centimetres from where the head's pose is centred makes well under a
//! tenth of a degree; the error fitted is an angle, so the distance hardly
//! matters otherwise.

use super::App;
use crate::world::Panel;
use fp_core::gaze_fit::{self, CHECKS, ROUNDS, Routine, Step, TARGETS};
use fp_render::Renderer;
use glam::{EulerRot, Mat4, Quat, Vec2, Vec3};

use super::FrameInput;

/// How far away the targets and the check dot are, metres, and how wide,
/// degrees.
const DISTANCE: f32 = 2.0;
const TARGET_DEGREES: f32 = 1.6;
const DOT_DEGREES: f32 = 0.35;
/// What's shown instead of the picture while calibrating, linear RGBA: a
/// plain dark grey, nothing at another depth to pull the eyes off a target.
pub(super) const BACKGROUND: [f32; 4] = [0.012, 0.012, 0.014, 1.0];

/// What's running.
pub(super) struct Calibrating {
    routine: Routine,
    check: bool,
}

/// The two panels it draws: the target, and the check's gaze dot.
pub(super) fn panels() -> (Panel, Panel) {
    let make = |px: u32, degrees: f32| {
        let w = 2.0 * DISTANCE * (0.5 * degrees).to_radians().tan();
        let mut p = Panel::new([px, px], Vec2::new(w, w), 1.0);
        p.display_only = true;
        p.ctx
            .style_mut(|s| s.visuals.panel_fill = egui::Color32::TRANSPARENT);
        p
    };
    (make(96, TARGET_DEGREES), make(32, DOT_DEGREES))
}

/// Where something `at` (yaw, pitch, degrees) from where the head faces
/// is, `distance` away, facing the eyes.
fn head_locked(head: (Vec3, Quat), at: [f32; 2], distance: f32) -> Mat4 {
    let turn =
        head.1 * Quat::from_euler(EulerRot::YXZ, -at[0].to_radians(), at[1].to_radians(), 0.0);
    Mat4::from_rotation_translation(turn, head.0 + turn * Vec3::NEG_Z * distance)
}

impl App {
    /// Starts, runs or finishes calibrating or checking the gaze; whether
    /// it's running this frame. Asked for from the Stereo tab.
    pub(super) fn gaze_calibration(&mut self, renderer: &mut Renderer, input: &FrameInput) -> bool {
        let ask = std::mem::take(&mut self.settings.autofocus_gaze.calibrate);
        if ask != 0 && self.calibrating.is_none() {
            let check = ask == 2 && self.settings.autofocus_gaze.gaze_fit.is_some();
            // Calibrating shows every target more than once: each showing
            // is a point of the fit.
            let targets: Vec<[f32; 2]> = if check {
                CHECKS.to_vec()
            } else {
                TARGETS.repeat(ROUNDS)
            };
            log::info!(
                "gaze {}: {} targets",
                if check { "check" } else { "calibration" },
                targets.len()
            );
            self.calibrating = Some(Calibrating {
                routine: Routine::new(&targets),
                check,
            });
            self.ui_visible = false;
        }
        let Some(c) = &mut self.calibrating else {
            self.gaze_target.visible = false;
            self.gaze_dot.visible = false;
            return false;
        };
        let Some(head) = input.head else {
            return true;
        };
        // The raw gaze, in the head's frame: what autofocus starts from.
        let raw = input.gaze.map(|g| (head.1.inverse() * g).to_array());
        match c.routine.step(raw, input.dt) {
            Step::Show(at) => {
                self.gaze_target.pose = head_locked(head, at, DISTANCE);
                self.gaze_target.visible = true;
                let t = &self.settings.autofocus_gaze;
                if c.check
                    && let (Some(fit), Some(raw)) = (t.gaze_fit, raw)
                {
                    let seen = gaze_fit::angles(fit.correct(raw));
                    self.gaze_dot.pose = head_locked(head, seen, DISTANCE - 0.02);
                    self.gaze_dot.visible = true;
                } else {
                    self.gaze_dot.visible = false;
                }
            }
            Step::Done => {
                let c = self.calibrating.take().expect("running");
                self.finish(c);
                // Back to the menus, where the result shows.
                self.ui_visible = true;
                self.gaze_target.visible = false;
                self.gaze_dot.visible = false;
                return false;
            }
        }
        for (panel, dot) in [(&mut self.gaze_target, false), (&mut self.gaze_dot, true)] {
            if panel.visible
                && panel.needs_paint()
                && let Err(e) = panel.paint(renderer, input.time, |ctx| picture(ctx, dot))
            {
                log::error!("gaze calibration: {e}");
            }
        }
        true
    }

    fn finish(&mut self, c: Calibrating) {
        let t = &mut self.settings.autofocus_gaze;
        let r = &c.routine;
        if c.check {
            let Some(fit) = t.gaze_fit else {
                return;
            };
            let errors: Vec<[f32; 2]> = r
                .seen
                .iter()
                .map(|(target, mean)| {
                    let seen = gaze_fit::angles(fit.correct(gaze_fit::direction(*mean)));
                    [target[0] - seen[0], target[1] - seen[1]]
                })
                .collect();
            let rms = (errors
                .iter()
                .map(|e| e[0] * e[0] + e[1] * e[1])
                .sum::<f32>()
                / errors.len().max(1) as f32)
                .sqrt();
            let each: Vec<String> = r
                .seen
                .iter()
                .zip(&errors)
                .map(|((at, _), e)| {
                    format!(
                        "at {:+.0}° {:+.0}°: {:+.2}° {:+.2}°",
                        at[0], at[1], e[0], e[1]
                    )
                })
                .collect();
            log::info!(
                "gaze check: error left RMS {rms:.2}° over {} targets, {} skipped; {}",
                r.seen.len(),
                r.skipped,
                each.join("; ")
            );
            return;
        }
        match gaze_fit::fit(&r.seen) {
            Some(report) => {
                let at: Vec<String> = TARGETS
                    .iter()
                    .map(|t| format!("{:+.0}° {:+.0}°", t[0], t[1]))
                    .collect();
                log::info!(
                    "gaze calibration: targets at {}, {ROUNDS} times; {report}; {} skipped",
                    at.join(", "),
                    r.skipped
                );
                t.gaze_fit = Some(report.fit);
            }
            None => log::warn!("gaze calibration: no target seen, nothing changed"),
        }
    }
}

/// A target (a ring with a dot) or the check's gaze dot.
fn picture(ctx: &egui::Context, dot: bool) {
    egui::CentralPanel::default()
        .frame(egui::Frame::new().fill(egui::Color32::TRANSPARENT))
        .show(ctx, |ui| {
            let rect = ui.max_rect();
            let c = rect.center();
            let r = 0.5 * rect.width().min(rect.height());
            let p = ui.painter();
            if dot {
                p.circle_filled(
                    c,
                    0.7 * r,
                    egui::Color32::from_rgba_unmultiplied(80, 220, 255, 200),
                );
            } else {
                p.circle_stroke(
                    c,
                    0.8 * r,
                    egui::Stroke::new(0.08 * r, egui::Color32::WHITE),
                );
                p.circle_filled(c, 0.12 * r, egui::Color32::WHITE);
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn targets_are_fixed_to_the_head_at_their_angle_and_distance() {
        let head = (Vec3::new(0.2, 1.6, -0.1), Quat::from_rotation_y(0.5));
        for at in [[0.0, 0.0], [24.0, -24.0], [-12.0, 0.0]] {
            let pose = head_locked(head, at, DISTANCE);
            let d = pose.w_axis.truncate() - head.0;
            assert!((d.length() - DISTANCE).abs() < 1e-4);
            let local = gaze_fit::angles((head.1.inverse() * d.normalize()).to_array());
            assert!(
                (local[0] - at[0]).abs() < 1e-3 && (local[1] - at[1]).abs() < 1e-3,
                "{local:?}"
            );
            // Facing the eyes.
            assert!((pose.transform_vector3(Vec3::Z).normalize() + d.normalize()).length() < 1e-4);
        }
    }
}
