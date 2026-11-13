//! Gaze calibration in the Stereo tab, under Scale while autofocus is on.

use super::widgets::{self, Rows};
use crate::settings::GazeCalibration;

/// Calibrate gaze, Check and Forget, and how the calibration stands.
pub fn calibration_rows(r: &mut Rows, c: &mut GazeCalibration) {
    r.content(|ui| {
        ui.horizontal_wrapped(|ui| {
            if widgets::chip(ui, tr!("adjust-gaze-calibrate"), false).clicked() {
                c.calibrate = 1;
            }
            if c.gaze_fit.is_some() {
                if widgets::chip(ui, tr!("adjust-gaze-check"), false).clicked() {
                    c.calibrate = 2;
                }
                if widgets::chip(ui, tr!("adjust-gaze-forget"), false).clicked() {
                    c.gaze_fit = None;
                }
            }
        });
        let fit = match c.gaze_fit {
            Some(f) => {
                let right = format!("{:+.1}", f.offset[0]);
                let up = format!("{:+.1}", f.offset[1]);
                let left = format!("{:.1}", f.rms);
                if f.rms > fp_core::gaze_fit::GOOD_RMS {
                    tr!("adjust-gaze-calibrate-again", right = right, up = up, left = left)
                } else {
                    tr!("adjust-gaze-calibrated", right = right, up = up, left = left)
                }
            }
            // Autofocus is on here: an uncorrected gaze aims it off.
            None => tr!("adjust-gaze-uncalibrated").to_string(),
        };
        ui.label(fit);
    });
}
