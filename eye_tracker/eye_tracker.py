"""
Eye-Tracking Mouse Controller
==============================
High-precision, calibrated eye-tracking mouse controller for Windows 11 (1920x1200 display).
Uses MediaPipe Tasks FaceLandmarker (RunningMode.VIDEO) with:
  - Anatomically Normalized Vertical Gaze Tracking (Eyelid-aperture normalized)
  - Decoupled Horizontal and Vertical Feature Mapping
  - Cursor Speed Scaling (CURSOR_SPEED_SCALE = 0.45) with velocity limiting
  - 1 Euro Adaptive Low-Pass Filter with resting dead-zone and saccade responsiveness
  - Visual 9-Point Verification Mode with separate Horizontal and Vertical Error reporting
  - 3-Second Center-Hold Resting Jitter and Variance Test
  - Safe Mode Progression: Preview Only (default) -> Cursor Test -> Full Control

Safety Modes:
  Mode A — Preview Only   (Default: safe, no mouse movement, no mouse clicks)
  Mode B — Cursor Test    (Moves cursor, all clicks disabled, requires 'B')
  Mode C — Full Control   (Moves cursor + wink-clicks, requires 'F' + confirmation)

Keyboard Shortcuts:
  SPACE   — Capture calibration target / record verification target / dismiss dialogs
  C       — Start 9-Point Calibration (with candidate model comparison on held-out samples)
  V       — Start Guided 9-Point Calibration Verification & Center-Hold Test
  B       --> Activate Mode B: Cursor Test (moves cursor, clicks disabled)
  F       --> Activate Mode C: Full Control (press F twice to confirm)
  P       --> Return immediately to Mode A: Preview Only (Safe)
  D       --> Toggle Extended Debug HUD
  ESC / Q --> Quit immediately and release all resources
"""

import sys
import time
import threading
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import pyautogui


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                             CONFIGURATION                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# --- Camera & Display ---
CAMERA_INDEX = 0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
SCREEN_WIDTH = 1920
SCREEN_HEIGHT = 1200

# --- Cursor Speed & Motion Control ---
# Speed scaling factor (0.1 to 1.0): 0.20 provides calm, slow, controllable movement
CURSOR_SPEED_SCALE = 0.20

# Maximum cursor velocity in pixels per second
MAX_CURSOR_SPEED_PX_SEC = 900.0

# Maximum single-frame cursor displacement in pixels
MAX_STEP_PX = 45.0

# Safe margin away from screen edges to avoid triggering PyAutoGUI failsafe
SAFE_MARGIN = 50

# Minimum position change before dispatching OS mouse move (prevents flooding OS queue)
MIN_DISPATCH_STEP_PX = 1.5

# --- Adaptive Filtering (1 Euro Filter) ---
# Low cutoff frequency during steady gaze (Hz). Lower = rock-solid resting cursor.
MIN_CUTOFF = 0.35

# Saccade responsiveness factor. Higher = fast, lag-free eye saccades.
BETA = 0.015

# Cutoff frequency for derivative estimation (Hz).
D_CUTOFF = 1.0

# Resting dead zone in screen pixels (suppresses involuntary micro-tremor).
DEAD_ZONE_PX = 14.0

# Head pose low-pass smoothing factor (heavy smoothing decouples head jitter).
HEAD_SMOOTH_ALPHA = 0.06

# Plausible screen coordinate margin for outlier detection
SCREEN_MARGIN_OUTLIER_PX = 300.0

# Freeze cursor if no face detected for this many consecutive frames
FACE_LOST_FREEZE_FRAMES = 4

# Maximum age of gaze estimate before freezing
STALE_GAZE_TIMEOUT_SEC = 0.35

# --- Wink Detection ---
WINK_CLOSE_RATIO = 0.55
WINK_OPEN_RATIO = 0.75
WINK_CONFIRM_FRAMES = 3
CLICK_COOLDOWN_SEC = 1.0
BASELINE_MIN_SAMPLES = 45

# --- 9-Point Calibration Targets (Fractions of Screen Width & Height) ---
CALIBRATION_TARGETS = [
    (0.08, 0.08),  # 1. Top-Left Corner
    (0.50, 0.08),  # 2. Top-Center Edge
    (0.92, 0.08),  # 3. Top-Right Corner
    (0.08, 0.50),  # 4. Mid-Left Edge
    (0.50, 0.50),  # 5. Screen Center
    (0.92, 0.50),  # 6. Mid-Right Edge
    (0.08, 0.92),  # 7. Bottom-Left Corner
    (0.50, 0.92),  # 8. Bottom-Center Edge
    (0.92, 0.92),  # 9. Bottom-Right Corner
]

CALIBRATION_COLLECT_SEC = 1.20
CALIBRATION_MIN_SAMPLES = 8

# Verification Sequence (9 Positions matching calibration grid)
VERIFICATION_SEQUENCE = [
    ("Center", 0.50, 0.50),
    ("Top-Left Corner", 0.08, 0.08),
    ("Top-Center Edge", 0.50, 0.08),
    ("Top-Right Corner", 0.92, 0.08),
    ("Mid-Left Edge", 0.08, 0.50),
    ("Mid-Right Edge", 0.92, 0.50),
    ("Bottom-Left Corner", 0.08, 0.92),
    ("Bottom-Center Edge", 0.50, 0.92),
    ("Bottom-Right Corner", 0.92, 0.92),
]

# --- PyAutoGUI Safety Defaults ---
pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.0


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                     MEDIAPIPE LANDMARK DEFINITIONS                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

LEFT_EYE_EAR_DEF = {
    "outer": 33,
    "inner": 133,
    "upper": (160, 158),
    "lower": (144, 153),
}

RIGHT_EYE_EAR_DEF = {
    "outer": 263,
    "inner": 362,
    "upper": (385, 387),
    "lower": (380, 373),
}

# Anatomical Eyelid Landmarks for Vertical Normalization
# Left eye in mirrored frame (user's physical left eye):
LEFT_EYELID_UPPER = 159
LEFT_EYELID_LOWER = 145
LEFT_IRIS_CENTER = 468
LEFT_IRIS_RING = [469, 470, 471, 472]

# Right eye in mirrored frame (user's physical right eye):
RIGHT_EYELID_UPPER = 386
RIGHT_EYELID_LOWER = 374
RIGHT_IRIS_CENTER = 473
RIGHT_IRIS_RING = [474, 475, 476, 477]

NOSE_TIP = 1

LEFT_EYE_CONTOUR = [
    33, 7, 163, 144, 145, 153, 154, 155, 133,
    173, 157, 158, 159, 160, 161, 246, 33,
]
RIGHT_EYE_CONTOUR = [
    362, 382, 381, 380, 374, 373, 390, 249,
    263, 466, 388, 387, 386, 385, 384, 398, 362,
]

FACE_OVAL = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
    397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10,
]


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   THREAD-SAFE LATEST-FRAME CAMERA                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class LatestFrameCamera:
    """
    Captures webcam frames asynchronously in a dedicated thread.
    Always maintains and provides the latest available frame without queue buildup.
    """

    def __init__(self, index=0, width=640, height=480):
        self.index = index
        self.width = width
        self.height = height

        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self.cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera device at index {index}. "
                "Ensure webcam is connected and not in use by another app."
            )

        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._frame = None
        self._frame_id = 0
        self._capture_timestamp = 0.0
        self._running = True
        self._fps = 0.0
        self._fps_timer = time.monotonic()
        self._fps_counter = 0

        self._thread = threading.Thread(target=self._capture_worker, daemon=True)
        self._thread.start()

    def _capture_worker(self):
        while self._running:
            ok, frame = self.cap.read()
            capture_time = time.monotonic()
            if ok and frame is not None and frame.size > 0:
                with self._lock:
                    self._frame = frame
                    self._frame_id += 1
                    self._capture_timestamp = capture_time
                    self._fps_counter += 1

                self._new_frame_event.set()

                now = time.monotonic()
                elapsed = now - self._fps_timer
                if elapsed >= 1.0:
                    with self._lock:
                        self._fps = self._fps_counter / elapsed
                    self._fps_counter = 0
                    self._fps_timer = now
            else:
                time.sleep(0.005)

    def read_latest(self):
        with self._lock:
            if self._frame is None:
                return None, 0, 0.0, self._fps
            return self._frame.copy(), self._frame_id, self._capture_timestamp, self._fps

    def release(self):
        self._running = False
        if self._thread.is_alive():
            self._thread.join(timeout=1.5)
        if self.cap.isOpened():
            self.cap.release()


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   ONE EURO ADAPTIVE FILTER (2D)                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class AdaptiveGazeFilter:
    """
    1 Euro Adaptive Low-Pass Filter with resting dead-zone.
    Allows intentional saccades to transition smoothly without artificial clamping.
    """

    def __init__(self, min_cutoff=MIN_CUTOFF, beta=BETA, d_cutoff=D_CUTOFF,
                 dead_zone_px=DEAD_ZONE_PX, screen_w=1920, screen_h=1200):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.dead_zone_px = float(dead_zone_px)
        self.screen_w = screen_w
        self.screen_h = screen_h

        self.t_prev = None
        self.x_prev = None
        self.y_prev = None
        self.dx_prev = 0.0
        self.dy_prev = 0.0

        # Resting anchor position
        self.anchor_x = None
        self.anchor_y = None

        self.outlier_count = 0
        self.last_outlier_reason = ""

    def _alpha(self, cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.t_prev = None
        self.x_prev = None
        self.y_prev = None
        self.dx_prev = 0.0
        self.dy_prev = 0.0
        self.anchor_x = None
        self.anchor_y = None
        self.outlier_count = 0
        self.last_outlier_reason = ""

    def filter(self, t, x, y):
        # 1. Plausible Screen Boundary Check
        margin = SCREEN_MARGIN_OUTLIER_PX
        if x < -margin or x > self.screen_w + margin or y < -margin or y > self.screen_h + margin:
            self.outlier_count += 1
            reason = f"Out of bounds: ({x:.0f}, {y:.0f})"
            self.last_outlier_reason = reason
            fallback_x = self.x_prev if self.x_prev is not None else self.screen_w / 2.0
            fallback_y = self.y_prev if self.y_prev is not None else self.screen_h / 2.0
            return fallback_x, fallback_y, fallback_x, fallback_y, True, reason

        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            self.y_prev = y
            self.anchor_x = x
            self.anchor_y = y
            return x, y, x, y, False, ""

        dt = max(t - self.t_prev, 1e-4)

        # 2. 1 Euro Filter for X
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        cutoff_x = self.min_cutoff + self.beta * abs(dx_hat)
        a_x = self._alpha(cutoff_x, dt)
        fx = a_x * x + (1.0 - a_x) * self.x_prev

        # 3. 1 Euro Filter for Y
        dy = (y - self.y_prev) / dt
        dy_hat = a_d * dy + (1.0 - a_d) * self.dy_prev
        cutoff_y = self.min_cutoff + self.beta * abs(dy_hat)
        a_y = self._alpha(cutoff_y, dt)
        fy = a_y * y + (1.0 - a_y) * self.y_prev

        self.t_prev = t
        self.x_prev = fx
        self.y_prev = fy
        self.dx_prev = dx_hat
        self.dy_prev = dy_hat

        # 4. Dead Zone Logic: locks cursor during resting tremor, re-anchors during saccades
        anchor_dist = np.hypot(fx - self.anchor_x, fy - self.anchor_y)
        if anchor_dist < self.dead_zone_px:
            out_x = self.anchor_x
            out_y = self.anchor_y
        else:
            self.anchor_x = fx
            self.anchor_y = fy
            out_x = fx
            out_y = fy

        return out_x, out_y, fx, fy, False, ""


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   FEATURE EXTRACTION & CALCULATIONS                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def landmark_to_normalized_xy(landmark):
    return np.array([landmark.x, landmark.y], dtype=np.float64)


def landmark_to_pixel_xy(landmark, width, height):
    px = int(np.clip(landmark.x * width, 0, width - 1))
    py = int(np.clip(landmark.y * height, 0, height - 1))
    return px, py


def compute_eye_aspect_ratio(landmarks, eye_def):
    outer = landmark_to_normalized_xy(landmarks[eye_def["outer"]])
    inner = landmark_to_normalized_xy(landmarks[eye_def["inner"]])

    u1 = landmark_to_normalized_xy(landmarks[eye_def["upper"][0]])
    u2 = landmark_to_normalized_xy(landmarks[eye_def["upper"][1]])
    l1 = landmark_to_normalized_xy(landmarks[eye_def["lower"][0]])
    l2 = landmark_to_normalized_xy(landmarks[eye_def["lower"][1]])

    horizontal_dist = np.linalg.norm(outer - inner)
    if horizontal_dist < 1e-4:
        return None

    v1 = np.linalg.norm(u1 - l1)
    v2 = np.linalg.norm(u2 - l2)

    ear = (v1 + v2) / (2.0 * horizontal_dist)
    if not np.isfinite(ear):
        return None
    return float(ear)


class GazeFeatureExtractor:
    """
    Extracts anatomically normalized horizontal and vertical gaze features.

    Key improvements for Vertical Tracking:
      - Normalizes vertical iris position by each eye's own eyelid aperture (height),
        providing wide, balanced dynamic range matching horizontal travel.
      - Uses bone-fixed canthi for horizontal center and eyelid midpoint for vertical center.
    """

    def __init__(self, head_smooth_alpha=HEAD_SMOOTH_ALPHA):
        self.head_smooth_alpha = head_smooth_alpha
        self.smooth_head = None

    def reset(self):
        self.smooth_head = None

    def extract(self, landmarks):
        left_ear = compute_eye_aspect_ratio(landmarks, LEFT_EYE_EAR_DEF)
        right_ear = compute_eye_aspect_ratio(landmarks, RIGHT_EYE_EAR_DEF)

        if left_ear is None or right_ear is None:
            return None

        # --- Left Eye in mirrored frame (user's physical left eye) ---
        l_outer = landmark_to_normalized_xy(landmarks[LEFT_EYE_EAR_DEF["outer"]])  # 33
        l_inner = landmark_to_normalized_xy(landmarks[LEFT_EYE_EAR_DEF["inner"]])  # 133
        l_upper = landmark_to_normalized_xy(landmarks[LEFT_EYELID_UPPER])          # 159
        l_lower = landmark_to_normalized_xy(landmarks[LEFT_EYELID_LOWER])          # 145
        l_iris = landmark_to_normalized_xy(landmarks[LEFT_IRIS_CENTER])            # 468

        l_width = max(abs(l_inner[0] - l_outer[0]), 0.025)
        l_height = max(abs(l_lower[1] - l_upper[1]), 0.012)
        l_center_x = (l_outer[0] + l_inner[0]) / 2.0
        l_eyelid_mid_y = (l_upper[1] + l_lower[1]) / 2.0
        l_center_y = l_eyelid_mid_y
        l_canthi_mid_y = (l_outer[1] + l_inner[1]) / 2.0

        u_l = (l_iris[0] - l_center_x) / l_width
        v_l_aperture = (l_iris[1] - l_eyelid_mid_y) / l_height
        v_l_canthi = (l_iris[1] - l_canthi_mid_y) / (l_width * 0.45)
        v_l = (v_l_aperture + v_l_canthi) / 2.0

        # --- Right Eye in mirrored frame (user's physical right eye) ---
        r_outer = landmark_to_normalized_xy(landmarks[RIGHT_EYE_EAR_DEF["outer"]])  # 263
        r_inner = landmark_to_normalized_xy(landmarks[RIGHT_EYE_EAR_DEF["inner"]])  # 362
        r_upper = landmark_to_normalized_xy(landmarks[RIGHT_EYELID_UPPER])          # 386
        r_lower = landmark_to_normalized_xy(landmarks[RIGHT_EYELID_LOWER])          # 374
        r_iris = landmark_to_normalized_xy(landmarks[RIGHT_IRIS_CENTER])            # 473

        r_width = max(abs(r_outer[0] - r_inner[0]), 0.025)
        r_height = max(abs(r_lower[1] - r_upper[1]), 0.012)
        r_center_x = (r_outer[0] + r_inner[0]) / 2.0
        r_eyelid_mid_y = (r_upper[1] + r_lower[1]) / 2.0
        r_center_y = r_eyelid_mid_y
        r_canthi_mid_y = (r_outer[1] + r_inner[1]) / 2.0

        u_r = (r_iris[0] - r_center_x) / r_width
        v_r_aperture = (r_iris[1] - r_eyelid_mid_y) / r_height
        v_r_canthi = (r_iris[1] - r_canthi_mid_y) / (r_width * 0.45)
        v_r = (v_r_aperture + v_r_canthi) / 2.0

        # Plausibility sanity checks: iris must be near eye aperture
        if abs(u_l) > 1.8 or abs(v_l) > 2.0 or abs(u_r) > 1.8 or abs(v_r) > 2.0:
            return None

        # Average both eyes to reduce sensor landmark noise
        gaze_u = (u_l + u_r) / 2.0
        gaze_v = (v_l + v_r) / 2.0
        gaze_v_aperture = (v_l_aperture + v_r_aperture) / 2.0
        gaze_v_canthi = (v_l_canthi + v_r_canthi) / 2.0

        # Head motion proxy: nose tip relative to inter-ocular midpoint
        eyes_mid = ((l_center_x + r_center_x) / 2.0, (l_center_y + r_center_y) / 2.0)
        iod = max(abs(r_center_x - l_center_x), 0.05)
        nose = landmark_to_normalized_xy(landmarks[NOSE_TIP])
        raw_head_x = (nose[0] - eyes_mid[0]) / iod
        raw_head_y = (nose[1] - eyes_mid[1]) / iod

        if self.smooth_head is None:
            self.smooth_head = np.array([raw_head_x, raw_head_y], dtype=np.float64)
        else:
            self.smooth_head[0] = (
                self.head_smooth_alpha * raw_head_x +
                (1.0 - self.head_smooth_alpha) * self.smooth_head[0]
            )
            self.smooth_head[1] = (
                self.head_smooth_alpha * raw_head_y +
                (1.0 - self.head_smooth_alpha) * self.smooth_head[1]
            )

        features = {
            "gaze_x": gaze_u,
            "gaze_y": gaze_v,
            "v_aperture": gaze_v_aperture,
            "v_canthi": gaze_v_canthi,
            "head_x": self.smooth_head[0],
            "head_y": self.smooth_head[1],
            "l_iris_raw": (l_iris[0], l_iris[1]),
            "r_iris_raw": (r_iris[0], r_iris[1]),
            "l_rel": (u_l, v_l),
            "r_rel": (u_r, v_r),
            "vector_4d": np.array([
                gaze_u, gaze_v,
                self.smooth_head[0], self.smooth_head[1],
            ], dtype=np.float64),
        }

        return features, left_ear, right_ear


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   CALIBRATION & MODEL SELECTION                          ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class CalibrationModel:
    """
    Fits decoupled, affine, and bilinear calibration mappings on calibration samples.
    Evaluates candidate models using Leave-One-Target-Out Spatial Cross-Validation (LOTO-CV)
    to eliminate data leakage and select the most reliable real-world mapping.
    Reports separate horizontal and vertical residual and cross-validation errors.
    """

    def __init__(self, ridge_lambda=1e-3):
        self.ridge_lambda = ridge_lambda
        self.selected_model_name = None
        self.model_type = None
        self.coeffs = None
        self.mean_fit_error = None
        self.target_errors_h = []
        self.target_errors_v = []
        self.target_errors_total = []
        self.loto_cv_error = None
        self.loto_cv_h = None
        self.loto_cv_v = None
        self.loto_target_errors = []
        self.screen_bounds = None

    def _fit_candidate(self, model_name, X, Y):
        n = len(X)
        if model_name == "Decoupled_Affine":
            Dx = np.column_stack([np.ones(n), X[:, 0]])
            cx = np.linalg.lstsq(Dx, Y[:, 0], rcond=None)[0]
            Dy = np.column_stack([np.ones(n), X[:, 1]])
            cy = np.linalg.lstsq(Dy, Y[:, 1], rcond=None)[0]
            return (cx, cy)
        elif model_name == "2D_Affine":
            D = np.column_stack([np.ones(n), X[:, 0], X[:, 1]])
            reg = self.ridge_lambda * np.eye(D.shape[1])
            coeffs = np.linalg.solve(D.T @ D + reg, D.T @ Y)
            return coeffs
        elif model_name == "Bilinear":
            D = np.column_stack([np.ones(n), X[:, 0], X[:, 1], X[:, 0] * X[:, 1]])
            reg = self.ridge_lambda * np.eye(D.shape[1])
            coeffs = np.linalg.solve(D.T @ D + reg, D.T @ Y)
            return coeffs
        return None

    def _predict_candidate(self, model_name, coeffs, feature_row):
        u = feature_row[0]
        v = feature_row[1]
        if model_name == "Decoupled_Affine":
            cx, cy = coeffs
            px = float(cx[0] + cx[1] * u)
            py = float(cy[0] + cy[1] * v)
            return px, py
        elif model_name == "2D_Affine":
            pred = np.array([1.0, u, v]) @ coeffs
            return float(pred[0]), float(pred[1])
        elif model_name == "Bilinear":
            pred = np.array([1.0, u, v, u * v]) @ coeffs
            return float(pred[0]), float(pred[1])
        return None

    def fit_and_select(self, train_samples_9, screen_targets_9, screen_w, screen_h):
        train_X = np.array(train_samples_9, dtype=np.float64)  # 9 x 4: [u, v, hx, hy]
        train_Y = np.array(screen_targets_9, dtype=np.float64)  # 9 x 2: [Tx, Ty]
        n_targets = len(train_X)

        candidate_models = ["Decoupled_Affine", "2D_Affine", "Bilinear"]

        print("\n  [Calibration] Evaluating Candidate Models via Leave-One-Target-Out Spatial Cross-Validation (LOTO-CV):")
        print("  (Zero Data Leakage: each target is evaluated exclusively by models trained without that target)")

        best_name = None
        best_coeffs = None
        best_cv_err = float("inf")
        best_cv_h = None
        best_cv_v = None
        best_cv_target_errs = None
        model_summaries = {}

        for model_name in candidate_models:
            cv_errs_tot = []
            cv_errs_h = []
            cv_errs_v = []

            for k in range(n_targets):
                mask = [i for i in range(n_targets) if i != k]
                X_train_k = train_X[mask]
                Y_train_k = train_Y[mask]

                coeffs_k = self._fit_candidate(model_name, X_train_k, Y_train_k)
                pred_k = self._predict_candidate(model_name, coeffs_k, train_X[k])

                err_h = abs(pred_k[0] - train_Y[k, 0])
                err_v = abs(pred_k[1] - train_Y[k, 1])
                err_tot = np.hypot(err_h, err_v)

                cv_errs_h.append(err_h)
                cv_errs_v.append(err_v)
                cv_errs_tot.append(err_tot)

            mean_cv_tot = float(np.mean(cv_errs_tot))
            mean_cv_h = float(np.mean(cv_errs_h))
            mean_cv_v = float(np.mean(cv_errs_v))

            # Train on all 9 targets for final fit
            full_coeffs = self._fit_candidate(model_name, train_X, train_Y)
            full_preds = [self._predict_candidate(model_name, full_coeffs, row) for row in train_X]
            fit_resids = [np.hypot(p[0] - t[0], p[1] - t[1]) for p, t in zip(full_preds, train_Y)]
            mean_fit = float(np.mean(fit_resids))

            model_summaries[model_name] = {
                "cv_tot": mean_cv_tot,
                "cv_h": mean_cv_h,
                "cv_v": mean_cv_v,
                "cv_targets": cv_errs_tot,
                "fit_tot": mean_fit,
                "full_coeffs": full_coeffs,
                "full_preds": full_preds,
            }

            print(
                f"    - {model_name:17s} -> LOTO-CV Err: {mean_cv_tot:5.1f}px (H:{mean_cv_h:4.1f}, V:{mean_cv_v:4.1f}) "
                f"| Train Fit: {mean_fit:5.1f}px"
            )

            if mean_cv_tot < best_cv_err:
                best_cv_err = mean_cv_tot
                best_cv_h = mean_cv_h
                best_cv_v = mean_cv_v
                best_name = model_name
                best_coeffs = full_coeffs
                best_cv_target_errs = cv_errs_tot

        self.selected_model_name = best_name
        self.coeffs = best_coeffs
        self.loto_cv_error = best_cv_err
        self.loto_cv_h = best_cv_h
        self.loto_cv_v = best_cv_v
        self.loto_target_errors = best_cv_target_errs
        self.screen_bounds = (screen_w, screen_h)

        # Compute final residual errors for all 9 positions
        summary = model_summaries[best_name]
        self.mean_fit_error = summary["fit_tot"]
        self.target_errors_total = []
        self.target_errors_h = []
        self.target_errors_v = []
        for idx in range(n_targets):
            pred_p = summary["full_preds"][idx]
            t_p = train_Y[idx]
            eh = float(abs(pred_p[0] - t_p[0]))
            ev = float(abs(pred_p[1] - t_p[1]))
            et = float(np.hypot(eh, ev))
            self.target_errors_h.append(eh)
            self.target_errors_v.append(ev)
            self.target_errors_total.append(et)

        print(f"\n  [Calibration] Selected Optimal Model: '{self.selected_model_name}' (Lowest LOTO-CV Spatial Error: {self.loto_cv_error:.1f}px)")
        print("  [Calibration] Residual Errors for All 9 Positions:")
        for idx in range(n_targets):
            print(
                f"    Point {idx + 1}: Residual={self.target_errors_total[idx]:5.1f}px (H:{self.target_errors_h[idx]:4.1f}, V:{self.target_errors_v[idx]:4.1f}) | "
                f"LOTO-CV={self.loto_target_errors[idx]:5.1f}px  (Target: {screen_targets_9[idx]})"
            )

        return True, ""

    def predict(self, feature_vector_4d):
        if not self.is_valid:
            return None

        u = feature_vector_4d[0]
        v = feature_vector_4d[1]

        if self.selected_model_name == "Decoupled_Affine":
            c_x, c_y = self.coeffs
            pred_x = float(c_x[0] + c_x[1] * u)
            pred_y = float(c_y[0] + c_y[1] * v)
        elif self.selected_model_name == "2D_Affine":
            pred = np.array([1.0, u, v]) @ self.coeffs
            pred_x, pred_y = float(pred[0]), float(pred[1])
        elif self.selected_model_name == "Bilinear":
            pred = np.array([1.0, u, v, u * v]) @ self.coeffs
            pred_x, pred_y = float(pred[0]), float(pred[1])
        else:
            return None

        if not np.isfinite(pred_x) or not np.isfinite(pred_y):
            return None

        sw, sh = self.screen_bounds
        if pred_x < -sw * 0.4 or pred_x > sw * 1.4 or pred_y < -sh * 0.4 or pred_y > sh * 1.4:
            return None

        return pred_x, pred_y

    @property
    def is_valid(self):
        return self.coeffs is not None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   WINK DETECTOR & GESTURE RECOGNIZER                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class WinkDetector:
    def __init__(self):
        self.baseline_left = None
        self.baseline_right = None
        self._samples_left = []
        self._samples_right = []
        self._left_wink_counter = 0
        self._right_wink_counter = 0
        self._gesture_latched = False
        self._last_click_timestamp = 0.0

    @property
    def baseline_ready(self):
        return self.baseline_left is not None and self.baseline_right is not None

    def add_baseline_sample(self, left_ear, right_ear):
        if self.baseline_ready:
            return
        if left_ear > 0.15 and right_ear > 0.15:
            self._samples_left.append(left_ear)
            self._samples_right.append(right_ear)

        if len(self._samples_left) >= BASELINE_MIN_SAMPLES:
            self.baseline_left = float(np.median(self._samples_left))
            self.baseline_right = float(np.median(self._samples_right))

    def reset(self):
        self.baseline_left = None
        self.baseline_right = None
        self._samples_left.clear()
        self._samples_right.clear()
        self._left_wink_counter = 0
        self._right_wink_counter = 0
        self._gesture_latched = False
        self._last_click_timestamp = 0.0

    def update(self, left_ear, right_ear):
        if not self.baseline_ready:
            return None

        left_closed = left_ear < (self.baseline_left * WINK_CLOSE_RATIO)
        right_closed = right_ear < (self.baseline_right * WINK_CLOSE_RATIO)
        left_open = left_ear > (self.baseline_left * WINK_OPEN_RATIO)
        right_open = right_ear > (self.baseline_right * WINK_OPEN_RATIO)

        both_closed = left_closed and right_closed
        neither_closed = (not left_closed) and (not right_closed)

        if both_closed or neither_closed:
            self._left_wink_counter = 0
            self._right_wink_counter = 0
            if neither_closed:
                self._gesture_latched = False
            return None

        if left_closed and right_open:
            self._right_wink_counter = 0
            self._left_wink_counter += 1
        elif right_closed and left_open:
            self._left_wink_counter = 0
            self._right_wink_counter += 1
        else:
            self._left_wink_counter = 0
            self._right_wink_counter = 0
            return None

        if self._gesture_latched:
            return None

        now = time.monotonic()
        if now - self._last_click_timestamp < CLICK_COOLDOWN_SEC:
            return None

        if self._left_wink_counter >= WINK_CONFIRM_FRAMES:
            self._gesture_latched = True
            self._last_click_timestamp = now
            self._left_wink_counter = 0
            return "left_wink"

        if self._right_wink_counter >= WINK_CONFIRM_FRAMES:
            self._gesture_latched = True
            self._last_click_timestamp = now
            self._right_wink_counter = 0
            return "right_wink"

        return None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   RATE-LIMITED CURSOR CONTROLLER                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

class CursorController:
    """
    Commands OS cursor movement towards calibrated absolute coordinates.
    Incorporates CURSOR_SPEED_SCALE (0.45) for smooth, calm, controllable tracking
    and eliminates cumulative drift.
    """

    def __init__(self, screen_w, screen_h, speed_scale=CURSOR_SPEED_SCALE):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.speed_scale = float(speed_scale)
        self.pos_x = screen_w / 2.0
        self.pos_y = screen_h / 2.0
        self.last_dispatched_x = int(self.pos_x)
        self.last_dispatched_y = int(self.pos_y)
        self.last_update_t = time.monotonic()

    def update(self, target_x, target_y, move_mouse=False):
        now = time.monotonic()
        dt = max(now - self.last_update_t, 1e-4)
        self.last_update_t = now

        margin = SAFE_MARGIN
        tx = float(np.clip(target_x, margin, self.screen_w - margin))
        ty = float(np.clip(target_y, margin, self.screen_h - margin))

        # Absolute asymptotic convergence with controlled speed scale
        step_x = (tx - self.pos_x) * self.speed_scale
        step_y = (ty - self.pos_y) * self.speed_scale
        step_dist = np.hypot(step_x, step_y)

        # Enforce maximum velocity and step limit
        max_allowed_dist = min(MAX_STEP_PX, MAX_CURSOR_SPEED_PX_SEC * dt)
        if step_dist > max_allowed_dist:
            scale = max_allowed_dist / step_dist
            step_x *= scale
            step_y *= scale

        self.pos_x = float(np.clip(self.pos_x + step_x, margin, self.screen_w - margin))
        self.pos_y = float(np.clip(self.pos_y + step_y, margin, self.screen_h - margin))

        if move_mouse:
            ix = int(self.pos_x)
            iy = int(self.pos_y)
            if np.hypot(ix - self.last_dispatched_x, iy - self.last_dispatched_y) >= MIN_DISPATCH_STEP_PX:
                try:
                    pyautogui.moveTo(ix, iy, _pause=False)
                    self.last_dispatched_x = ix
                    self.last_dispatched_y = iy
                except pyautogui.FailSafeException:
                    print("\n  [SAFETY] PyAutoGUI FailSafe triggered. Cursor touched corner.")
                    raise

        return self.pos_x, self.pos_y


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                   UI, MINI DISPLAY & VISUALIZATIONS                      ║
# ╚══════════════════════════════════════════════════════════════════════════╝

MODE_COLORS = {
    "preview":      (255, 200, 0),    # Cyan/Aqua
    "cursor_test":  (0, 180, 255),    # Amber/Orange
    "full_control": (0, 0, 255),      # Bright Red
    "calibration":  (0, 255, 0),      # Bright Green
    "verify_mode":  (255, 120, 255),  # Magenta
}

MODE_LABELS = {
    "preview":      "MODE A: PREVIEW ONLY (Safe - Mouse Disabled)",
    "cursor_test":  "MODE B: CURSOR TEST (Clicks Disabled)",
    "full_control": "MODE C: FULL CONTROL ACTIVE (Cursor + Clicks)",
    "calibration":  "9-POINT CALIBRATION IN PROGRESS",
    "verify_mode":  "9-POINT CALIBRATION VERIFICATION & CENTER HOLD",
}


def draw_face_and_iris_landmarks(frame, landmarks, width, height):
    for i in range(len(FACE_OVAL) - 1):
        pt1 = landmark_to_pixel_xy(landmarks[FACE_OVAL[i]], width, height)
        pt2 = landmark_to_pixel_xy(landmarks[FACE_OVAL[i + 1]], width, height)
        cv2.line(frame, pt1, pt2, (140, 140, 140), 1)

    for i in range(len(LEFT_EYE_CONTOUR) - 1):
        pt1 = landmark_to_pixel_xy(landmarks[LEFT_EYE_CONTOUR[i]], width, height)
        pt2 = landmark_to_pixel_xy(landmarks[LEFT_EYE_CONTOUR[i + 1]], width, height)
        cv2.line(frame, pt1, pt2, (0, 220, 0), 1)

    for i in range(len(RIGHT_EYE_CONTOUR) - 1):
        pt1 = landmark_to_pixel_xy(landmarks[RIGHT_EYE_CONTOUR[i]], width, height)
        pt2 = landmark_to_pixel_xy(landmarks[RIGHT_EYE_CONTOUR[i + 1]], width, height)
        cv2.line(frame, pt1, pt2, (0, 220, 0), 1)

    for idx in LEFT_IRIS_RING + RIGHT_IRIS_RING:
        pt = landmark_to_pixel_xy(landmarks[idx], width, height)
        cv2.circle(frame, pt, 2, (255, 255, 0), -1)

    l_iris = landmark_to_pixel_xy(landmarks[LEFT_IRIS_CENTER], width, height)
    r_iris = landmark_to_pixel_xy(landmarks[RIGHT_IRIS_CENTER], width, height)
    cv2.circle(frame, l_iris, 5, (0, 0, 0), -1)
    cv2.circle(frame, l_iris, 4, (0, 255, 255), -1)
    cv2.circle(frame, r_iris, 5, (0, 0, 0), -1)
    cv2.circle(frame, r_iris, 4, (0, 255, 255), -1)

    nose = landmark_to_pixel_xy(landmarks[NOSE_TIP], width, height)
    cv2.circle(frame, nose, 3, (255, 120, 0), -1)


def draw_mini_display_map(frame, screen_w, screen_h, raw_x, raw_y, filt_x, filt_y, cur_x=None, cur_y=None, targets_list=None):
    fh, fw = frame.shape[:2]
    map_w = 210
    map_h = int(map_w * (screen_h / screen_w))
    x1 = fw - map_w - 12
    y1 = fh - map_h - 75
    x2 = x1 + map_w
    y2 = y1 + map_h

    cv2.rectangle(frame, (x1 - 2, y1 - 18), (x2 + 2, y2 + 2), (30, 30, 30), -1)
    cv2.rectangle(frame, (x1 - 2, y1 - 18), (x2 + 2, y2 + 2), (180, 180, 180), 1)
    cv2.putText(frame, f"Screen Map ({screen_w}x{screen_h})", (x1 + 4, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (10, 10, 10), -1)

    for fx, fy in CALIBRATION_TARGETS:
        tx = int(x1 + fx * map_w)
        ty = int(y1 + fy * map_h)
        cv2.circle(frame, (tx, ty), 2, (100, 100, 100), -1)

    # 1. Raw Gaze (Magenta X)
    if raw_x is not None and raw_y is not None:
        rx = int(x1 + (raw_x / screen_w) * map_w)
        ry = int(y1 + (raw_y / screen_h) * map_h)
        if x1 <= rx <= x2 and y1 <= ry <= y2:
            cv2.drawMarker(frame, (rx, ry), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 8, 1)

    # 2. Filtered Gaze (Cyan Circle)
    fx_pt = None
    if filt_x is not None and filt_y is not None:
        fx = int(x1 + (filt_x / screen_w) * map_w)
        fy = int(y1 + (filt_y / screen_h) * map_h)
        if x1 <= fx <= x2 and y1 <= fy <= y2:
            cv2.circle(frame, (fx, fy), 4, (0, 255, 255), -1)
            cv2.circle(frame, (fx, fy), 5, (0, 0, 0), 1)
            fx_pt = (fx, fy)

    # 3. Cursor Position (Yellow Plus) & Lag Line
    if cur_x is not None and cur_y is not None:
        cx = int(x1 + (cur_x / screen_w) * map_w)
        cy = int(y1 + (cur_y / screen_h) * map_h)
        if x1 <= cx <= x2 and y1 <= cy <= y2:
            cv2.drawMarker(frame, (cx, cy), (0, 255, 255), cv2.MARKER_CROSS, 7, 1)
            if fx_pt is not None and np.hypot(filt_x - cur_x, filt_y - cur_y) > 20.0:
                cv2.line(frame, fx_pt, (cx, cy), (0, 255, 255), 1)


def draw_hud(frame, lines, mode):
    fh, fw = frame.shape[:2]
    color = MODE_COLORS.get(mode, (255, 255, 255))
    label = MODE_LABELS.get(mode, mode.upper())

    cv2.rectangle(frame, (0, 0), (fw, 28), (25, 25, 25), -1)
    cv2.putText(frame, label, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

    line_h = 16
    panel_h = line_h * len(lines) + 10
    y_start = fh - panel_h
    cv2.rectangle(frame, (0, y_start), (fw, fh), (20, 20, 20), -1)

    for i, line in enumerate(lines):
        y = y_start + 13 + i * line_h
        cv2.putText(frame, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)


def create_calibration_canvas(screen_w, screen_h, target_px, index, total, pip_frame=None, message=""):
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    canvas[:] = (25, 25, 25)
    tx, ty = target_px

    cv2.circle(canvas, (tx, ty), 36, (255, 255, 255), 2)
    cv2.circle(canvas, (tx, ty), 20, (0, 200, 255), 2)
    cv2.circle(canvas, (tx, ty), 7, (0, 255, 0), -1)

    cv2.putText(
        canvas, f"Calibration Point {index + 1} of {total}",
        (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (240, 240, 240), 2
    )
    cv2.putText(
        canvas, "Focus directly on target dot, keep head steady, then press SPACE",
        (50, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (190, 190, 190), 1
    )

    if message:
        cv2.putText(canvas, message, (50, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 255, 255), 2)

    cv2.putText(
        canvas, "C: Restart Calibration  |  ESC/Q: Cancel and Exit",
        (50, screen_h - 35), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (130, 130, 130), 1
    )

    if pip_frame is not None:
        pip_w = 260
        pip_h = int(pip_frame.shape[0] * (pip_w / pip_frame.shape[1]))
        resized_pip = cv2.resize(pip_frame, (pip_w, pip_h))

        pip_y1 = 30 if ty > screen_h // 2 else screen_h - pip_h - 30
        pip_x1 = 40 if tx > screen_w // 2 else screen_w - pip_w - 40
        pip_y2 = pip_y1 + pip_h
        pip_x2 = pip_x1 + pip_w

        cv2.rectangle(canvas, (pip_x1 - 2, pip_y1 - 2), (pip_x2 + 2, pip_y2 + 2), (0, 220, 255), 2)
        canvas[pip_y1:pip_y2, pip_x1:pip_x2] = resized_pip
        cv2.putText(
            canvas, "Live Preview", (pip_x1 + 6, pip_y1 + 18),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1
        )

    return canvas


def create_verification_canvas(screen_w, screen_h, stage, step_idx, total_steps,
                               target_name, target_xy, live_pred_xy=None,
                               center_hold_pct=0.0, results_lines=None, reliability_str=""):
    canvas = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    canvas[:] = (20, 20, 20)

    if stage == "center_hold":
        cx, cy = screen_w // 2, screen_h // 2
        cv2.circle(canvas, (cx, cy), 46, (255, 255, 255), 2)
        cv2.circle(canvas, (cx, cy), 24, (255, 120, 255), 2)
        cv2.circle(canvas, (cx, cy), 8, (0, 255, 0), -1)

        cv2.putText(
            canvas, "STAGE 1: CENTER-HOLD RESTING JITTER TEST (3 Seconds)",
            (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 120, 255), 2
        )
        cv2.putText(
            canvas, "Keep head still and fixate on the green center dot to measure resting eye noise.",
            (50, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1
        )

        bar_w = int(center_hold_pct * 400)
        cv2.rectangle(canvas, (50, 130), (50 + bar_w, 150), (255, 120, 255), -1)
        cv2.rectangle(canvas, (50, 130), (450, 150), (180, 180, 180), 1)
        cv2.putText(
            canvas, f"Progress: {center_hold_pct * 100:.0f}%",
            (465, 146), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1
        )

    elif stage == "grid_verification":
        tx, ty = target_xy
        cv2.circle(canvas, (tx, ty), 42, (255, 255, 255), 2)
        cv2.circle(canvas, (tx, ty), 22, (255, 120, 255), 2)
        cv2.circle(canvas, (tx, ty), 8, (0, 255, 0), -1)

        cv2.putText(
            canvas, f"STAGE 2: ACCURACY VERIFICATION — Point {step_idx + 1}/{total_steps}: {target_name}",
            (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.88, (0, 255, 255), 2
        )
        cv2.putText(
            canvas, "Look steadily at the target dot, then press SPACE to evaluate horizontal and vertical accuracy.",
            (50, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (200, 200, 200), 1
        )

        if live_pred_xy is not None:
            px, py = int(live_pred_xy[0]), int(live_pred_xy[1])
            err_x = abs(px - tx)
            err_y = abs(py - ty)
            err_tot = np.hypot(err_x, err_y)
            cv2.putText(
                canvas,
                f"Live Pred: ({px}, {py}) | Target: ({tx}, {ty}) | "
                f"Horiz Err: {err_x}px | Vert Err: {err_y}px | Total: {err_tot:.0f}px",
                (50, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 220, 255), 2
            )
            cv2.drawMarker(canvas, (px, py), (0, 255, 255), cv2.MARKER_CROSS, 24, 2)

    elif stage == "results":
        cv2.putText(
            canvas, "CALIBRATION ACCURACY & RESTING NOISE VERIFICATION REPORT",
            (50, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (0, 255, 255), 2
        )
        if results_lines:
            for idx, rline in enumerate(results_lines):
                y_pos = 110 + idx * 26
                col = (0, 255, 0) if "PASSED" in rline else (
                    (0, 0, 255) if "UNRELIABLE" in rline else (230, 230, 230)
                )
                cv2.putText(canvas, rline, (50, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.50, col, 1)

        cv2.putText(
            canvas, "Press SPACE to return to Preview Mode",
            (50, screen_h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2
        )

    cv2.putText(
        canvas, "ESC / Q: Cancel Verification",
        (50, screen_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (120, 120, 120), 1
    )

    return canvas


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║                             MAIN PIPELINE                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def main():
    print("=" * 70)
    print("      Eye-Tracking Mouse Controller (Windows 11 / Python 3.13)       ")
    print("=" * 70)

    model_path = Path(__file__).with_name("face_landmarker.task")
    if not model_path.exists():
        print(f"\n[FATAL] Model not found: {model_path}")
        sys.exit(1)
    print(f"[OK] Model found: {model_path.name} ({model_path.stat().st_size:,} bytes)")

    screen_w, screen_h = pyautogui.size()
    print(f"[OK] Detected Screen Resolution: {screen_w} x {screen_h} pixels")

    print("[INIT] Opening webcam with CAP_DSHOW...")
    try:
        camera = LatestFrameCamera(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT)
    except RuntimeError as e:
        print(f"[FATAL] Camera init error: {e}")
        sys.exit(1)

    # Warmup
    first_frame = None
    for _ in range(50):
        first_frame, _, _, _ = camera.read_latest()
        if first_frame is not None:
            break
        time.sleep(0.04)

    if first_frame is None:
        print("[FATAL] Camera failed to capture frames.")
        camera.release()
        sys.exit(1)

    print(f"[OK] Camera stream active: {first_frame.shape[1]}x{first_frame.shape[0]} uint8")

    print("[INIT] Initializing MediaPipe FaceLandmarker...")
    base_options = mp.tasks.BaseOptions(model_asset_path=str(model_path))
    landmarker_options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    landmarker = mp.tasks.vision.FaceLandmarker.create_from_options(landmarker_options)
    print("[OK] FaceLandmarker initialized.")

    mode = "preview"  # Always start in Mode A: Preview Only
    feature_extractor = GazeFeatureExtractor(head_smooth_alpha=HEAD_SMOOTH_ALPHA)
    calibration = CalibrationModel()
    gaze_filter = AdaptiveGazeFilter(
        min_cutoff=MIN_CUTOFF, beta=BETA, d_cutoff=D_CUTOFF,
        dead_zone_px=DEAD_ZONE_PX, screen_w=screen_w, screen_h=screen_h
    )
    wink_detector = WinkDetector()
    cursor = CursorController(screen_w, screen_h, speed_scale=CURSOR_SPEED_SCALE)

    # Calibration Runtime Variables
    cal_targets_px = [
        (int(fx * screen_w), int(fy * screen_h))
        for fx, fy in CALIBRATION_TARGETS
    ]
    cal_index = 0
    cal_gaze_samples_median = []
    cal_screen_targets = []
    cal_collecting = False
    cal_buffer = []
    cal_collect_start_time = 0.0
    cal_status_message = ""

    # Verification Mode Variables
    verify_stage = "center_hold"  # 'center_hold' -> 'grid_verification' -> 'results'
    verify_center_start_t = 0.0
    verify_center_raw_pts = []
    verify_center_pure_1euro_pts = []
    verify_center_filt_pts = []
    verify_center_cursor_pts = []
    verify_index = 0
    verify_recorded_samples = []
    verify_results_lines = None
    verify_reliability_str = ""

    last_processed_frame_id = -1
    prev_timestamp_ms = 0
    consecutive_no_face_frames = 0
    last_valid_gaze_time = 0.0
    active_wink_label = ""
    active_wink_expiry = 0.0
    last_status_message = ""
    show_extended_diagnostics = False

    t_infer_ms = 0.0
    t_feat_ms = 0.0
    t_draw_ms = 0.0
    frame_age_ms = 0.0
    infer_counter = 0
    infer_timer = time.monotonic()
    infer_fps = 0.0

    raw_pred_x, raw_pred_y = None, None
    filt_pred_x, filt_pred_y = None, None
    pure_pred_x, pure_pred_y = None, None
    is_outlier_frame = False
    outlier_reason_str = ""

    last_log_t = time.monotonic()

    CAMERA_WINDOW = "Eye Tracker - Camera Preview"
    CALIB_WINDOW = "Eye Tracker - Calibration"
    VERIFY_WINDOW = "Eye Tracker - Verification"
    calib_win_open = False
    verify_win_open = False

    print("\n" + "=" * 70)
    print("                 SYSTEM ACTIVE - PREVIEW ONLY (SAFE)               ")
    print("=" * 70)
    print("  Controls:")
    print("    C       --> Run 9-Point Calibration (Fits Decoupled / Affine Models)")
    print("    V       --> Run 9-Point Verification Test (Reports Horiz & Vert Errors)")
    print("    B       --> Cursor Test Mode (moves cursor, clicks disabled)")
    print("    F       --> Full Control Mode (confirm with F)")
    print("    P       --> Return to Mode A: Preview Only (Safe)")
    print("    D       --> Toggle Extended Debug HUD")
    print("    ESC / Q --> Exit")
    print("=" * 70 + "\n")

    try:
        while True:
            loop_start = time.perf_counter()

            # ── 1. Acquire Fresh Camera Frame ──────────────────────
            raw_frame, frame_id, cap_time, cam_fps = camera.read_latest()
            if raw_frame is None:
                time.sleep(0.003)
                continue

            is_new_frame = (frame_id != last_processed_frame_id)
            if not is_new_frame:
                time.sleep(0.002)
                continue

            last_processed_frame_id = frame_id
            frame_age_ms = (time.monotonic() - cap_time) * 1000.0

            frame = cv2.flip(raw_frame, 1)  # Mirrored display
            frame_h, frame_w = frame.shape[:2]

            # ── 2. MediaPipe Video Inference ────────────────────────
            rgb_frame = np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

            now_ms = int(time.monotonic() * 1000)
            timestamp_ms = max(prev_timestamp_ms + 1, now_ms)
            prev_timestamp_ms = timestamp_ms

            t_inf_start = time.perf_counter()
            inference_ok = False
            result = None
            try:
                result = landmarker.detect_for_video(mp_image, timestamp_ms)
                inference_ok = True
            except Exception as e:
                last_status_message = f"Inference Error: {e}"
            t_infer_ms = (time.perf_counter() - t_inf_start) * 1000.0

            infer_counter += 1
            if time.monotonic() - infer_timer >= 1.0:
                infer_fps = infer_counter / (time.monotonic() - infer_timer)
                infer_counter = 0
                infer_timer = time.monotonic()

            # ── 3. Gaze & Landmark Extraction ───────────────────────
            t_feat_start = time.perf_counter()
            face_detected = False
            iris_detected = False
            num_landmarks = 0
            landmarks = None
            feat_dict = None
            left_ear, right_ear = None, None

            if inference_ok and result and result.face_landmarks:
                landmarks = result.face_landmarks[0]
                num_landmarks = len(landmarks)
                face_detected = True
                iris_detected = num_landmarks >= 478

                if iris_detected:
                    extracted = feature_extractor.extract(landmarks)
                    if extracted is not None:
                        feat_dict, left_ear, right_ear = extracted
                        last_valid_gaze_time = time.monotonic()
                        consecutive_no_face_frames = 0
            else:
                consecutive_no_face_frames += 1

            t_feat_ms = (time.perf_counter() - t_feat_start) * 1000.0

            # ── 4. Predict Gaze & Adaptive Filter ───────────────────
            gaze_is_stale = (time.monotonic() - last_valid_gaze_time) > STALE_GAZE_TIMEOUT_SEC
            face_is_lost = consecutive_no_face_frames > FACE_LOST_FREEZE_FRAMES
            is_outlier_frame = False
            outlier_reason_str = ""

            raw_pred_x, raw_pred_y = None, None
            filt_pred_x, filt_pred_y = None, None
            pure_pred_x, pure_pred_y = None, None

            if calibration.is_valid and feat_dict is not None and not gaze_is_stale and not face_is_lost:
                raw_pt = calibration.predict(feat_dict["vector_4d"])
                if raw_pt is not None:
                    raw_pred_x, raw_pred_y = raw_pt
                    filt_pred_x, filt_pred_y, pure_pred_x, pure_pred_y, is_outlier_frame, outlier_reason_str = gaze_filter.filter(
                        time.monotonic(), raw_pred_x, raw_pred_y
                    )

            # ── 5. Mode Handlers ────────────────────────────────────

            # --- SUB-MODE: 9-Point Calibration Verification & Center Hold (Key 'V') ---
            if mode == "verify_mode":
                if not verify_win_open:
                    cv2.namedWindow(VERIFY_WINDOW, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(VERIFY_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    verify_win_open = True

                if verify_stage == "center_hold":
                    elapsed_hold = time.monotonic() - verify_center_start_t
                    hold_pct = min(elapsed_hold / 3.0, 1.0)

                    if raw_pred_x is not None and filt_pred_x is not None:
                        verify_center_raw_pts.append((raw_pred_x, raw_pred_y))
                        verify_center_pure_1euro_pts.append((pure_pred_x, pure_pred_y))
                        verify_center_filt_pts.append((filt_pred_x, filt_pred_y))
                        verify_center_cursor_pts.append((cursor.pos_x, cursor.pos_y))

                    if hold_pct >= 1.0:
                        verify_stage = "grid_verification"
                        verify_index = 0
                        verify_recorded_samples.clear()

                    v_canvas = create_verification_canvas(
                        screen_w, screen_h, verify_stage, 0, len(VERIFICATION_SEQUENCE),
                        "Center Hold", (screen_w // 2, screen_h // 2),
                        center_hold_pct=hold_pct
                    )
                    cv2.imshow(VERIFY_WINDOW, v_canvas)

                elif verify_stage == "grid_verification":
                    if verify_index < len(VERIFICATION_SEQUENCE):
                        t_name, fx, fy = VERIFICATION_SEQUENCE[verify_index]
                        t_pos = (int(fx * screen_w), int(fy * screen_h))
                        live_pred = (filt_pred_x, filt_pred_y) if filt_pred_x is not None else None

                        v_canvas = create_verification_canvas(
                            screen_w, screen_h, verify_stage, verify_index, len(VERIFICATION_SEQUENCE),
                            t_name, t_pos, live_pred_xy=live_pred
                        )
                        cv2.imshow(VERIFY_WINDOW, v_canvas)
                    else:
                        verify_stage = "results"

                elif verify_stage == "results":
                    v_canvas = create_verification_canvas(
                        screen_w, screen_h, verify_stage, verify_index, len(VERIFICATION_SEQUENCE),
                        "", (0, 0), results_lines=verify_results_lines,
                        reliability_str=verify_reliability_str
                    )
                    cv2.imshow(VERIFY_WINDOW, v_canvas)

            # --- SUB-MODE: 9-Point Calibration ---
            elif mode == "calibration":
                if verify_win_open:
                    cv2.destroyWindow(VERIFY_WINDOW)
                    verify_win_open = False

                if not calib_win_open:
                    cv2.namedWindow(CALIB_WINDOW, cv2.WINDOW_NORMAL)
                    cv2.setWindowProperty(CALIB_WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    calib_win_open = True

                if cal_index < len(cal_targets_px):
                    current_target = cal_targets_px[cal_index]
                    canvas = create_calibration_canvas(
                        screen_w, screen_h, current_target,
                        cal_index, len(cal_targets_px),
                        pip_frame=frame, message=cal_status_message
                    )

                    if cal_collecting:
                        elapsed_collect = time.monotonic() - cal_collect_start_time

                        if feat_dict is not None and face_detected and iris_detected:
                            cal_buffer.append(feat_dict["vector_4d"].copy())
                            if left_ear is not None and right_ear is not None:
                                wink_detector.add_baseline_sample(left_ear, right_ear)

                        pct = min(elapsed_collect / CALIBRATION_COLLECT_SEC, 1.0)
                        bar_w = int(pct * 360)
                        cv2.rectangle(canvas, (50, 175), (50 + bar_w, 195), (0, 220, 255), -1)
                        cv2.rectangle(canvas, (50, 175), (410, 195), (200, 200, 200), 1)
                        cv2.putText(
                            canvas,
                            f"Collecting... {len(cal_buffer)} samples (need >= {CALIBRATION_MIN_SAMPLES})",
                            (50, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1
                        )

                        if elapsed_collect >= CALIBRATION_COLLECT_SEC:
                            if len(cal_buffer) >= CALIBRATION_MIN_SAMPLES:
                                # Use median feature vector across the window to filter blink artifacts
                                robust_feature = np.median(cal_buffer, axis=0)
                                cal_gaze_samples_median.append(robust_feature)
                                cal_screen_targets.append(current_target)

                                cal_index += 1
                                cal_status_message = f"Point {cal_index} OK ({len(cal_buffer)} samples)"
                                print(f"  [Calibration] Captured point {cal_index}/{len(cal_targets_px)} -> Median Feature: u={robust_feature[0]:+.3f}, v={robust_feature[1]:+.3f}")
                            else:
                                reason = "No face/iris" if not face_detected else "Eyes blinked"
                                cal_status_message = f"Rejected: {len(cal_buffer)} samples ({reason}). Retry."
                                print(f"  [Calibration] {cal_status_message}")

                            cal_collecting = False
                            cal_buffer.clear()

                    cv2.imshow(CALIB_WINDOW, canvas)

                else:
                    # All 9 points captured: fit models and select best via LOTO-CV spatial cross-validation
                    ok, msg = calibration.fit_and_select(
                        cal_gaze_samples_median, cal_screen_targets,
                        screen_w, screen_h
                    )
                    if ok:
                        print("  [Calibration] Selected optimal model successfully!")
                        mode = "preview"
                        cal_status_message = ""
                        cv2.destroyWindow(CALIB_WINDOW)
                        calib_win_open = False
                        gaze_filter.reset()
                        cursor.pos_x = screen_w / 2.0
                        cursor.pos_y = screen_h / 2.0
                    else:
                        cal_index = 0
                        cal_gaze_samples_median.clear()
                        cal_screen_targets.clear()
                        cal_status_message = f"Calibration failed ({msg}). Press SPACE to retry."

            # --- NORMAL TRACKING (Preview / Cursor Test / Full Control) ---
            else:
                if calib_win_open:
                    cv2.destroyWindow(CALIB_WINDOW)
                    calib_win_open = False
                if verify_win_open:
                    cv2.destroyWindow(VERIFY_WINDOW)
                    verify_win_open = False

                # Move cursor with speed scaling
                if filt_pred_x is not None:
                    allow_mouse_move = (mode in ("cursor_test", "full_control"))
                    cursor.update(filt_pred_x, filt_pred_y, move_mouse=allow_mouse_move)

                detected_gesture = None
                if left_ear is not None and right_ear is not None and not face_is_lost:
                    detected_gesture = wink_detector.update(left_ear, right_ear)

                if detected_gesture is not None:
                    now_t = time.monotonic()
                    if detected_gesture == "right_wink":
                        active_wink_label = "GESTURE: RIGHT WINK --> (Left Click)"
                        active_wink_expiry = now_t + 1.2
                        if mode == "full_control":
                            try:
                                pyautogui.click(button="left", _pause=False)
                                print("  [ACTION] Right Wink --> Left Click executed")
                            except pyautogui.FailSafeException:
                                raise
                    elif detected_gesture == "left_wink":
                        active_wink_label = "GESTURE: LEFT WINK --> (Right Click)"
                        active_wink_expiry = now_t + 1.2
                        if mode == "full_control":
                            try:
                                pyautogui.click(button="right", _pause=False)
                                print("  [ACTION] Left Wink --> Right Click executed")
                            except pyautogui.FailSafeException:
                                raise

            # ── 6. Render Camera Preview, Mini Map & HUD ────────────
            t_draw_start = time.perf_counter()

            if landmarks is not None and iris_detected:
                draw_face_and_iris_landmarks(frame, landmarks, frame_w, frame_h)

            # Mini Display Representation
            draw_mini_display_map(
                frame, screen_w, screen_h,
                raw_pred_x, raw_pred_y, filt_pred_x, filt_pred_y,
                cur_x=cursor.pos_x, cur_y=cursor.pos_y, targets_list=cal_targets_px
            )

            # Center alert
            if not face_detected and not face_is_lost:
                cv2.putText(
                    frame, "NO FACE DETECTED",
                    (frame_w // 2 - 140, frame_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.80, (0, 0, 255), 2
                )
            elif face_is_lost:
                cv2.putText(
                    frame, "FACE LOST - CURSOR FROZEN",
                    (frame_w // 2 - 180, frame_h // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2
                )

            # Comprehensive HUD
            hud_lines = []
            hud_lines.append(
                f"Cam FPS: {cam_fps:.1f} | Infer FPS: {infer_fps:.1f} ({t_infer_ms:.1f}ms) | Age: {frame_age_ms:.1f}ms"
            )

            if feat_dict is not None:
                hud_lines.append(
                    f"Gaze Norm: Horiz(u)={feat_dict['gaze_x']:+.3f} | Vert(v)={feat_dict['gaze_y']:+.3f} (Aper:{feat_dict['v_aperture']:+.3f}, Canthi:{feat_dict['v_canthi']:+.3f}) | Head: ({feat_dict['head_x']:+.2f}, {feat_dict['head_y']:+.2f})"
                )

            cal_name = calibration.selected_model_name if calibration.is_valid else "NONE"
            val_str = f"LOTO-CV={calibration.loto_cv_error:.0f}px (H:{calibration.loto_cv_h:.0f}, V:{calibration.loto_cv_v:.0f})" if calibration.is_valid else "NOT CALIBRATED"
            hud_lines.append(f"Model: {cal_name} | {val_str} | SpeedScale: {cursor.speed_scale:.2f}")

            if raw_pred_x is not None and filt_pred_x is not None:
                diff_x = abs(raw_pred_x - filt_pred_x)
                diff_y = abs(raw_pred_y - filt_pred_y)
                diff_tot = np.hypot(diff_x, diff_y)
                cur_lag = np.hypot(filt_pred_x - cursor.pos_x, filt_pred_y - cursor.pos_y)
                hud_lines.append(
                    f"Raw: ({raw_pred_x:.0f}, {raw_pred_y:.0f}) | Filt: ({filt_pred_x:.0f}, {filt_pred_y:.0f}) | Cursor: ({cursor.pos_x:.0f}, {cursor.pos_y:.0f}) | Lag: {cur_lag:.0f}px"
                )
                hud_lines.append(
                    f"JitterDelta: {diff_tot:.0f}px (dX:{diff_x:.0f}, dY:{diff_y:.0f}) | Outliers: {gaze_filter.outlier_count} | Mode: {mode}"
                )
            else:
                hud_lines.append(f"Outlier Count: {gaze_filter.outlier_count} | Mode: {mode}")

            if time.monotonic() < active_wink_expiry:
                hud_lines.append(active_wink_label)

            hud_lines.append("Shortcuts: C=Calib  V=Verify(9pts)  B=CursorTest  F=Full  P=Preview  Q=Quit")

            draw_hud(frame, hud_lines, mode)
            t_draw_ms = (time.perf_counter() - t_draw_start) * 1000.0

            if time.monotonic() - last_log_t >= 5.0:
                print(
                    f"[PROFILER] Cam: {cam_fps:.1f} FPS | Infer: {infer_fps:.1f} FPS | "
                    f"Model: {calibration.selected_model_name} | Outliers: {gaze_filter.outlier_count}"
                )
                last_log_t = time.monotonic()

            cv2.imshow(CAMERA_WINDOW, frame)

            # ── 7. Process Keyboard Commands ────────────────────────
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                print("\n[USER] Quit command received. Terminating safely...")
                break

            elif key == ord("d"):
                show_extended_diagnostics = not show_extended_diagnostics

            elif key == ord("v"):
                if not calibration.is_valid:
                    print("  [NOTICE] Calibrate first (press C) before running the Verification Mode.")
                else:
                    print("  [VERIFY] Starting Calibration Verification & Center-Hold Test...")
                    mode = "verify_mode"
                    verify_stage = "center_hold"
                    verify_center_start_t = time.monotonic()
                    verify_center_raw_pts.clear()
                    verify_center_pure_1euro_pts.clear()
                    verify_center_filt_pts.clear()
                    verify_center_cursor_pts.clear()
                    verify_index = 0
                    verify_recorded_samples.clear()
                    verify_results_lines = None

            elif key == 32 and mode == "verify_mode":
                if verify_stage == "grid_verification":
                    if verify_index < len(VERIFICATION_SEQUENCE):
                        t_name, fx, fy = VERIFICATION_SEQUENCE[verify_index]
                        tx, ty = int(fx * screen_w), int(fy * screen_h)
                        rp_x = raw_pred_x if raw_pred_x is not None else screen_w / 2
                        rp_y = raw_pred_y if raw_pred_y is not None else screen_h / 2
                        fp_x = filt_pred_x if filt_pred_x is not None else screen_w / 2
                        fp_y = filt_pred_y if filt_pred_y is not None else screen_h / 2
                        cp_x = cursor.pos_x
                        cp_y = cursor.pos_y

                        err_x = abs(fp_x - tx)
                        err_y = abs(fp_y - ty)
                        err_tot = np.hypot(err_x, err_y)
                        cur_lag = np.hypot(fp_x - cp_x, fp_y - cp_y)

                        raw_u = feat_dict["gaze_x"] if feat_dict else 0.0
                        raw_v = feat_dict["gaze_y"] if feat_dict else 0.0

                        verify_recorded_samples.append({
                            "name": t_name,
                            "target": (tx, ty),
                            "raw_pred": (int(rp_x), int(rp_y)),
                            "filt_pred": (int(fp_x), int(fp_y)),
                            "cursor": (int(cp_x), int(cp_y)),
                            "gaze_feature": (raw_u, raw_v),
                            "err_x": err_x,
                            "err_y": err_y,
                            "err_tot": err_tot,
                            "cur_lag": cur_lag,
                        })
                        print(
                            f"  [Verify {verify_index + 1}/9] {t_name:18s} -> Target: ({tx:4d}, {ty:4d}) | Pred: ({int(fp_x):4d}, {int(fp_y):4d}) | "
                            f"Cursor: ({int(cp_x):4d}, {int(cp_y):4d}) | Features: (u={raw_u:+.3f}, v={raw_v:+.3f}) | "
                            f"Horiz Err: {err_x:4.1f}px | Vert Err: {err_y:4.1f}px | Total: {err_tot:4.1f}px | Lag: {cur_lag:4.1f}px"
                        )
                        verify_index += 1

                        if verify_index >= len(VERIFICATION_SEQUENCE):
                            verify_stage = "results"
                            # Compute Statistics
                            h_errs = [s["err_x"] for s in verify_recorded_samples]
                            v_errs = [s["err_y"] for s in verify_recorded_samples]
                            tot_errs = [s["err_tot"] for s in verify_recorded_samples]
                            lags = [s["cur_lag"] for s in verify_recorded_samples]

                            mean_h = np.mean(h_errs)
                            mean_v = np.mean(v_errs)
                            mean_tot = np.mean(tot_errs)
                            max_tot = np.max(tot_errs)
                            mean_lag = np.mean(lags)

                            # Center hold stats
                            r_xs = [p[0] for p in verify_center_raw_pts]
                            r_ys = [p[1] for p in verify_center_raw_pts]
                            f_xs = [p[0] for p in verify_center_pure_1euro_pts]
                            f_ys = [p[1] for p in verify_center_pure_1euro_pts]
                            c_xs = [p[0] for p in verify_center_cursor_pts]
                            c_ys = [p[1] for p in verify_center_cursor_pts]

                            raw_std = np.hypot(np.std(r_xs), np.std(r_ys)) if r_xs else 0.0
                            raw_spread_x = (np.max(r_xs) - np.min(r_xs)) if r_xs else 0.0
                            raw_spread_y = (np.max(r_ys) - np.min(r_ys)) if r_ys else 0.0
                            pure_1euro_std = np.hypot(np.std(f_xs), np.std(f_ys)) if f_xs else 0.0
                            cur_std = np.hypot(np.std(c_xs), np.std(c_ys)) if c_xs else 0.0

                            # Net 3-second drift
                            if len(r_xs) >= 20:
                                drift_x = np.mean(r_xs[-10:]) - np.mean(r_xs[:10])
                                drift_y = np.mean(r_ys[-10:]) - np.mean(r_ys[:10])
                                net_drift = float(np.hypot(drift_x, drift_y))
                            else:
                                net_drift = 0.0

                            is_reliable = (mean_tot <= 190.0 and mean_v <= 200.0)
                            status_label = "RELIABLE (PASSED)" if is_reliable else "UNRELIABLE (FLAGGED: Recalibration Advised)"

                            verify_results_lines = [
                                f"Verification Status: {status_label}",
                                f"Model Evaluated: {calibration.selected_model_name} | Training Residual: {calibration.mean_fit_error:.1f}px | LOTO-CV Spatial Val: {calibration.loto_cv_error:.1f}px",
                                f"Center-Hold Jitter (3s): Raw Std={raw_std:.1f}px (Spread: {raw_spread_x:.0f}x{raw_spread_y:.0f}px) | 1Euro Std={pure_1euro_std:.1f}px | Cursor Std={cur_std:.1f}px",
                                f"Center-Hold Drift (3s): {net_drift:.1f}px | Mean Dynamic Cursor Lag: {mean_lag:.1f}px",
                                f"Overall Mean Accuracy: {mean_tot:.1f}px  (Horiz Mean: {mean_h:.1f}px | Vert Mean: {mean_v:.1f}px)",
                                f"Maximum Target Error: {max_tot:.1f}px",
                                "-" * 75,
                            ]

                            for s in verify_recorded_samples:
                                verify_results_lines.append(
                                    f"{s['name']:18s} Target: {str(s['target']):12s} Pred: {str(s['filt_pred']):12s} "
                                    f"HorizErr: {s['err_x']:4.1f}px | VertErr: {s['err_y']:4.1f}px | Total: {s['err_tot']:4.1f}px | Lag: {s['cur_lag']:4.1f}px"
                                )

                            print("\n" + "=" * 70)
                            print("         CALIBRATION VERIFICATION REPORT")
                            print("=" * 70)
                            for rline in verify_results_lines:
                                print(f"  {rline}")
                            print("=" * 70 + "\n")

                elif verify_stage == "results":
                    mode = "preview"
                    if verify_win_open:
                        cv2.destroyWindow(VERIFY_WINDOW)
                        verify_win_open = False
                    print("  [Verify] Verification test dismissed. Returned to Preview Only mode.")

            elif key == ord("c"):
                if not face_detected or not iris_detected:
                    print("  [NOTICE] Face or iris not detected. Position face before calibrating.")
                else:
                    print("  [CALIBRATION] Starting 9-point calibration...")
                    mode = "calibration"
                    cal_index = 0
                    cal_gaze_samples_median.clear()
                    cal_screen_targets.clear()
                    cal_collecting = False
                    cal_buffer.clear()
                    cal_status_message = "Look directly at the target dot and press SPACE"
                    wink_detector.reset()
                    feature_extractor.reset()
                    calibration = CalibrationModel()

            elif key == ord("p"):
                mode = "preview"
                print("  [MODE] Returned to Mode A: Preview Only (Safe).")

            elif key == ord("b"):
                if not calibration.is_valid:
                    print("  [NOTICE] Calibration required first. Press 'C' to calibrate.")
                else:
                    mode = "cursor_test"
                    print("  [MODE] Mode B: CURSOR TEST ACTIVE (Clicks Disabled).")

            elif key == ord("f"):
                if not calibration.is_valid:
                    print("  [NOTICE] Calibration required first. Press 'C' to calibrate.")
                elif not wink_detector.baseline_ready:
                    print("  [NOTICE] Complete calibration first to set baseline.")
                else:
                    print("\n  [SAFETY] Press 'F' again to confirm FULL CONTROL, or any other key to cancel.")
                    confirm_img = frame.copy()
                    cv2.rectangle(
                        confirm_img,
                        (frame_w // 2 - 220, frame_h // 2 - 40),
                        (frame_w // 2 + 220, frame_h // 2 + 40),
                        (0, 0, 180), -1
                    )
                    cv2.putText(
                        confirm_img, "Press 'F' again to confirm FULL CONTROL",
                        (frame_w // 2 - 200, frame_h // 2 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2
                    )
                    cv2.putText(
                        confirm_img, "Any other key to cancel",
                        (frame_w // 2 - 100, frame_h // 2 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1
                    )
                    cv2.imshow(CAMERA_WINDOW, confirm_img)
                    ck = cv2.waitKey(0) & 0xFF
                    if ck == ord("f"):
                        mode = "full_control"
                        print("  [MODE] Mode C: FULL CONTROL ACTIVE.")
                    else:
                        print("  [SAFETY] Full Control cancelled.")

            elif key == 32 and mode == "calibration":
                if cal_index < len(cal_targets_px) and not cal_collecting:
                    cal_collecting = True
                    cal_buffer.clear()
                    cal_collect_start_time = time.monotonic()
                    cal_status_message = f"Collecting Point {cal_index + 1}... Keep eyes fixed on dot."

    except pyautogui.FailSafeException:
        print("\n[SAFETY] PyAutoGUI FailSafe triggered. Cursor moved to corner.")
    except KeyboardInterrupt:
        print("\n[USER] KeyboardInterrupt received.")
    finally:
        print("[SHUTDOWN] Releasing camera, landmarker, and window resources...")
        try:
            landmarker.close()
        except Exception:
            pass
        camera.release()
        cv2.destroyAllWindows()
        print("[SHUTDOWN] Cleanup complete.")


if __name__ == "__main__":
    main()
