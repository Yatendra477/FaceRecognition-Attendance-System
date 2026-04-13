"""
liveness.py
-----------
Multi-check liveness detection using MediaPipe Face Mesh.
Provides blink detection (EAR), head pose estimation, and
texture analysis to guard against spoofing attacks.
"""

import cv2
import numpy as np
import time

try:
    import mediapipe as mp
    _MP_AVAILABLE = True
except ImportError:
    _MP_AVAILABLE = False


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe Face Mesh landmark indices for eye contours
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# EAR threshold and blink requirements
EAR_THRESHOLD = 0.22
MIN_BLINKS = 1
CONSECUTIVE_FRAMES = 2

# Head pose landmarks (nose tip, chin, left eye corner, right eye corner,
# left mouth corner, right mouth corner)
POSE_LANDMARKS = [1, 152, 33, 263, 61, 291]

# 3D model points for head pose estimation (generic face model)
MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),     # Nose tip
    (0.0,   -330.0, -65.0),    # Chin
    (-225.0, 170.0, -135.0),   # Left eye corner
    (225.0,  170.0, -135.0),   # Right eye corner
    (-150.0, -150.0, -125.0),  # Left mouth corner
    (150.0,  -150.0, -125.0),  # Right mouth corner
], dtype=np.float64)

# Liveness window
LIVENESS_WINDOW = 4.0  # seconds to accumulate evidence
MIN_YAW_VARIATION = 4.0  # degrees

# Texture analysis
LAPLACIAN_THRESHOLD = 40.0  # below this → likely flat photo


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────
def _eye_aspect_ratio(landmarks, eye_indices, w, h):
    """
    Compute the Eye Aspect Ratio (EAR) for a set of eye landmark indices.
    """
    pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]

    # Vertical distances
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    # Horizontal distance
    h1 = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))

    if h1 == 0:
        return 0.3  # Default open-eye value
    return (v1 + v2) / (2.0 * h1)


def _get_head_pose(landmarks, w, h):
    """
    Estimate head yaw, pitch, roll from face landmarks using solvePnP.
    Returns (yaw, pitch, roll) in degrees, or None on failure.
    """
    image_points = np.array([
        (landmarks[idx].x * w, landmarks[idx].y * h)
        for idx in POSE_LANDMARKS
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rotation_vec, _ = cv2.solvePnP(
        MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return None

    rmat, _ = cv2.Rodrigues(rotation_vec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles  # (pitch, yaw, roll) in degrees


def _texture_variance(face_bgr):
    """
    Compute Laplacian variance — a measure of image sharpness.
    Flat photos/screens tend to have lower variance than real 3D faces.
    """
    if face_bgr is None or face_bgr.size == 0:
        return 999.0  # Pass by default if no face crop
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (128, 128))
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


# ─────────────────────────────────────────────────────────────────────────────
# Main Liveness Checker
# ─────────────────────────────────────────────────────────────────────────────
class LivenessChecker:
    """
    Accumulates per-frame evidence over a sliding window and reports
    whether the face in front of the camera is a live human.
    """

    def __init__(self):
        if not _MP_AVAILABLE:
            raise ImportError("mediapipe is required for liveness detection.")

        self._face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Tracking state
        self._blink_count = 0
        self._ear_below_count = 0
        self._yaw_history = []
        self._texture_scores = []
        self._start_time = time.time()
        self._last_result = None

    def reset(self):
        """Clear all accumulated evidence."""
        self._blink_count = 0
        self._ear_below_count = 0
        self._yaw_history = []
        self._texture_scores = []
        self._start_time = time.time()
        self._last_result = None

    @property
    def elapsed(self):
        return time.time() - self._start_time

    def update(self, frame_bgr, face_box=None):
        """
        Process a single frame and update internal liveness evidence.

        Parameters
        ----------
        frame_bgr : Full BGR frame from the camera.
        face_box  : Optional (x, y, w, h) to crop face for texture check.

        Returns
        -------
        dict with keys:
          - status: "checking" | "live" | "spoof"
          - message: human-readable reason
          - progress: float 0.0–1.0 (how far through the liveness window)
          - blinks: int
          - yaw_range: float (degrees)
          - texture_ok: bool
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        progress = min(self.elapsed / LIVENESS_WINDOW, 1.0)

        if not results.multi_face_landmarks:
            return {
                "status": "checking",
                "message": "No face detected by mesh",
                "progress": progress,
                "blinks": self._blink_count,
                "yaw_range": 0.0,
                "texture_ok": True,
            }

        landmarks = results.multi_face_landmarks[0].landmark

        # ── 1. Blink detection ───────────────────────────────────────────────
        left_ear = _eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear = _eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0

        if avg_ear < EAR_THRESHOLD:
            self._ear_below_count += 1
        else:
            if self._ear_below_count >= CONSECUTIVE_FRAMES:
                self._blink_count += 1
            self._ear_below_count = 0

        # ── 2. Head pose ─────────────────────────────────────────────────────
        pose = _get_head_pose(landmarks, w, h)
        if pose is not None:
            yaw = pose[1]
            self._yaw_history.append(yaw)

        yaw_range = 0.0
        if len(self._yaw_history) > 2:
            yaw_range = max(self._yaw_history) - min(self._yaw_history)

        # ── 3. Texture analysis ──────────────────────────────────────────────
        if face_box is not None:
            x, y, fw, fh = face_box
            crop = frame_bgr[
                max(0, y): min(h, y + fh),
                max(0, x): min(w, x + fw),
            ]
            variance = _texture_variance(crop)
            self._texture_scores.append(variance)

        texture_ok = True
        if self._texture_scores:
            avg_texture = np.mean(self._texture_scores)
            texture_ok = avg_texture > LAPLACIAN_THRESHOLD

        # ── Decision ─────────────────────────────────────────────────────────
        if self.elapsed < LIVENESS_WINDOW:
            return {
                "status": "checking",
                "message": f"Verifying… ({self._blink_count} blinks, yaw {yaw_range:.1f}°)",
                "progress": progress,
                "blinks": self._blink_count,
                "yaw_range": yaw_range,
                "texture_ok": texture_ok,
            }

        # Window complete — make final decision
        is_live = True
        reasons = []

        if self._blink_count < MIN_BLINKS:
            is_live = False
            reasons.append(f"No blinks detected ({self._blink_count}/{MIN_BLINKS})")

        if yaw_range < MIN_YAW_VARIATION:
            is_live = False
            reasons.append(f"Insufficient head movement ({yaw_range:.1f}°)")

        if not texture_ok:
            is_live = False
            reasons.append("Flat texture detected (possible photo/screen)")

        status = "live" if is_live else "spoof"
        message = "✅ Live face confirmed" if is_live else "❌ " + "; ".join(reasons)

        self._last_result = {
            "status": status,
            "message": message,
            "progress": 1.0,
            "blinks": self._blink_count,
            "yaw_range": yaw_range,
            "texture_ok": texture_ok,
        }
        return self._last_result

    def close(self):
        """Release MediaPipe resources."""
        if self._face_mesh:
            self._face_mesh.close()


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────
_checker = None


def get_liveness_checker():
    """Return (and lazily initialise) the liveness checker singleton."""
    global _checker
    if _checker is None:
        _checker = LivenessChecker()
    return _checker
