"""
face_detection.py
-----------------
Provides face detection utilities using OpenCV Haar Cascade detector
(with optional DNN upgrade if weights are present).
"""

import os
import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
HAAR_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# Confidence threshold for DNN detector
DNN_CONFIDENCE_THRESHOLD = 0.5

# Padding added around detected face (fraction of bbox size)
FACE_PADDING = 0.10

# Possible DNN model file locations (cross-platform)
_CV2_DATA = os.path.dirname(cv2.data.haarcascades)   # .../cv2/data/
_DNN_PROTO_CANDIDATES = [
    os.path.join(_CV2_DATA, "dnn", "deploy.prototxt"),
    os.path.join(_CV2_DATA, "deploy.prototxt"),
]
_DNN_MODEL_CANDIDATES = [
    os.path.join(_CV2_DATA, "dnn", "res10_300x300_ssd_iter_140000.caffemodel"),
    os.path.join(_CV2_DATA, "res10_300x300_ssd_iter_140000.caffemodel"),
]


class FaceDetector:
    """
    Detects faces in an image using OpenCV's deep-learning face detector.
    Falls back to Haar Cascade if the DNN model files are unavailable.
    """

    def __init__(self):
        self._dnn_net = None
        self._haar = None
        self._mode = self._init_detector()

    # ── Initialization ──────────────────────────────────────────────────────
    def _init_detector(self):
        """Try to load DNN detector; fallback to Haar Cascade."""
        proto_path = next((p for p in _DNN_PROTO_CANDIDATES if os.path.isfile(p)), None)
        model_path = next((p for p in _DNN_MODEL_CANDIDATES if os.path.isfile(p)), None)

        if proto_path and model_path:
            try:
                net = cv2.dnn.readNetFromCaffe(proto_path, model_path)
                self._dnn_net = net
                return "dnn"
            except Exception:
                pass

        # Haar Cascade fallback (always available with OpenCV)
        haar = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
        if haar.empty():
            raise RuntimeError(
                "Neither DNN face detector nor Haar Cascade could be loaded. "
                "Please check your OpenCV installation."
            )
        self._haar = haar
        return "haar"

    # ── Public API ──────────────────────────────────────────────────────────
    def detect(self, frame):
        """
        Detect faces in *frame* (BGR uint8).

        Returns list of (x, y, w, h) bounding boxes.
        """
        if self._mode == "dnn":
            return self._detect_dnn(frame)
        return self._detect_haar(frame)

    # ── Private helpers ──────────────────────────────────────────────────────
    def _detect_dnn(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)),
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
        )
        self._dnn_net.setInput(blob)
        detections = self._dnn_net.forward()

        boxes = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < DNN_CONFIDENCE_THRESHOLD:
                continue
            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)
            bw, bh = x2 - x1, y2 - y1
            pad_x = int(bw * FACE_PADDING)
            pad_y = int(bh * FACE_PADDING)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            bw = min(w - x1, bw + 2 * pad_x)
            bh = min(h - y1, bh + 2 * pad_y)
            boxes.append((x1, y1, bw, bh))
        return boxes

    def _detect_haar(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(faces) == 0:
            return []
        return [(int(x), int(y), int(w), int(h)) for x, y, w, h in faces]

    # ── Drawing helper ───────────────────────────────────────────────────────
    @staticmethod
    def draw_results(frame, boxes, labels):
        """
        Draw bounding boxes and name/confidence labels on *frame*.
        Returns annotated copy.
        """
        out = frame.copy()
        for (x, y, w, h), label in zip(boxes, labels):
            colour = (0, 255, 0) if "Unknown" not in label else (0, 0, 255)
            cv2.rectangle(out, (x, y), (x + w, y + h), colour, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            cv2.rectangle(out, (x, y - th - 8), (x + tw + 4, y), colour, -1)
            cv2.putText(
                out, label, (x + 2, y - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA,
            )
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton
# ─────────────────────────────────────────────────────────────────────────────
_detector = None


def get_detector():
    """Return (and lazily initialise) the module-level FaceDetector singleton."""
    global _detector
    if _detector is None:
        _detector = FaceDetector()
    return _detector


def detect_faces(frame):
    """Convenience wrapper — detect faces using the singleton detector."""
    return get_detector().detect(frame)
