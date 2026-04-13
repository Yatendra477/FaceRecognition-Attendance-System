"""
arcface.py
----------
Loads a pre-trained ArcFace ONNX model and generates normalised
512-dimensional face embeddings using ONNX Runtime.
"""

import os
import numpy as np
import cv2
import onnxruntime as ort

# ─────────────────────────────────────────────────────────────────────────────
# Model path
# ─────────────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_BASE_DIR, "models", "arcface.onnx")

# ArcFace expects 112x112 RGB input
INPUT_SIZE = (112, 112)


class ArcFaceModel:
    """Wrapper around the ArcFace ONNX model."""

    def __init__(self, model_path=MODEL_PATH):
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                "ArcFace model not found at '{}'. "
                "Run 'python download_model.py' to download it.".format(model_path)
            )
        providers = ["CPUExecutionProvider"]
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._session = ort.InferenceSession(model_path, providers=providers)
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def get_embedding(self, face_bgr):
        """
        Generate a normalised 512-dim embedding for *face_bgr*.

        Parameters
        ----------
        face_bgr : BGR uint8 image of the detected face region.

        Returns numpy array of shape (512,) with L2-normalised float32 values.
        """
        blob = self._preprocess(face_bgr)
        outputs = self._session.run([self._output_name], {self._input_name: blob})
        embedding = outputs[0][0]  # shape (512,)
        return self._l2_normalize(embedding)

    @staticmethod
    def _preprocess(face_bgr):
        """Resize, convert BGR->RGB, normalise to [-1, 1], add batch dim."""
        face = cv2.resize(face_bgr, INPUT_SIZE)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32)
        face = (face - 127.5) / 127.5
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)
        return face

    @staticmethod
    def _l2_normalize(vec):
        """Return L2-normalised vector."""
        norm = np.linalg.norm(vec)
        if norm == 0:
            return vec
        return vec / norm


# ─────────────────────────────────────────────────────────────────────────────
# Module-level singleton (lazy initialisation)
# ─────────────────────────────────────────────────────────────────────────────
_model = None


def get_model():
    """Return (and lazily initialise) the module-level ArcFaceModel singleton."""
    global _model
    if _model is None:
        _model = ArcFaceModel()
    return _model


def get_embedding(face_bgr):
    """Convenience wrapper — generate an embedding via the singleton model."""
    return get_model().get_embedding(face_bgr)
