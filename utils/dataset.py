"""
dataset.py
----------
Handles capturing face images for new users and generating / storing
their ArcFace embeddings into embeddings.json.
"""

import os
import json
import time
import numpy as np
import cv2

from utils.arcface import get_embedding

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(_BASE_DIR, "dataset")
EMBEDDINGS_PATH = os.path.join(_BASE_DIR, "embeddings", "embeddings.json")

# Minimum images required before generating an embedding
MIN_IMAGES = 3


# ─────────────────────────────────────────────────────────────────────────────
# Image capture
# ─────────────────────────────────────────────────────────────────────────────
def save_face_image(
    name: str,
    frame: np.ndarray,
    face_box: tuple[int, int, int, int],
) -> str:
    """
    Crop the detected face from *frame* and save it to
    ``dataset/<name>/img_<timestamp>.jpg``.

    Parameters
    ----------
    name     : Student / person name (used as folder name).
    frame    : Full BGR webcam frame.
    face_box : (x, y, w, h) bounding box in pixel coordinates.

    Returns
    -------
    Absolute path to the saved image file.
    """
    user_dir = os.path.join(DATASET_DIR, _sanitise(name))
    os.makedirs(user_dir, exist_ok=True)

    x, y, w, h = face_box
    # Clamp to frame boundaries
    x, y = max(0, x), max(0, y)
    w = min(w, frame.shape[1] - x)
    h = min(h, frame.shape[0] - y)

    face_crop = frame[y: y + h, x: x + w]
    timestamp = int(time.time() * 1000)
    filename = f"img_{timestamp}.jpg"
    filepath = os.path.join(user_dir, filename)
    cv2.imwrite(filepath, face_crop)
    return filepath


def count_images(name: str) -> int:
    """Return how many images have been captured for *name*."""
    user_dir = os.path.join(DATASET_DIR, _sanitise(name))
    if not os.path.isdir(user_dir):
        return 0
    return len([
        f for f in os.listdir(user_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Embedding generation
# ─────────────────────────────────────────────────────────────────────────────
def generate_embeddings_for_user(name: str) -> tuple[bool, str]:
    """
    Read all images for *name*, generate ArcFace embeddings, average them
    (gives a more robust representation), L2-normalise, and store in JSON.

    Returns
    -------
    (success: bool, message: str)
    """
    user_dir = os.path.join(DATASET_DIR, _sanitise(name))
    if not os.path.isdir(user_dir):
        return False, f"No dataset folder found for '{name}'."

    image_files = [
        os.path.join(user_dir, f)
        for f in os.listdir(user_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    if len(image_files) < MIN_IMAGES:
        return False, (
            f"Only {len(image_files)} image(s) found for '{name}'. "
            f"Please capture at least {MIN_IMAGES}."
        )

    embeddings = []
    failed = 0
    for img_path in image_files:
        img = cv2.imread(img_path)
        if img is None:
            failed += 1
            continue
        try:
            emb = get_embedding(img)
            embeddings.append(emb)
        except Exception:
            failed += 1

    if not embeddings:
        return False, "Failed to generate any embeddings. Check image quality."

    # Average embeddings → more robust single representation
    avg_emb = np.mean(embeddings, axis=0).astype(np.float32)
    # Re-normalise after averaging
    norm = np.linalg.norm(avg_emb)
    if norm > 0:
        avg_emb /= norm

    # Load existing store, add / update this user, write back
    os.makedirs(os.path.dirname(EMBEDDINGS_PATH), exist_ok=True)
    existing: dict = {}
    if os.path.exists(EMBEDDINGS_PATH):
        try:
            with open(EMBEDDINGS_PATH, "r") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, ValueError):
            existing = {}

    existing[name] = avg_emb.tolist()
    with open(EMBEDDINGS_PATH, "w") as f:
        json.dump(existing, f, indent=2)

    msg = (
        f"✅ Embedding generated for '{name}' from {len(embeddings)} image(s)."
    )
    if failed:
        msg += f" ({failed} image(s) skipped due to errors.)"
    return True, msg


# ─────────────────────────────────────────────────────────────────────────────
# Registered users
# ─────────────────────────────────────────────────────────────────────────────
def list_registered_users() -> list[str]:
    """Return names of all users who have a sub-folder in the dataset dir."""
    if not os.path.isdir(DATASET_DIR):
        return []
    return [
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ]


def delete_student(name: str) -> tuple[bool, str]:
    """
    Remove a student entirely — delete their image folder and
    remove their embedding from the JSON store.

    Returns (success, message).
    """
    import shutil
    from utils.recognition import get_store

    user_dir = os.path.join(DATASET_DIR, _sanitise(name))
    removed_images = False
    removed_embedding = False

    # Remove image folder
    if os.path.isdir(user_dir):
        shutil.rmtree(user_dir)
        removed_images = True

    # Remove embedding
    store = get_store()
    if store.remove_user(name):
        removed_embedding = True

    if removed_images or removed_embedding:
        return True, f"✅ Student '{name}' has been removed."
    return False, f"⚠️ No data found for '{name}'."


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _sanitise(name: str) -> str:
    """Strip characters that are unsafe in directory names."""
    return "".join(c for c in name.strip() if c.isalnum() or c in "_ -")

