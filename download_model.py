"""
download_model.py
-----------------
Downloads the ArcFace (r100) ONNX model from the InsightFace ONNX model zoo
and saves it to models/arcface.onnx.

Run once before starting the application:
    python download_model.py
"""

import os
import sys
import urllib.request

# ─────────────────────────────────────────────────────────────────────────────
# Model source
# We use the buffalo_l / w600k_r50 ArcFace ONNX from InsightFace's model hub.
# This is a publicly hosted, lightweight (~166 MB) model.
# ─────────────────────────────────────────────────────────────────────────────
MODEL_URLS = [
    # Primary: InsightFace ONNX model zoo (w600k_r50 ArcFace)
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
    # Fallback: direct ONNX from community mirror
    "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/buffalo_l/w600k_r50.onnx",
]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODELS_DIR, "arcface.onnx")


def _progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "█" * int(pct // 2) + "░" * (50 - int(pct // 2))
        mb_done = downloaded / 1_048_576
        mb_total = total_size / 1_048_576
        sys.stdout.write(
            f"\r  [{bar}] {pct:5.1f}%  {mb_done:.1f}/{mb_total:.1f} MB"
        )
        sys.stdout.flush()
    if downloaded >= total_size:
        print()


def download_direct_onnx():
    """Download the ONNX file directly from HuggingFace mirror."""
    url = MODEL_URLS[1]
    print(f"Downloading from:\n  {url}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    urllib.request.urlretrieve(url, MODEL_PATH, reporthook=_progress)
    print(f"\n✅ Model saved to: {MODEL_PATH}")


def download_zip_and_extract():
    """Download the InsightFace buffalo_l zip and extract w600k_r50.onnx."""
    import zipfile
    import tempfile

    url = MODEL_URLS[0]
    print(f"Downloading buffalo_l pack from:\n  {url}")
    os.makedirs(MODELS_DIR, exist_ok=True)

    with tempfile.NamedTemporaryFile(suffix=".zip", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        urllib.request.urlretrieve(url, tmp_path, reporthook=_progress)
        print("\nExtracting w600k_r50.onnx …")
        with zipfile.ZipFile(tmp_path, "r") as zf:
            # Find the ArcFace ONNX inside the archive
            candidates = [n for n in zf.namelist() if n.endswith(".onnx") and "w600k_r50" in n]
            if not candidates:
                # Fallback: take any .onnx
                candidates = [n for n in zf.namelist() if n.endswith(".onnx")]
            if not candidates:
                raise FileNotFoundError("No .onnx file found inside the archive.")
            chosen = candidates[0]
            print(f"  Extracting: {chosen}")
            with zf.open(chosen) as src, open(MODEL_PATH, "wb") as dst:
                dst.write(src.read())
        print(f"✅ Model saved to: {MODEL_PATH}")
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / 1_048_576
        print(f"✅ Model already exists at:\n   {MODEL_PATH}  ({size_mb:.1f} MB)")
        answer = input("Re-download? [y/N]: ").strip().lower()
        if answer != "y":
            print("Skipping download.")
            return

    print("=" * 60)
    print(" ArcFace ONNX Model Downloader")
    print("=" * 60)

    # Try direct ONNX first (smaller, faster)
    try:
        download_direct_onnx()
    except Exception as e:
        print(f"\n⚠️  Direct download failed: {e}")
        print("Trying zip bundle …")
        try:
            download_zip_and_extract()
        except Exception as e2:
            print(f"\n❌ Both download methods failed: {e2}")
            print(
                "\nManual download instructions:\n"
                "  1. Visit https://github.com/deepinsight/insightface/wiki/Model-Zoo\n"
                "  2. Download the 'buffalo_l' pack\n"
                "  3. Extract 'w600k_r50.onnx' and rename it to 'arcface.onnx'\n"
                "  4. Place it in the 'models/' directory of this project.\n"
            )
            sys.exit(1)

    # Sanity check
    if os.path.exists(MODEL_PATH):
        size_mb = os.path.getsize(MODEL_PATH) / 1_048_576
        print(f"\nFile size: {size_mb:.1f} MB")
        if size_mb < 10:
            print("⚠️  Warning: File seems too small — it may be corrupted.")
        else:
            print("✅ Download complete! You can now run:\n   streamlit run app.py")
    else:
        print("❌ Model file not found after download — something went wrong.")
        sys.exit(1)


if __name__ == "__main__":
    main()
