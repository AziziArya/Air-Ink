from pathlib import Path
from urllib.request import urlretrieve

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"

def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 1_000_000:
        print(f"Model already exists: {MODEL_PATH}")
        return

    print("Downloading the official MediaPipe Hand Landmarker model...")
    try:
        urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception:
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
        raise

    print(f"Model saved to: {MODEL_PATH}")
    print(f"File size: {MODEL_PATH.stat().st_size:,} bytes")

if __name__ == "__main__":
    main()
