
import cv2
import mediapipe as mp
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ---------------- CONFIG ----------------

MODEL_PATH = Path("models/hand_landmarker.task")

SAMPLES_PER_DIGIT = 20
IMAGE_SIZE = 28
CAMERA_INDEX = 0

RAW_DIR = Path("dataset_raw")
IMAGE_DIR = Path("dataset_images")

# -----------------------------------------


def make_image(points):
    """Create a 28x28 image without connecting separate strokes."""

    drawing_points = [
        p for p in points
        if p["pen_down"] and p["detected"]
    ]

    if len(drawing_points) < 2:
        return None

    coords = np.array(
        [[p["x"], p["y"]] for p in drawing_points],
        dtype=np.float32
    )

    min_xy = coords.min(axis=0)
    max_xy = coords.max(axis=0)

    width_height = max_xy - min_xy
    largest = max(float(width_height[0]), float(width_height[1]))

    if largest < 0.001:
        return None

    # Normalize while preserving aspect ratio.
    margin = 3
    usable = IMAGE_SIZE - 2 * margin
    scale = usable / largest

    normalized = (coords - min_xy) * scale

    offset_x = (IMAGE_SIZE - width_height[0] * scale) / 2
    offset_y = (IMAGE_SIZE - width_height[1] * scale) / 2

    canvas = np.zeros(
        (IMAGE_SIZE, IMAGE_SIZE),
        dtype=np.uint8
    )

    previous = None

    for p in points:
        if not p["detected"] or not p["pen_down"]:
            previous = None
            continue

        x = (p["x"] - min_xy[0]) * scale + offset_x
        y = (p["y"] - min_xy[1]) * scale + offset_y

        current = (
            int(round(x)),
            int(round(y))
        )

        if previous is not None:
            cv2.line(
                canvas,
                previous,
                current,
                255,
                thickness=2,
                lineType=cv2.LINE_AA
            )

        previous = current

    return canvas


def save_sample(digit, sample_number, points):
    raw_folder = RAW_DIR / str(digit)
    image_folder = IMAGE_DIR / str(digit)

    raw_folder.mkdir(parents=True, exist_ok=True)
    image_folder.mkdir(parents=True, exist_ok=True)

    filename = f"sample_{sample_number:03d}"

    raw_path = raw_folder / f"{filename}.json"
    image_path = image_folder / f"{filename}.png"

    image = make_image(points)

    if image is None:
        print("Sample has insufficient drawing points.")
        return False

    data = {
        "digit": digit,
        "sample": sample_number,
        "created_at": datetime.now().isoformat(),
        "point_count": len(points),
        "points": points
    }

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    cv2.imwrite(str(image_path), image)

    return True


def main():

    if not MODEL_PATH.exists():
        print(f"Model not found: {MODEL_PATH}")
        return

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("Cannot open webcam.")
        return

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5
    )

    landmarker = vision.HandLandmarker.create_from_options(options)

    digit = 0
    sample_number = 1

    recording = False
    pen_down = True
    points = []

    last_timestamp = 0

    print("\nAir-Ink Dataset Capture V2")
    print("--------------------------")
    print("ENTER : Start recording")
    print("D     : Toggle pen up/down")
    print("SPACE : Finish sample")
    print("R     : Redo current sample")
    print("N     : Skip current sample")
    print("Q     : Save and quit")
    print("--------------------------\n")

    try:
        while digit < 10:

            success, frame = camera.read()

            if not success:
                print("Camera frame error.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            timestamp = time.monotonic_ns() // 1_000_000
            timestamp = max(timestamp, last_timestamp + 1)
            last_timestamp = timestamp

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            result = landmarker.detect_for_video(
                mp_image,
                timestamp
            )

            detected = len(result.hand_landmarks) > 0
            tip_x = None
            tip_y = None

            if detected:
                tip = result.hand_landmarks[0][8]
                tip_x = float(tip.x)
                tip_y = float(tip.y)

                px = int(tip_x * width)
                py = int(tip_y * height)

                cv2.circle(
                    frame,
                    (px, py),
                    9,
                    (0, 255, 0),
                    -1
                )

                if recording:
                    points.append({
                        "time_ms": timestamp,
                        "x": tip_x,
                        "y": tip_y,
                        "detected": True,
                        "pen_down": pen_down
                    })

            elif recording:
                points.append({
                    "time_ms": timestamp,
                    "x": None,
                    "y": None,
                    "detected": False,
                    "pen_down": False
                })

            status = "RECORDING" if recording else "READY"
            pen_status = "DOWN" if pen_down else "UP"

            cv2.putText(
                frame,
                f"Digit: {digit}   Sample: {sample_number}/{SAMPLES_PER_DIGIT}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Status: {status}   Pen: {pen_status}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                "ENTER Start | D Pen | SPACE Finish | R Redo | N Skip | Q Quit",
                (15, height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1
            )

            cv2.imshow("Air-Ink Dataset Capture V2", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Quitting...")
                break

            elif key in (10, 13) and not recording:
                points = []
                pen_down = True
                recording = True
                print(f"Recording digit {digit}, sample {sample_number}")

            elif key == ord("d") and recording:
                pen_down = not pen_down
                print("Pen:", "DOWN" if pen_down else "UP")

            elif key == ord("r"):
                points = []
                recording = False
                pen_down = True
                print("Current sample reset.")

            elif key == ord("n"):
                points = []
                recording = False
                sample_number += 1

                if sample_number > SAMPLES_PER_DIGIT:
                    digit += 1
                    sample_number = 1

                print("Sample skipped.")

            elif key == 32 and recording:
                recording = False

                saved = save_sample(
                    digit,
                    sample_number,
                    points
                )

                if saved:
                    print(
                        f"Saved digit {digit}, "
                        f"sample {sample_number}"
                    )

                    sample_number += 1

                    if sample_number > SAMPLES_PER_DIGIT:
                        digit += 1
                        sample_number = 1

                else:
                    print("Sample not saved. Press R to retry.")

                points = []
                pen_down = True

    finally:
        camera.release()
        landmarker.close()
        cv2.destroyAllWindows()

    print("\nCapture session ended.")
    print(f"Raw data: {RAW_DIR.resolve()}")
    print(f"Images:   {IMAGE_DIR.resolve()}")


if __name__ == "__main__":
    main()
