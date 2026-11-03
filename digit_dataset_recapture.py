import cv2
import mediapipe as mp
import numpy as np
import json
import time
from pathlib import Path
from datetime import datetime

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ---------------- SETTINGS ----------------

MODEL_PATH = Path("models/hand_landmarker.task")

# فقط ارقامی که نیاز به ضبط مجدد دارند
DIGITS_TO_CAPTURE = [0, 3, 5]

SAMPLES_PER_DIGIT = 20
IMAGE_SIZE = 28
CAMERA_INDEX = 0

OUTPUT_DIR = Path("dataset_manual")

# -------------------------------------------


def create_digit_image(points):
    """Convert separated finger strokes into a 28x28 image."""

    valid = [
        p for p in points
        if p["detected"] and p["pen_down"]
        and p["x"] is not None and p["y"] is not None
    ]

    if len(valid) < 3:
        return None

    coords = np.array(
        [[p["x"], p["y"]] for p in valid],
        dtype=np.float32
    )

    min_xy = coords.min(axis=0)
    max_xy = coords.max(axis=0)

    dimensions = max_xy - min_xy
    largest = float(max(dimensions))

    if largest < 0.001:
        return None

    margin = 3
    usable = IMAGE_SIZE - 2 * margin
    scale = usable / largest

    offset_x = (IMAGE_SIZE - dimensions[0] * scale) / 2
    offset_y = (IMAGE_SIZE - dimensions[1] * scale) / 2

    image = np.zeros((IMAGE_SIZE, IMAGE_SIZE), dtype=np.uint8)

    previous = None

    for p in points:
        if (
            not p["detected"]
            or not p["pen_down"]
            or p["x"] is None
            or p["y"] is None
        ):
            previous = None
            continue

        x = (p["x"] - min_xy[0]) * scale + offset_x
        y = (p["y"] - min_xy[1]) * scale + offset_y

        current = (int(round(x)), int(round(y)))

        if previous is not None:
            cv2.line(
                image, previous, current,
                255, 2, cv2.LINE_AA
            )

        previous = current

    return image


def next_sample_number(folder):
    folder.mkdir(parents=True, exist_ok=True)

    existing = []

    for path in folder.glob("sample_*.json"):
        try:
            existing.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass

    return max(existing, default=0) + 1


def save_sample(digit, sample_number, points):
    folder = OUTPUT_DIR / str(digit)
    folder.mkdir(parents=True, exist_ok=True)

    image = create_digit_image(points)

    if image is None:
        print("Not enough valid drawing points. Please retry.")
        return False

    name = f"sample_{sample_number:03d}"

    raw_path = folder / f"{name}.json"
    image_path = folder / f"{name}.png"

    data = {
        "digit": digit,
        "sample": sample_number,
        "created_at": datetime.now().isoformat(),
        "points": points
    }

    with open(raw_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)

    cv2.imwrite(str(image_path), image)

    return True


def draw_live_path(frame, points):
    """Draw only pen-down sections; never connect separate strokes."""

    height, width = frame.shape[:2]
    previous = None
    first_point = None
    last_point = None

    for p in points:
        if (
            not p["detected"]
            or not p["pen_down"]
            or p["x"] is None
            or p["y"] is None
        ):
            previous = None
            continue

        current = (
            int(p["x"] * width),
            int(p["y"] * height)
        )

        if first_point is None:
            first_point = current

        if previous is not None:
            cv2.line(
                frame, previous, current,
                (0, 255, 255), 3, cv2.LINE_AA
            )

        previous = current
        last_point = current

    if first_point is not None:
        cv2.circle(frame, first_point, 8, (0, 255, 0), -1)

    if last_point is not None:
        cv2.circle(frame, last_point, 8, (0, 0, 255), -1)

    return frame


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

    try:
        for digit in DIGITS_TO_CAPTURE:

            folder = OUTPUT_DIR / str(digit)
            sample_number = next_sample_number(folder)

            while sample_number <= SAMPLES_PER_DIGIT:

                points = []
                recording = False
                reviewing = False
                pen_down = True
                last_timestamp = 0

                print(
                    f"\nDigit {digit} | "
                    f"Sample {sample_number}/{SAMPLES_PER_DIGIT}"
                )

                while True:
                    success, frame = camera.read()

                    if not success:
                        print("Camera error.")
                        return

                    frame = cv2.flip(frame, 1)
                    height, width = frame.shape[:2]

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                    timestamp = time.monotonic_ns() // 1_000_000
                    timestamp = max(timestamp, last_timestamp + 1)
                    last_timestamp = timestamp

                    result = landmarker.detect_for_video(
                        mp.Image(
                            image_format=mp.ImageFormat.SRGB,
                            data=rgb
                        ),
                        timestamp
                    )

                    detected = len(result.hand_landmarks) > 0

                    if detected:
                        tip = result.hand_landmarks[0][8]
                        x = float(tip.x)
                        y = float(tip.y)

                        cv2.circle(
                            frame,
                            (int(x * width), int(y * height)),
                            8,
                            (0, 255, 0),
                            -1
                        )

                        if recording:
                            points.append({
                                "time_ms": timestamp,
                                "x": x,
                                "y": y,
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

                    if recording or reviewing:
                        frame = draw_live_path(frame, points)

                    status = (
                        "RECORDING" if recording
                        else "REVIEW - S SAVE / R RETRY" if reviewing
                        else "READY - ENTER TO START"
                    )

                    pen_status = "DOWN" if pen_down else "UP"

                    cv2.putText(
                        frame,
                        f"Digit: {digit}  Sample: {sample_number}/{SAMPLES_PER_DIGIT}",
                        (12, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        f"{status} | PEN: {pen_status}",
                        (12, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.58,
                        (0, 255, 255),
                        2
                    )

                    cv2.putText(
                        frame,
                        "ENTER Start | D Pen | SPACE Finish | S Save | R Retry | Q Quit",
                        (12, height - 18),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.43,
                        (255, 255, 255),
                        1
                    )

                    cv2.imshow("Air-Ink Manual Recapture", frame)

                    if reviewing:
                        preview = create_digit_image(points)

                        if preview is not None:
                            enlarged = cv2.resize(
                                preview,
                                (280, 280),
                                interpolation=cv2.INTER_NEAREST
                            )
                            cv2.imshow("28x28 Preview - Check Before Saving", enlarged)
                        else:
                            blank = np.zeros((280, 280), dtype=np.uint8)
                            cv2.imshow("28x28 Preview - Check Before Saving", blank)

                    key = cv2.waitKey(1) & 0xFF

                    if key == ord("q"):
                        print("Capture stopped.")
                        return

                    if reviewing:
                        if key == ord("s"):
                            if save_sample(digit, sample_number, points):
                                print("Sample saved.")
                                sample_number += 1
                                break

                        elif key == ord("r"):
                            print("Retrying this sample.")
                            points = []
                            reviewing = False
                            recording = False
                            pen_down = True
                            cv2.destroyWindow("28x28 Preview - Check Before Saving")

                        continue

                    if key in (10, 13) and not recording:
                        points = []
                        pen_down = True
                        recording = True
                        print("Recording started.")

                    elif key == ord("d") and recording:
                        pen_down = not pen_down
                        print("Pen:", "DOWN" if pen_down else "UP")

                    elif key == 32 and recording:
                        recording = False
                        reviewing = True
                        print("Review the path and 28x28 preview.")

                    elif key == ord("r"):
                        points = []
                        recording = False
                        pen_down = True
                        print("Sample cleared. Press ENTER to start again.")

        print("\nAll requested digits have been captured.")

    finally:
        camera.release()
        landmarker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()