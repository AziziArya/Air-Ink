
import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path
from datetime import datetime

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# ==================================================
# PATHS AND SETTINGS
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"
OUTPUT_DIR = BASE_DIR / "digit_capture_output"

DIGITS = list(range(10))
CAMERA_INDEX = 0
WARMUP_REQUIRED = 10

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# DATA
# ==================================================

captures = {
    digit: {
        "status": "not_recorded",
        "duration": 0.0,
        "records": []
    }
    for digit in DIGITS
}

current_digit_index = 0

warmup_count = 0
warmup_complete = False

recording = False
paused = False

active_elapsed = 0.0
last_tick = None

last_timestamp_ms = -1
saved = False

test_start_time = datetime.now().strftime(
    "%Y-%m-%d %H:%M:%S"
)


# ==================================================
# HELPER FUNCTIONS
# ==================================================

def current_digit():
    if current_digit_index >= len(DIGITS):
        return None
    return DIGITS[current_digit_index]


def get_points(records):
    return [
        (row["x"], row["y"])
        for row in records
        if row["detected"]
    ]


def calculate_path_length(points):
    total = 0.0

    for i in range(1, len(points)):
        x1, y1 = points[i - 1]
        x2, y2 = points[i]

        total += float(np.sqrt(
            (x2 - x1) ** 2 + (y2 - y1) ** 2
        ))

    return total


def get_statistics(data):
    records = data["records"]

    total = len(records)
    detected = sum(1 for row in records if row["detected"])
    missing = total - detected

    rate = detected / total * 100 if total else 0.0

    points = get_points(records)
    path_length = calculate_path_length(points)

    bbox = None
    displacement = 0.0

    if points:
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]

        bbox = (
            min(xs), max(xs),
            min(ys), max(ys)
        )

        displacement = float(np.sqrt(
            (points[-1][0] - points[0][0]) ** 2 +
            (points[-1][1] - points[0][1]) ** 2
        ))

    return {
        "total": total,
        "detected": detected,
        "missing": missing,
        "rate": rate,
        "path_length": path_length,
        "displacement": displacement,
        "bbox": bbox
    }


# ==================================================
# RECORDING CONTROLS
# ==================================================

def start_digit():
    global recording, paused
    global active_elapsed, last_tick

    digit = current_digit()

    if digit is None:
        print("All digits are completed.")
        return

    captures[digit] = {
        "status": "recording",
        "duration": 0.0,
        "records": []
    }

    active_elapsed = 0.0
    last_tick = time.perf_counter()

    recording = True
    paused = False

    print(f"\nRecording digit {digit}...")


def finish_digit():
    global recording, paused, current_digit_index

    digit = current_digit()

    if digit is None or not recording:
        return

    captures[digit]["duration"] = active_elapsed
    captures[digit]["status"] = "recorded"

    recording = False
    paused = False

    stats = get_statistics(captures[digit])

    print(f"\nDigit {digit} saved in memory.")
    print(f"Frames: {stats['total']}")
    print(f"Detected: {stats['detected']}")
    print(f"Detection rate: {stats['rate']:.2f}%")
    print(f"Duration: {active_elapsed:.2f} seconds")

    current_digit_index += 1

    if current_digit_index == len(DIGITS):
        print("\nAll digits completed!")


def redo_digit():
    global recording, paused, active_elapsed, last_tick

    digit = current_digit()

    if digit is None:
        return

    recording = False
    paused = False
    active_elapsed = 0.0
    last_tick = None

    captures[digit] = {
        "status": "not_recorded",
        "duration": 0.0,
        "records": []
    }

    print(f"\nDigit {digit} cleared. Press ENTER to record again.")


def skip_digit():
    global recording, paused, current_digit_index

    digit = current_digit()

    if digit is None:
        return

    recording = False
    paused = False

    captures[digit]["status"] = "skipped"
    captures[digit]["duration"] = active_elapsed

    current_digit_index += 1

    print(f"\nDigit {digit} skipped.")


# ==================================================
# SAVE REPORT
# ==================================================

def save_report():
    global saved, recording, paused

    if saved:
        return

    # Preserve a recording that was still active at exit.
    if recording:
        digit = current_digit()

        if digit is not None:
            captures[digit]["duration"] = active_elapsed
            captures[digit]["status"] = "partial"

        recording = False
        paused = False

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    filename = (
        "digit_trajectory_"
        + datetime.now().strftime("%Y%m%d_%H%M%S")
        + ".txt"
    )

    output_path = OUTPUT_DIR / filename

    with open(output_path, "w", encoding="utf-8") as file:
        file.write("AIR-INK DIGIT TRAJECTORY CAPTURE TEST\n")
        file.write("=" * 65 + "\n")
        file.write(f"Started: {test_start_time}\n")
        file.write(
            f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        file.write("Landmark: index fingertip (MediaPipe landmark 8)\n")
        file.write("Coordinates: normalized x/y, approximately 0 to 1\n")
        file.write(
            "Note: Recording windows are manually started and stopped. "
            "Pen-up/pen-down is not automatically detected.\n"
        )

        file.write("\n\nSUMMARY\n")
        file.write("=" * 65 + "\n")

        for digit in DIGITS:
            data = captures[digit]
            stats = get_statistics(data)

            file.write(f"\nDIGIT: {digit}\n")
            file.write(f"Status: {data['status']}\n")
            file.write(f"Duration: {data['duration']:.4f} seconds\n")
            file.write(f"Total frames: {stats['total']}\n")
            file.write(f"Detected frames: {stats['detected']}\n")
            file.write(f"Missing frames: {stats['missing']}\n")
            file.write(f"Detection rate: {stats['rate']:.2f}%\n")
            file.write(f"Path length: {stats['path_length']:.6f}\n")
            file.write(
                f"Start-to-end displacement: "
                f"{stats['displacement']:.6f}\n"
            )

            if stats["bbox"] is not None:
                min_x, max_x, min_y, max_y = stats["bbox"]
                file.write(
                    f"Bounding box: x={min_x:.6f} to {max_x:.6f}, "
                    f"y={min_y:.6f} to {max_y:.6f}\n"
                )
            else:
                file.write("Bounding box: unavailable\n")

        file.write("\n\nFRAME-BY-FRAME DATA\n")
        file.write("=" * 65 + "\n")

        for digit in DIGITS:
            data = captures[digit]

            file.write(f"\n\nDIGIT {digit}\n")
            file.write(
                "frame | elapsed | detected | x | y | handedness\n"
            )

            for index, row in enumerate(data["records"]):
                if row["detected"]:
                    x_value = f"{row['x']:.6f}"
                    y_value = f"{row['y']:.6f}"
                else:
                    x_value = "NA"
                    y_value = "NA"

                file.write(
                    f"{index} | "
                    f"{row['elapsed']:.4f} | "
                    f"{row['detected']} | "
                    f"{x_value} | "
                    f"{y_value} | "
                    f"{row['handedness']}\n"
                )

    saved = True

    print("\n" + "=" * 60)
    print("REPORT SUCCESSFULLY SAVED")
    print(f"Full path:\n{output_path.resolve()}")
    print("=" * 60)


# ==================================================
# MAIN
# ==================================================

def main():
    global warmup_count, warmup_complete
    global recording, paused
    global active_elapsed, last_tick
    global last_timestamp_ms

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found:\n{MODEL_PATH}"
        )

    print("Loading MediaPipe model...")

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

    cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        landmarker.close()
        raise RuntimeError(
            "Could not open webcam. Check CAMERA_INDEX and camera connection."
        )

    print("\nAIR-INK DIGIT CAPTURE")
    print("---------------------")
    print("Show your hand for warm-up.")
    print("ENTER = start current digit")
    print("SPACE = finish current digit")
    print("P = pause / resume")
    print("R = redo current digit")
    print("N = skip current digit")
    print("Q = save and quit")
    print("---------------------\n")

    try:
        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read a camera frame.")
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb
            )

            timestamp_ms = time.perf_counter_ns() // 1_000_000
            timestamp_ms = max(timestamp_ms, last_timestamp_ms + 1)
            last_timestamp_ms = timestamp_ms

            result = landmarker.detect_for_video(
                mp_image,
                timestamp_ms
            )

            detected = len(result.hand_landmarks) > 0

            x = None
            y = None
            handedness = "unknown"

            if detected:
                tip = result.hand_landmarks[0][8]
                x = float(tip.x)
                y = float(tip.y)

                if result.handedness:
                    handedness = result.handedness[0][0].category_name

                cv2.circle(
                    frame,
                    (int(x * width), int(y * height)),
                    9,
                    (0, 255, 0),
                    -1
                )

            # Warm-up
            if not warmup_complete:
                if detected:
                    warmup_count += 1
                else:
                    warmup_count = 0

                if warmup_count >= WARMUP_REQUIRED:
                    warmup_complete = True
                    print("Warm-up completed. Ready to record.")

            # Record current digit
            digit = current_digit()

            if recording and not paused and digit is not None:
                now = time.perf_counter()

                if last_tick is not None:
                    dt = min(max(now - last_tick, 0.0), 0.25)
                    active_elapsed += dt

                last_tick = now

                captures[digit]["records"].append({
                    "elapsed": active_elapsed,
                    "detected": detected,
                    "x": x,
                    "y": y,
                    "handedness": handedness
                })

            # Draw current trajectory
            if digit is not None:
                points = get_points(captures[digit]["records"])

                pixel_points = [
                    (int(px * width), int(py * height))
                    for px, py in points
                ]

                for i in range(1, len(pixel_points)):
                    cv2.line(
                        frame,
                        pixel_points[i - 1],
                        pixel_points[i],
                        (255, 150, 0),
                        3
                    )

            # Display status
            if not warmup_complete:
                status = f"WARM-UP: {warmup_count}/{WARMUP_REQUIRED}"
            elif digit is None:
                status = "ALL DIGITS COMPLETED"
            elif paused:
                status = f"DIGIT {digit}: PAUSED"
            elif recording:
                status = f"DIGIT {digit}: RECORDING"
            else:
                status = f"DIGIT {digit}: READY"

            cv2.putText(
                frame, status, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                (0, 255, 255), 2
            )

            if digit is not None:
                cv2.putText(
                    frame,
                    f"Elapsed: {active_elapsed:.2f}s",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            cv2.putText(
                frame,
                "ENTER Start | SPACE Finish | P Pause",
                (20, height - 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.putText(
                frame,
                "R Redo | N Skip | Q Save & Quit",
                (20, height - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )

            cv2.imshow("AIR-INK DIGIT CAPTURE", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                print("Q pressed. Saving report...")
                break

            if not warmup_complete:
                continue

            if digit is None:
                break

            if key in (13, 10):
                if not recording:
                    start_digit()

            elif key == 32:
                if recording:
                    finish_digit()

            elif key in (ord("p"), ord("P")):
                if recording and not paused:
                    paused = True
                    print("Paused.")
                elif recording and paused:
                    paused = False
                    last_tick = time.perf_counter()
                    print("Resumed.")

            elif key in (ord("r"), ord("R")):
                redo_digit()

            elif key in (ord("n"), ord("N")):
                skip_digit()

    finally:
        try:
            save_report()
        finally:
            cap.release()
            cv2.destroyAllWindows()
            landmarker.close()
            print("Camera and MediaPipe closed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nPROGRAM ERROR:")
        print(repr(error))
        raise
