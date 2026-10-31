
import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================================================
# AIR-INK TRACKING QUALITY TEST 3
# SINGLE TXT OUTPUT
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "hand_landmarker.task"
OUTPUT_DIR = BASE_DIR / "quality_test_3_output"

CAMERA_INDEX = 0
WARMUP_FRAMES = 10
STEPS = ["UP", "LEFT", "DOWN", "RIGHT", "CIRCLES"]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )


# =========================================================
# MEDIAPIPE
# =========================================================

options = vision.HandLandmarkerOptions(
    base_options=python.BaseOptions(
        model_asset_path=str(MODEL_PATH)
    ),
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

landmarker = vision.HandLandmarker.create_from_options(options)


# =========================================================
# CAMERA
# =========================================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    landmarker.close()
    raise RuntimeError("Could not open webcam.")


# =========================================================
# TEST STATE
# =========================================================

state = "WARMUP"
step_index = 0

warmup_count = 0
active_elapsed = 0.0
last_tick = time.perf_counter()
last_timestamp_ms = 0

running = True
trail = []

all_samples = []
step_results = {}
pause_events = []
gap_events = []

current_step_samples = []
current_step_start = 0.0
current_gap_start = None

test_started_at = None
test_finished_at = None


# =========================================================
# HELPERS
# =========================================================

def current_step():
    if step_index >= len(STEPS):
        return "FINISHED"
    return STEPS[step_index]


def circle_sweep(points):
    if len(points) < 10:
        return 0.0

    points = np.array(points, dtype=np.float32)
    center = np.mean(points, axis=0)

    angles = np.arctan2(
        points[:, 1] - center[1],
        points[:, 0] - center[0]
    )

    angles = np.unwrap(angles)

    return float(abs(angles[-1] - angles[0]))


def calculate_result(name, samples):
    total = len(samples)
    detected = [
        s for s in samples if s["detected"]
    ]

    count = len(detected)
    missing = total - count

    rate = count / total * 100 if total else 0.0

    duration = (
        samples[-1]["elapsed"] - samples[0]["elapsed"]
        if total > 1 else 0.0
    )

    points = [
        (s["x"], s["y"])
        for s in detected
        if s["x"] is not None and s["y"] is not None
    ]

    displacement = 0.0
    sweep = 0.0

    if len(points) >= 2:
        displacement = float(
            np.linalg.norm(
                np.array(points[-1]) - np.array(points[0])
            )
        )

    if name == "CIRCLES":
        sweep = circle_sweep(points)

    return {
        "name": name,
        "frames": total,
        "detected": count,
        "missing": missing,
        "rate": rate,
        "duration": duration,
        "displacement": displacement,
        "sweep": sweep,
    }


def add_sample(name, detected, x, y, handedness):
    sample = {
        "elapsed": active_elapsed,
        "step": name,
        "detected": detected,
        "x": x,
        "y": y,
        "handedness": handedness,
    }

    all_samples.append(sample)
    current_step_samples.append(sample)


def start_step():
    global state
    global current_step_samples
    global current_step_start
    global last_tick
    global trail
    global current_gap_start

    current_step_samples = []
    current_step_start = active_elapsed
    current_gap_start = None
    trail = []

    state = "ACTIVE"
    last_tick = time.perf_counter()

    print("Started:", current_step())


def finish_gap_if_open():
    global current_gap_start

    if current_gap_start is not None:
        gap_events.append({
            "step": current_step(),
            "start": current_gap_start,
            "end": active_elapsed,
            "duration": active_elapsed - current_gap_start,
        })
        current_gap_start = None


def finish_step():
    global state
    global step_index
    global current_step_samples
    global current_gap_start
    global test_finished_at

    finish_gap_if_open()

    name = current_step()

    step_results[name] = calculate_result(
        name, current_step_samples
    )

    print("Finished:", name)
    print("Detection rate:",
          round(step_results[name]["rate"], 2), "%")

    step_index += 1
    current_step_samples = []

    if step_index >= len(STEPS):
        state = "FINISHED"
        test_finished_at = datetime.now()
        save_report()
    else:
        state = "READY"


def start_pause():
    global state
    global last_tick
    global current_gap_start

    if current_gap_start is not None:
        gap_events.append({
            "step": current_step(),
            "start": current_gap_start,
            "end": active_elapsed,
            "duration": active_elapsed - current_gap_start,
            "ended_by": "pause",
        })
        current_gap_start = None

    pause_events.append({
        "event": "PAUSE",
        "step": current_step(),
        "elapsed": active_elapsed,
        "clock_time": datetime.now().isoformat(),
    })

    state = "PAUSED"


def resume_pause():
    global state
    global last_tick

    pause_events.append({
        "event": "RESUME",
        "step": current_step(),
        "elapsed": active_elapsed,
        "clock_time": datetime.now().isoformat(),
    })

    state = "ACTIVE"
    last_tick = time.perf_counter()


# =========================================================
# SINGLE TXT REPORT
# =========================================================

def save_report():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"AIR_INK_TEST_3_{timestamp}.txt"

    total = len(all_samples)
    detected = sum(1 for s in all_samples if s["detected"])
    missing = total - detected

    overall_rate = (
        detected / total * 100 if total else 0.0
    )

    lines = []

    def write(text=""):
        lines.append(str(text))

    write("=" * 65)
    write("AIR-INK TRACKING QUALITY TEST 3")
    write("COMPLETE SINGLE-FILE REPORT")
    write("=" * 65)

    write()
    write("TEST INFORMATION")
    write("-" * 65)
    write(f"Started: {test_started_at}")
    write(f"Finished: {test_finished_at or datetime.now()}")
    write(f"Warm-up required frames: {WARMUP_FRAMES}")
    write(f"Warm-up achieved: {warmup_count >= WARMUP_FRAMES}")
    write(f"Active measurement time: {active_elapsed:.4f} seconds")

    write()
    write("OVERALL RESULTS")
    write("-" * 65)
    write(f"Total measured frames: {total}")
    write(f"Detected frames: {detected}")
    write(f"Missing frames: {missing}")
    write(f"Detection rate: {overall_rate:.2f}%")

    write()
    write("STEP RESULTS")
    write("-" * 65)

    for name in STEPS:
        if name not in step_results:
            write(f"{name}: NOT COMPLETED")
            continue

        r = step_results[name]

        write()
        write(f"STEP: {name}")
        write(f"Frames: {r['frames']}")
        write(f"Detected frames: {r['detected']}")
        write(f"Missing frames: {r['missing']}")
        write(f"Detection rate: {r['rate']:.2f}%")
        write(f"Duration: {r['duration']:.4f} seconds")
        write(f"Displacement: {r['displacement']:.5f}")
        write(f"Angular sweep: {r['sweep']:.4f} radians")

    write()
    write("TRACKING GAP EVENTS")
    write("-" * 65)

    if not gap_events:
        write("No tracking gaps recorded.")
    else:
        for i, gap in enumerate(gap_events, 1):
            write(
                f"{i}. Step={gap['step']} | "
                f"Start={gap['start']:.4f}s | "
                f"End={gap['end']:.4f}s | "
                f"Duration={gap['duration']:.4f}s"
            )

    write()
    write("PAUSE / RESUME EVENTS")
    write("-" * 65)

    if not pause_events:
        write("No pause events.")
    else:
        for i, event in enumerate(pause_events, 1):
            write(
                f"{i}. {event['event']} | "
                f"Step={event['step']} | "
                f"Elapsed={event['elapsed']:.4f}s | "
                f"Clock={event['clock_time']}"
            )

    write()
    write("FRAME-BY-FRAME DATA")
    write("-" * 65)
    write(
        "Frame | Elapsed(s) | Step | Detected | "
        "X normalized | Y normalized | Handedness"
    )

    for i, s in enumerate(all_samples, 1):
        x_text = (
            f"{s['x']:.5f}"
            if s["x"] is not None else "N/A"
        )
        y_text = (
            f"{s['y']:.5f}"
            if s["y"] is not None else "N/A"
        )

        write(
            f"{i} | {s['elapsed']:.4f} | "
            f"{s['step']} | {s['detected']} | "
            f"{x_text} | {y_text} | {s['handedness']}"
        )

    write()
    write("=" * 65)
    write("END OF REPORT")
    write("=" * 65)

    output_path.write_text(
        "\n".join(lines),
        encoding="utf-8"
    )

    print("\nREPORT SAVED:")
    print(output_path)


# =========================================================
# MAIN LOOP
# =========================================================

print("\nAIR-INK TRACKING QUALITY TEST 3")
print("--------------------------------")
print("Show your hand and wait for warm-up.")
print("ENTER = start step")
print("SPACE = finish step")
print("P = pause / resume")
print("Q = save and quit")
print("--------------------------------")

try:
    while running:

        ret, frame = cap.read()

        if not ret:
            print("Camera read failed.")
            break

        frame = cv2.flip(frame, 1)

        now = time.perf_counter()

        timestamp_ms = int(now * 1000)
        if timestamp_ms <= last_timestamp_ms:
            timestamp_ms = last_timestamp_ms + 1
        last_timestamp_ms = timestamp_ms

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        hand_detected = len(result.hand_landmarks) > 0

        x = None
        y = None
        handedness = "unknown"

        if hand_detected:
            landmarks = result.hand_landmarks[0]
            x = float(landmarks[8].x)
            y = float(landmarks[8].y)

            if result.handedness:
                handedness = (
                    result.handedness[0][0].category_name
                )

        # WARM-UP
        if state == "WARMUP":
            if hand_detected:
                warmup_count += 1
            else:
                warmup_count = 0

            if warmup_count >= WARMUP_FRAMES:
                state = "READY"
                test_started_at = datetime.now()
                print("\nWarm-up complete. Press ENTER.")

        # ACTIVE MEASUREMENT
        elif state == "ACTIVE":
            dt = max(0.0, now - last_tick)
            active_elapsed += dt
            last_tick = now

            add_sample(
                current_step(),
                hand_detected,
                x,
                y,
                handedness
            )

            if hand_detected:
                if current_gap_start is not None:
                    gap_events.append({
                        "step": current_step(),
                        "start": current_gap_start,
                        "end": active_elapsed,
                        "duration": active_elapsed - current_gap_start,
                    })
                    current_gap_start = None

                px = int(x * frame.shape[1])
                py = int(y * frame.shape[0])

                trail.append((px, py))
                if len(trail) > 300:
                    trail.pop(0)

            else:
                if current_gap_start is None:
                    current_gap_start = active_elapsed

        # DRAW
        display = frame.copy()

        for i in range(1, len(trail)):
            cv2.line(
                display,
                trail[i - 1],
                trail[i],
                (0, 255, 255),
                2
            )

        if hand_detected:
            px = int(x * display.shape[1])
            py = int(y * display.shape[0])
            cv2.circle(
                display,
                (px, py),
                8,
                (0, 255, 0),
                -1
            )

        if state == "WARMUP":
            title = "WARM-UP"
            info = f"Hand frames: {warmup_count}/{WARMUP_FRAMES}"

        elif state == "READY":
            title = f"READY: {current_step()}"
            info = "Press ENTER to start"

        elif state == "ACTIVE":
            title = f"ACTIVE: {current_step()}"
            info = "SPACE = finish | P = pause"

        elif state == "PAUSED":
            title = f"PAUSED: {current_step()}"
            info = "P = resume"

        else:
            title = "TEST COMPLETED"
            info = "Q = close"

        cv2.rectangle(
            display,
            (0, 0),
            (display.shape[1], 105),
            (25, 25, 25),
            -1
        )

        cv2.putText(
            display,
            title,
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 255),
            2
        )

        cv2.putText(
            display,
            info,
            (20, 78),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        if state == "ACTIVE":
            status = (
                "HAND DETECTED"
                if hand_detected
                else "HAND NOT DETECTED"
            )

            color = (
                (0, 255, 0)
                if hand_detected
                else (0, 0, 255)
            )

            cv2.putText(
                display,
                status,
                (20, display.shape[0] - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2
            )

            cv2.putText(
                display,
                f"Active time: {active_elapsed:.2f}s",
                (display.shape[1] - 280, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        cv2.imshow("Air-Ink Test 3", display)

        key = cv2.waitKey(1) & 0xFF

        if key in (ord("q"), ord("Q")):
            if state == "ACTIVE":
                finish_gap_if_open()
                step_results[current_step()] = calculate_result(
                    current_step(), current_step_samples
                )

            if all_samples:
                test_finished_at = datetime.now()
                save_report()

            running = False

        elif key in (ord("p"), ord("P")):
            if state == "ACTIVE":
                start_pause()
            elif state == "PAUSED":
                resume_pause()

        elif key in (13, 10):
            if state == "READY":
                start_step()

        elif key == 32:
            if state == "ACTIVE":
                finish_step()

finally:
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

print("\nProgram closed.")
