from collections import deque
from pathlib import Path
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# -------------------- Configuration --------------------
CAMERA_INDEX = 0          # Try 1 if your external webcam is not camera 0.
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_TRAIL_POINTS = 300

MODEL_PATH = (
    Path(__file__).resolve().parent
    / "models"
    / "hand_landmarker.task"
)

# MediaPipe hand landmark index 8 = index fingertip.
INDEX_FINGER_TIP = 8


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "MediaPipe model not found.\n"
            "Run this first: python download_model.py"
        )

    # Open the webcam.
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera index {CAMERA_INDEX}. "
            "Check camera permissions or try CAMERA_INDEX = 1."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    # VIDEO mode processes frames synchronously and requires increasing timestamps.
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

    trail = deque(maxlen=MAX_TRAIL_POINTS)
    start_time = time.perf_counter()
    previous_frame_time = start_time

    print("Air-Ink Tracking MVP is running.")
    print("Show one hand to the camera.")
    print("Press 'c' to clear the trajectory.")
    print("Press 'q' to quit.")

    try:
        with vision.HandLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = cap.read()
                if not success:
                    print("Warning: could not read a frame from the webcam.")
                    break

                # Mirror the preview so movement feels natural.
                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]

                # OpenCV uses BGR; MediaPipe image input is SRGB/RGB.
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                # Monotonically increasing timestamp in milliseconds.
                timestamp_ms = int(
                    (time.perf_counter() - start_time) * 1000
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    timestamp_ms
                )

                fingertip_pixel = None

                if result.hand_landmarks:
                    hand = result.hand_landmarks[0]

                    # Draw all 21 detected hand landmarks for visual debugging.
                    for landmark in hand:
                        px = int(landmark.x * width)
                        py = int(landmark.y * height)
                        cv2.circle(
                            frame,
                            (px, py),
                            3,
                            (255, 180, 0),
                            -1
                        )

                    # Extract the index fingertip (landmark 8).
                    tip = hand[INDEX_FINGER_TIP]
                    tip_x = max(0, min(width - 1, int(tip.x * width)))
                    tip_y = max(0, min(height - 1, int(tip.y * height)))
                    fingertip_pixel = (tip_x, tip_y)

                    trail.append(fingertip_pixel)

                    cv2.circle(
                        frame,
                        fingertip_pixel,
                        9,
                        (0, 0, 255),
                        -1
                    )
                    cv2.putText(
                        frame,
                        "INDEX TIP",
                        (tip_x + 12, tip_y - 12),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 0, 255),
                        2
                    )
                    status = "Hand detected"
                else:
                    # None breaks the line so separate writing segments
                    # are not accidentally connected across tracking loss.
                    trail.append(None)
                    status = "No hand detected"

                # Draw the fingertip trajectory.
                for i in range(1, len(trail)):
                    previous_point = trail[i - 1]
                    current_point = trail[i]

                    if previous_point is not None and current_point is not None:
                        cv2.line(
                            frame,
                            previous_point,
                            current_point,
                            (0, 255, 0),
                            3
                        )

                # FPS estimate for basic performance monitoring.
                now = time.perf_counter()
                delta = now - previous_frame_time
                fps = 1.0 / delta if delta > 0 else 0.0
                previous_frame_time = now

                cv2.putText(
                    frame,
                    f"Status: {status}",
                    (12, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f}",
                    (12, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )
                cv2.putText(
                    frame,
                    "C: clear trail | Q: quit",
                    (12, height - 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("Air-Ink | Tracking MVP", frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                elif key == ord("c"):
                    trail.clear()

    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("Air-Ink Tracking MVP stopped.")


if __name__ == "__main__":
    main()
