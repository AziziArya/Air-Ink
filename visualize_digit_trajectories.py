
from pathlib import Path
import cv2
import numpy as np
import re

OUTPUT_DIR = Path("digit_capture_output")

reports = sorted(
    OUTPUT_DIR.glob("*.txt"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)

if not reports:
    raise FileNotFoundError("No TXT report found.")

digits = {str(i): [] for i in range(10)}
current_digit = None

# Read trajectory data
with reports[0].open("r", encoding="utf-8-sig") as file:
    for raw_line in file:
        line = raw_line.strip()

        match = re.match(r"^DIGIT\s+([0-9])\s*$", line, re.I)
        if match:
            current_digit = match.group(1)
            continue

        if current_digit is None or "|" not in line:
            continue

        parts = [p.strip() for p in line.split("|")]

        if len(parts) < 6:
            continue

        try:
            int(parts[0])

            if parts[2].lower() != "true":
                continue

            x = float(parts[3])
            y = float(parts[4])

            if 0 <= x <= 1 and 0 <= y <= 1:
                digits[current_digit].append((x, y))

        except ValueError:
            continue

# Canvas settings
CELL_W = 250
CELL_H = 270
canvas = np.full((CELL_H * 2, CELL_W * 5, 3), 248, np.uint8)

for number in range(10):
    digit = str(number)
    points = digits[digit]

    row, col = divmod(number, 5)
    x0 = col * CELL_W
    y0 = row * CELL_H

    left = x0 + 22
    top = y0 + 48
    width = CELL_W - 44
    height = CELL_H - 85

    # Cell background
    cv2.rectangle(
        canvas,
        (x0 + 3, y0 + 3),
        (x0 + CELL_W - 3, y0 + CELL_H - 3),
        (225, 225, 225),
        -1
    )

    cv2.putText(
        canvas, f"DIGIT {digit}",
        (x0 + 12, y0 + 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65, (20, 20, 20), 2
    )

    cv2.rectangle(
        canvas,
        (left, top),
        (left + width, top + height),
        (170, 170, 170), 1
    )

    if len(points) < 2:
        cv2.putText(
            canvas, "No data",
            (left + 10, top + height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 0, 200), 1
        )
        continue

    # Convert to numpy array
    pts = np.array(points, dtype=np.float32)

    # Crop to actual trajectory bounds
    min_x, min_y = pts.min(axis=0)
    max_x, max_y = pts.max(axis=0)

    span_x = max_x - min_x
    span_y = max_y - min_y

    # Avoid division by zero for nearly straight lines
    span_x = max(span_x, 0.015)
    span_y = max(span_y, 0.015)

    # Add padding around the trajectory
    pad_x = span_x * 0.18
    pad_y = span_y * 0.18

    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y

    span_x = max_x - min_x
    span_y = max_y - min_y

    # Fit trajectory into the box, preserving aspect ratio
    scale = min(
        (width - 20) / span_x,
        (height - 20) / span_y
    )

    draw_w = span_x * scale
    draw_h = span_y * scale

    offset_x = left + (width - draw_w) / 2
    offset_y = top + (height - draw_h) / 2

    pixels = []

    for x, y in pts:
        px = int(offset_x + (x - min_x) * scale)
        py = int(offset_y + (y - min_y) * scale)
        pixels.append((px, py))

    # Draw trajectory
    for i in range(1, len(pixels)):
        cv2.line(
            canvas,
            pixels[i - 1],
            pixels[i],
            (210, 90, 35),
            2,
            cv2.LINE_AA
        )

    # Start and end markers
    cv2.circle(canvas, pixels[0], 5, (0, 160, 0), -1)
    cv2.circle(canvas, pixels[-1], 5, (0, 0, 220), -1)

    cv2.putText(
        canvas,
        f"{len(points)} points",
        (left + 5, top + height + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.4,
        (60, 60, 60),
        1
    )

# Save full-resolution image
output_path = OUTPUT_DIR / "digit_trajectories_cropped.png"

if not cv2.imwrite(str(output_path), canvas):
    raise RuntimeError("Could not save image.")

print("\nPoints per digit:")
for i in range(10):
    print(f"{i}: {len(digits[str(i)])}")

print(f"\nSaved image: {output_path}")

# Display
cv2.imshow("Air-Ink - Cropped Digits 0-9", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()
