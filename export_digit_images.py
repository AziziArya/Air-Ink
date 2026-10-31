
from pathlib import Path
import numpy as np
import cv2
import re

OUTPUT_DIR = Path("digit_capture_output")
IMAGE_DIR = OUTPUT_DIR / "digit_images_28x28"
IMAGE_DIR.mkdir(parents=True, exist_ok=True)

reports = sorted(
    OUTPUT_DIR.glob("*.txt"),
    key=lambda p: p.stat().st_mtime,
    reverse=True
)

if not reports:
    raise FileNotFoundError("TXT report not found.")

digits = {str(i): [] for i in range(10)}
current_digit = None

# Read trajectory data
with reports[0].open("r", encoding="utf-8-sig") as f:
    for raw in f:
        line = raw.strip()

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


def make_digit_image(points):
    image = np.zeros((28, 28), dtype=np.uint8)

    if len(points) < 2:
        return image

    pts = np.array(points, dtype=np.float32)

    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)

    size = max(max_xy - min_xy)
    if size < 0.001:
        return image

    # Normalize while preserving aspect ratio
    normalized = (pts - min_xy) / size

    # Fit inside 20x20 area
    pixels = normalized * 20 + 3

    pixels[:, 0] += (28 - 26) / 2 - 3
    pixels[:, 1] += (28 - 26) / 2 - 3

    pixels = np.round(pixels).astype(np.int32)

    # Draw continuous trajectory
    for i in range(1, len(pixels)):
        cv2.line(
            image,
            tuple(pixels[i - 1]),
            tuple(pixels[i]),
            255,
            2,
            cv2.LINE_AA
        )

    # Slight smoothing
    image = cv2.GaussianBlur(image, (3, 3), 0.5)

    return image


# Generate images
preview = np.full((2 * 160, 5 * 160, 3), 245, dtype=np.uint8)

for n in range(10):
    digit = str(n)
    points = digits[digit]

    image = make_digit_image(points)

    output_path = IMAGE_DIR / f"{digit}.png"
    cv2.imwrite(str(output_path), image)

    # Create enlarged preview
    enlarged = cv2.resize(
        image, (120, 120),
        interpolation=cv2.INTER_NEAREST
    )

    row, col = divmod(n, 5)
    x0, y0 = col * 160, row * 160

    cv2.putText(
        preview, f"DIGIT {digit}",
        (x0 + 10, y0 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6, (20, 20, 20), 2
    )

    preview[y0 + 30:y0 + 150, x0 + 20:x0 + 140] = cv2.cvtColor(
        enlarged, cv2.COLOR_GRAY2BGR
    )

    print(f"Digit {digit}: {len(points)} points -> saved")

# Save preview
preview_path = IMAGE_DIR / "all_digits_preview.png"
cv2.imwrite(str(preview_path), preview)

print("\nAll images generated.")
print(f"Images folder: {IMAGE_DIR}")
print(f"Preview: {preview_path}")

cv2.imshow("Air-Ink 28x28 Dataset", preview)
cv2.waitKey(0)
cv2.destroyAllWindows()
