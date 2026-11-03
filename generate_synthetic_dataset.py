import cv2
import numpy as np
from pathlib import Path
import random


# ---------------- SETTINGS ----------------

OUTPUT_DIR = Path("dataset_synthetic")

IMAGES_PER_DIGIT = 300
IMAGE_SIZE = 28
CANVAS_SIZE = 64

FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
]

random.seed(42)
np.random.seed(42)

# -------------------------------------------


def create_base_digit(digit):
    canvas = np.zeros(
        (CANVAS_SIZE, CANVAS_SIZE),
        dtype=np.uint8
    )

    font = random.choice(FONTS)
    scale = random.uniform(1.0, 1.45)
    thickness = random.randint(2, 4)

    text = str(digit)

    text_size, baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )

    x = (CANVAS_SIZE - text_size[0]) // 2
    y = (CANVAS_SIZE + text_size[1]) // 2

    cv2.putText(
        canvas,
        text,
        (x, y),
        font,
        scale,
        255,
        thickness,
        cv2.LINE_AA
    )

    return canvas


def augment_image(image):
    result = image.copy()

    center = (CANVAS_SIZE / 2, CANVAS_SIZE / 2)

    # Small rotation only; no upside-down digits.
    angle = random.uniform(-7, 7)

    matrix = cv2.getRotationMatrix2D(
        center,
        angle,
        random.uniform(0.88, 1.12)
    )

    # Small horizontal and vertical shear.
    matrix[0, 1] += random.uniform(-0.07, 0.07)
    matrix[1, 0] += random.uniform(-0.04, 0.04)

    # Small translation.
    matrix[0, 2] += random.uniform(-3, 3)
    matrix[1, 2] += random.uniform(-3, 3)

    result = cv2.warpAffine(
        result,
        matrix,
        (CANVAS_SIZE, CANVAS_SIZE),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0
    )

    # Slightly change stroke thickness.
    if random.random() < 0.5:
        kernel = np.ones((2, 2), np.uint8)
        result = cv2.dilate(result, kernel, iterations=1)
    else:
        kernel = np.ones((2, 2), np.uint8)
        result = cv2.erode(result, kernel, iterations=1)

    # Small blur variation.
    if random.random() < 0.3:
        result = cv2.GaussianBlur(result, (3, 3), 0.5)

    # Resize to 28x28.
    result = cv2.resize(
        result,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA
    )

    return result


def main():
    total = 0

    for digit in range(10):

        digit_folder = OUTPUT_DIR / str(digit)
        digit_folder.mkdir(parents=True, exist_ok=True)

        for index in range(IMAGES_PER_DIGIT):

            base = create_base_digit(digit)
            augmented = augment_image(base)

            filename = digit_folder / f"synthetic_{index + 1:04d}.png"

            cv2.imwrite(str(filename), augmented)

            total += 1

        print(
            f"Digit {digit}: "
            f"{IMAGES_PER_DIGIT} images generated."
        )

    print("\nSynthetic dataset generation complete.")
    print(f"Total images: {total}")
    print(f"Output: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()