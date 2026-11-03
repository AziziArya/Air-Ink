from pathlib import Path
import cv2
import numpy as np

def load_images(dataset_dir, image_size=28):
    X, y, paths = [], [], []
    root = Path(dataset_dir)
    for label in range(10):
        folder = root / str(label)
        for path in sorted(folder.glob("*.png")):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None:
                continue
            image = cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)
            X.append(image.astype("float32") / 255.0)
            y.append(label)
            paths.append(str(path))
    if not X:
        raise RuntimeError("No readable PNG images found in {}".format(root.resolve()))
    X = np.asarray(X, dtype=np.float32)[..., np.newaxis]
    y = np.asarray(y, dtype=np.int64)
    return X, y, paths
