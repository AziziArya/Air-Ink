from pathlib import Path
import cv2

def validate_dataset(dataset_dir, expected_classes=range(10), min_per_class=2):
    root = Path(dataset_dir)
    report = {}
    errors = []
    for label in expected_classes:
        folder = root / str(label)
        paths = sorted(folder.glob("*.png")) if folder.exists() else []
        valid = 0
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is None or image.size == 0:
                errors.append("Unreadable image: {}".format(path))
            else:
                valid += 1
        report[str(label)] = {"files": len(paths), "valid": valid}
        if valid < min_per_class:
            errors.append("Class {} has {} valid images; minimum is {}".format(label, valid, min_per_class))
    return report, errors
