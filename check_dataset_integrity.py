from pathlib import Path
from PIL import Image
import hashlib
import numpy as np
from collections import defaultdict

root = Path("dataset_manual")
classes = sorted(p for p in root.iterdir() if p.is_dir())

hashes = defaultdict(list)
print("CLASS | FILES | SHAPES | MIN | MAX | MEAN")

for folder in classes:
    files = list(folder.glob("*.png"))
    shapes = set()
    values = []

    for path in files:
        img = Image.open(path).convert("L")
        arr = np.array(img)
        shapes.add(arr.shape)
        values.extend([arr.min(), arr.max(), arr.mean()])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        hashes[digest].append(str(path))

    print(folder.name, len(files), shapes,
          round(min(values), 2), round(max(values), 2),
          round(float(np.mean(values)), 2))

duplicates = [paths for paths in hashes.values() if len(paths) > 1]

print("\nEXACT DUPLICATE GROUPS:", len(duplicates))
for group in duplicates[:20]:
    print(group)
