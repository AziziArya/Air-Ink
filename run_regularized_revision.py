from pathlib import Path
import json

from data.validation import validate_dataset
from data.preprocessing import load_images
from training.trainer import train_cnn
from models.cnn_experimental import build_cnn as build_experimental_cnn

DATASET = Path("dataset_manual")
OUTPUT = Path("experiments/regularized_revision_v2")

def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    report, errors = validate_dataset(DATASET, min_per_class=2)
    if errors:
        raise RuntimeError("\n".join(errors))

    X, y, paths = load_images(DATASET)
    print("Dataset shape:", X.shape)

    results = []

    for seed in (42, 123, 2026):
        result = train_cnn(
            "regularized_v2",
            X,
            y,
            OUTPUT / "cnn",
            seed=seed,
            epochs=120,
            batch_size=16,
            model_builder=lambda: build_experimental_cnn("regularized")
        )

        results.append({
            key: result[key]
            for key in (
                "model", "seed", "accuracy", "macro_f1",
                "epochs_ran", "n_train", "n_validation", "n_test"
            )
        })

        print("RESULT:", results[-1])

    with open(OUTPUT / "summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Finished:", OUTPUT.resolve())

if __name__ == "__main__":
    main()
