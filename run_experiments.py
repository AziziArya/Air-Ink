
from pathlib import Path
import json
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from data.validation import validate_dataset
from data.preprocessing import load_images
from models.classical_ml import extract_hog, get_classical_models
from training.trainer import train_cnn


DATASET = Path("dataset_manual")
OUTPUT = Path("experiments/image_baselines_shared_split")

SPLIT_SEED = 42
TRAINING_SEEDS = (42, 123, 2026)

EPOCHS = 120
BATCH_SIZE = 16


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    report, errors = validate_dataset(
        DATASET,
        min_per_class=2
    )

    print("Dataset check:", json.dumps(report, indent=2))

    if errors:
        raise RuntimeError(
            "Dataset validation failed:\n" + "\n".join(errors)
        )

    X, y, paths = load_images(DATASET)

    print("Loaded:", X.shape)
    print(
        "Class counts:",
        np.bincount(y, minlength=10).tolist()
    )

    # Shared stratified split: 64% train, 16% validation, 20% test
    indices = np.arange(len(y))

    trainval_idx, test_idx = train_test_split(
        indices,
        test_size=0.20,
        stratify=y,
        random_state=SPLIT_SEED
    )

    train_idx, val_idx = train_test_split(
        trainval_idx,
        test_size=0.20,
        stratify=y[trainval_idx],
        random_state=SPLIT_SEED
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    print("Split sizes:")
    print("Train:", len(train_idx))
    print("Validation:", len(val_idx))
    print("Test:", len(test_idx))

    # Classical ML: fit only on training data.
    features = extract_hog(X)

    classical = {}

    for name, estimator in get_classical_models().items():
        estimator.fit(features[train_idx], y_train)

        pred = estimator.predict(features[test_idx])

        classical[name] = {
            "accuracy": float(
                accuracy_score(y_test, pred)
            ),
            "macro_f1": float(
                f1_score(y_test, pred, average="macro")
            )
        }

        joblib.dump(
            estimator,
            OUTPUT / (name + ".joblib")
        )

        print("Classical:", name, classical[name])

    with open(
        OUTPUT / "classical_metrics.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(classical, f, indent=2)

    # CNN: same train, validation and test samples for every model.
    cnn_results = []

    for kind in ("baseline", "deep", "regularized"):
        for seed in TRAINING_SEEDS:
            result = train_cnn(
                kind,
                X_train,
                y_train,
                X_val,
                y_val,
                X_test,
                y_test,
                OUTPUT / "cnn",
                seed=seed,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE
            )

            selected = {
                key: result[key]
                for key in (
                    "model",
                    "seed",
                    "accuracy",
                    "macro_f1",
                    "best_epoch",
                    "best_val_loss",
                    "epochs_ran",
                    "n_train",
                    "n_validation",
                    "n_test"
                )
            }

            cnn_results.append(selected)
            print("CNN result:", selected)

    with open(
        OUTPUT / "cnn_summary.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(cnn_results, f, indent=2)

    print("Finished. Results:", OUTPUT.resolve())


if __name__ == "__main__":
    main()
