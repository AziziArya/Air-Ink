
from pathlib import Path
import json
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from training.callbacks import make_callbacks


def train_cnn(
    kind,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    output_dir,
    seed=42,
    epochs=120,
    batch_size=16,
    model_builder=None
):
    tf.keras.utils.set_random_seed(seed)

    if model_builder is None:
        from models.cnn_baseline import build_cnn
        model = build_cnn(kind)
    else:
        model = model_builder()

    run_dir = Path(output_dir) / "{}_seed{}".format(kind, seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=make_callbacks(run_dir),
        verbose=2
    )

    weights = run_dir / "best.weights.h5"
    if weights.exists():
        model.load_weights(str(weights))

    probs = model.predict(X_test, verbose=0)
    pred = np.argmax(probs, axis=1)

    best_epoch = int(np.argmin(history.history["val_loss"]) + 1)
    best_val_loss = float(np.min(history.history["val_loss"]))

    metrics = {
        "model": kind,
        "seed": seed,
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(
            f1_score(y_test, pred, average="macro")
        ),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_ran": len(history.history["loss"]),
        "n_train": int(len(X_train)),
        "n_validation": int(len(X_val)),
        "n_test": int(len(X_test)),
        "classification_report": classification_report(
            y_test,
            pred,
            output_dict=True,
            zero_division=0
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            pred,
            labels=list(range(10))
        ).tolist()
    }

    model.save(str(run_dir / "final_model.h5"), save_format="h5")

    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
