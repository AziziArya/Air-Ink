
import cv2
import numpy as np
import tensorflow as tf
import json
import random

from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ---------------- CONFIG ----------------

IMAGE_SIZE = 28
DATASET_DIR = Path("dataset_manual")
MODEL_DIR = Path("trained_models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "air_ink_cnn.keras"

EPOCHS = 40
BATCH_SIZE = 16
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# ----------------------------------------


def load_dataset():
    images = []
    labels = []

    for digit in range(10):
        folder = DATASET_DIR / str(digit)

        if not folder.exists():
            print(f"Warning: digit {digit} folder not found.")
            continue

        for path in sorted(folder.glob("*.png")):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)

            if image is None:
                print(f"Could not read: {path}")
                continue

            image = cv2.resize(
                image,
                (IMAGE_SIZE, IMAGE_SIZE)
            )

            image = image.astype("float32") / 255.0

            images.append(image)
            labels.append(digit)

    if not images:
        raise RuntimeError("No images found in dataset_manual.")

    X = np.array(images, dtype=np.float32)[..., np.newaxis]
    y = np.array(labels, dtype=np.int64)

    return X, y


def build_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28, 1)),

        tf.keras.layers.Conv2D(
            32, (3, 3),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(
            64, (3, 3),
            activation="relu",
            padding="same"
        ),
        tf.keras.layers.MaxPooling2D((2, 2)),

        tf.keras.layers.Conv2D(
            128, (3, 3),
            activation="relu",
            padding="same"
        ),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.35),

        tf.keras.layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model


def main():
    print("Loading manual dataset...")

    X, y = load_dataset()

    print("\nDataset distribution:")

    for digit in range(10):
        print(f"Digit {digit}: {np.sum(y == digit)}")

    # Split while preserving digit proportions.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=SEED,
        stratify=y
    )

    print(f"\nTraining images: {len(X_train)}")
    print(f"Testing images: {len(X_test)}")

    model = build_model()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=7,
            restore_best_weights=True
        ),

        tf.keras.callbacks.ModelCheckpoint(
            str(MODEL_PATH),
            monitor="val_loss",
            save_best_only=True
        )
    ]

    print("\nTraining CNN...")

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.20,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    print("\nEvaluating model...")

    loss, accuracy = model.evaluate(
        X_test,
        y_test,
        verbose=0
    )

    predictions = model.predict(X_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    print(f"\nTest loss: {loss:.4f}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predicted_labels,
            labels=list(range(10)),
            zero_division=0
        )
    )

    print("\nConfusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predicted_labels,
            labels=list(range(10))
        )
    )

    model.save(MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()
