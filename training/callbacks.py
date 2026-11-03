from pathlib import Path
import tensorflow as tf

def make_callbacks(run_dir):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    return [
        tf.keras.callbacks.ModelCheckpoint(str(run_dir / "best.weights.h5"), monitor="val_loss", save_best_only=True, save_weights_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, min_delta=1e-4, restore_best_weights=True, verbose=1),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, min_lr=1e-6, verbose=1),
        tf.keras.callbacks.CSVLogger(str(run_dir / "history.csv")),
    ]
