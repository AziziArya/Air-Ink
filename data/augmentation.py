import tensorflow as tf

def make_augmentation():
    # Conservative, label-preserving transforms for handwritten digits.
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(0.035, fill_mode="constant", fill_value=0.0),
        tf.keras.layers.RandomTranslation(0.06, 0.06, fill_mode="constant", fill_value=0.0),
        tf.keras.layers.RandomZoom(0.08, fill_mode="constant", fill_value=0.0),
    ], name="train_only_augmentation")
