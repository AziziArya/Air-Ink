import tensorflow as tf
from data.augmentation import make_augmentation

def build_cnn(kind="baseline", input_shape=(28, 28, 1), classes=10):
    layers = [
        tf.keras.layers.Input(shape=input_shape),
        make_augmentation()
    ]

    if kind == "baseline":
        layers += [
            tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu"),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(96, activation="relu"),
            tf.keras.layers.Dropout(0.25)
        ]

    elif kind == "deep":
        for filters in (32, 64, 128):
            layers += [
                tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu"),
                tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu"),
                tf.keras.layers.MaxPooling2D()
            ]
        layers += [
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(128, activation="relu")
        ]

    elif kind == "regularized":
        for filters in (32, 64):
            layers += [
                tf.keras.layers.Conv2D(filters, 3, padding="same", use_bias=False),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Activation("relu"),
                tf.keras.layers.MaxPooling2D()
            ]

        layers += [
            tf.keras.layers.Conv2D(96, 3, padding="same", use_bias=False),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Activation("relu"),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(96, activation="relu"),
            tf.keras.layers.Dropout(0.25)
        ]

    else:
        raise ValueError("Unknown CNN kind: {}".format(kind))

    layers.append(tf.keras.layers.Dense(classes, activation="softmax"))

    model = tf.keras.Sequential(layers, name="air_ink_" + kind)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model
