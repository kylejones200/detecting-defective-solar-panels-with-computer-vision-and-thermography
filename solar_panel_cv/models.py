"""Model builders for solar panel defect classification."""

import tensorflow as tf


def create_mobilenet_model(num_classes: int, img_height: int, img_width: int):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    return tf.keras.Sequential(
        [
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes),
        ]
    ), base


def create_binary_model(img_height: int, img_width: int):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(img_height, img_width, 3),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False
    model = tf.keras.Sequential(
        [
            base,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    return model, base


def create_vgg16_model(num_classes: int, img_height: int, img_width: int):
    base_model = tf.keras.applications.VGG16(
        include_top=False,
        weights="imagenet",
        input_shape=(img_height, img_width, 3),
    )
    base_model.trainable = False
    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    x = tf.keras.applications.vgg16.preprocess_input(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes)(x)
    return tf.keras.Model(inputs, outputs), base_model
