import tensorflow as tf
from tensorflow.keras import layers, models


def build_model(num_classes):

    # ========================================================
    # MobileNetV2 pretrained on ImageNet
    # ========================================================

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet"
    )

    # Freeze pretrained layers initially
    base_model.trainable = False

    # ========================================================
    # Classification Head
    # ========================================================

    model = models.Sequential([

        base_model,

        layers.GlobalAveragePooling2D(),

        layers.Dropout(0.3),

        layers.Dense(
            128,
            activation="relu"
        ),

        layers.Dropout(0.2),

        # Automatically becomes Dense(10)
        # for our Tomato-only dataset
        layers.Dense(
            num_classes,
            activation="softmax"
        )
    ])

    return model