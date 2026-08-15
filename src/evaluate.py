import os
import tensorflow as tf
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)
import seaborn as sns
import matplotlib.pyplot as plt


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

test_dir = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "test"
)

model_path = os.path.join(
    PROJECT_DIR,
    "src",
    "plant_disease_model.h5"
)


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


print("==============================================")
print("🍅 TOMATO DISEASE MODEL EVALUATION")
print("==============================================")


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\n📂 Loading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(

    test_dir,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False
)


# ============================================================
# CLASS NAMES
# ============================================================

class_names = test_ds.class_names

print("\n🍅 Test classes:")

for i, class_name in enumerate(class_names):

    print(f"{i}: {class_name}")

print(f"\n✅ Number of classes: {len(class_names)}")


if len(class_names) != 10:

    print("\n❌ ERROR: Expected 10 classes!")

    exit()


# ============================================================
# MOBILENETV2 PREPROCESSING
# ============================================================

preprocess_input = (
    tf.keras.applications.mobilenet_v2.preprocess_input
)


def preprocess_dataset(images, labels):

    images = preprocess_input(images)

    return images, labels


test_ds = test_ds.map(
    preprocess_dataset,
    num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.prefetch(
    buffer_size=tf.data.AUTOTUNE
)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n🧠 Loading trained model...")

model = tf.keras.models.load_model(
    model_path
)

print("✅ Model loaded successfully.")

print("\nModel output shape:")
print(model.output_shape)


# ============================================================
# GET TRUE LABELS
# ============================================================

print("\n🔍 Generating predictions...")

y_true = np.concatenate(
    [
        labels.numpy()
        for images, labels in test_ds
    ],
    axis=0
)


# ============================================================
# PREDICTIONS
# ============================================================

y_pred_probs = model.predict(
    test_ds
)

y_pred = np.argmax(
    y_pred_probs,
    axis=1
)


# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n" + "=" * 60)

print(
    f"🍅 TEST ACCURACY: {accuracy:.2%}"
)

print("=" * 60)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n📊 CLASSIFICATION REPORT:\n")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

plt.figure(
    figsize=(12, 10)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.title(
    "Tomato Disease Detection - Confusion Matrix"
)

plt.xticks(
    rotation=45,
    ha="right"
)

plt.yticks(
    rotation=0
)

plt.tight_layout()

plt.show()


print("\n✅ Evaluation complete!")