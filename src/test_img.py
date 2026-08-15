import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from pathlib import Path
from tkinter import Tk, filedialog


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "src" / "plant_disease_model.h5"
LABELS_FILE = BASE_DIR / "src" / "labels.txt"


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)

CONF_THRESHOLD = 70


# ============================================================
# LOAD LABELS
# ============================================================

if not LABELS_FILE.exists():

    print("❌ labels.txt not found:")
    print(LABELS_FILE)

    exit()


with open(LABELS_FILE, "r") as f:

    class_names = [
        line.strip()
        for line in f
        if line.strip()
    ]


print("🍅 Tomato Classes:")

for i, name in enumerate(class_names):

    print(f"{i}: {name}")


print("\nTotal Classes:", len(class_names))


# ============================================================
# SAFETY CHECK
# ============================================================

if len(class_names) != 10:

    print("\n❌ ERROR!")
    print("Expected 10 Tomato classes.")
    print("Found:", len(class_names))

    exit()


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():

    print("\n❌ Model not found:")
    print(MODEL_PATH)

    exit()


model = tf.keras.models.load_model(
    MODEL_PATH
)

print("\n✅ Tomato disease model loaded successfully.")

print("Model output shape:", model.output_shape)


# ============================================================
# FORMAT CLASS NAME
# ============================================================

def format_class_name(name):

    name = name.replace("___", " - ")
    name = name.replace("__", " ")
    name = name.replace("_", " ")

    return name


# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_disease(img_path):

    print("\n📷 Image selected:")
    print(img_path)

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    img = image.load_img(
        img_path,
        target_size=IMG_SIZE
    )

    # --------------------------------------------------------
    # Convert image to array
    # --------------------------------------------------------

    img_array = image.img_to_array(img)

    # Add batch dimension
    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Use SAME preprocessing as training
    # --------------------------------------------------------

    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(
        img_array
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    pred_probs = model.predict(
        img_array,
        verbose=0
    )

    pred_index = np.argmax(
        pred_probs[0]
    )

    confidence = (
        pred_probs[0][pred_index] * 100
    )

    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if pred_index >= len(class_names):

        print("\n❌ Prediction index out of range!")

        return

    pred_class = class_names[pred_index]

    formatted_name = format_class_name(
        pred_class
    )

    # --------------------------------------------------------
    # Display result
    # --------------------------------------------------------

    print("\n" + "=" * 55)

    if confidence < CONF_THRESHOLD:

        print("⚠️ LOW CONFIDENCE")

    else:

        print("🌿 TOMATO DISEASE PREDICTION")

    print("=" * 55)

    print(f"\nPrediction : {formatted_name}")
    print(f"Confidence : {confidence:.2f}%")

    # --------------------------------------------------------
    # Show top 3 predictions
    # --------------------------------------------------------

    top_indices = np.argsort(
        pred_probs[0]
    )[-3:][::-1]

    print("\nTop 3 Predictions:")

    for rank, index in enumerate(
        top_indices,
        start=1
    ):

        name = format_class_name(
            class_names[index]
        )

        score = (
            pred_probs[0][index] * 100
        )

        print(
            f"{rank}. {name} "
            f"({score:.2f}%)"
        )

    print("\n" + "=" * 55)


# ============================================================
# FILE SELECTION
# ============================================================

if __name__ == "__main__":

    root = Tk()

    root.withdraw()

    print("\n📂 Please select a tomato leaf image...")

    file_path = filedialog.askopenfilename(

        title="Select Tomato Leaf Image",

        filetypes=[
            ("Image Files", "*.png;*.jpg;*.jpeg;*.bmp")
        ]
    )

    if file_path:

        predict_disease(file_path)

    else:

        print("\n❌ No image selected.")