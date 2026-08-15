from flask import Flask, request, jsonify
from flask_cors import CORS

import tensorflow as tf
import numpy as np

from PIL import Image
from pathlib import Path


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

CORS(app)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = BASE_DIR / "src" / "plant_disease_model.h5"
LABELS_PATH = BASE_DIR / "src" / "labels.txt"


# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)

CONFIDENCE_THRESHOLD = 70.0


# ============================================================
# VERIFY MODEL AND LABEL FILE
# ============================================================

if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"❌ Model not found:\n{MODEL_PATH}"
    )


if not LABELS_PATH.exists():

    raise FileNotFoundError(
        f"❌ Labels file not found:\n{LABELS_PATH}"
    )


# ============================================================
# LOAD LABELS
# ============================================================

with open(LABELS_PATH, "r") as file:

    CLASS_NAMES = [
        line.strip()
        for line in file
        if line.strip()
    ]


print("\n🍅 TOMATO CLASSES")

for index, name in enumerate(CLASS_NAMES):

    print(f"{index}: {name}")


if len(CLASS_NAMES) != 10:

    raise ValueError(
        f"Expected 10 classes, "
        f"but found {len(CLASS_NAMES)}."
    )


# ============================================================
# LOAD MODEL
# ============================================================

print("\n🧠 Loading MobileNetV2 model...")

MODEL = tf.keras.models.load_model(
    MODEL_PATH
)

print("✅ Model loaded successfully.")

print(
    "Model output shape:",
    MODEL.output_shape
)


# ============================================================
# FORMAT CLASS NAME
# ============================================================

def format_class_name(name):

    name = name.replace(
        "Tomato__",
        ""
    )

    name = name.replace(
        "Tomato_",
        ""
    )

    name = name.replace(
        "__",
        " "
    )

    name = name.replace(
        "_",
        " "
    )

    return name.strip().title()


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify({

        "success": True,

        "application":
            "Tomato Guard AI",

        "status":
            "healthy",

        "plant":
            "Tomato",

        "model":
            "MobileNetV2",

        "classes":
            len(CLASS_NAMES)

    })


# ============================================================
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if "image" not in request.files:

        return jsonify({

            "success": False,

            "error":
                "No image uploaded."

        }), 400


    file = request.files["image"]


    # --------------------------------------------------------
    # Check filename
    # --------------------------------------------------------

    if file.filename == "":

        return jsonify({

            "success": False,

            "error":
                "No image selected."

        }), 400


    try:

        # ----------------------------------------------------
        # Open image
        # ----------------------------------------------------

        image = Image.open(
            file.stream
        ).convert("RGB")


        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = image.resize(
            IMG_SIZE
        )


        # ----------------------------------------------------
        # Convert to NumPy
        # ----------------------------------------------------

        image_array = np.array(
            image,
            dtype=np.float32
        )


        # Add batch dimension

        image_array = np.expand_dims(
            image_array,
            axis=0
        )


        # ----------------------------------------------------
        # MobileNetV2 preprocessing
        # ----------------------------------------------------

        image_array = (
            tf.keras.applications
            .mobilenet_v2
            .preprocess_input(
                image_array
            )
        )


        # ----------------------------------------------------
        # Predict
        # ----------------------------------------------------

        predictions = MODEL.predict(
            image_array,
            verbose=0
        )[0]


        # ----------------------------------------------------
        # Best prediction
        # ----------------------------------------------------

        predicted_index = int(
            np.argmax(predictions)
        )

        confidence = float(
            predictions[predicted_index] * 100
        )

        predicted_class = (
            CLASS_NAMES[predicted_index]
        )


        # ----------------------------------------------------
        # Top 3
        # ----------------------------------------------------

        top_indices = np.argsort(
            predictions
        )[-3:][::-1]


        top_predictions = []


        for index in top_indices:

            top_predictions.append({

                "name":
                    format_class_name(
                        CLASS_NAMES[index]
                    ),

                "confidence":
                    round(
                        float(
                            predictions[index]
                            * 100
                        ),
                        2
                    )

            })


        # ----------------------------------------------------
        # Healthy?
        # ----------------------------------------------------

        is_healthy = (
            "healthy"
            in predicted_class.lower()
        )


        # ----------------------------------------------------
        # Confidence status
        # ----------------------------------------------------

        low_confidence = (
            confidence
            < CONFIDENCE_THRESHOLD
        )


        # ----------------------------------------------------
        # Send JSON
        # ----------------------------------------------------

        return jsonify({

            "success": True,

            "plant":
                "Tomato",

            "prediction":
                format_class_name(
                    predicted_class
                ),

            "confidence":
                round(
                    confidence,
                    2
                ),

            "healthy":
                is_healthy,

            "low_confidence":
                low_confidence,

            "top_predictions":
                top_predictions

        })


    except Exception as error:

        print(
            "\n❌ Prediction error:",
            error
        )


        return jsonify({

            "success": False,

            "error":
                "Unable to process the image."

        }), 500


# ============================================================
# START SERVER
# ============================================================

if __name__ == "__main__":

    print("\n" + "=" * 60)

    print("🍅 TOMATO GUARD AI")

    print("=" * 60)

    print(
        "AI-Powered Tomato Disease Detection"
    )

    print(
        "\n🍅 Plant scope: TOMATO ONLY"
    )

    print(
        "🧠 Model: MobileNetV2"
    )

    print(
        "📊 Classes: 10"
    )

    print(
        "\n🌐 Backend:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print("=" * 60)


    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )