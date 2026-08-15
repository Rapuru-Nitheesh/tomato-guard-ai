import tensorflow as tf
from model import build_model

# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = tf.io.gfile.join(
    tf.compat.v1.resource_loader.get_data_files_path(),
    ""
)

# Use absolute paths based on project structure
import os

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_dir = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "train"
)

val_dir = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "val"
)

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
SEED = 123

print("==============================================")
print("🍅 TOMATO DISEASE MODEL TRAINING")
print("==============================================")

print("\nTraining directory:")
print(train_dir)

print("\nValidation directory:")
print(val_dir)

# ============================================================
# LOAD TRAINING DATA
# ============================================================

train_ds = tf.keras.utils.image_dataset_from_directory(

    train_dir,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=True,

    seed=SEED
)

# ============================================================
# LOAD VALIDATION DATA
# ============================================================

val_ds = tf.keras.utils.image_dataset_from_directory(

    val_dir,

    image_size=IMG_SIZE,

    batch_size=BATCH_SIZE,

    shuffle=False
)

# ============================================================
# CLASS INFORMATION
# ============================================================

class_names = train_ds.class_names

num_classes = len(class_names)

print("\n🍅 Classes found:")

for i, class_name in enumerate(class_names):

    print(f"{i}: {class_name}")

print(f"\n✅ Number of classes: {num_classes}")

# Safety check
if num_classes != 10:

    print("\n❌ ERROR!")
    print("Expected exactly 10 Tomato classes.")
    print("Found:", num_classes)

    exit()

# ============================================================
# MOBILENETV2 PREPROCESSING
# ============================================================

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

def preprocess_dataset(images, labels):

    images = preprocess_input(images)

    return images, labels


train_ds = train_ds.map(
    preprocess_dataset,
    num_parallel_calls=tf.data.AUTOTUNE
)

val_ds = val_ds.map(
    preprocess_dataset,
    num_parallel_calls=tf.data.AUTOTUNE
)

# ============================================================
# PREFETCH
# ============================================================

train_ds = train_ds.prefetch(
    buffer_size=tf.data.AUTOTUNE
)

val_ds = val_ds.prefetch(
    buffer_size=tf.data.AUTOTUNE
)

# ============================================================
# BUILD MODEL
# ============================================================

print("\n🧠 Building MobileNetV2 model...")

model = build_model(num_classes)

# ============================================================
# COMPILE
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=["accuracy"]
)

# ============================================================
# MODEL SUMMARY
# ============================================================

print("\n📋 Model Summary:\n")

model.summary()

# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True
)

# ============================================================
# MODEL CHECKPOINT
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(

    "plant_disease_model.h5",

    monitor="val_accuracy",

    save_best_only=True,

    verbose=1
)

# ============================================================
# TRAIN
# ============================================================

print("\n🚀 Starting training...\n")

history = model.fit(

    train_ds,

    validation_data=val_ds,

    epochs=EPOCHS,

    callbacks=[
        early_stopping,
        checkpoint
    ]
)

# ============================================================
# FINAL SAVE
# ============================================================

model.save("plant_disease_model.h5")

print("\n" + "=" * 60)
print("🍅 TRAINING COMPLETE!")
print("=" * 60)

print("\n✅ Model saved as:")
print("plant_disease_model.h5")

print(f"\n✅ Number of classes: {num_classes}")

print("\n🍅 Tomato disease model is ready for evaluation!")