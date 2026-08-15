import os
import random
import shutil
from pathlib import Path
from PIL import Image

# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================================
# DATASET PATHS
# ============================================================

raw_data_dir = BASE_DIR / "data" / "raw" / "PlantVillage"
processed_data_dir = BASE_DIR / "data" / "processed"

train_dir = processed_data_dir / "train"
val_dir = processed_data_dir / "val"
test_dir = processed_data_dir / "test"

# ============================================================
# PARAMETERS
# ============================================================

IMG_SIZE = (224, 224)

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

random.seed(42)

# ============================================================
# CHECK RAW DATASET
# ============================================================

if not raw_data_dir.exists():
    print("❌ ERROR: PlantVillage dataset not found!")
    print(f"Expected location:")
    print(raw_data_dir)
    exit()

print("📂 Raw dataset found at:")
print(raw_data_dir)

# ============================================================
# FIND TOMATO CLASSES AUTOMATICALLY
# ============================================================

all_classes = [
    folder.name
    for folder in raw_data_dir.iterdir()
    if folder.is_dir()
]

tomato_classes = sorted([
    class_name
    for class_name in all_classes
    if class_name.lower().startswith("tomato")
])

print("\n🍅 Tomato classes found:")

for i, class_name in enumerate(tomato_classes, start=1):
    print(f"{i:2}. {class_name}")

print(f"\n✅ Total Tomato classes found: {len(tomato_classes)}")

# Safety check
if len(tomato_classes) != 10:
    print("\n⚠️ WARNING!")
    print("Expected 10 Tomato classes, but found:", len(tomato_classes))
    print("\nPlease check your PlantVillage folder before continuing.")
    exit()

# ============================================================
# CLEAN OLD PROCESSED DATA
# ============================================================

if processed_data_dir.exists():
    print("\n🧹 Removing old processed dataset...")
    shutil.rmtree(processed_data_dir)

# Recreate directories
train_dir.mkdir(parents=True, exist_ok=True)
val_dir.mkdir(parents=True, exist_ok=True)
test_dir.mkdir(parents=True, exist_ok=True)

print("✅ Fresh processed directories created.")

# ============================================================
# IMAGE PROCESSING FUNCTION
# ============================================================

def resize_and_save(image_path, save_path):

    try:

        with Image.open(image_path) as img:

            # Convert to RGB
            img = img.convert("RGB")

            # Resize
            img = img.resize(IMG_SIZE)

            # Save
            img.save(save_path)

    except Exception as e:

        print(f"❌ Failed to process {image_path}: {e}")


# ============================================================
# PROCESS TOMATO CLASSES
# ============================================================

print("\n🍅 Processing Tomato dataset...\n")

for class_name in tomato_classes:

    class_path = raw_data_dir / class_name

    # Get image files
    images = [
        img
        for img in class_path.iterdir()
        if img.is_file()
    ]

    # Shuffle
    random.shuffle(images)

    total = len(images)

    # Calculate split positions
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]

    print(f"📁 {class_name}")
    print(f"   Total : {total}")
    print(f"   Train : {len(train_images)}")
    print(f"   Val   : {len(val_images)}")
    print(f"   Test  : {len(test_images)}")

    # Create directories
    train_class_dir = train_dir / class_name
    val_class_dir = val_dir / class_name
    test_class_dir = test_dir / class_name

    train_class_dir.mkdir(parents=True, exist_ok=True)
    val_class_dir.mkdir(parents=True, exist_ok=True)
    test_class_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # TRAIN
    # ========================================================

    for img_path in train_images:

        save_path = train_class_dir / img_path.name

        resize_and_save(
            img_path,
            save_path
        )

    # ========================================================
    # VALIDATION
    # ========================================================

    for img_path in val_images:

        save_path = val_class_dir / img_path.name

        resize_and_save(
            img_path,
            save_path
        )

    # ========================================================
    # TEST
    # ========================================================

    for img_path in test_images:

        save_path = test_class_dir / img_path.name

        resize_and_save(
            img_path,
            save_path
        )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("🍅 TOMATO DATASET PREPARATION COMPLETE!")
print("=" * 60)

print("\nProcessed dataset location:")
print(processed_data_dir)

print("\nTrain:")
print(train_dir)

print("\nValidation:")
print(val_dir)

print("\nTest:")
print(test_dir)

print("\nClasses used:")

for i, class_name in enumerate(tomato_classes, start=1):
    print(f"{i:2}. {class_name}")

print(f"\n✅ Total Tomato classes: {len(tomato_classes)}")
print("\n🎉 Ready for Tomato-only model training!")