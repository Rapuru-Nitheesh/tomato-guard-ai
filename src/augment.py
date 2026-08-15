import os
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
import random

# ============================================================
# PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

train_data_dir = BASE_DIR / "data" / "processed" / "train"

# ============================================================
# SETTINGS
# ============================================================

IMG_SIZE = (224, 224)

# Number of augmented versions for each original image
AUGS_PER_IMAGE = 2

random.seed(42)

# ============================================================
# AUGMENTATION FUNCTION
# ============================================================

def augment_image(img):

    augmented = []

    # Random rotation
    angle = random.choice([90, 180, 270])
    augmented.append(img.rotate(angle))

    # Random horizontal flip
    augmented.append(ImageOps.mirror(img))

    # Random brightness
    brightness = ImageEnhance.Brightness(
        img
    ).enhance(random.uniform(0.8, 1.2))

    augmented.append(brightness)

    # Random contrast
    contrast = ImageEnhance.Contrast(
        img
    ).enhance(random.uniform(0.8, 1.2))

    augmented.append(contrast)

    return augmented


# ============================================================
# CHECK TRAIN DIRECTORY
# ============================================================

if not train_data_dir.exists():

    print("❌ Training directory not found:")
    print(train_data_dir)
    exit()

print("📂 Training directory:")
print(train_data_dir)

print("\n🍅 Augmenting TRAINING images only...\n")


# ============================================================
# PROCESS EACH CLASS
# ============================================================

total_original = 0
total_augmented = 0

for class_folder in sorted(train_data_dir.iterdir()):

    if not class_folder.is_dir():
        continue

    print(f"📁 {class_folder.name}")

    # IMPORTANT:
    # Only select original image files.
    # Ignore files containing "_aug".
    image_files = [
        img
        for img in class_folder.iterdir()
        if img.is_file()
        and img.suffix.lower() in [".jpg", ".jpeg", ".png"]
        and "_aug" not in img.stem
    ]

    class_original_count = len(image_files)

    print(f"   Original images: {class_original_count}")

    total_original += class_original_count

    # ========================================================
    # AUGMENT EACH ORIGINAL IMAGE
    # ========================================================

    for img_path in image_files:

        try:

            with Image.open(img_path) as img:

                img = img.convert("RGB")
                img = img.resize(IMG_SIZE)

                augmented_images = augment_image(img)

                # Only create requested number
                # of augmented images
                selected_augmentations = augmented_images[:AUGS_PER_IMAGE]

                for idx, aug_img in enumerate(selected_augmentations):

                    output_name = (
                        f"{img_path.stem}_aug{idx}"
                        f"{img_path.suffix}"
                    )

                    output_path = class_folder / output_name

                    aug_img.save(output_path)

                    total_augmented += 1

        except Exception as e:

            print(f"   ⚠️ Skipping {img_path.name}: {e}")

    print(f"   Added augmented images: "
          f"{class_original_count * AUGS_PER_IMAGE}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("🍅 DATA AUGMENTATION COMPLETE!")
print("=" * 60)

print(f"\nOriginal training images : {total_original}")
print(f"Augmented images added  : {total_augmented}")
print(f"Total training images   : "
      f"{total_original + total_augmented}")

print("\n⚠️ Validation and test datasets were NOT modified.")
print("✅ Ready for model training!")