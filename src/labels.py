from pathlib import Path

# ============================================================
# PROJECT ROOT
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# ============================================================
# TRAIN DIRECTORY
# ============================================================

train_dir = BASE_DIR / "data" / "processed" / "train"

# ============================================================
# LABELS FILE
# ============================================================

labels_file = BASE_DIR / "src" / "labels.txt"

# ============================================================
# CHECK TRAIN DIRECTORY
# ============================================================

if not train_dir.exists():

    print("❌ Training directory not found:")
    print(train_dir)
    exit()

# ============================================================
# GET CLASS NAMES
# ============================================================

class_names = sorted([
    folder.name
    for folder in train_dir.iterdir()
    if folder.is_dir()
])

# ============================================================
# SAFETY CHECK
# ============================================================

if len(class_names) != 10:

    print("❌ ERROR!")
    print("Expected 10 Tomato classes.")
    print("Found:", len(class_names))

    exit()

# ============================================================
# SAVE LABELS
# ============================================================

with open(labels_file, "w") as f:

    for name in class_names:
        f.write(name + "\n")

# ============================================================
# RESULT
# ============================================================

print("✅ labels.txt created successfully!")

print("📍 Location:")
print(labels_file)

print("\n🍅 Classes:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print(f"\n✅ Total classes: {len(class_names)}")