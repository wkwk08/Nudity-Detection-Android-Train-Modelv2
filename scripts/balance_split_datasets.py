import random
import shutil
from pathlib import Path
from config import DATASETS_DIR, OUTPUT_DIR
import random, shutil
from pathlib import Path

BASE_PATH = DATASETS_DIR
OUTPUT_PATH = OUTPUT_DIR
random.seed(42)

# Upper-bound targets; script will auto-adjust to available counts per source.
TARGETS = {
    "Objective_1": {
        # Merge mid-light + mid-dark into a single 'medium' pool
        "ducnguyen168/data_skintone": {"light": 4000, "medium": 4000, "dark": 4000},
        # Copy Pratheepan face categories (used later for fairness/context)
        "Face_Dataset/Pratheepan_Dataset": {"FacePhoto": 2000, "FamilyPhoto": 2000},
    },
    "Objective_2": {
        # COCO images are flat dirs; we treat each dir as one pool (no class subfolders)
        # We'll split the image pool into train/val/test, not per 'human' class.
        "train2017/train2017": {"_pool": 10000},
        "val2017/val2017": {"_pool": 5000},  # optional: include val images in the pool
        # MPII images actually live under archive/mpii_human_pose_v1/images (based on your path)
        "archive/mpii_human_pose_v1/images": {"_pool": 6000},
    },
    "Objective_3": {
        # WIDER: treat all activity folders under images/ as a single pool
        "WIDER_train/WIDER_train/images": {"_pool": 14000},
        # CelebA nested folder (img_align_celeba/img_align_celeba); treat as one pool
        "Img-20251116T172027Z-1-001/Img/img_align_celeba/img_align_celeba": {"_pool": 6000},
    },
    "Objective_4": {
        # P2datasetFull already has train/val/test splits and classes 1/2; copy-through, no re-split.
        "Adult content dataset/P2datasetFull": {"_passthrough": True},
        # Auto-generated detector dataset: flat IMAGES folder, treat as one pool and split
        "DETECTOR_AUTO_GENERATED_DATA/DETECTOR_AUTO_GENERATED_DATA/IMAGES": {"_pool": 6000},
    },
}

IMG_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")

def list_images(dir_path: Path, recursive: bool = True):
    if not dir_path.exists():
        return []
    files = []
    if recursive:
        for p in IMG_PATTERNS:
            files.extend(dir_path.rglob(p))
    else:
        for p in IMG_PATTERNS:
            files.extend(dir_path.glob(p))
    # Skip __MACOSX and hidden/system
    return [f for f in files if "__MACOSX" not in str(f)]

def copy_files(files, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        try:
            shutil.copy(f, out_dir / f.name)
        except Exception as e:
            # Skip problematic files but keep going
            print(f"Copy error: {f} -> {e}")

def split_pool(files, count):
    random.shuffle(files)
    files = files[:count]
    train = int(0.7 * count)
    val = int(0.15 * count)
    test = count - train - val
    return {
        "train": files[:train],
        "val": files[train:train + val],
        "test": files[train + val:],
    }, train, val, test

def process_objective_1(dataset_name, class_targets):
    source = BASE_PATH / "Objective_1" / dataset_name

    for cls, target in class_targets.items():
        if dataset_name == "ducnguyen168/data_skintone" and cls == "medium":
            mid_light = list_images(source / "mid-light")
            mid_dark = list_images(source / "mid-dark")
            files = mid_light + mid_dark
        else:
            files = list_images(source / cls)

        available = len(files)
        if available == 0:
            print(f"Objective_1/{dataset_name}/{cls}: no files found.")
            continue

        use = min(target, available)
        splits, tr, va, te = split_pool(files, use)
        for split_name, split_files in splits.items():
            out = OUTPUT_PATH / "Objective_1" / split_name / dataset_name.replace("data_skintone", "") / cls
            copy_files(split_files, out)
        print(f"Objective_1/{dataset_name}/{cls}: req {target}, avail {available}, used {use} → train {tr}, val {va}, test {te}.")

def process_objective_2(dataset_name, class_targets):
    source = BASE_PATH / "Objective_2" / dataset_name
    # Build a single pool of images from this dataset path (it’s a flat dir or nested activity dirs)
    files = list_images(source)
    available = len(files)
    if available == 0:
        print(f"Objective_2/{dataset_name}: no images found.")
        return
    # There is only one _pool key; get its target
    target = class_targets.get("_pool", available)
    use = min(target, available)
    splits, tr, va, te = split_pool(files, use)
    for split_name, split_files in splits.items():
        out = OUTPUT_PATH / "Objective_2" / split_name / dataset_name
        copy_files(split_files, out)
    print(f"Objective_2/{dataset_name}: req {target}, avail {available}, used {use} → train {tr}, val {va}, test {te}.")

def process_objective_3(dataset_name, class_targets):
    source = BASE_PATH / "Objective_3" / dataset_name
    files = list_images(source)
    available = len(files)
    if available == 0:
        print(f"Objective_3/{dataset_name}: no images found.")
        return
    target = class_targets.get("_pool", available)
    use = min(target, available)
    splits, tr, va, te = split_pool(files, use)
    for split_name, split_files in splits.items():
        out = OUTPUT_PATH / "Objective_3" / split_name / dataset_name
        copy_files(split_files, out)
    print(f"Objective_3/{dataset_name}: req {target}, avail {available}, used {use} → train {tr}, val {va}, test {te}.")

def process_objective_4(dataset_name, class_targets):
    source = BASE_PATH / "Objective_4" / dataset_name

    # P2datasetFull: pass-through existing splits and classes (1, 2)
    if class_targets.get("_passthrough"):
        for split in ["train", "val", "test"]:
            split_dir = source / split
            if not split_dir.exists():
                print(f"Objective_4/{dataset_name}: missing split {split_dir}")
                continue
            for cls in ["1", "2"]:
                class_dir = split_dir / cls
                files = list_images(class_dir)
                out = OUTPUT_PATH / "Objective_4" / split / dataset_name / cls
                copy_files(files, out)
                print(f"Objective_4/{dataset_name}/{split}/{cls}: copied {len(files)} files (pass-through).")
        return

    # Flat IMAGES pool: split
    files = list_images(source)
    available = len(files)
    if available == 0:
        print(f"Objective_4/{dataset_name}: no images found.")
        return
    target = class_targets.get("_pool", available)
    use = min(target, available)
    splits, tr, va, te = split_pool(files, use)
    for split_name, split_files in splits.items():
        out = OUTPUT_PATH / "Objective_4" / split_name / dataset_name
        copy_files(split_files, out)
    print(f"Objective_4/{dataset_name}: req {target}, avail {available}, used {use} → train {tr}, val {va}, test {te}.")

def main():
    for objective, datasets in TARGETS.items():
        for dataset_name, class_targets in datasets.items():
            if objective == "Objective_1":
                process_objective_1(dataset_name, class_targets)
            elif objective == "Objective_2":
                process_objective_2(dataset_name, class_targets)
            elif objective == "Objective_3":
                process_objective_3(dataset_name, class_targets)
            elif objective == "Objective_4":
                process_objective_4(dataset_name, class_targets)

    print("\n✅ All datasets processed. Output saved to:", OUTPUT_PATH)

if __name__ == "__main__":
    main()