"""Make Mask From Annotations"""

import os
from pathlib import Path
import cv2
import numpy as np
import glob as glob
"""
Train_Images = dataset/train/images
Val_Images = dataset/val/images

Train_Masks = dataset/train/mask
Val_masks = dataset/val/mask

Train_Annotations = dataset/train/annotations
Val_Annotations = dataset/val/annotations
"""
image_train_path = Path("dataset/train/images")
image_val_path = Path("dataset/val/images")
mask_train_path = Path("dataset/train/mask")
mask_val_path = Path("dataset/val/mask")

# --- Collect image files ---
image_train_path_list = sorted([p for p in image_train_path.glob("*") if p.suffix.lower() in ['.jpg', '.png', '.jpeg']])
image_val_path_list   = sorted([p for p in image_val_path.glob("*") if p.suffix.lower() in ['.jpg', '.png', '.jpeg']])

# --- Function to make masks ---
def make_binary_masks(image_list, split_name, mask_dir):
    annotations_dir = Path(f"dataset/{split_name}/annotations")
    os.makedirs(mask_dir, exist_ok=True)

    for img_file in image_list:
        label_file = annotations_dir / f"{img_file.stem}.txt"

        img = cv2.imread(str(img_file))
        if img is None:
            print(f"⚠️ Could not read image: {img_file}")
            continue

        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        if not label_file.exists():
            cv2.imwrite(str(mask_dir / f"{img_file.stem}.png"), mask)
            print(f"⚠️ No label found for {img_file.name}, saved empty mask.")
            continue

        with open(label_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue

                cls = int(float(parts[0]))
                coords = np.array(parts[1:], dtype=np.float32).reshape(-1, 2)
                coords[:, 0] *= w
                coords[:, 1] *= h
                coords = coords.astype(np.int32)

                # Fill polygon for each class region
                cv2.fillPoly(mask, [coords], color=cls + 1)

        out_path = mask_dir / f"{img_file.stem}.png"
        cv2.imwrite(str(out_path), mask)

    print(f"✅ Polygon masks saved for {split_name} in: {mask_dir}")

# --- Run for train and val splits ---
make_binary_masks(image_train_path_list, "train", mask_train_path)
make_binary_masks(image_val_path_list, "val", mask_val_path)
