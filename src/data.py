"""Data module - skeleton"""
import os
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import transforms

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
"""
Train_Images = dataset/train/images
Val_Images = dataset/val/images

Train_Masks = dataset/train/masks
Val_masks = dataset/val/masks

Train_Annotations = dataset/train/annotations
Val_Annotations = dataset/val/annotations
"""
# -----------------------------
# Dataset Class
# -----------------------------
class RoadSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, train=True, image_size=(224, 224)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))
        self.train = train
        self.image_size = image_size

        assert len(self.image_files) == len(self.mask_files), (
            f"Image and mask count mismatch: {len(self.image_files)} vs {len(self.mask_files)}")

        # Normalization (ImageNet)
        self.normalize = transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                              std=(0.229, 0.224, 0.225))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        # --- Read with OpenCV ---
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        # --- Resize both ---
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

        # --- Data augmentations (applied to both) ---
        if self.train:
            # Random horizontal flip
            if torch.rand(1) < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()

            # Random rotation
            if torch.rand(1) < 0.5:
                angle = np.random.uniform(-10, 10)
                (h, w) = image.shape[:2]
                M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
                image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR)
                mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST)

        # --- Convert to Tensor ---
        image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0  # (C,H,W)
        mask = torch.from_numpy(mask).float().unsqueeze(0) / 255.0  # (1,H,W)

        # --- Normalize image ---
        image = self.normalize(image)

        return image, mask

# -----------------------------
# Dataloader function
# -----------------------------
def get_dataloaders(train_img_dir, train_mask_dir, val_img_dir, val_mask_dir,
                    batch_size=4, num_workers=0, image_size=(224, 224)):
    """Returns PyTorch dataloaders for training and validation datasets."""

    train_dataset = RoadSegmentationDataset(
        train_img_dir, train_mask_dir, train=True, image_size=image_size)

    val_dataset = RoadSegmentationDataset(
        val_img_dir, val_mask_dir, train=False, image_size=image_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

