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

class RoadSegmentationDataset(Dataset):
    def __init__(self,image_dir,mask_dir,transforms):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(self.image_dir))
        self.masks = sorted(os.listdir(self.mask_dir))
        self.transforms = transforms
    
    def __len__(self):return len(self.images)

    def __getitem__(self,index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.masks[index])
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))

    # Load mask as grayscale
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (224, 224), interpolation=cv2.INTER_NEAREST)  # ← CRITICAL

    # Convert to binary: 0 or 1 (float32 for BCELoss)
        mask = (mask > 0).astype(np.float32)  # ← 0.0 or 1.0

    # Apply transforms only to image
        if self.transforms:
            img = self.transforms(img)

    # Return: img (tensor), mask (float tensor, not long)
        return img, torch.from_numpy(mask)

    def choose(self):return self[torch.randint(0, len(self))]

    @staticmethod
    def collate_fn(batch):
        imgs, masks = zip(*batch)
        imgs = torch.stack(imgs).float()
        masks = torch.stack(masks).float()
        return imgs, masks

tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])


def get_dataloaders(train_img_dir, train_mask_dir,
                    val_img_dir, val_mask_dir,
                    batch_size,num_workers,
                    collate_fn=RoadSegmentationDataset.collate_fn):
    train_dataset = RoadSegmentationDataset(train_img_dir, train_mask_dir,tfms)
    val_dataset = RoadSegmentationDataset(val_img_dir, val_mask_dir,tfms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers,
                            collate_fn=collate_fn)

    return train_loader, val_loader