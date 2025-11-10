import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from data import get_dataloaders
from model import UNet, save_model
from metrics import dice_score


EPOCHS = 10
BATCH_SIZE = 4
LR = 1e-4
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")

# dataset paths (change these to your actual folders)
TRAIN_IMG_DIR = "dataset/train/images"
TRAIN_MASK_DIR = "dataset/train/masks"
VAL_IMG_DIR = "dataset/val/images"
VAL_MASK_DIR = "dataset/val/masks"

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_dice = 0

    for imgs, masks in tqdm(loader, desc="Training", leave=False):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)
        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        epoch_dice += dice_score(preds, masks).item()

    return epoch_loss / len(loader), epoch_dice / len(loader)

def validate_one_epoch(model, loader, criterion):
    model.eval()
    val_loss, val_dice = 0, 0

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Validation", leave=False):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            val_loss += criterion(preds, masks).item()
            val_dice += dice_score(preds, masks).item()

    return val_loss / len(loader), val_dice / len(loader)

def main():
    print("Starting training...")
    train_loader, val_loader = get_dataloaders(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR,
        VAL_IMG_DIR, VAL_MASK_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS)

    model = UNet(out_channels=1).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()  # for binary segmentation

    best_dice = 0
    os.makedirs("models", exist_ok=True)

    for epoch in range(EPOCHS):
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_dice = validate_one_epoch(model, val_loader, criterion)

        print(f"Epoch [{epoch+1}/{EPOCHS}] "
              f"Train Loss: {train_loss:.4f}  Train Dice: {train_dice:.4f} "
              f"Val Loss: {val_loss:.4f}  Val Dice: {val_dice:.4f}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            save_model(model, "models/unet_best.pt")
            print(f"Saved new best model (Dice: {best_dice:.4f})")

    print("Training complete!")

if __name__ == "__main__":
    main()