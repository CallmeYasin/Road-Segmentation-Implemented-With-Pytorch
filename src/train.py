import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import argparse

from data import get_dataloaders
from model import UNet, save_model, load_model
from metrics import dice_coeff, dice_loss

NUM_WORKERS = 0
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else "cpu"
)

# dataset paths (change to your actual folders)
TRAIN_IMG_DIR = "dataset/train/images"
TRAIN_MASK_DIR = "dataset/train/masks"
VAL_IMG_DIR = "dataset/val/images"
VAL_MASK_DIR = "dataset/val/masks"


# ----------------------------
# Helper: compute pos_weight
# ----------------------------
def compute_pos_weight(loader):
    """Estimate class imbalance ratio for BCEWithLogitsLoss."""
    pos, neg = 0.0, 0.0
    for _, masks in loader:
        m = masks
        if m.ndim == 4:
            m = m.squeeze(1)
        pos += m.sum().item()
        neg += (1.0 - m).sum().item()
    ratio = neg / (pos + 1e-6)
    return torch.tensor([ratio])

# ----------------------------
# Training / Validation Epochs
# ----------------------------
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    epoch_loss, epoch_dice = 0, 0
    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    for imgs, masks in tqdm(loader, desc="Training", leave=False):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        if masks.ndim == 3:
            masks = masks.unsqueeze(1)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            preds = model(imgs)
            loss = criterion(preds, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()
        epoch_dice += dice_coeff(preds, masks).item()

    return epoch_loss / len(loader), epoch_dice / len(loader)


def validate_one_epoch(model, loader, criterion):
    model.eval()
    val_loss, val_dice = 0, 0
    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Validation", leave=False):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            val_loss += criterion(preds, masks).item()
            val_dice += dice_coeff(preds, masks).item()
    return val_loss / len(loader), val_dice / len(loader)


# ----------------------------
# Main training entry point
# ----------------------------
def main():
    parser = argparse.ArgumentParser(description="Road Segmentation Training")
    parser.add_argument('--model', type=str, default=None,
                        help='Path to pretrained model (.pth) if available')
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size for training (default: 4)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    args = parser.parse_args()

    print("Starting training...")
    train_loader, val_loader = get_dataloaders(
        TRAIN_IMG_DIR, TRAIN_MASK_DIR,
        VAL_IMG_DIR, VAL_MASK_DIR,
        batch_size=args.batch_size,
        num_workers=NUM_WORKERS
    )

    print(f"Using device: {DEVICE}")

    # Load model (if given)
    if args.model and os.path.exists(args.model):
        print(f"Loading model from: {args.model}")
        model = load_model(UNet, args.model, device=DEVICE, out_channels=1)
    else:
        print("No pretrained model found — initializing new model.")
        model = UNet(out_channels=1).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2, verbose=True)

    # Compute or set pos_weight
    print("Computing pos_weight for class imbalance...")
    pos_weight = compute_pos_weight(train_loader).to(DEVICE)
    print(f"pos_weight = {pos_weight.item():.4f}")

    bce_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def combined_loss(logits, targets):
        return bce_criterion(logits, targets) + dice_loss(logits, targets)

    best_dice = 0.0
    os.makedirs("models", exist_ok=True)

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_dice = train_one_epoch(model, train_loader, optimizer, combined_loss)
        val_loss, val_dice = validate_one_epoch(model, val_loader, combined_loss)

        scheduler.step(val_dice)

        print(f"Train Loss: {train_loss:.4f} | Train Dice: {train_dice:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}")

        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            save_model(model, "models/unet_best.pth")
            print(f"Saved new best model (Dice: {best_dice:.4f})")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
