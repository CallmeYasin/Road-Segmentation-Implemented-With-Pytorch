import os
import torch
import torch.nn as nn
import tqdm as tqdm
import matplotlib.pyplot as plt

from src.model import UNet,load_model
from src.data import get_dataloaders
from src.metrics import dice_score

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")

MODEL_PATH = "models/unet_best.pt"
VAL_IMG_DIR = "dataset/val/images"
VAL_MASK_DIR = "dataset/val/masks"
BATCH_SIZE = 2
NUM_WORKERS = 0

def evaluate():
    print("Loading model for evaluation...")
    model = load_model(UNet, MODEL_PATH, device=DEVICE, out_channels=1)

    _, val_loader = get_dataloaders(
        VAL_IMG_DIR, VAL_MASK_DIR,
        VAL_IMG_DIR, VAL_MASK_DIR,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS)

    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    val_loss, val_dice = 0, 0

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Evaluating"):
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            preds = model(imgs)
            val_loss += criterion(preds, masks).item()
            val_dice += dice_score(preds, masks).item()

    avg_loss = val_loss / len(val_loader)
    avg_dice = val_dice / len(val_loader)

    print(f"\n Validation Loss: {avg_loss:.4f}")
    print(f" Dice Score: {avg_dice:.4f}")

    # visualize one prediction
    imgs, masks = next(iter(val_loader))
    imgs = imgs.to(DEVICE)
    preds = torch.sigmoid(model(imgs))
    preds = (preds > 0.5).float().cpu()

    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs[0].imshow(imgs[0].permute(1, 2, 0).cpu())
    axs[0].set_title("Image")
    axs[1].imshow(masks[0].cpu(), cmap="gray")
    axs[1].set_title("Ground Truth")
    axs[2].imshow(preds[0][0], cmap="gray")
    axs[2].set_title("Prediction")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()