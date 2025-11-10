import torch
import torch.nn.functional as F

def dice_coeff(logits, targets, smooth=1e-6):
    """Returns Dice coefficient (0..1), higher is better."""
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    targets = targets.view(targets.size(0), -1)
    inter = (probs * targets).sum(dim=1)
    denom = probs.sum(dim=1) + targets.sum(dim=1)
    dice = ((2 * inter + smooth) / (denom + smooth))
    return dice.mean()

def dice_loss(logits, targets, smooth=1e-6):
    return 1.0 - dice_coeff(logits, targets, smooth)