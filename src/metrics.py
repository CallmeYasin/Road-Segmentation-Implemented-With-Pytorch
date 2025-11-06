"""Metrics module - skeleton"""
import torch

def dice_score(preds,targets,threshold=0.5,smooth=1e-6):
    probs = torch.sigmoid(preds)
    inter = (probs * targets).sum()
    return 1 - (2*inter + smooth) / (probs.sum() + targets.sum() + smooth)

def iou_score(preds,targets,threshold=0.5,smooth=1e-8):
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum()
    union = preds.sum() + targets.sum() - intersection
    return intersection / (union + smooth)