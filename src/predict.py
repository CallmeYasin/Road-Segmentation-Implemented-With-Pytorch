"""Predict utilities - skeleton"""
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.model import UNet, load_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")

def predict_single_image(model, image_path, output_path=None, size=(224, 224)):
    """Predict segmentation for a single image"""
    # Load and preprocess image
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_size = image.shape[:2]
    
    # Resize and normalize
    image_resized = cv2.resize(image, size)
    image_tensor = torch.tensor(image_resized).permute(2, 0, 1).float() / 255.0
    
    # ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    image_tensor = (image_tensor - mean) / std
    
    image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
    
    # Predict
    model.eval()
    with torch.no_grad():
        pred = model(image_tensor)
        mask = torch.sigmoid(pred).squeeze().cpu().numpy()
        binary_mask = (mask > 0.5).astype(np.uint8)
    
    # Resize mask back to original size
    binary_mask = cv2.resize(binary_mask, (original_size[1], original_size[0]), 
                           interpolation=cv2.INTER_NEAREST)
    
    # Save if output path provided
    if output_path:
        cv2.imwrite(str(output_path), binary_mask * 255)
    
    return image, binary_mask

def visualize_prediction(image, mask, save_path=None):
    """Visualize original image and prediction"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Prediction mask
    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Prediction Mask')
    axes[1].axis('off')
    
    # Overlay
    axes[2].imshow(image)
    axes[2].imshow(mask, alpha=0.3, cmap='jet')
    axes[2].set_title('Overlay')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Example usage"""
    model_path = "models/unet_best.pt"
    image_path = "path/to/your/image.jpg"  # Change this
    
    # Load model
    model = load_model(UNet, model_path, device=DEVICE, out_channels=1)
    
    # Predict
    image, mask = predict_single_image(model, image_path, "prediction_mask.png")
    
    # Visualize
    visualize_prediction(image, mask, "prediction_visualization.png")

if __name__ == "__main__":
    main()