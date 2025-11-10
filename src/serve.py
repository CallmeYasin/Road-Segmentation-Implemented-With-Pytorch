import io
import torch
import torch.nn.functional as F
import numpy as np
import argparse
import sys
from PIL import Image
import cv2
import os

from model import UNet, load_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
IMG_SIZE = (224, 224)

def preprocess_image(image_path):
    """Load and preprocess image from file path"""
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img)
    img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0).to(DEVICE)
    return img

def postprocess_mask(pred):
    """Convert model output to binary mask"""
    pred = torch.sigmoid(pred)
    mask = (pred > 0.5).float().squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    return mask

def main():
    # Set up command line arguments
    parser = argparse.ArgumentParser(description='Road Segmentation Prediction')
    parser.add_argument('--input', '-i', type=str, required=True, 
                        help='Input image file path')
    parser.add_argument('--output', '-o', type=str, 
                        help='Output mask file path (default: same as input with _mask suffix)')
    parser.add_argument('--show', '-s', action='store_true',
                        help='Display the result using matplotlib')
    parser.add_argument('--model', '-m', type=str, default="models/unet_best.pth",
                        help='Path to the model file (default: models/unet_best.pth)')
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found!")
        sys.exit(1)
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file '{args.model}' not found!")
        sys.exit(1)
    
    # Set default output path if not provided
    if args.output is None:
        base_name = os.path.splitext(args.input)[0]
        args.output = f"{base_name}_mask.png"
    
    # Load model
    print("Loading UNet model for inference...")
    try:
        model = load_model(UNet, args.model, device=DEVICE, out_channels=1)
        model.eval()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Process image
    try:
        print(f"Processing image: {args.input}")
        img_tensor = preprocess_image(args.input)

        with torch.no_grad():
            pred = model(img_tensor)

        mask = postprocess_mask(pred)
        
        # Save mask
        cv2.imwrite(args.output, mask)
        print(f"Mask saved to: {args.output}")
        
        # Display if requested
        if args.show:
            try:
                import matplotlib.pyplot as plt
                
                # Load original image for display
                original_img = Image.open(args.input).convert("RGB")
                original_img = original_img.resize(IMG_SIZE)
                
                # Create visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                
                ax1.imshow(original_img)
                ax1.set_title('Original Image')
                ax1.axis('off')
                
                ax2.imshow(mask, cmap='gray')
                ax2.set_title('Road Segmentation Mask')
                ax2.axis('off')
                
                plt.tight_layout()
                plt.show()
                
            except ImportError:
                print("Matplotlib not installed. Install with: pip install matplotlib")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()