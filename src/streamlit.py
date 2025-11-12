import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image
import os
from model import UNet, load_model
from predict import predict_single_image
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

def uploaded_file_to_path(uploaded_file):
    """Convert Streamlit UploadedFile to temporary file path"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_file:
        uploaded_file.seek(0)
        tmp_file.write(uploaded_file.read())
        return tmp_file.name

def overlay_mask_on_image(image, mask, alpha=0.7, color=[0, 255, 0]):
    """
    Overlay binary mask on image
    """
    # Convert to numpy if tensors
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    # Handle dimensions
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    
    # Ensure mask is 2D and properly binary
    if mask.ndim == 3:
        mask = mask.squeeze()
    
    # Normalize image to 0-255 if needed
    if image.max() <= 1.0:
        image = (image * 255).astype(np.uint8)
    
    # Ensure mask is binary (0 or 1)
    if mask.max() > 1.0:
        binary_mask = (mask > 127).astype(np.float32)
    else:
        binary_mask = (mask > 0.5).astype(np.float32)
    
    # Create colored mask
    colored_mask = np.zeros_like(image)
    colored_mask[:, :, 0] = color[0] * binary_mask  # R
    colored_mask[:, :, 1] = color[1] * binary_mask  # G  
    colored_mask[:, :, 2] = color[2] * binary_mask  # B
    
    # Blend image and mask
    overlayed = cv2.addWeighted(image, 1 - alpha, colored_mask.astype(np.uint8), alpha, 0)
    
    return overlayed, binary_mask

@st.cache_resource
def load_model_cached(model_path):
    """Cache the model"""
    if os.path.exists(model_path):
        st.write(f"Loading model from: {model_path}")
        model = load_model(UNet, model_path, device=DEVICE, out_channels=1)
        return model
    else:
        st.write("No pretrained model found — initializing new model.")
        return UNet(out_channels=1).to(DEVICE)

def main():
    st.title("Road Segmentation App")
    st.write("Upload an image to segment roads")
    
    # Use the same model path that works in predict.py
    model_path = r"C:\Users\Parsian 09352252262\Downloads\unet_best.pth"
    
    # Load model
    model = load_model_cached(model_path)
    
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            # Display original image
            original_image = Image.open(uploaded_file)
            st.image(original_image, caption="Uploaded Image", use_column_width=True)
            
            # Process image
            with st.spinner("Processing image..."):
                # Convert to temporary file
                temp_path = uploaded_file_to_path(uploaded_file)
                
                try:
                    # Get prediction - use the same size as predict.py (224, 224)
                    image_result, mask = predict_single_image(
                        model=model,
                        image_path=temp_path,
                        output_path=None,
                        size=(224, 224)  # Changed from 256 to match predict.py
                    )
                    
                    # Debug: Show mask statistics
                    st.write(f"**Mask Statistics:**")
                    mask_np = mask if isinstance(mask, np.ndarray) else mask.cpu().numpy()
                    
                    st.write(f"- Min value: {mask_np.min():.4f}")
                    st.write(f"- Max value: {mask_np.max():.4f}")
                    st.write(f"- Mean value: {mask_np.mean():.4f}")
                    st.write(f"- Non-zero pixels: {np.count_nonzero(mask_np > 0.5)}")
                    st.write(f"- Unique values: {np.unique(mask_np)}")
                    
                    # Ensure mask is properly binary for display
                    if mask_np.max() <= 1.0:
                        display_mask = (mask_np * 255).astype(np.uint8)
                    else:
                        display_mask = mask_np.astype(np.uint8)
                    
                    # Create overlays
                    overlay1, binary_mask1 = overlay_mask_on_image(
                        image=image_result, 
                        mask=mask, 
                        alpha=0.7, 
                        color=[0, 255, 0]  # Green
                    )
                    
                    overlay2, binary_mask2 = overlay_mask_on_image(
                        image=image_result, 
                        mask=mask, 
                        alpha=0.8, 
                        color=[255, 0, 0]  # Red
                    )
                    
                    # Display results
                    tab1, tab2, tab3, tab4 = st.tabs([
                        "Original", "Predicted Mask", "Overlay (Green)", "Overlay (Red)"
                    ])
                    
                    with tab1:
                        st.image(image_result, use_column_width=True, caption="Processed Image")
                    
                    with tab2:
                        # Display binary mask in grayscale
                        st.image(display_mask, 
                                use_column_width=True, 
                                caption="Binary Mask (Road=White, Background=Black)",
                                clamp=True)
                    
                    with tab3:
                        st.image(overlay1, use_column_width=True, caption="Green Overlay (Alpha: 0.7)")
                    
                    with tab4:
                        st.image(overlay2, use_column_width=True, caption="Red Overlay (Alpha: 0.8)")
                        
                finally:
                    # Clean up
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
            st.error(f"Full error: {repr(e)}")

if __name__ == "__main__":
    main()