import io
import torch
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import cv2

from model import UNet, load_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else "cpu")
MODEL_PATH = "models/unet_best.pth"
IMG_SIZE = (224, 224)

# Load model once at startup
print("Loading UNet model for inference...")
model = load_model(UNet, MODEL_PATH, device=DEVICE, out_channels=1)
model.eval()

# Create FastAPI app
app = FastAPI(title="Road Segmentation API", version="1.0")

def preprocess_image(file_bytes):
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img)
    img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
    img = img.unsqueeze(0).to(DEVICE)
    return img


def postprocess_mask(pred):
    pred = torch.sigmoid(pred)
    mask = (pred > 0.5).float().squeeze().cpu().numpy()
    mask = (mask * 255).astype(np.uint8)
    return mask


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        img_tensor = preprocess_image(file_bytes)

        with torch.no_grad():
            pred = model(img_tensor)

        mask = postprocess_mask(pred)

        # encode mask as base64 image (for JSON response)
        _, buffer = cv2.imencode(".png", mask)
        mask_bytes = buffer.tobytes()
        mask_b64 = np.frombuffer(mask_bytes, dtype=np.uint8).tolist()

        return JSONResponse({
            "message": "Prediction successful",
            "mask_data": mask_b64[:100],  # partial data preview (for testing)
            "shape": mask.shape
        })
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
