# api.py - Fixed for Render deployment
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import io
import base64
import numpy as np
from typing import Dict, Any
import uvicorn

app = FastAPI(title="Brain Tumor Classification API")

# CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class_to_idx = {'glioma_tumor': 0, 'meningioma_tumor': 1, 'pituitary_tumor': 2, 'Normal': 3}
idx_to_class = {v: k for k, v in class_to_idx.items()}
target_names = ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor', 'Normal']

# Load your model
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
checkpoint = torch.load('brain_tumor_model_finetuned.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(device)
model.eval()

# Calibration temperature
TEMPERATURE = 2.375

# Class-specific temperatures
CLASS_TEMPERATURES = {
    'glioma_tumor': 2.2,
    'meningioma_tumor': 3.0,
    'pituitary_tumor': 2.0,
    'Normal': 3.0
}

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ============= FIXED: Grad-CAM without OpenCV =============
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_image, target_class=None):
        self.model.zero_grad()
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        
        return cam.squeeze().cpu().numpy(), target_class

# Initialize Grad-CAM
target_layer = model.layer4[-1]
gradcam = GradCAM(model, target_layer)

def create_heatmap_overlay(image_tensor, heatmap, alpha=0.5):
    """Convert heatmap to base64 - NO OpenCV/Matplotlib needed!"""
    # Denormalize image
    img = image_tensor.squeeze().cpu().numpy().transpose(1, 2, 0)
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)
    
    # Resize heatmap using simple numpy interpolation
    h, w = img.shape[:2]
    heatmap_resized = np.array(Image.fromarray(heatmap).resize((w, h), Image.Resampling.BILINEAR))
    
    # Create colormap (jet colormap from scratch - no OpenCV!)
    def jet_colormap(x):
        """Jet colormap implementation"""
        x = np.clip(x, 0, 1)
        r = np.clip(1.5 - np.abs(4 * x - 2), 0, 1)
        g = np.clip(1.5 - np.abs(4 * x - 1), 0, 1)
        b = np.clip(1.5 - np.abs(4 * x), 0, 1)
        return np.stack([r, g, b], axis=-1)
    
    heatmap_colored = jet_colormap(heatmap_resized)
    
    # Overlay
    overlay = (1 - alpha) * img + alpha * heatmap_colored
    overlay = np.clip(overlay, 0, 1)
    
    # Convert to base64 without matplotlib
    from PIL import Image as PILImage
    img_pil = PILImage.fromarray((overlay * 255).astype(np.uint8))
    buf = io.BytesIO()
    img_pil.save(buf, format='PNG')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

@app.get("/")
async def root():
    return {"message": "Brain Tumor Classification API", "status": "running"}

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    use_class_temperature: bool = True,
    include_heatmap: bool = False
) -> Dict[str, Any]:
    """Predict brain tumor type from uploaded MRI image"""
    try:
        # Read and preprocess image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)
        
        # Predict with temperature scaling
        with torch.no_grad():
            logits = model(img_tensor)
            
            if use_class_temperature:
                scaled_logits = logits.clone()
                for i, class_name in enumerate(target_names):
                    scaled_logits[0, i] = logits[0, i] / CLASS_TEMPERATURES[class_name]
            else:
                scaled_logits = logits / TEMPERATURE
            
            probs = F.softmax(scaled_logits, dim=1)
            confidence, pred = torch.max(probs, dim=1)
        
        predicted_class = idx_to_class[pred.item()]
        confidence_score = confidence.item()
        
        # Prepare response
        response = {
            "success": True,
            "prediction": predicted_class,
            "confidence": confidence_score,
            "confidence_percent": f"{confidence_score:.1%}",
            "probabilities": {
                class_name: float(probs[0][i].cpu().numpy())
                for i, class_name in enumerate(target_names)
            },
            "calibration_info": {
                "temperature_used": CLASS_TEMPERATURES if use_class_temperature else TEMPERATURE,
                "is_class_specific": use_class_temperature
            }
        }
        
        # Add Grad-CAM if requested
        if include_heatmap:
            try:
                heatmap, _ = gradcam.generate(img_tensor, pred.item())
                response["heatmap"] = create_heatmap_overlay(img_tensor, heatmap)
                print(f"Heatmap generated successfully for {predicted_class}")
            except Exception as e:
                print(f"Heatmap error: {e}")
                response["heatmap_error"] = str(e)
        
        return response
    
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)