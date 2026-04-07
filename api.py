# api.py - Deploy this to Render
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class_to_idx = {'glioma_tumor': 0, 'meningioma_tumor': 1, 'pituitary_tumor': 2, 'Normal': 3}
idx_to_class = {v: k for k, v in class_to_idx.items()}
target_names = ['glioma_tumor', 'meningioma_tumor', 'pituitary_tumor', 'Normal']

# Load model
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load('brain_tumor_model_finetuned.pth', map_location=device)['model_state_dict'])
model = model.to(device)
model.eval()

# Calibration temperatures
TEMPERATURE = 2.375
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


# ── Grad-CAM ──────────────────────────────────────────────────────────────────
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)   # ✅ not deprecated

    def save_activation(self, module, input, output):
        # Do NOT detach here — activations must stay in graph until backward()
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_image, target_class=None):
        if self.gradients is not None:
            self.gradients = None
        if self.activations is not None:
            self.activations = None

        with torch.enable_grad():                          # ✅ gradients ON
            img = input_image.detach().requires_grad_(True)
            self.model.zero_grad()
            output = self.model(img)

            if target_class is None:
                target_class = output.argmax(dim=1).item()

            one_hot = torch.zeros_like(output)
            one_hot[0][target_class] = 1
            output.backward(gradient=one_hot)

        if self.gradients is None or self.activations is None:
            raise ValueError("Grad-CAM hooks did not capture gradients/activations.")

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations.detach()).sum(dim=1, keepdim=True)  # ✅ detach activations after backward
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy(), target_class


# Initialise Grad-CAM once at startup
target_layer = model.layer4[-1]
gradcam = GradCAM(model, target_layer)


# ── Heatmap overlay ───────────────────────────────────────────────────────────
def create_heatmap_overlay(image_tensor, heatmap, alpha=0.5):
    import cv2
    import matplotlib.pyplot as plt

    # ✅ .detach() before .numpy() — tensor may still be in grad graph
    img = image_tensor.squeeze().cpu().detach().numpy().transpose(1, 2, 0)
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img = np.clip(img, 0, 1)

    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
    overlay = (1 - alpha) * img + alpha * heatmap_colored

    buf = io.BytesIO()
    plt.imsave(buf, overlay, format='png')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
async def root():
    return {"message": "Brain Tumor Classification API", "status": "running"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": True}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    use_class_temperature: bool = True,
    include_heatmap: bool = True,
) -> Dict[str, Any]:
    """Predict brain tumor type from an uploaded MRI image."""
    try:
        # ── Read & preprocess ──────────────────────────────────────────────
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        img_tensor = transform(image).unsqueeze(0).to(device)

        # ── Inference (no grad needed for prediction) ──────────────────────
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
        pred_class_idx = pred.item()   # plain int — safe to use anywhere

        # ── Build response ─────────────────────────────────────────────────
        response: Dict[str, Any] = {
            "success": True,
            "prediction": predicted_class,
            "confidence": confidence_score,
            "confidence_percent": f"{confidence_score:.1%}",
            "probabilities": {
                class_name: float(probs[0][i].cpu().detach().numpy())  # ✅ detach
                for i, class_name in enumerate(target_names)
            },
            "calibration_info": {
                "temperature_used": CLASS_TEMPERATURES if use_class_temperature else TEMPERATURE,
                "is_class_specific": use_class_temperature,
            },
            "heatmap": None,
            "heatmap_error": None,
        }

        # ── Grad-CAM (fresh forward+backward pass with grads enabled) ──────
        if include_heatmap:
            try:
                # Fresh tensor from the original PIL image — not from the no_grad pass
                heatmap_tensor = transform(image).unsqueeze(0).to(device)
                heatmap, _ = gradcam.generate(heatmap_tensor, pred_class_idx)
                response["heatmap"] = create_heatmap_overlay(heatmap_tensor, heatmap)
            except Exception as heatmap_err:
                # Prediction is still valid — degrade gracefully
                response["heatmap_error"] = str(heatmap_err)

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)