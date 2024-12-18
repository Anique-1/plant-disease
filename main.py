import os
import streamlit as st

# Add error handling for torch import
try:
    import torch
except ImportError:
    st.warning("PyTorch not found. Attempting to use CPU-only version.")
    import sys
    import subprocess

    # Attempt to install torch if not present
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'torch'])
    import torch

from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np

# Rest of the previous Streamlit code remains the same...

# Optional: Add a device check
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

# Modify your model loading and prediction functions to use the device
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("wambugu71/crop_leaf_diseases_vit")
    model = AutoModelForImageClassification.from_pretrained("wambugu71/crop_leaf_diseases_vit")
    
    # Move model to appropriate device
    device = get_device()
    model.to(device)
    
    return processor, model

# Update prediction function to use device
def predict_image(image, processor, model):
    device = get_device()
    
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits.cpu()  # Move back to CPU for further processing
        predicted_class_idx = logits.argmax(1).item()
    
    # Get class label and confidence
    class_labels = model.config.id2label
    predicted_label = class_labels[predicted_class_idx]
    confidence = torch.softmax(logits, dim=1).max().item() * 100
    
    return predicted_label, round(confidence, 2)

# Rest of the previous Streamlit code remains the same...
