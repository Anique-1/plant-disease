import os
import torch
import streamlit as st
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import numpy as np

# Load pre-trained model and processor
@st.cache_resource
def load_model():
    processor = AutoImageProcessor.from_pretrained("wambugu71/crop_leaf_diseases_vit")
    model = AutoModelForImageClassification.from_pretrained("wambugu71/crop_leaf_diseases_vit")
    return processor, model

# Function to predict image class
def predict_image(image, processor, model):
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    
    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_idx = logits.argmax(1).item()
    
    # Get class label and confidence
    class_labels = model.config.id2label
    predicted_label = class_labels[predicted_class_idx]
    confidence = torch.softmax(logits, dim=1).max().item() * 100
    
    return predicted_label, round(confidence, 2)

# Streamlit app
def main():
    # Set page configuration
    st.set_page_config(
        page_title="Crop Leaf Disease Classifier", 
        page_icon=":seedling:",
        layout="centered"
    )
    
    # Custom CSS for styling
    st.markdown("""
    <style>
    .main-container {
        max-width: 800px;
        margin: auto;
        padding: 20px;
        background-color: #f0f2f6;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .title {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 20px;
    }
    .upload-box {
        background-color: #ffffff;
        border: 2px dashed #3498db;
        border-radius: 10px;
        padding: 20px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .upload-box:hover {
        border-color: #2980b9;
        background-color: #f1f8ff;
    }
    .result-box {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        margin-top: 20px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main content
    st.markdown('<div class="main-container">', unsafe_allow_html=True)
    st.markdown('<h1 class="title">🍃 Crop Leaf Disease Classifier</h1>', unsafe_allow_html=True)
    
    # Load model
    processor, model = load_model()
    
    # File uploader
    st.markdown('<div class="upload-box">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload a crop leaf image", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image of a crop leaf to classify its disease"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Prediction section
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        st.markdown('<div class="result-box">', unsafe_allow_html=True)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Make prediction
        try:
            predicted_label, confidence = predict_image(image, processor, model)
            
            # Display prediction results
            st.markdown(f"**Predicted Disease:** {predicted_label}")
            st.markdown(f"**Confidence:** {confidence}%")
            
            # Color-coded confidence indicator
            if confidence > 80:
                st.success("High confidence prediction")
            elif confidence > 60:
                st.warning("Moderate confidence prediction")
            else:
                st.error("Low confidence prediction")
        
        except Exception as e:
            st.error(f"Error processing image: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()