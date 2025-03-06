import os
import torch
import torch.nn.functional as F
import streamlit as st
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path

# --- Ensure Streamlit runs without auto-reload issues ---
os.environ["STREAMLIT_SERVER_ENABLE_WATCHER"] = "false"

# --- Define class labels for YOLOv5 classification ---
CLASS_LABELS = {
    "Paddy": ["brown_spot", "leaf_blast", "rice_hispa", "sheath_blight"],
    "GroundNut": ["alternaria_leaf_spot", "leaf_spot", "rosette", "rust"],
    "Cotton": ["bacterial_blight", "curl_virus", "herbicide_growth_damage", "leaf_hopper_jassids", "leaf_redding", "leaf_variegation"]
}

# --- Model Download Links (Replace with your links) ---
MODEL_URLS = {
    "Paddy": "classification_4Disease_best.pt",
    "GroundNut": "groundnut_best.pt",
    "Cotton": "re_do_cotton_2best.pt"
}

# --- Function to Download Model ---
def download_model(crop_type):
    model_path = Path(f"{crop_type}.pt")
    if not model_path.exists():
        st.info(f"Downloading {crop_type} model...")
        torch.hub.download_url_to_file(MODEL_URLS[crop_type], str(model_path))
    return model_path

# --- Load YOLOv5 Classification Model ---
@st.cache_resource
def load_model(crop_type):
    model_path = download_model(crop_type)
    try:
        model = torch.load(model_path, map_location=torch.device('cpu'))  # Load in CPU mode
        model.eval()
        return model
    except Exception as e:
        st.error(f"Failed to load {crop_type} model: {e}")
        return None

# --- Image Preprocessing for YOLOv5 ---
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize for YOLOv5 classification input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# --- Classification Function ---
def classify_image(image, crop_type):
    model = load_model(crop_type)
    if model is None:
        return None, None

    image_tensor = preprocess_image(image)

    with torch.no_grad():
        results = model(image_tensor)  # Perform inference

    if isinstance(results, torch.Tensor):
        results = results.squeeze(0)  # Remove batch dimension if necessary

    probabilities = F.softmax(results, dim=0)  # Convert logits to probabilities

    predicted_idx = torch.argmax(probabilities).item()
    predicted_label = CLASS_LABELS[crop_type][predicted_idx]

    return predicted_label, probabilities.tolist()

# --- Streamlit UI ---
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Crop Disease Classification</h1>", unsafe_allow_html=True)

st.markdown("### Select the Crop Type")
crop_selection = st.selectbox("Select the crop", ["Paddy", "GroundNut", "Cotton"], label_visibility="hidden")
st.write(f"Selected Crop: {crop_selection}")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Uploaded Image.", use_container_width=True)

    if st.button("Classify Disease"):
        with st.spinner("Classifying..."):
            predicted_label, probabilities = classify_image(img, crop_selection)

            if predicted_label:
                st.success(f"Predicted Disease: {predicted_label}")
                st.write(f"Confidence Scores: {probabilities}")
            else:
                st.error("Error in classification.")
