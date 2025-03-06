import os
import pathlib
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from pathlib import Path

# Ensure compatibility with Windows paths
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Disable Streamlit watcher to prevent reload issues
os.environ["STREAMLIT_SERVER_ENABLE_WATCHER"] = "false"

# Define class labels for each crop type
CLASS_LABELS = {
    "Paddy": ["brown_spot", "leaf_blast", "rice_hispa", "sheath_blight"],
    "GroundNut": ["alternaria_leaf_spot", "leaf_spot", "rosette", "rust"],
    "Cotton": ["bacterial_blight", "curl_virus", "herbicide_growth_damage", "leaf_hopper_jassids", "leaf_redding", "leaf_variegation"]
}

# Define model paths for each crop (use Pathlib for OS compatibility)
MODEL_PATHS = {
    "Paddy": Path(r"classification_4Disease_best.pt"),
    "GroundNut": Path(r"groundnut_best.pt"),
    "Cotton": Path(r"re_do_cotton_2best.pt")
}

# Load the appropriate YOLOv5 classification model
@st.cache_resource
def load_model(crop_type):
    try:
        model_path = MODEL_PATHS.get(crop_type)
        if not model_path or not model_path.exists():
            st.error(f"Model not found: {model_path}")
            return None

        model = torch.hub.load("ultralytics/yolov5", "custom", path=str(model_path), force_reload=True)
        model.eval()  # Set to evaluation mode
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# Image preprocessing
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize for YOLOv5 classification input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Standard normalization
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension

# Classification function
def classify_image(image, crop_type):
    model = load_model(crop_type)
    if model is None:
        return None, None

    image_tensor = preprocess_image(image)

    with torch.no_grad():
        results = model(image_tensor)  # Perform inference

    # Ensure results are formatted correctly
    if isinstance(results, torch.Tensor):
        results = results.squeeze(0)  # Remove batch dimension if necessary

    # Convert logits to probabilities
    probabilities = F.softmax(results, dim=0)  # YOLO may return single-dim tensor

    # Get the predicted class index
    predicted_idx = torch.argmax(probabilities).item()
    predicted_label = CLASS_LABELS[crop_type][predicted_idx]  # Map index to label

    return predicted_label, probabilities.tolist()

# Streamlit UI
st.markdown("""
    <style>
    .title { text-align: center; color: #4CAF50; font-size: 36px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">Crop Disease Classification</div>', unsafe_allow_html=True)

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





















# import os
# import torch
# import streamlit as st
# from PIL import Image
# from torchvision import transforms  # For preprocessing the image before inference
# import pathlib
# temp = pathlib.PosixPath
# pathlib.PosixPath = pathlib.WindowsPath
# # Set up environment variables
# os.environ["STREAMLIT_SERVER_ENABLE_WATCHER"] = "false"  # Disable problematic watcher

# # Mapping of crop names to their YOLOv5 classification models
# crop_model_mapping = {
#     "Paddy": "classification_4Disease_best.pt",
#     "Cotton": "re_do_cotton_2best.pt",
#     "Groundnut": "groundnut_best.pt"
# }

# # Define class labels for each crop
# CLASS_LABELS = {
#     "Paddy": ["brown_spot", "leaf_blast", "rice_hispa", "sheath_blight"],
#     "Groundnut": ["alternaria_leaf_spot", "leaf_spot", "rosette", "rust"],
#     "Cotton": ["bacterial_blight", "curl_virus", "herbicide_growth_damage",
#                "leaf_hopper_jassids", "leaf_redding", "leaf_variegation"]
# }

# # Cache model loading to avoid reloading on every classification
# @st.cache_resource
# def load_model(crop_name):
#     """Loads the YOLOv5 model only once per crop type."""
#     try:
#         model_path = crop_model_mapping.get(crop_name, None)
#         if model_path is None:
#             raise ValueError(f"No model found for crop: {crop_name}")

#         # Load model on CPU for inference
#         model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False, device='cpu')
#         model.eval()  # Set model to evaluation mode
#         return model
#     except Exception as e:
#         st.error(f"Model loading failed: {str(e)}")
#         return None

# # Preprocess image for model input
# def preprocess_image(img):
#     img = img.convert('RGB')  # Ensure the image is in RGB format
#     preprocess = transforms.Compose([
#         transforms.Resize((640, 640)),  # Resize to match YOLOv5 input size
#         transforms.ToTensor(),
#     ])
#     img_tensor = preprocess(img).unsqueeze(0)  # Add batch dimension
#     return img_tensor

# # Perform classification
# def classify_image(img, crop_name):
#     model = load_model(crop_name)  # Load the model only once per crop
#     if model is None:
#         return None, None

#     img_tensor = preprocess_image(img)  # Preprocess the image

#     # Perform inference
#     with torch.no_grad():
#         results = model(img_tensor)

#     # Extract class predictions
#     output = results[0]  # This contains the raw class scores
#     confidence, class_idx = torch.max(output, dim=0)  # Get highest confidence class

#     # Map index to class label
#     try:
#         class_label = CLASS_LABELS[crop_name][class_idx.item()]
#     except KeyError:
#         st.error(f"Error: '{crop_name}' not found in class labels. Please check the crop name.")
#         return None, None

#     return class_label, confidence.item()

# # Streamlit UI
# st.title("Crop Disease Detection")

# # Select crop type
# crop_selection = st.selectbox("Select the crop", ["Paddy", "Cotton", "Groundnut"])
# st.write(f"Selected Crop: {crop_selection}")

# # Upload image
# uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

# if uploaded_image:
#     img = Image.open(uploaded_image).convert("RGB")
#     st.image(img, caption="Uploaded Image", use_column_width=True)

#     if st.button("Run Classification"):
#         with st.spinner("Classifying..."):
#             predicted_class, confidence = classify_image(img, crop_selection)

#             if predicted_class:
#                 st.success(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})")
#             else:
#                 st.error("Classification failed.")

