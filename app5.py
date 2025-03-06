# import os
# import torch
# import streamlit as st
# from PIL import Image
# from torchvision import transforms  # For preprocessing the image before inference

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












import os
import torch
import streamlit as st
import subprocess
from PIL import Image
from torchvision import transforms

# 📌 Ensure PyTorch is installed (important for Streamlit Cloud)
try:
    import torch
except ModuleNotFoundError:
    subprocess.run(["pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cpu/"])
    import torch  # Import again after installation

# 🌱 Set Streamlit environment variables
os.environ["STREAMLIT_SERVER_ENABLE_WATCHER"] = "false"

# 🏷️ Class labels for each crop type
CLASS_LABELS = {
    "Paddy": ["brown_spot", "leaf_blast", "rice_hispa", "sheath_blight"],
    "Groundnut": ["alternaria_leaf_spot", "leaf_spot", "rosette", "rust"],
    "Cotton": ["bacterial_blight", "curl_virus", "herbicide_growth_damage",
               "leaf_hopper_jassids", "leaf_redding", "leaf_variegation"]
}

# 📌 Model paths (Ensure your models are accessible from GitHub or cloud storage)
MODEL_PATHS = {
    "Paddy": "classification_4Disease_best.pt",
    "Cotton": "re_do_cotton_2best.pt",
    "Groundnut": "groundnut_best.pt"
}

# 🚀 Cache model loading for performance
@st.cache_resource
def load_model(crop_name):
    """Load the YOLOv5 classification model once."""
    try:
        model_path = MODEL_PATHS.get(crop_name, None)
        if model_path is None:
            st.error(f"No model found for {crop_name}. Check the model path.")
            return None

        # Ensure the model file exists
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            return None

        # Load model from Ultralytics YOLOv5 repo
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
        model.eval()  # Set model to evaluation mode
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# 🖼️ Image preprocessing
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((640, 640)),  # Resize to match YOLOv5 input size
        transforms.ToTensor(),
    ])
    return transform(img).unsqueeze(0)  # Add batch dimension

# 🔍 Classification function
def classify_image(img, crop_name, confidence_threshold=0.5):
    model = load_model(crop_name)
    if model is None:
        return None, None

    img_tensor = preprocess_image(img)

    # Perform inference
    with torch.no_grad():
        results = model(img_tensor)

    # Extract class predictions
    output = results[0]  # This contains the raw class scores
    confidence, class_idx = torch.max(output, dim=0)  # Get highest confidence class

    # Only return results above confidence threshold
    if confidence.item() < confidence_threshold:
        return "Uncertain prediction", confidence.item()

    # Map index to class label
    predicted_label = CLASS_LABELS[crop_name][class_idx.item()]
    return predicted_label, confidence.item()

# 🎨 Streamlit UI
st.title("🌾 Crop Disease Classification with YOLOv5")

# 🌱 Crop selection
crop_selection = st.selectbox("Select the crop", list(CLASS_LABELS.keys()))
st.write(f"Selected Crop: {crop_selection}")

# 📤 Upload image
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    img = Image.open(uploaded_image).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("🔍 Run Classification"):
        with st.spinner("Processing..."):
            predicted_class, confidence = classify_image(img, crop_selection)

            if predicted_class:
                st.success(f"✅ Prediction: {predicted_class} (Confidence: {confidence:.2f})")

                # 🔽 Provide a downloadable result file
                result_text = f"Crop: {crop_selection}\nPrediction: {predicted_class}\nConfidence: {confidence:.2f}"
                st.download_button("Download Results", result_text, file_name="classification_result.txt")
            else:
                st.error("❌ Classification failed. Try again.")
