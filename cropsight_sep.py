import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import numpy as np

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CNNModel(nn.Module):
    def __init__(self, n_classes=10):
        super(CNNModel, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout2d(0.3),
            nn.MaxPool2d(2)
        )

        self.fc_layers = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # Global Average Pooling
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

# Transformations for the input image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dictionary to map plant names to model paths and class names
plant_models = {
    "Cotton": {"model_path": "Cotton_cnn_model_state_dict.pth", "classes": ["bacterial_blight", "curl_virus", "fussarium_wilt", "healthy"]},
    "Potato": {"model_path": "Potato_cnn_model_state_dict.pth", "classes": ["Early_Blight", "Healthy", "Late_Blight"]},
    "Wheat": {"model_path": "Wheat_cnn_model_state_dict.pth", "classes": ["Brown_Rust", "Healthy", "Yellow_Rust"]}
}

# Streamlit App
st.title("Plant Disease Detection")

# Plant Selection
plant_name = st.selectbox("Select Plant", list(plant_models.keys()))
model_info = plant_models[plant_name]

# Load Model
model = CNNModel(n_classes=len(model_info["classes"]))
model.load_state_dict(torch.load(model_info["model_path"], map_location=device))
model.to(device)
model.eval()

# Image Upload
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    

    # Preprocess Image
    image_tensor = transform(image).unsqueeze(0).to(device)

    # Prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted_idx = torch.max(probabilities, 1)

    confidence = confidence.item() * 100
    predicted_class = model_info["classes"][predicted_idx.item()].replace("_", " ")  # Removing underscores

    # Display Results
    st.header(f"**Plant:** {plant_name}")
    if confidence < 90:
        st.subheader("**Disease Predicted:** Undefined")

    else:
        st.subheader(f"**Disease Predicted:** {predicted_class}")
        
    st.subheader(f"**Confidence:** {confidence:.2f}%")
    st.image(image, caption="Uploaded Image", use_container_width=True)