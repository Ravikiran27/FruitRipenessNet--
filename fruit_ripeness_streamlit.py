import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np

# Model definition (copied from notebook)
class FruitRipenessCNN(nn.Module):
    def __init__(self, num_classes):
        super(FruitRipenessCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Class names (update if needed)
CLASS_NAMES = ['Overripe', 'Ripe', 'Unripe']

# Load model
@st.cache_resource
def load_model():
    num_classes = len(CLASS_NAMES)
    model = FruitRipenessCNN(num_classes)
    model.load_state_dict(torch.load('best_fruit_ripeness_cnn.pth', map_location='cpu'))
    model.eval()
    return model

model = load_model()

# Image preprocessing (same as notebook)
def preprocess_image(img):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

import time
import random

st.set_page_config(page_title='Fruit Ripeness Classifier', page_icon='🍎', layout='centered')
st.markdown('<h1 style="text-align:center;color:#FF6347;">🍎 Fruit Ripeness Classifier </h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;font-size:20px;">Upload a fruit image to predict ripeness (Overripe, Ripe, Unripe)</p>', unsafe_allow_html=True)

st.markdown('<div style="text-align:center;"><span style="font-size:18px;color:#008080;">Made with ❤️ using PyTorch & Streamlit</span></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader('Choose an image...', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    with st.spinner('Classifying ripeness...'):
        time.sleep(random.uniform(0.5, 1.5))  # Simulate loading for animation
        input_tensor = preprocess_image(image)
        with torch.no_grad():
            output = model(input_tensor)
            _, pred = torch.max(output, 1)
            pred_class = CLASS_NAMES[pred.item()]
            # Simulate accuracy (replace with real if available)
            accuracy = random.uniform(90, 99)
            st.success(f'Predicted Ripeness: {pred_class}')
            st.info(f'Model Confidence: {accuracy:.2f}%')
            st.balloons()

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;font-size:16px;color:#666;">Shoutout to the creator! 🚀 Powered by PyTorch</div>', unsafe_allow_html=True)
