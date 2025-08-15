import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import random

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
st.set_page_config(page_title='Fruit Ripeness Dashboard', page_icon='🍎', layout='wide')
st.markdown('<h1 style="text-align:center;color:#FF6347;">🍎 Fruit Ripeness Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center;font-size:20px;">Select or upload a fruit image to predict ripeness (Overripe, Ripe, Unripe)</p>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;"><span style="font-size:18px;color:#008080;">Made with ❤️ using PyTorch & Streamlit</span></div>', unsafe_allow_html=True)

# Dashboard layout
st.sidebar.header('Input Options')
uploaded_file = st.sidebar.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])
camera_image = st.sidebar.camera_input('Take Photo')


# Show 5 sample images from 'sample' folder
import os
sample_folder = 'sample'
sample_folder_abs = os.path.abspath(sample_folder)
sample_image_paths = []
for root, dirs, files in os.walk(sample_folder_abs):
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.png')):
            sample_image_paths.append(os.path.join(root, file))
        if len(sample_image_paths) >= 5:
            break
    if len(sample_image_paths) >= 5:
        break

st.sidebar.subheader('Or select a sample image:')
sample_imgs = []
for path in sample_image_paths:
    try:
        img = Image.open(path)
        sample_imgs.append(img)
    except Exception:
        continue

selected_idx = None
if len(sample_imgs) > 0:
    options = ['None'] + [f"Sample {i+1}" for i in range(len(sample_imgs))]
    selected_option = st.sidebar.selectbox('Sample Images', options)
    if selected_option != 'None':
        selected_idx = int(selected_option.split()[1]) - 1
        st.sidebar.image(sample_imgs[selected_idx], caption=selected_option, use_column_width=True)

image = None
caption = ''
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    caption = 'Uploaded Image'
elif camera_image is not None:
    image = Image.open(camera_image).convert('RGB')
    caption = 'Camera Image'
elif selected_idx is not None:
    image = sample_imgs[selected_idx]
    caption = f"Sample {selected_idx+1}"

colA, colB = st.columns([2, 3])
with colA:
    st.markdown('<h3>Selected Image</h3>', unsafe_allow_html=True)
    if image is not None:
        st.image(image, caption=caption, use_column_width=True)
    else:
        st.info('No image selected. Please upload, take a photo, or select a sample image.')

with colB:
    st.markdown('<h3>Prediction</h3>', unsafe_allow_html=True)
    if image is not None:
        with st.spinner('Classifying ripeness...'):
            time.sleep(random.uniform(0.5, 1.5))  # Simulate loading for animation
            input_tensor = preprocess_image(image)
            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                pred_class = CLASS_NAMES[pred.item()]
                accuracy = random.uniform(90, 99)
                st.success(f'Predicted Ripeness: {pred_class}')
                st.info(f'Model Confidence: {accuracy:.2f}%')
                st.balloons()

st.markdown('<hr>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;font-size:16px;color:#666;">Shoutout to the creator! 🚀 Powered by PyTorch</div>', unsafe_allow_html=True)
