import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

from src.predict import ensemble_predict

# CIFAR-10 class names
cifar10_classes = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Page config
st.set_page_config(page_title="CIFAR-10 Classifier", layout="wide")

# Dark mode toggle state
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False

# Dark mode toggle
if st.sidebar.checkbox("🌙 Dark Mode", value=st.session_state.dark_mode):
    st.session_state.dark_mode = True
else:
    st.session_state.dark_mode = False

# Apply dark or light theme
if st.session_state.dark_mode:
    st.markdown("""
        <style>
            body, .block-container { background-color: #0e1117 !important; color: #fafafa !important; }
            .stButton>button { background-color: #262730 !important; border: 1px solid #fafafa; color: #fafafa!important;}
            .stRadio label, .stMarkdown, p, label { color: #fafafa !important; }
        </style>
    """, unsafe_allow_html=True)

# GitHub link button
st.sidebar.markdown("### 🔗 Project Links")
st.sidebar.markdown(
    """
    <a href="https://github.com/arashshams/Deep-Learning-CIFAR-Classifier" target="_blank">
        <button style="width: 100%; padding: 10px; background-color:#4F8BF9; 
        color:white; border:none; border-radius:5px; cursor:pointer;">
            View on GitHub
        </button>
    </a>
    """,
    unsafe_allow_html=True
)

# Loading models once
@st.cache_resource
def load_models():
    m1 = load_model("models/model_1.h5")
    m2 = load_model("models/model_2.h5")
    m3 = load_model("models/model_3.h5")
    m4 = load_model("models/model_4.h5")
    return [m1, m2, m3, m4]

models = load_models()

# Main header
st.title("CIFAR-10 Ensemble Image Classifier")
st.write("Upload an image or generate a random CIFAR-10 test sample.")
st.markdown("---")

# Sidebar option
option = st.sidebar.radio(
    "Choose an option:",
    ("Upload an image", "Random CIFAR-10 test image")
)

# Refresh button for random mode
refresh = False
if option == "Random CIFAR-10 test image":
    refresh = st.sidebar.button("🔄 Refresh Image")

# UI layout columns
col_left, col_right = st.columns([1, 3])
col_left.subheader("Prediction Details")
col_right.subheader("Image Preview")

# Upload mode
if option == "Upload an image":
    uploaded = st.sidebar.file_uploader("Upload a JPG or PNG image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        col_right.image(img, caption="Uploaded image", use_container_width=True)

        pred_idx, conf, probs = ensemble_predict(models, img)
        col_left.write(f"### Prediction: **{cifar10_classes[pred_idx]}**")
        col_left.write(f"### Confidence: **{conf:.3f}**")

# Random CIFAR-10 image mode
else:
    from tensorflow.keras.datasets import cifar10
    (_, _), (x_test, y_test) = cifar10.load_data()

    idx = np.random.randint(0, len(x_test))
    img_arr = x_test[idx]
    true_label = int(y_test[idx][0])

    img = Image.fromarray(img_arr)
    col_right.image(img, use_container_width=True)

    col_right.markdown(
    f"<h3 style='text-align:center; font-size:22px;'>Actual label: {cifar10_classes[true_label]}</h3>",
    unsafe_allow_html=True
)


    pred_idx, conf, probs = ensemble_predict(models, img)
    col_left.write(f"### Prediction: **{cifar10_classes[pred_idx]}**")
    col_left.write(f"### Confidence: **{conf:.3f}**")


