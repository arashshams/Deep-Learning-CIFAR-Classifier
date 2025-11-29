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

@st.cache_resource
def load_models():
    m1 = load_model("models/model_1.h5")
    m2 = load_model("models/model_2.h5")
    m3 = load_model("models/model_3.h5")
    m4 = load_model("models/model_4.h5")
    return [m1, m2, m3, m4]

models = load_models()

st.title("CIFAR-10 Ensemble Image Classifier")

option = st.radio(
    "Choose an option:",
    ("Upload an image", "Random CIFAR-10 test image")
)

if option == "Upload an image":
    uploaded = st.file_uploader("Upload a JPG or PNG image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Uploaded image", use_column_width=True)

        pred_idx, conf, probs = ensemble_predict(models, img)
        st.write(f"**Prediction:** {cifar10_classes[pred_idx]}")
        st.write(f"**Confidence:** {conf:.3f}")

else:
    # Using CIFAR-10 test set from Keras
    from tensorflow.keras.datasets import cifar10
    (_, _), (x_test, y_test) = cifar10.load_data()

    idx = np.random.randint(0, len(x_test))
    img_arr = x_test[idx]
    true_label = int(y_test[idx][0])

    img = Image.fromarray(img_arr)
    st.image(img, caption=f"Actual label: {cifar10_classes[true_label]}", use_column_width=True)

    pred_idx, conf, probs = ensemble_predict(models, img)
    st.write(f"**Prediction:** {cifar10_classes[pred_idx]}")
    st.write(f"**Confidence:** {conf:.3f}")
