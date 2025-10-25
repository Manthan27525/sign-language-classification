import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import json
from PIL import Image


MODEL_PATH = "model/gesture_model.tflite"
CLASS_MAP_PATH = "notebooks/class_names.json"


@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model


@st.cache_data
def load_classes():
    with open(CLASS_MAP_PATH, "r") as f:
        classes = json.load(f)
    if isinstance(classes, dict):
        classes = [classes[str(i)] for i in range(len(classes))]
    return classes


model = load_model()
classes = load_classes()


def preprocess_image(image):
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
    img = cv2.resize(img, (64, 64))
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=(0, -1))
    return img


st.set_page_config(
    page_title="Sign Language Classifier", page_icon="🤟", layout="centered"
)
st.title("🖐️ Sign Language Gesture Recognition")

st.sidebar.header("Choose Input Source")
option = st.sidebar.radio("Select one:", ["📁 Upload Image", "📸 Use Webcam"])

st.sidebar.markdown("---")
st.sidebar.write("Model: **gesture_model.h5**")
st.sidebar.write(f"Classes Detected: {len(classes)}")


if option == "📁 Upload Image":
    uploaded_file = st.file_uploader(
        "Upload an image of a hand sign:", type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        img_in = preprocess_image(image)
        preds = model.predict(img_in)
        top_idx = np.argmax(preds[0])
        top_prob = preds[0][top_idx]
        label = classes[top_idx]

        st.subheader("🧠 Prediction Result")
        st.success(f"**{label}**  ({top_prob * 100:.2f}%)")


elif option == "📸 Use Webcam":
    st.info("Click below to capture a live image")
    camera_input = st.camera_input("Capture Hand Gesture")

    if camera_input is not None:
        image = Image.open(camera_input).convert("RGB")
        st.image(image, caption="Captured Image", use_container_width=True)

        img_in = preprocess_image(image)
        preds = model.predict(img_in)
        top_idx = np.argmax(preds[0])
        top_prob = preds[0][top_idx]
        label = classes[top_idx]

        st.subheader("🧠 Prediction Result")
        st.success(f"**{label}**  ({top_prob * 100:.2f}%)")
