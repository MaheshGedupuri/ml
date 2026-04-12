import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.set_page_config(page_title="Cancer Detection", layout="centered")

st.title("🩺 Cancer Detection using X-ray")
st.write("Upload a chest X-ray image to detect possible cancer presence.")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cancer_model.h5")

try:
    model = load_model()
except Exception as e:
    st.error("Model not found or failed to load.")
    st.stop()

uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image = image.resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    st.subheader("Prediction Result")

    if prediction > 0.7:
        st.error(f"Cancer Detected (Confidence: {prediction:.2f})")
    else:
        st.success(f"Normal (Confidence: {1 - prediction:.2f})")

st.markdown("---")
st.caption("Academic Prototype - Not for Medical Use")
