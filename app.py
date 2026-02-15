import streamlit as st
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("xray_model.h5")


st.title("Cancer Detection from X-ray")

uploaded_file = st.file_uploader("Upload X-ray", type=["jpg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((224,224))
    st.image(img)

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    if prediction > 0.5:
        st.write("Prediction: Cancer Detected")
    else:
        st.write("Prediction: Normal")
