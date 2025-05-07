import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load your trained model
model = tf.keras.models.load_model("your_model.h5")

# Get model's expected input shape (e.g., (32, 32))
input_shape = model.input_shape[1:3]

st.title("🔥 Wildfire Image Classifier")
st.write("Upload an image to check if it shows a **Wildfire** or **Non-Wildfire**.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Uploaded Image", use_container_width=True)

    # Resize and normalize
    image_resized = image.resize(input_shape)
    image_array = np.array(image_resized) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)[0][0]
    class_label = "🔥 Wildfire" if prediction > 0.5 else "🌿 Non-Wildfire"

    # Show prediction
    st.subheader(f"Prediction: **{class_label}**")
    st.write(f"Confidence Score: `{prediction:.2f}`")

    # Interpretability Additions
    threshold = 0.5
    delta = abs(prediction - threshold)

    # Prediction Strength
    if prediction >= 0.9 or prediction <= 0.1:
        strength = "🟢 Very Certain"
    elif 0.7 <= prediction <= 0.9 or 0.1 <= prediction <= 0.3:
        strength = "🟡 Moderately Certain"
    else:
        strength = "🔴 Uncertain Prediction"

    # Show extra insights
    st.write(f"Threshold for classification: `{threshold}`")
    st.write(f"Prediction Strength: {strength}")
    st.write(f"Distance from decision boundary: `{delta:.2f}`")

    # Explanation
    if prediction > 0.85:
        reason = "High flame or smoke patterns detected — very likely a wildfire scene."
    elif prediction > 0.5:
        reason = "Some fire-like features present — could be a wildfire."
    elif prediction < 0.2:
        reason = "Clear or safe-looking terrain — no obvious wildfire indicators."
    else:
        reason = "Uncertain features — doesn't clearly match wildfire patterns."

    st.info(f"🧠 **Reason:** {reason}")
