import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model

# Load your trained model
model = load_model("fairface_cnn_model_with_orb_taketwo.h5")

# Streamlit UI setup
st.title("Live Model Prediction App")
st.write("Upload an image or take a picture to make a prediction.")

# Option 1: Use camera input
camera_image = st.camera_input("Take a picture")

# Option 2: File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# Load and preprocess the image from either source
if camera_image or uploaded_file:
    file_bytes = None
    if camera_image:
        file_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
    elif uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

    # Decode image and preprocess
    if file_bytes is not None:
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # Resize image to match the model's input shape
        resized_image = cv2.resize(image, (224, 224))  # Adjust size as per your model
        preprocessed_image = resized_image / 255.0  # Normalize to [0, 1]
        preprocessed_image = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension

        # Check model input shape
        input_shape = model.input_shape  # (None, 7) or (None, 224, 224, 3)
        st.write(f"Model expects input shape: {input_shape}")

        # Adjust preprocessing based on the model's expected input shape
        if input_shape == (None, 224, 224, 3):  # Model expects images
            # Use the preprocessed image (already resized and normalized)
            pass
        elif input_shape == (None, 7):  # Model expects a vector of size 7
            preprocessed_image = resized_image.flatten()[:7]  # Adjust to required size
            preprocessed_image = np.expand_dims(preprocessed_image, axis=0)
        else:
            st.error("Unsupported model input shape. Please check the model.")
            preprocessed_image = None

        if preprocessed_image is not None:
            # Make a prediction
            prediction = model.predict(preprocessed_image)

            # Display prediction
            st.write(f"Prediction: {np.argmax(prediction)}")
            st.write(f"Confidence: {np.max(prediction) * 100:.2f}%")
