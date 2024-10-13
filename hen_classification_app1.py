import streamlit as st
import cv2
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model (make sure the model file is in the same directory)
model = tf.keras.models.load_model('C:/path/to/quantized_hen_classification_model.tflite')

# Function to preprocess the frame (resize and normalize)
def preprocess_frame(frame):
    img_resized = cv2.resize(frame, (128, 128))
    img_normalized = img_resized / 255.0
    img_input = np.expand_dims(img_normalized, axis=0)
    return img_input

# Function to make prediction
def predict(frame):
    processed_frame = preprocess_frame(frame)
    prediction = model.predict(processed_frame)[0][0]
    label = "Healthy" if prediction < 0.5 else "Suffering"
    return label

# Streamlit app layout
st.title("Hen Classification: Healthy or Suffering")
st.write("This app uses a camera feed to classify a hen as either 'Healthy' or 'Suffering'.")

# Capture live camera feed
cam_feed = st.camera_input("Take a live photo")

if cam_feed is not None:
    # Convert the image from the live camera feed
    image = Image.open(cam_feed)
    frame = np.array(image)
    
    # Convert from RGB (used by PIL) to BGR (used by OpenCV)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Display the image in Streamlit
    st.image(frame, channels="BGR")

    # Make a prediction
    prediction = predict(frame)

    # Show the result
    st.write(f"Prediction: The hen is {prediction}")
