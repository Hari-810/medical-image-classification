
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load ultrasound model
ultrasound_model = load_model('ultrasound_classifier_model.h5')

# Define class names for the ultrasound model
ultrasound_class_names = ["External test images", "benign", "images", "malignant"]

# Streamlit interface
st.title("Ultrasound Image Classification")

def predict_image(model, class_names, uploaded_file):
    if uploaded_file is not None:
        # Read the image file buffer and convert to RGB
        img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Assuming the model expects scaled images

        # Check if the model expects flattened input
        if len(model.input_shape) == 2 and model.input_shape[1] != 224 * 224 * 3:
            img_array = np.reshape(img_array, (1, -1))

        prediction = model.predict(img_array)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        predicted_class_name = class_names[predicted_class_index]

        return predicted_class_name
    return None

# File uploader for ultrasound images
uploaded_file = st.file_uploader("Choose an ultrasound image...", type=["jpg", "jpeg", "png"])

if st.button("Predict"):
    result = predict_image(ultrasound_model, ultrasound_class_names, uploaded_file)
    if result:
        st.success(f"Predicted Class: {result}")
    else:
        st.error("Please upload an image first.")

if __name__ == "__main__":
    st.write("Streamlit Ultrasound Image Classification App")
