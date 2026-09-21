import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model("fruit_model.keras")

# Class names
class_names = ["fresh", "rotten"]

# Streamlit title
st.title("🍎 Fruit Freshness Classifier")
st.write("Upload a fruit image and the model will predict whether it is fresh or rotten.")

# Upload image
uploaded_file = st.file_uploader(
    "Upload a fruit image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Open image
    image = Image.open(uploaded_file)

    # Display image
    st.image(
        image,
        caption="Uploaded Fruit",
        use_container_width=True
    )

    # Resize image
    image = image.resize((224, 224))

    # Convert image to NumPy array
    image_array = np.array(image)

    # Convert RGBA to RGB
    if image_array.shape[-1] == 4:
        image_array = image_array[:, :, :3]

    # Normalize
    image_array = image_array / 255.0

    # Add batch dimension
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array)

    # Get predicted class
    predicted_class = np.argmax(prediction[0])

    # Get confidence
    confidence = np.max(prediction[0]) * 100

    # Display result
    if predicted_class == 1:
        result = "🍎 Fruit is Fresh"
    else:
        result = "🍎 Fruit is Rotten"

    st.success(result)

    # Display confidence
    st.write(f"Confidence: {confidence:.2f}%")