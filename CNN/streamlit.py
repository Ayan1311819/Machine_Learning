import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import gdown
import os

file_id = '1-AQGIP_4wIGacaTcHmNg0WdbzvXFwk8W'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'my_model.h5'
gdown.download(url, output, quiet=False)
model = tf.keras.models.load_model('my_model.h5')

# Check if the file exists
if os.path.exists(output):
    print("File downloaded successfully")
else:
    print("Failed to download file")
# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((227, 227))
    image = np.array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image
  
# Streamlit
st.title('CNN Model Deployment with Streamlit')
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)
    st.write(f'Prediction: {np.argmax(prediction)}')
