
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import tensorflow as tf
from PIL import Image

import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import easyocr
import pickle


# Load your trained model and tokenizer
model = load_model('news_classification_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)


max_seq_length = 24  # Update this to the max sequence length from your training

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])  # Ensure you've downloaded the necessary language models

def predict_news(text):
    sequence = tokenizer.texts_to_sequences([text])
    pad = pad_sequences(sequence, maxlen=max_seq_length)
    prediction = model.predict(pad)
    label = np.argmax(prediction, axis=1)
    return 'Real' if label == 1 else 'Fake'

def extract_text_from_image(image):
    detected_text = reader.readtext(np.array(image), paragraph=True)
    extracted_text = " ".join([text[1] for text in detected_text])
    return extracted_text
def save_uploaded_file(directory, uploaded_file, save_as):
    # Create the directory if it doesn't exist
    if not os.path.exists(directory):
        os.makedirs(directory)
    # Save the uploaded file to the specified directory with the specified name
    file_path = os.path.join(directory, save_as)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


# Streamlit UI
st.title('Fake News Detection App')
st.write("Upload an image containing text or enter text manually to check if it's real or fake news.")

# Image Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
     # Specify the directory and the filename you want to save the uploaded file as
    directory = "uploaded_images"
    save_as = "uploaded_image.jpg"  # Specify the name you want to save the file as

    # Save the uploaded image
    saved_path = save_uploaded_file(directory, uploaded_file, save_as)

    if saved_path:  # If the image is saved successfully
        st.success(f"Image successfully saved at: {saved_path}")
        # Display the saved image
        image = Image.open(saved_path)

    #user_input = extract_text_from_image(uploaded_file)
    result = reader.readtext(saved_path)
    #print(result)
    text=[]
    # Print the extracted text
    for detection in result:
        print(detection[1])  # The detected text
        text.append(detection[1])
    user_input = ' '.join(text)
    st.write("Extracted Text:")
    st.write(user_input)
else:
    user_input = st.text_area("Or Enter Text Here", "")

if st.button('Predict'):
    if user_input:
        prediction = predict_news(user_input)
        st.write(f'Prediction: *{prediction}*')
    else:
        st.write("Please upload an image or enter text to predict.")