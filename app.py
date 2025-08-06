import os
import json
import cv2
import numpy as np
import tensorflow as tf
import requests
from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Google Drive file ID for your model
GOOGLE_DRIVE_FILE_ID = '1MVPWJK71yKIdM9xZDTMtp_Oo9pYQfSL5'
MODEL_PATH = "crop_classification_model.h5"

def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {'id': id, 'confirm': token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:
                f.write(chunk)

# Download model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    download_file_from_google_drive(GOOGLE_DRIVE_FILE_ID, MODEL_PATH)
    print("Download completed.")

# Load the model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load crop info JSON
with open('crop_info.json', 'r') as f:
    crop_info = json.load(f)

# Crop names (order must match model's output classes)
crop_names = ['Apple', 'Banana', 'Cotton', 'Grapes', 'Jute', 'Maize',
              'Mango', 'Millets', 'Orange', 'Paddy', 'Papaya', 'Sugarcane',
              'Tea', 'Tomato', 'Wheat']

img_height, img_width = 224, 224

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('result.html', prediction="Model not loaded.", info="")

    if 'file' not in request.files:
        return render_template('result.html', prediction="No file part in the request.", info="")

    file = request.files['file']

    if file.filename == '':
        return render_template('result.html', prediction="No selected file.", info="")

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        img = cv2.imread(filepath)
        if img is None:
            return render_template('result.html', prediction="Could not read image.", info="")

        img = cv2.resize(img, (img_width, img_height))
        img = np.expand_dims(img, axis=0)

        predictions = model.predict(img)
        predicted_class_index = np.argmax(predictions)
        predicted_crop = crop_names[predicted_class_index]

        info = crop_info.get(predicted_crop, {})
        info_text = ""
        for k, v in info.items():
            info_text += f"<b>{k.replace('_', ' ').title()}:</b> {v}<br>"

        return render_template('result.html', prediction=predicted_crop, info=info_text)

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {e}", info="")

if __name__ == '__main__':
    app.run(debug=True)
