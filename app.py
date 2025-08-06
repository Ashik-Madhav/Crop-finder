import os
import json
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import requests

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'crop_classification_model.h5'
MODEL_URL = 'https://drive.google.com/uc?id=1MVPWJK71yKIdM9xZDTMtp_Oo9pYQfSL5'

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Download the model if not already present
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        try:
            response = requests.get(MODEL_URL)
            with open(MODEL_PATH, 'wb') as f:
                f.write(response.content)
            print("Download completed.")
        except Exception as e:
            print(f"Failed to download model: {e}")

# Download and load the model
download_model()
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Load crop information
try:
    with open('crop_info.json', 'r') as f:
        crop_info = json.load(f)
except:
    crop_info = {}

# Crop class labels (must match model training order)
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
        return render_template('result.html', prediction="No file uploaded.", info="")

    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', prediction="No file selected.", info="")

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        img = cv2.imread(filepath)
        if img is None:
            return render_template('result.html', prediction="Invalid image format.", info="")

        img = cv2.resize(img, (img_width, img_height))
        img = img / 255.0  # Normalize image
        img = np.expand_dims(img, axis=0)

        predictions = model.predict(img)
        predicted_index = np.argmax(predictions)
        predicted_crop = crop_names[predicted_index]

        # Get crop info
        info = crop_info.get(predicted_crop, {})
        info_text = ""
        for k, v in info.items():
            info_text += f"<b>{k.replace('_', ' ').title()}:</b> {v}<br>"

        return render_template('result.html', prediction=predicted_crop, info=info_text)

    except Exception as e:
        return render_template('result.html', prediction=f"Error: {e}", info="")

if __name__ == '__main__':
    app.run(debug=True)
