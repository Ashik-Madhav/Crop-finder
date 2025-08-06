import os
import json
import cv2
import numpy as np
import tensorflow as tf
import gdown
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'crop_classification_model.h5'
MODEL_URL = 'https://drive.google.com/uc?id=1MVPWJK71yKIdM9xZDTMtp_Oo9pYQfSL5'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive using gdown...")
        try:
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
            print("Download completed.")
        except Exception as e:
            print("Failed to download model:", e)

# Download and load model
download_model()

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully.")
except Exception as e:
    print("Error loading model:", e)
    model = None

# Load crop info
with open('crop_info.json', 'r') as f:
    crop_info = json.load(f)

# Crop class names
crop_names = ['Apple', 'Banana', 'Cotton', 'Grapes', 'Jute', 'Maize',
              'Mango', 'Millets', 'Orange', 'Paddy', 'Papaya',
              'Sugarcane', 'Tea', 'Tomato', 'Wheat']

IMG_H, IMG_W = 224, 224

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return render_template('result.html', prediction="Model not loaded.", info="")

    if 'file' not in request.files:
        return render_template('result.html', prediction="No file provided.", info="")

    file = request.files['file']
    if file.filename == '':
        return render_template('result.html', prediction="No file selected.", info="")

    fname = secure_filename(file.filename)
    path = os.path.join(UPLOAD_FOLDER, fname)
    file.save(path)

    img = cv2.imread(path)
    if img is None:
        return render_template('result.html', prediction="Invalid image.", info="")

    img = cv2.resize(img, (IMG_W, IMG_H))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    idx = np.argmax(preds)
    crop = crop_names[idx]

    info = crop_info.get(crop, {})
    info_str = "".join(f"<b>{k.replace('_',' ').title()}:</b> {v}<br>" for k, v in info.items())

    return render_template('result.html', prediction=crop, info=info_str)

if __name__ == "__main__":
    port = int(os.getenv('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
