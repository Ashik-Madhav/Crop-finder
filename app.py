from flask import Flask, request, jsonify, render_template_string
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import os
import json

app = Flask(__name__)

# Load the model
MODEL_PATH = 'crop_classification_model.h5'
model = load_model(MODEL_PATH)

# Load crop info from JSON
with open('crop_info.json', 'r') as f:
    crop_info = json.load(f)

# Crop class names
CROP_CLASSES = ['Apple', 'Banana', 'Cotton', 'Grapes', 'Jute', 'Maize',
                'Mango', 'Millets', 'Orange', 'Paddy', 'Papaya', 'Sugarcane',
                'Tea', 'Tomato', 'Wheat']

# HTML UI
HTML_TEMPLATE = '''
<!doctype html>
<title>Crop Identifier</title>
<h1>Upload an image to identify the crop</h1>
<form method=post enctype=multipart/form-data>
  <input type=file name=file>
  <input type=submit value=Upload>
</form>
{% if prediction %}
<h2>Prediction: {{ prediction }}</h2>
<h3>Scientific Name: {{ info.scientific_name }}</h3>
<h3>Season: {{ info.season }}</h3>
<h3>Growth Duration: {{ info.growth_duration }}</h3>
<h3>Climate:</h3>
<ul>
    <li>Temperature: {{ info.climate.temperature }}</li>
    <li>Rainfall: {{ info.climate.rainfall }}</li>
    <li>Soil Type: {{ info.climate.soil_type }}</li>
</ul>
{% endif %}
'''

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    info = None
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Process image
            image = Image.open(file).resize((224, 224)).convert('RGB')
            img_array = np.array(image) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Predict
            prediction_index = np.argmax(model.predict(img_array))
            prediction = CROP_CLASSES[prediction_index]

            # Get crop info
            for crop in crop_info["crops"]:
                if crop["name"].lower() == prediction.lower():
                    info = crop
                    break

    return render_template_string(HTML_TEMPLATE, prediction=prediction, info=info)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 10000)))
