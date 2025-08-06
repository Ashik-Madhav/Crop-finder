import os
import requests
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

MODEL_PATH = "crop_classification_model.h5"
DRIVE_URL = "https://drive.google.com/uc?export=download&id=1MVPWJK71yKIdM9xZDTMtp_Oo9pYQfSL5"

# Automatically download the model if not present
if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    response = requests.get(DRIVE_URL, timeout=120)
    response.raise_for_status()
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    print("Model downloaded successfully.")

# Load the model
model = tf.keras.models.load_model(MODEL_PATH)

# Define class names
class_names = [
    "Apple", "Banana", "Brinjal", "Cabbage", "Carrot",
    "Cauliflower", "Chili", "Corn", "Cucumber", "Onion",
    "Potato", "Rice", "Soybean", "Tomato", "Wheat"
]

def preprocess_image(image):
    image = image.resize((224, 224))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    image = Image.open(file.stream).convert("RGB")
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    predicted_class = class_names[np.argmax(predictions)]

    return jsonify({"predicted_crop": predicted_class})

@app.route("/", methods=["GET"])
def home():
    return "Crop Identifier is running!"

if __name__ == "__main__":
    app.run(debug=True)
