from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
from PIL import Image
import os

app = Flask(__name__)
model = load_model("crop_classification_model.h5")
class_names = ['Apple', 'Banana', 'Corn', 'Cotton', 'Grapes', 'Jute', 'Maize', 'Mango', 'Millets', 'Orange', 'Rice', 'Sugarcane', 'Tea', 'Wheat', 'Watermelon']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400
    file = request.files['image']
    if file.filename == '':
        return "No image selected", 400

    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)

    img = Image.open(img_path).resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = class_names[np.argmax(prediction)]

    return render_template('result.html', prediction=predicted_class)

if __name__ == '__main__':
    app.run(debug=True)
