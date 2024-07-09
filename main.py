from flask import Flask, request, jsonify, render_template
import numpy as np
from torchvision.models import resnet18, ResNet18_Weights
import torch 
import os
from PIL import Image
from torchvision import datasets
import torchvision.transforms as transforms
import ast
import psycopg2
from create_db import add_row

app = Flask(__name__)

model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model.eval()

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def classify_image(image):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
    return predicted.item()

def get_labels():
    with open('imagenet1000_clsidx_to_labels.txt', 'r') as file:
        data = file.read()

    labels_dict = ast.literal_eval(data)
    return labels_dict

labels = get_labels()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        image = Image.open(file)
        prediction = classify_image(image)
        add_row(prediction=labels[prediction], 
                prediction_id=prediction)
        
        return jsonify({'prediction': labels[prediction]})

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
    