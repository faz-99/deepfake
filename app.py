from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import gdown

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
MODEL_PATH = 'deepfake_ensemble_model.h5'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Download model from Google Drive if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        print("Downloading model from Google Drive...")
        url = "https://drive.google.com/uc?id=1CCuGDjPm5Av_6W23xKMcalfSIn2H65pK"
        gdown.download(url, MODEL_PATH, quiet=False)

# Load the model
download_model()
model = load_model(MODEL_PATH)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part in request'

        file = request.files['file']
        if file.filename == '':
            return 'No selected file'

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Preprocess image
            img = image.load_img(filepath, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            pred = model.predict(img_array)[0][0]
            prediction = 'Fake' if pred > 0.5 else 'Real'
            image_path = filepath

    return render_template('index.html', prediction=prediction, image_path=image_path)
