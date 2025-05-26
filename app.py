from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained ensemble model
model = load_model('ensemble_model.h5')

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    file_path = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)

            img_tensor = preprocess_image(file_path)
            pred = model.predict(img_tensor)[0][0]
            prediction = "Fake" if pred >= 0.5 else "Real"

    return render_template('index.html', prediction=prediction, image_path=file_path)

if __name__ == '__main__':
    app.run(debug=True)
