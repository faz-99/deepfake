Deepfake Image Detector
A simple Flask web app that detects whether an uploaded image is a deepfake or real using a pre-trained Keras deep learning model.
Features
Upload an image via web form
Predict if the image is Fake or Real
Display the uploaded image and prediction result
Setup Instructions
1. Clone this repository
Bash
git clone https://github.com/faz-99/deepfake.git
cd deepfake
2. Install dependencies
Bash
pip install -r requirements.txt
If you don’t have requirements.txt, install manually:
Bash
pip install flask tensorflow numpy werkzeug
3. Prepare the model file
The model file deepfake_ensemble_model.h5 (~300MB) cannot be pushed to GitHub due to size limits. To use the app:
Download the model from your Google Drive at /content/drive/MyDrive/deepfake_ensemble_model.h5
Place it inside a folder called model in the project root:
Code
deepfake/
├── app.py
├── model/
│   └── deepfake_ensemble_model.h5
└── templates/
Make sure your app loads the model with:
Python
model = load_model('model/deepfake_ensemble_model.h5')
4. Run the Flask app
On Linux/macOS terminal:
Bash
export FLASK_APP=app.py
export FLASK_ENV=development   # optional, for debug mode
flask run
On Windows CMD:
Bash
set FLASK_APP=app.py
set FLASK_ENV=development
flask run
5. Use the app
Open your browser and visit http://127.0.0.1:5000 to upload images and get predictions.
