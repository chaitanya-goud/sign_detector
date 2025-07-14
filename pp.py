import numpy as np
import pickle
import os
from flask import Flask, request, render_template, jsonify

# Ensure templates directory exists for Flask
if not os.path.exists('templates'):
    os.makedirs('templates')

class HandSignDetector:
    def __init__(self):
        self.sign_names = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
        self.model = None
        self.load_model()

    def load_model(self):
        try:
            with open('hand_sign_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            try:
                with open('hand_sign_data.pkl', 'rb') as f2:
                    data_dict = pickle.load(f2)
                    self.sign_names = data_dict.get('sign_names', self.sign_names)
            except Exception:
                pass
        except FileNotFoundError:
            print("No trained model found. Please train a model first.")
            self.model = None

detector = HandSignDetector()

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html', prediction=None)

@app.route('/realtime')
def realtime_page():
    return render_template('realtime.html')

@app.route('/predict_landmarks', methods=['POST'])
def predict_landmarks():
    data = request.json
    if not isinstance(data, dict):
        return jsonify({'error': 'No JSON data'}), 400
    landmarks = data.get('landmarks')
    if not landmarks or not isinstance(landmarks, list):
        return jsonify({'prediction': 'No hand detected.'})
    if detector.model:
        try:
            arr = np.array(landmarks).reshape(1, -1)
            pred = detector.model.predict(arr)[0]
            prediction = detector.sign_names.get(pred, str(pred))
        except Exception as e:
            prediction = f'Error: {str(e)}'
    else:
        prediction = 'Model not loaded.'
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True) 
