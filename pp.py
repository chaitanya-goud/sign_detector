import cv2
import mediapipe as mp
import numpy as np
import pickle
import io
import os
from flask import Flask, request, render_template, jsonify
from PIL import Image
import base64

# Ensure templates directory exists for Flask
if not os.path.exists('templates'):
    os.makedirs('templates')

class HandSignDetector:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.sign_names = {1: "1", 2: "2", 3: "3", 4: "4", 5: "5"}
        self.model = None
        self.load_model()

    def extract_landmarks(self, hand_landmarks):
        landmarks = []
        if hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks

    def load_model(self):
        try:
            with open('hand_sign_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            # Try to load sign_names from data if available
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

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    data = request.json
    if not isinstance(data, dict):
        return jsonify({'error': 'No JSON data'}), 400
    img_data = data.get('image')
    if not img_data:
        return jsonify({'error': 'No image data'}), 400
    if ',' in img_data:
        img_data = img_data.split(',')[1]
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    hands = detector.mp_hands.Hands(static_image_mode=True, max_num_hands=1)
    results = hands.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
    prediction = 'No hand detected.'
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = detector.extract_landmarks(hand_landmarks)
            if landmarks and detector.model:
                pred = detector.model.predict([landmarks])[0]
                prediction = detector.sign_names.get(pred, str(pred))
            else:
                prediction = 'Could not extract landmarks.'
    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    import sys
    app.run(debug=True) 