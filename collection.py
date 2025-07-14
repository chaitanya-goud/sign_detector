import cv2
import mediapipe as mp
import numpy as np
import time
from collections import Counter
import pickle
import os

class HandSignDetector:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # For data collection and training
        self.data = []
        self.labels = []
        self.current_label = None
        self.collecting = False
        
        # For prediction
        self.model = None
        self.prediction_buffer = []
        self.buffer_size = 10
        
        # Sign labels (customize these)
        self.sign_names = {
            
            1: "1", 
            2: "2",
            3: "3",
            4: "4",
            5: "5"
        }
        
    def extract_landmarks(self, hand_landmarks):
        """Extract normalized landmark coordinates"""
        landmarks = []
        if hand_landmarks:
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
        return landmarks
    
    def collect_data(self):
        """Data collection mode"""
        print("=== DATA COLLECTION MODE ===")
        print("Available signs:", list(self.sign_names.values()))
        print("Controls:")
        print("- Press number keys (0-5) to start collecting for that sign")
        print("- Press SPACE to stop collecting current sign")
        print("- Press 's' to save data")
        print("- Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Collect data if in collection mode
                    if self.collecting and self.current_label is not None:
                        landmarks = self.extract_landmarks(hand_landmarks)
                        if landmarks:
                            self.data.append(landmarks)
                            self.labels.append(self.current_label)
                            
            # Display info
            status_text = f"Collecting: {self.sign_names.get(self.current_label, 'None') if self.collecting else 'Stopped'}"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Samples: {len(self.data)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Hand Sign Data Collection', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Handle key presses
            if key == ord('q'):
                break
            elif key == ord(' '):
                self.collecting = False
                print(f"Stopped collecting. Total samples: {len(self.data)}")
            elif key == ord('s'):
                self.save_data()
            elif key in [ord(str(i)) for i in range(1,6)]:
                label = int(chr(key))
                if label in self.sign_names:
                    self.current_label = label
                    self.collecting = True
                    print(f"Started collecting for: {self.sign_names[label]}")
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_data(self):
        """Save collected data"""
        if len(self.data) > 0:
            data_dict = {
                'data': np.array(self.data),
                'labels': np.array(self.labels),
                'sign_names': self.sign_names
            }
            with open('hand_sign_data.pkl', 'wb') as f:
                pickle.dump(data_dict, f)
            print(f"Saved {len(self.data)} samples to hand_sign_data.pkl")
        else:
            print("No data to save!")
    
    def load_data(self):
        """Load saved data"""
        try:
            with open('hand_sign_data.pkl', 'rb') as f:
                data_dict = pickle.load(f)
            self.data = data_dict['data'].tolist()
            self.labels = data_dict['labels'].tolist()
            self.sign_names = data_dict['sign_names']
            print(f"Loaded {len(self.data)} samples")
            return True
        except FileNotFoundError:
            print("No saved data found. Please collect data first.")
            return False
    
    def train_model(self):
        """Train a simple classifier"""
        if len(self.data) == 0:
            print("No data available for training!")
            return
            
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score
        
        X = np.array(self.data)
        y = np.array(self.labels)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)
        
        # Test accuracy
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Model trained! Accuracy: {accuracy:.2f}")
        
        # Save model
        with open('hand_sign_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        print("Model saved to hand_sign_model.pkl")
    
    def load_model(self):
        """Load trained model"""
        try:
            with open('hand_sign_model.pkl', 'rb') as f:
                self.model = pickle.load(f)
            print("Model loaded successfully!")
            return True
        except FileNotFoundError:
            print("No trained model found. Please train a model first.")
            return False
    
    def predict_realtime(self):
        """Real-time prediction mode"""
        if self.model is None:
            print("No model loaded!")
            return
            
        print("=== REAL-TIME PREDICTION MODE ===")
        print("Press 'q' to quit")
        
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            
            prediction_text = "No hand detected"
            confidence_text = ""
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Draw landmarks
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Make prediction
                    landmarks = self.extract_landmarks(hand_landmarks)
                    if landmarks:
                        prediction = self.model.predict([landmarks])[0]
                        probabilities = self.model.predict_proba([landmarks])[0]
                        confidence = max(probabilities)
                        
                        # Use buffer for stable predictions
                        self.prediction_buffer.append(prediction)
                        if len(self.prediction_buffer) > self.buffer_size:
                            self.prediction_buffer.pop(0)
                        
                        # Get most common prediction
                        if len(self.prediction_buffer) >= 5:
                            stable_prediction = Counter(self.prediction_buffer).most_common(1)[0][0]
                            prediction_text = f"Sign: {self.sign_names.get(stable_prediction, 'Unknown')}"
                            confidence_text = f"Confidence: {confidence:.2f}"
            
            # Display prediction
            cv2.putText(frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, confidence_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Hand Sign Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()

# === FLASK WEB APP FOR HAND SIGN DETECTION ===
from flask import Flask, request, render_template, Response, jsonify
from werkzeug.utils import secure_filename
import io
from PIL import Image
import base64

app = Flask(__name__)

# Ensure templates directory exists for Flask
if not os.path.exists('templates'):
    os.makedirs('templates')

def get_latest_model_and_sign_names():
    detector = HandSignDetector()
    detector.load_model()
    # Try to load sign_names from data if available
    try:
        with open('hand_sign_data.pkl', 'rb') as f:
            data_dict = pickle.load(f)
            detector.sign_names = data_dict.get('sign_names', detector.sign_names)
    except Exception:
        pass
    return detector

@app.route('/', methods=['GET', 'POST'])
def upload_predict():
    prediction = None
    if request.method == 'POST':
        if 'file' not in request.files:
            prediction = 'No file uploaded.'
        else:
            file = request.files['file']
            if file.filename == '':
                prediction = 'No file selected.'
            else:
                filename = secure_filename(file.filename)
                img_bytes = file.read()
                img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
                img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                detector = get_latest_model_and_sign_names()
                mp_hands = detector.mp_hands
                hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
                results = hands.process(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        landmarks = detector.extract_landmarks(hand_landmarks)
                        if landmarks and detector.model:
                            pred = detector.model.predict([landmarks])[0]
                            prediction = detector.sign_names.get(pred, str(pred))
                        else:
                            prediction = 'Could not extract landmarks.'
                else:
                    prediction = 'No hand detected.'
    return render_template('index.html', prediction=prediction)

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
    # Remove header if present
    if ',' in img_data:
        img_data = img_data.split(',')[1]
    img_bytes = base64.b64decode(img_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    detector = get_latest_model_and_sign_names()
    mp_hands = detector.mp_hands
    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
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

# Only run Flask if not running as CLI
if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'web':
        app.run(debug=True)