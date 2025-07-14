# Hand Sign Detection Flask App

This project is a real-time hand sign detection web app using Flask, OpenCV, and MediaPipe.

## Features
- Real-time hand sign detection using your webcam
- Simple web interface
- Powered by a trained scikit-learn model

## Requirements
- Python 3.7+
- See `requirements.txt` for dependencies

## Local Development
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   python pp.py
   ```
3. Open [http://localhost:5000/realtime](http://localhost:5000/realtime) in your browser.

## Deploying to Render (Free Tier)

1. **Push your code to a GitHub repository.**
2. **Create a Render account** at [https://render.com](https://render.com).
3. **Create a new Web Service**:
   - Connect your GitHub repo.
   - Select Python 3 environment.
   - Render will auto-detect `requirements.txt` and `Procfile`.
   - Set the Start Command to:
     ```
     python pp.py
     ```
   - (Or leave as is, since the `Procfile` is present.)
4. **Deploy!**
   - Wait for the build to finish.
   - Visit the generated URL (e.g., `https://your-app.onrender.com/realtime`).

## Notes
- Make sure `hand_sign_model.pkl` (and optionally `hand_sign_data.pkl`) are in the root directory.
- The free Render tier will sleep after 15 minutes of inactivity.
- If you need to collect new data or retrain, do so locally and upload the new model file.

---

**Enjoy your real-time hand sign detection app!** 
