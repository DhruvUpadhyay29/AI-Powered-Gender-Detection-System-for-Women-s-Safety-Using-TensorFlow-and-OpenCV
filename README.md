
# 🛡️ AI-Powered Gender Detection System for Women’s Safety using TensorFlow and OpenCV

This project leverages AI, computer vision, and gender classification to enhance women’s safety. It detects faces from a live webcam stream, classifies gender using a trained model, and logs alerts when males are detected in female-only zones.

---

## 🚀 Features

- 👁️ Real-time face detection using OpenCV
- 🧠 Gender classification with a TensorFlow-trained CNN model
- 🔔 Alert system with logging (JSON format)
- 🌐 Web interface with live video feed
- 📊 Logs male entries for monitoring

---

## 🗂️ Project Structure

```
AI-Powered-Gender-Detection/
├── ap8.py                   # Main application script
├── train1.py                # Model training script
├── alert_log.json           # Stores logged alerts
├── template/
│   ├── index.html           # Landing page
│   ├── nextpage.html        # Post-detection interface
│   ├── video_feed.html      # Live feed display
│   ├── script.js            # Frontend JavaScript
│   └── styles.css           # Styling
└── .gitattributes
```

---

## 💻 How to Run

### 🧰 Requirements

- Python 3.7+
- OpenCV
- TensorFlow
- Flask
- NumPy
- Keras

Install dependencies:
```bash
pip install -r requirements.txt
```

> You can create `requirements.txt` from your environment:
```bash
pip freeze > requirements.txt
```

### ▶️ Running the App

```bash
python ap8.py
```

- Open your browser and navigate to: `http://127.0.0.1:5000`
- Allow webcam access
- The system will display gender prediction
- Male detections are logged into `alert_log.json`

---

## 🧠 How It Works

- The webcam feed is processed frame-by-frame using OpenCV.
- Faces are detected and cropped.
- Cropped face images are fed into a CNN model trained to classify gender.
- Detected males trigger an alert and are logged for auditing.

---

## 📈 Model Training

To retrain or fine-tune the model, use:

```bash
python train1.py
```

Ensure training datasets are structured correctly and labeled appropriately.

---

## 📸 Web UI Pages

- `index.html` – Welcome screen with webcam access
- `nextpage.html` – Dashboard interface
- `video_feed.html` – Shows real-time feed
- `script.js` – Frontend logic for navigation and interaction
- `styles.css` – Interface styling

---

## 👨‍💻 Authors

- Dhruv Upadhyay

---

## 📄 License

This project is licensed under the MIT License — feel free to use, adapt, and improve it.

---

## 🙏 Acknowledgements

- TensorFlow & Keras for deep learning tools
- OpenCV for real-time computer vision
- Flask for lightweight web serving
