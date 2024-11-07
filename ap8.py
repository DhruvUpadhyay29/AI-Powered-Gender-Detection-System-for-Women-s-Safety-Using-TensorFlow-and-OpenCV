import os
import datetime
from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cvlib as cv
import json

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained models for gender detection and SOS gesture recognition
gender_model = load_model('gender_detection_head.h5')
gesture_model = load_model('sos_gesture_model.h5')  # Load your SOS gesture model
classes = ['MALE', 'FEMALE']
gesture_classes = ['NOT_SOS', 'SOS']  # Assuming binary classification for SOS gesture

# File to store past alerts
ALERT_LOG_FILE = 'alert_log.json'

# Function to log alerts
def log_alert(alert_type, location):
    alert_data = {
        "alert_type": alert_type,
        "location": location,
        "timestamp": datetime.datetime.now().isoformat()
    }
    if os.path.exists(ALERT_LOG_FILE):
        with open(ALERT_LOG_FILE, 'r+') as file:
            data = json.load(file)
            data.append(alert_data)
            file.seek(0)
            json.dump(data, file, indent=4)
    else:
        with open(ALERT_LOG_FILE, 'w') as file:
            json.dump([alert_data], file, indent=4)

# Function to identify hotspots based on past alerts
def identify_hotspots():
    if os.path.exists(ALERT_LOG_FILE):
        with open(ALERT_LOG_FILE, 'r') as file:
            data = json.load(file)
            # Analyze the data to identify frequent locations (basic example)
            location_count = {}
            for alert in data:
                location = alert['location']
                if location in location_count:
                    location_count[location] += 1
                else:
                    location_count[location] = 1
            # Return locations with high alert frequency
            hotspots = [loc for loc, count in location_count.items() if count >= 3]  # Example threshold
            return hotspots
    return []

# Function to detect SOS gesture
def detect_sos_gesture(frame):
    # Preprocess the frame for the SOS gesture recognition model
    gesture_crop = cv2.resize(frame, (224,224))  # Assuming the model expects 64x64 input size
    gesture_crop = gesture_crop.astype("float") / 255.0
    gesture_crop = img_to_array(gesture_crop)
    gesture_crop = np.expand_dims(gesture_crop, axis=0)

    # Predict using the gesture model
    conf = gesture_model.predict(gesture_crop)[0]
    idx = np.argmax(conf)
    label = gesture_classes[idx]

    # Return True if the SOS gesture is detected
    return label == 'SOS'

# Video capture
camera = cv2.VideoCapture(0)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Detect faces in the frame
            faces, confidence = cv.detect_face(frame)

            # Initialize counters for men and women
            men_count = 0
            women_count = 0
            lone_woman_alert = False
            surrounded_alert = False
            sos_alert = False

            # Store coordinates of detected men and women
            women_coords = []
            men_coords = []

            # Loop through detected faces
            for idx, f in enumerate(faces):
                (startX, startY) = f[0], f[1]
                (endX, endY) = f[2], f[3]

                # Expand the bounding box to approximate the entire head
                padding = int(0.5 * (endY - startY))
                startX = max(0, startX - padding)
                startY = max(0, startY - padding)
                endX = min(frame.shape[1], endX + padding)
                endY = min(frame.shape[0], endY + padding)

                # Draw rectangle over the head region
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)

                # Crop the detected head region
                head_crop = np.copy(frame[startY:endY, startX:endX])

                if head_crop.shape[0] < 10 or head_crop.shape[1] < 10:
                    continue

                # Preprocess for gender detection model
                head_crop = cv2.resize(head_crop, (224, 224))
                head_crop = head_crop.astype("float") / 255.0
                head_crop = img_to_array(head_crop)
                head_crop = np.expand_dims(head_crop, axis=0)

                # Predict gender
                conf = gender_model.predict(head_crop)[0]
                idx = np.argmax(conf)
                label = classes[idx]

                # Update gender count and store coordinates
                if label == 'MALE':
                    men_count += 1
                    men_coords.append((startX, startY, endX, endY))
                else:
                    women_count += 1
                    women_coords.append((startX, startY, endX, endY))

                Y = startY - 20 if startY - 20 > 20 else startY + 20

                # Display the label in larger font above the head rectangle
                cv2.putText(frame, label, (startX, Y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

            # Lone woman detection
            if women_count == 1 and men_count == 0:
                lone_woman_alert = True
                log_alert('Lone Woman Detected', 'TUMKUR')  # Example location

            # Woman surrounded by three or more men detection
            if women_count == 1 and men_count >= 2:
                surrounded_alert = True
                log_alert('Woman Surrounded by 2+ Men', 'TUMKUR')  # Example location

            # SOS gesture detection
            sos_alert = detect_sos_gesture(frame)
            if sos_alert:
                log_alert('SOS Gesture Detected', 'TUMKUR')  # Example location

            # Display the count of men and women on the screen
            count_text = f"Men: {men_count}  Women: {women_count}"
            cv2.putText(frame, count_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 3)

            # Display alerts
            if lone_woman_alert:
                cv2.putText(frame, "ALERT: Lone Woman Detected!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            if surrounded_alert:
                cv2.putText(frame, "ALERT: Woman Surrounded by 2+ Men!", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            if sos_alert:
                cv2.putText(frame, "ALERT: SOS Gesture Detected!", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            # Encode the frame in JPEG format
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            # Yield the frame in byte format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/nextpage')
def next_page():
    hotspots = identify_hotspots()
    return render_template('nextpage.html', hotspots=hotspots)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)

