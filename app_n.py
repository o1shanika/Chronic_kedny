from flask import Flask, jsonify, render_template, Response
import cv2
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

# Initialize pedestrian detection model
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Simple ML model for predicting crossing time
def estimate_crossing_time(num_pedestrians):
    base_time = 5  # Base time in seconds
    additional_time_per_person = 2  # Additional seconds per pedestrian
    return base_time + num_pedestrians * additional_time_per_person

# Webcam video stream
def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect pedestrians
        pedestrians, _ = hog.detectMultiScale(frame, winStride=(8, 8), padding=(16, 16), scale=1.05)

        # Draw rectangles around pedestrians
        for (x, y, w, h) in pedestrians:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Encode the frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        # Yield the frame in bytes for video streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to display the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# API endpoint to get crossing time
@app.route('/predict_crossing_time', methods=['GET'])
def predict_crossing_time():
    # For simplicity, assume we count pedestrians directly here
    num_pedestrians = 5  # Hardcoded for demo; replace with actual detection count
    crossing_time = estimate_crossing_time(num_pedestrians)
    return jsonify({'crossing_time': crossing_time})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
