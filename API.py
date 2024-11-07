from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from PIL import Image
from io import BytesIO
import pandas as pd
import os

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables for storing data
global_data_buffer = []
global_timestamps = []
bpm_history_forehead = []

# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Butterworth bandpass filter design
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Function to apply Butterworth bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

# Smoothing function for BPM values using a weighted average
def weighted_smooth_bpm(bpm_values, weights):
    weights = np.array(weights[-len(bpm_values):])
    return np.average(bpm_values, weights=weights)

@app.route('/process_video', methods=['POST'])
def process_video():
    global global_data_buffer
    global global_timestamps
    global bpm_history_forehead

    fps = 30

    # Check that an image file was sent
    if 'frame' not in request.files:
        return jsonify({'error': 'No se recibió el archivo de imagen'}), 400

    try:
        # Read the image file
        file = request.files['frame'].read()
        img = Image.open(BytesIO(file)).convert('RGB')
        frame = np.array(img)

        # Convert to grayscale and detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return jsonify({'bpm': 'No se detecta el rostro'})

        for (x, y, w, h) in faces:
            # Define the forehead ROI
            forehead_roi = frame[y:y + h // 7, x + w // 4:x + 3 * w // 4]

            if forehead_roi.size == 0:
                continue

            # Blur and calculate the mean green channel value in the ROI
            forehead_roi = cv2.GaussianBlur(forehead_roi, (5, 5), 0)
            mean_forehead = np.mean(forehead_roi[:, :, 1])

            # Add values to buffer
            global_data_buffer.append(mean_forehead)
            global_timestamps.append(len(global_timestamps) / fps)

            # Process data if buffer has data for at least 15 seconds
            if len(global_data_buffer) > fps * 15:
                # Keep only the last 15 seconds of data
                global_data_buffer = global_data_buffer[-fps * 15:]
                global_timestamps = global_timestamps[-fps * 15:]

                # Bandpass filter the forehead data
                filtered_forehead = butter_bandpass_filter(global_data_buffer, 0.8333, 2.5, fps, order=6)

                # Perform FFT on the filtered signal
                fft_forehead = np.fft.rfft(filtered_forehead)
                freqs = np.fft.rfftfreq(len(filtered_forehead), 1.0 / fps)
                peak_freq_forehead = freqs[np.argmax(np.abs(fft_forehead))]

                # Calculate BPM
                bpm_forehead = peak_freq_forehead * 60.0
                bpm_history_forehead.append(bpm_forehead)

                # Define weights for weighted average (recent values have higher weights)
                weights = np.linspace(1, 2, len(bpm_history_forehead))
                smoothed_bpm_forehead = weighted_smooth_bpm(bpm_history_forehead, weights)

                return jsonify({
                    'bpm': round(smoothed_bpm_forehead, 2),
                    'histogram': normalize_histogram(cv2.calcHist([frame], [0], None, [256], [0, 256]).flatten())
                })

        return jsonify({'bpm': 'Estimando...'})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Normalize histogram for debugging purposes
def normalize_histogram(hist):
    min_val = np.min(hist)
    max_val = np.max(hist)
    normalized_hist = np.round((hist - min_val) / (max_val - min_val) * 255).astype(int)
    return normalized_hist.tolist()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))

