import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from PIL import Image
from io import BytesIO
import pandas as pd
import os

app = Flask(__name__)
CORS(app)

global_data_buffer = []
global_timestamps = []
bpm_history_forehead = []

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y

def smooth_bpm(bpm_values, window_size=10):
    if len(bpm_values) < window_size:
        return np.mean(bpm_values)
    return np.mean(bpm_values[-window_size:])

@app.route('/process_video', methods=['POST'])
def process_video():
    global global_data_buffer
    global global_timestamps
    global bpm_history_forehead

    fps = 30

    file = request.files['frame'].read()
    img = Image.open(BytesIO(file)).convert('RGB')
    frame = np.array(img)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({'bpm': 'No se detecta el rostro'})

    for (x, y, w, h) in faces:
        forehead_roi = frame[y:y + h // 7, x + w // 4:x + 3 * w // 4]

        if forehead_roi.size == 0:
            continue

        forehead_roi = cv2.GaussianBlur(forehead_roi, (5, 5), 0)
        mean_forehead = np.mean(forehead_roi[:, :, 1])
        global_data_buffer.append(mean_forehead)
        global_timestamps.append(len(global_timestamps) / fps)

        if len(global_data_buffer) > fps * 15:
            global_data_buffer = global_data_buffer[-fps * 15:]
            global_timestamps = global_timestamps[-fps * 15:]

            filtered_forehead = butter_bandpass_filter(global_data_buffer, 0.8333, 2, fps, order=5)
            fft_forehead = np.fft.rfft(filtered_forehead)
            freqs = np.fft.rfftfreq(len(filtered_forehead), 1.0 / fps)
            peak_freq_forehead = freqs[np.argmax(np.abs(fft_forehead))]
            bpm_forehead = peak_freq_forehead * 60.0

            bpm_history_forehead.append(bpm_forehead)
            smoothed_bpm_forehead = smooth_bpm(bpm_history_forehead)

            return jsonify({'bpm': int(smoothed_bpm_forehead), 'histogram': normalize_histogram(cv2.calcHist([frame], [0], None, [256], [0, 256]).flatten())})

    return jsonify({'bpm': 'Estimando...'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
