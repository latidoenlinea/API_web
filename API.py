from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from PIL import Image
from io import BytesIO
import pandas as pd
import os
import threading
from collections import deque
import time

app = Flask(name)
CORS(app, resources={r"/": {"origins": ""}})

# Thread-safe data storage using session-based approach
class SessionData:
    def init(self):
        self.data_buffer = deque(maxlen=450)  # 15 seconds at 30 fps
        self.timestamps = deque(maxlen=450)
        self.bpm_history = deque(maxlen=10)   # Keep last 10 BPM readings
        self.start_time = time.time()
        self.lock = threading.Lock()

    def add_data(self, value):
        with self.lock:
            current_time = time.time() - self.start_time
            self.data_buffer.append(value)
            self.timestamps.append(current_time)

    def get_data(self):
        with self.lock:
            return list(self.data_buffer), list(self.timestamps)

    def add_bpm(self, bpm):
        with self.lock:
            self.bpm_history.append(bpm)

    def get_bpm_history(self):
        with self.lock:
            return list(self.bpm_history)

    def has_enough_data(self, fps=30, min_seconds=15):
        return len(self.data_buffer) >= fps * min_seconds
# Global session storage (in production, use Redis or database)
sessions = {}
# Load the Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Butterworth bandpass filter design
def butter_bandpass(lowcut, highcut, fs, order=5):
    """Design Butterworth bandpass filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """Apply Butterworth bandpass filter"""
    if len(data) < order * 3:  # Minimum data points for filter
        return data

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y
def weighted_smooth_bpm(bpm_values, weights=None):
    """Smoothing function for BPM values using a weighted average"""
    if not bpm_values:
        return None

    if weights is None:
        # Recent values have higher weights
        weights = np.linspace(1, 2, len(bpm_values))

    weights = np.array(weights[-len(bpm_values):])
    return np.average(bpm_values, weights=weights)
def validate_bpm(bpm):
    """Validate that BPM is in physiological range"""
    if 30 <= bpm <= 200:
        return bpm
    return None
def validate_image_file(file_content):
    """Validate that the file is a valid image"""
    try:
        img = Image.open(BytesIO(file_content))
        img.verify()
        return True
    except Exception:
        return False
def extract_forehead_roi(frame, x, y, w, h):
    """Extract forehead ROI with better dimensioning"""
    # Make ROI larger and safer
    forehead_height = max(h // 5, 15)  # Minimum 15 pixels
    forehead_width = max(w // 2, 30)   # Minimum 30 pixels

    roi_y = y + h // 10  # Start a bit lower
    roi_x = x + w // 4

    # Ensure boundaries
    roi_y = max(0, roi_y)
    roi_x = max(0, roi_x)
    roi_h = min(forehead_height, frame.shape[0] - roi_y)
    roi_w = min(forehead_width, frame.shape[1] - roi_x)

    if roi_h <= 0 or roi_w <= 0:
        return None

    return frame[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
def normalize_histogram(hist):
    """Normalize histogram safely avoiding division by zero"""
    if hist is None or len(hist) == 0:
        return []

    min_val = np.min(hist)
    max_val = np.max(hist)

    # Avoid division by zero
    if max_val == min_val:
        return [0] * len(hist)

    normalized_hist = np.round((hist - min_val) / (max_val - min_val) * 255).astype(int)
    return normalized_hist.tolist()
def get_session_data(session_id):
    """Get or create session data"""
    if session_id not in sessions:
        sessions[session_id] = SessionData()
    return sessions[session_id]
@app.route('/process_video', methods=['POST'])
def process_video():
    fps = 30
    session_id = request.form.get('session_id', 'default')

    # Check that an image file was sent
    if 'frame' not in request.files:
        return jsonify({'error': 'No se recibió el archivo de imagen'}), 400
    try:
        # Read and validate image file
        file = request.files['frame']
        file_content = file.read()

        # Validate image
        if not validate_image_file(file_content):
            return jsonify({'error': 'Archivo no es una imagen válida'}), 400

        # Load and process image
        img = Image.open(BytesIO(file_content)).convert('RGB')
        frame = np.array(img)

        # Convert to grayscale for face detection (correct RGB to GRAY conversion)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0:
            return jsonify({'bpm': 'No se detecta el rostro'})
        # Process only the first (largest) face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]

        # Extract forehead ROI
        forehead_roi = extract_forehead_roi(frame, x, y, w, h)

        if forehead_roi is None or forehead_roi.size == 0:
            return jsonify({'bpm': 'ROI de frente muy pequeña'})
        # Blur and calculate mean green channel value (RGB format)
        forehead_roi = cv2.GaussianBlur(forehead_roi, (5, 5), 0)
        mean_forehead = np.mean(forehead_roi[:, :, 1])  # Green channel in RGB

        # Get session data
        session_data = get_session_data(session_id)

        # Add data to session buffer
        session_data.add_data(mean_forehead)
        # Process data if buffer has enough data
        if session_data.has_enough_data(fps, 15):
            data_buffer, timestamps = session_data.get_data()

            # Apply bandpass filter
            try:
                filtered_data = butter_bandpass_filter(
                    data_buffer, 0.8333, 2, fps, order=6
                )
            except Exception as e:
                return jsonify({'error': f'Error en filtrado: {str(e)}'})
            # Perform FFT on the filtered signal
            fft_result = np.fft.rfft(filtered_data)
            freqs = np.fft.rfftfreq(len(filtered_data), 1.0 / fps)

            # Find peak frequency (ignore DC component)
            valid_indices = freqs > 0
            valid_freqs = freqs[valid_indices]
            valid_fft = np.abs(fft_result[valid_indices])

            if len(valid_fft) == 0:
                return jsonify({'bpm': 'Error en análisis de frecuencia'})

            peak_freq = valid_freqs[np.argmax(valid_fft)]

            # Calculate BPM
            bpm_raw = peak_freq * 60.0

            # Validate BPM
            bpm_validated = validate_bpm(bpm_raw)

            if bpm_validated is None:
                return jsonify({'bpm': 'BPM fuera de rango fisiológico'})

            # Add to history and smooth
            session_data.add_bpm(bpm_validated)
            bpm_history = session_data.get_bpm_history()

            # Calculate smoothed BPM
            smoothed_bpm = weighted_smooth_bpm(bpm_history)

            # Calculate histogram safely
            try:
                hist = cv2.calcHist([frame], [0], None, [256], [0, 256]).flatten()
                normalized_hist = normalize_histogram(hist)
            except Exception:
                normalized_hist = []
            return jsonify({
                'bpm': round(smoothed_bpm, 2),
                'raw_bpm': round(bpm_raw, 2),
                'samples': len(data_buffer),
                'histogram': normalized_hist,
                'status': 'success'
            })
        else:
            # Still collecting data
            data_buffer, * = session*data.get_data()
            progress = len(data_buffer) / (fps * 15) * 100

            return jsonify({
                'bpm': 'Estimando...',
                'progress': round(progress, 1),
                'samples': len(data_buffer),
                'required_samples': fps * 15,
                'status': 'collecting'
            })
    except Exception as e:
        return jsonify({'error': f'Error interno: {str(e)}'}, 500)
@app.route('/reset_session', methods=['POST'])
def reset_session():
    """Reset session data"""
    session_id = request.form.get('session_id', 'default')

    if session_id in sessions:
        del sessions[session_id]

    return jsonify({'status': 'Session reset successfully'})
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'active_sessions': len(sessions),
        'timestamp': time.time()
    })
# Clean up old sessions periodically
def cleanup_old_sessions():
    """Remove sessions older than 1 hour"""
    current_time = time.time()
    to_remove = []

    for session_id, session_data in sessions.items():
        if current_time - session_data.start_time > 3600:  # 1 hour
            to_remove.append(session_id)

    for session_id in to_remove:
        del sessions[session_id]
# Set up periodic cleanup (in production, use a proper scheduler)
import atexit
cleanup_timer = None
def start_cleanup_timer():
    global cleanup_timer
    cleanup_old_sessions()
    cleanup_timer = threading.Timer(300.0, start_cleanup_timer)  # Every 5 minutes
    cleanup_timer.daemon = True
    cleanup_timer.start()
if name == '__main__':
    start_cleanup_timer()
    atexit.register(lambda: cleanup_timer and cleanup_timer.cancel())
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
