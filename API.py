from flask import Flask, jsonify, request
from flask_cors import CORS
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from io import BytesIO
from PIL import Image
import logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
CORS(app)  # Habilita CORS para todas las rutas y orígenes

# Carga el clasificador Haar para la detección de rostros
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Diseño de filtro pasa banda de Butterworth
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

@app.route('/process_video', methods=['POST'])
def process_video():
    file = request.files['frame'].read()
    img = Image.open(BytesIO(file)).convert('RGB')
    frame = np.array(img)

    print(f"DEBUG:root:Tamaño de la imagen recibida: {frame.shape}")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    if len(faces) == 0:
        print("DEBUG:root:No se detectaron rostros en la imagen")
        return jsonify({'bpm': 'No se pudo calcular BPM, rostro no detectado'})

    data_buffer = []
    fps = 30  # Define cuadros por segundo

    for (x, y, w, h) in faces:
        forehead_roi = frame[y:y + h // 7, x + w // 4:x + 3 * w // 4]

        mean_forehead = np.mean(forehead_roi[:, :, 1])
        data_buffer.append(mean_forehead)

        if len(data_buffer) >= fps * 10:
            data_buffer = data_buffer[-fps * 10:]
            filtered = butter_bandpass_filter(data_buffer, 0.75, 3.5, fps, order=5)

            fft = np.fft.rfft(filtered)
            freqs = np.fft.rfftfreq(len(filtered), 1.0 / fps)
            peak_freq = freqs[np.argmax(np.abs(fft))]
            bpm = peak_freq * 60.0

            return jsonify({'bpm': int(bpm)})

    return jsonify({'bpm': 'No se pudo calcular BPM, rostro no detectado'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
