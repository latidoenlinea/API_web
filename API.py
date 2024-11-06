from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
from scipy.signal import butter, filtfilt
from PIL import Image
from io import BytesIO
import pandas as pd

app = Flask(__name__)
CORS(app)

# Variables globales
global_data_buffer = []
global_timestamps = []
bpm_history = []

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

def smooth_bpm(bpm_values, window_size=10):
    if len(bpm_values) < window_size:
        return np.mean(bpm_values)
    smoothed_bpm = np.mean(bpm_values[-window_size:])
    return smoothed_bpm

def normalize_histogram(hist):
    min_val = np.min(hist)
    max_val = np.max(hist)
    normalized_hist = np.round((hist - min_val) / (max_val - min_val) * 255).astype(int)
    return normalized_hist.tolist()

@app.route('/process_video', methods=['POST'])
def process_video():
    global global_data_buffer
    global global_timestamps
    global bpm_history

    fps = 30  # Cuadros por segundo

    # Lee el archivo de imagen enviado
    file = request.files['frame'].read()
    img = Image.open(BytesIO(file)).convert('RGB')
    frame = np.array(img)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return jsonify({'bpm': 'No se detectaron rostros en la imagen'})

    for (x, y, w, h) in faces:
        # Calcula la región de la frente
        forehead_roi = frame[y:y + h // 7, x + w // 4:x + 3 * w // 4]

        # Calcula las regiones de los pómulos
        cheek_left_roi = frame[y + h // 3:y + h // 2, x + w // 8:x + w // 2]
        cheek_right_roi = frame[y + h // 3:y + h // 2, x + w // 2:x + 7 * w // 8]

        if forehead_roi.size == 0 or cheek_left_roi.size == 0 or cheek_right_roi.size == 0:
            continue

        # Promedia los valores de las tres regiones
        mean_forehead = np.mean(forehead_roi[:, :, 1])
        mean_cheek_left = np.mean(cheek_left_roi[:, :, 1])
        mean_cheek_right = np.mean(cheek_right_roi[:, :, 1])

        mean_combined = np.mean([mean_forehead, mean_cheek_left, mean_cheek_right])

        global_data_buffer.append(mean_combined)
        global_timestamps.append(len(global_timestamps) / fps)

        print(f"\n Global Data Buffer: {len(global_data_buffer)}")

        if len(global_data_buffer) > fps * 15:
            global_data_buffer = global_data_buffer[-fps * 15:]
            global_timestamps = global_timestamps[-fps * 15:]

            # Filtro pasa banda para aislar frecuencias relevantes
            filtered = butter_bandpass_filter(global_data_buffer, 0.8333, 2, fps, order=4)

            fft = np.fft.rfft(filtered)
            freqs = np.fft.rfftfreq(len(filtered), 1.0 / fps)

            peak_freq = freqs[np.argmax(np.abs(fft))]
            bpm = peak_freq * 60.0

            # Agregar BPM al historial y suavizar
            bpm_history.append(bpm)
            smoothed_bpm = smooth_bpm(bpm_history)

            # Calcula el histograma de la imagen
            hist = cv2.calcHist([frame], [0], None, [256], [0, 256]).flatten()  # Histograma original
            normalized_hist = normalize_histogram(hist)  # Normaliza el histograma

            if 60 <= smoothed_bpm <= 120:
                return jsonify({'bpm': int(smoothed_bpm) + 8, 'histogram': normalized_hist})

    if bpm_history:
        return jsonify({'bpm': int(bpm_history[-1]) + 8})
    else:
        return jsonify({'bpm': 'Estimando...'})

# Nueva ruta para guardar datos en un archivo Excel
@app.route('/guardar_datos_excel', methods=['POST'])
def guardar_datos_excel():
    data = request.json

    # Crear un DataFrame a partir de los datos recibidos
    df = pd.DataFrame(data)

    # Guardar el archivo Excel en la misma carpeta
    file_path = 'datos_bpm.xlsx'
    df.to_excel(file_path, index=False)

    # Enviar el archivo Excel de vuelta al cliente
    return send_file(file_path, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
