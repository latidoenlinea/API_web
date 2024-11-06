from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from scipy.signal import butter, lfilter, find_peaks
import time

app = Flask(__name__)
CORS(app)

# Variables globales para almacenar los datos de la señal
signal_data = []
timestamps = []

# Parámetros de filtrado
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut=0.75, highcut=3, fs=30, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Procesar la señal y calcular BPM
def calculate_bpm():
    global signal_data, timestamps

    # Filtrar la señal
    filtered_signal = bandpass_filter(signal_data)

    # Encontrar picos
    peaks, _ = find_peaks(filtered_signal, distance=30)
    peak_intervals = np.diff([timestamps[i] for i in peaks])
    
    # Calcular BPM
    if len(peak_intervals) > 0:
        avg_interval = np.mean(peak_intervals)
        bpm = 60 / avg_interval
        return bpm
    return None

@app.route('/process_video', methods=['POST'])
def process_frame():
    global signal_data, timestamps

    # Obtener el frame enviado
    file = request.files['frame']
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)

    # Convertir a espacio de color YCrCb
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y_channel = ycrcb_img[:, :, 0]

    # Definir la región de interés (ROI) en la frente
    height, width = y_channel.shape
    roi = y_channel[height // 5:height // 3, width // 3:2 * width // 3]  # Frente

    # Calcular el promedio de intensidad en la ROI
    avg_intensity = np.mean(roi)
    signal_data.append(avg_intensity)
    timestamps.append(time.time())

    # Limitar los datos a los últimos 10 segundos para análisis en tiempo real
    if len(timestamps) > 300:  # Mantener 10 segundos de datos a 30 FPS
        signal_data.pop(0)
        timestamps.pop(0)

    bpm = calculate_bpm()
    
    # Respuesta JSON
    return jsonify({'bpm': bpm if bpm else "Estimando..."}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
