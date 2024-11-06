from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import time
import os

CORS(app)
app = Flask(__name__)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

if face_cascade.empty():
    print("Error al cargar el clasificador de rostro.")

pulses = []
pulse_count = 0
last_bpm_time = time.time()  # Para controlar el tiempo entre cálculos

def calculate_bpm(pulses):
    if len(pulses) < 2:
        return None
    intervals = np.diff(pulses)
    bpm = 60 / np.mean(intervals) if len(intervals) > 0 else None
    return bpm

@app.route('/process_video', methods=['POST'])
def process_video():
    global pulse_count, last_bpm_time
    if request.method == 'POST':
        image_file = request.files.get('image')
        if image_file:
            try:
                image = np.frombuffer(image_file.read(), np.uint8)
                img = cv2.imdecode(image, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (320, 240))  # Cambiar resolución

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

                if len(faces) == 0:
                    return jsonify({"bpm_forehead": None, "mensaje": "No se detectó rostro."})

                for (x, y, w, h) in faces:
                    roi = img[y:y + h, x:x + w]
                    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                    mean_val = np.mean(roi_gray)

                    if mean_val < 70:
                        pulses.append(time.time())
                        pulse_count += 1

                    # Calcular BPM cada segundo
                    current_time = time.time()
                    if current_time - last_bpm_time >= 1:  # Cada segundo
                        bpm = calculate_bpm(pulses)
                        if bpm is not None:
                            last_bpm_time = current_time
                            pulses.clear()
                            pulse_count = 0
                            return jsonify({"bpm": bpm, "mensaje": "Ritmo cardíaco medido."})
                        else:
                            return jsonify({"bpm": "No se pudo calcular BPM."})

                return jsonify({"bpm": None, "mensaje": "Ritmo cardíaco no medido."})

            except Exception as e:
                print(f"Error al procesar la imagen: {str(e)}")
                return jsonify({'error': 'Error en el procesamiento'}), 500

        return jsonify({"bpm": "Error al procesar la imagen."})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
