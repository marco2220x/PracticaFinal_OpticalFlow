import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Ruta del video
video_path = 'videos/Video.mp4'
# Leer el video
cap = cv2.VideoCapture(video_path)
# Leer el primer frame
ret, frame = cap.read()
zone = frame[250:900, 100:1000]

# Crear HSV y hacer que el valor sea una constante
hsv = np.zeros_like(zone)
hsv[..., 1] = 255

# Preprocesamiento para el método exacto
zone = cv2.cvtColor(zone, cv2.COLOR_BGR2GRAY)

# Inicializar el detector de fondo
deteccion = cv2.createBackgroundSubtractorMOG2(history=10000, varThreshold=12)

# Lista para almacenar las detecciones en cada frame
detections_list = []

while True:
    # Leer el frame
    ret, new_frame = cap.read()
    if not ret:
        break
    
    # Seleccionar una zona
    new_zone = new_frame[250:900, 100:1000]

    cv2.imshow('Zona', new_zone)

    # Preprocesamiento para el método exacto
    new_zone_gray = cv2.cvtColor(new_zone, cv2.COLOR_BGR2GRAY)

    # Calcular el flujo óptico
    flow = cv2.calcOpticalFlowFarneback(zone, new_zone_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Codificación: convierte la salida del algoritmo en coordenadas polares
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Utilizar tono y saturación para codificar el flujo óptico
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    # Convertir la imagen HSV a BGR 
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('Optical Flow', bgr)
