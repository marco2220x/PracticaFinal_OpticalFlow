import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Ruta del video
video_path = 'Videos/Video.mp4'

# Inicializar la captura de video
cap = cv2.VideoCapture(video_path)

# Parámetros para la detección de esquinas Shi-Tomasi
feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)

# Parámetros para el flujo óptico de Lucas-Kanade
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)

# Crear algunos colores aleatorios
color = np.random.randint(0, 255, (100, 3))

# Tomar el primer frame y encontrar esquinas en él
ret, old_frame = cap.read()
old_frame = old_frame[250:900, 100:1000]
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Crear una máscara para dibujar
mask = np.zeros_like(old_frame)

# Lista para almacenar las detecciones en cada frame
detections_list = []

while True:
    # Leer el siguiente frame del video
    ret, frame = cap.read()
    if not ret:
        break

    # Seleccionar una zona de interés
    frame = frame[250:900, 100:1000]
    cv2.imshow('Zona', frame)

    # Convertir a escala de grises
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcular el flujo óptico de Lucas-Kanade
    flow = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, winSize=(15, 15), maxLevel=2,
                                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Seleccionar puntos buenos
    good_new = flow[0]
    good_old = p0

    # Crear una máscara binaria para resaltar el movimiento
    mask_bin = np.zeros_like(frame)

    # Dibujar las trayectorias
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)
        mask = cv2.line(mask, (a, b), (c, d), (0, 255, 0), 2)
        frame = cv2.circle(frame, (a, b), 5, (0, 255, 0), -1)

    # Combinar la imagen original con la máscara
    img = cv2.add(frame, mask)
    cv2.imshow("Optcial Flow", img)