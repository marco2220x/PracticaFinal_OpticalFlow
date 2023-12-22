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

    # Segmentación

    # Convertir frame actual a escala de grises
    frame_actual_gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Umbralizar
    _, thresholded = cv2.threshold(frame_actual_gray, 30, 255, cv2.THRESH_BINARY)

    # Eliminar ruido
    kernel = np.ones((8, 8), np.uint8)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    segmentation = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    cv2.imshow('Segmentacion', segmentation)

    # Encontrar contornos
    contours, _ = cv2.findContours(segmentation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Almacenar detecciones que superan un área mínima
    detecciones = []
    for cont in contours:
        area = cv2.contourArea(cont)
        if area > 6000:
            x, y, ancho, alto = cv2.boundingRect(cont)
            detecciones.append([x, y, ancho, alto])

    # Verificar si hay suficientes detecciones para formar clusters
    if len(detecciones) >= 4:
        detections_array = np.array(detecciones)
        detections_list.append(detecciones)

        silhouette_scores = []
        max_k = min(11, len(detections_array) - 1)
        for k in range(2, max_k):
            kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(detections_array)
            try:
                silhouette_scores.append(silhouette_score(detections_array, kmeans.labels_))
            except ValueError:
                print(f"Skipping k={k} due to ValueError in silhouette_score")
                continue

        if silhouette_scores:
            # Seleccionar el valor óptimo de k
            optimal_k = np.argmax(silhouette_scores) + 2  # Sumar 2 ya que empezamos desde k=2
            #print(f"Optimal k: {optimal_k}")

            # Aplicar K-medias con el valor óptimo de k
            kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init=10).fit(detections_array)

            # Pintar cada rectángulo perteneciente a cada cluster
            for i, (x, y, ancho, alto) in enumerate(detecciones):
                cluster_label = kmeans.predict([[x, y, ancho, alto]])[0]
                color = (255, 0, 0) if cluster_label == 0 else (0, 255, 0) if cluster_label == 1 else (0, 0, 255)
                cv2.rectangle(new_zone, (x, y), (x + ancho, y + alto), color, 3)

            # Visualizar los resultados con clusters coloreados
            cv2.imshow('Zone (Clustered)', new_zone)

    zone = new_zone_gray

    key = cv2.waitKey(5)
    if key == 27:
        break

# Aplicar K-medias y evaluar con el puntaje de la silueta
detections_array = np.vstack(detections_list)
silhouette_scores = []

# Prueba diferentes valores de k
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(detections_array)
    silhouette_scores.append(silhouette_score(detections_array, kmeans.labels_))

# Visualizar el método del codo
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Método del Codo para Determinar k')
plt.show()

cv2.destroyAllWindows()
