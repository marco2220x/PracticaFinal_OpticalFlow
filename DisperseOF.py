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
        # Segmentación

    # Convertir imagen resultante a escala de grises
    segmented_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Umbralizar
    _, thresholded = cv2.threshold(segmented_img, 80, 255, cv2.THRESH_BINARY_INV)

    # Filtrar ruido utilizando operaciones morfológicas
    kernel = np.ones((8, 8), np.uint8)
    thresholded = cv2.morphologyEx(thresholded, cv2.MORPH_OPEN, kernel)
    segmentation = cv2.morphologyEx(thresholded, cv2.MORPH_CLOSE, kernel)

    cv2.imshow("Segmentacion", segmentation)

    # Encontrar contornos
    contours, _ = cv2.findContours(segmentation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Almacenar detecciones que superan un área mínima
    detecciones = []
    for cont in contours:
        area = cv2.contourArea(cont)
        if area > 1000:
            x, y, ancho, alto = cv2.boundingRect(cont)
            cv2.rectangle(frame, (x, y), (x + ancho, y + alto), (255, 255, 0), 3)
            detecciones.append([x, y, ancho, alto])

    # Verificar si hay suficientes detecciones para formar clusters
    if len(detecciones) >= 2:
        detections_array = np.array(detecciones)
        detections_list.append(detections_array)

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
            print(f"Optimal k: {optimal_k}")

            # Aplicar K-medias con el valor óptimo de k
            kmeans = KMeans(n_clusters=optimal_k, random_state=0, n_init=10).fit(detections_array)

            # Pintar cada rectángulo perteneciente a cada cluster
            for i, (x, y, ancho, alto) in enumerate(detecciones):
                cluster_label = kmeans.predict([[x, y, ancho, alto]])[0]
                color = (255, 0, 0) if cluster_label == 0 else (0, 255, 0) if cluster_label == 1 else (0, 0, 255)
                cv2.rectangle(frame, (x, y), (x + ancho, y + alto), color, 3)

            # Visualizar los resultados con clusters coloreados
            cv2.imshow('Zone (Clustered)', frame)

    k = cv2.waitKey(25) & 0xFF
    if k == 27:
        break
    if k == ord("c"):
        mask = np.zeros_like(old_frame)

    # Actualizar el frame anterior y los puntos anteriores
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

# Aplicar K-medias y evaluar con el puntaje de la silueta
detections_array = np.vstack(detections_list)
silhouette_scores = []

# Probar diferentes valores de k
for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10).fit(detections_array)
    silhouette_scores.append(silhouette_score(detections_array, kmeans.labels_))

# Visualizar el método del codo
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Número de clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Método del Codo para Determinar k')
plt.show()

# Liberar los recursos de la captura de video
cap.release()
cv2.destroyAllWindows()