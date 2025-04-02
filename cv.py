import cv2
import numpy as np
import math
from scipy.spatial import distance
from tkinter import Tk
from tkinter.filedialog import askopenfilename

# Abrir un cuadro de diálogo para seleccionar la imagen
Tk().withdraw()  # Ocultar la ventana principal de Tkinter
image_path = askopenfilename(title="Selecciona una imagen", filetypes=[("Imagenes", "*.png;*.jpg;*.jpeg")])
if not image_path:
    raise Exception("No se seleccionó ninguna imagen.")

# Cargar la imagen
image = cv2.imread(image_path)
if image is None:
    raise Exception("Error al cargar la imagen. Verifica la ruta y el nombre del archivo.")

# Obtener dimensiones de la imagen
height, width, _ = image.shape

# Convertir a HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Definir rangos para detección
green_lower = np.array([40, 100, 50])
green_upper = np.array([80, 255, 255])
orange_lower = np.array([5, 100, 100])
orange_upper = np.array([20, 255, 255])

# Crear máscaras
green_mask = cv2.inRange(hsv, green_lower, green_upper)
orange_mask = cv2.inRange(hsv, orange_lower, orange_upper)

# Aplicar un blur gaussiano a la máscara naranja para homologar contornos cercanos
orange_mask = cv2.GaussianBlur(orange_mask, (7, 7), 0)
kernel = np.ones((5, 5), np.uint8)
orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)

# --- DETECCIÓN DEL PUNTO VERDE MÁS A LA IZQUIERDA ---
green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if len(green_contours) == 0:
    raise Exception("No se encontró ningún contorno verde.")

leftmost_green_point = None
min_x = float('inf')
for c in green_contours:
    M = cv2.moments(c)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        # Escoger el contorno cuyo centro esté más a la izquierda (menor X)
        if cx < min_x:
            min_x = cx
            leftmost_green_point = (cx, cy)
if leftmost_green_point is None:
    raise Exception("No se encontró un punto verde con momento válido.")

# --- DETECCIÓN DE PUNTOS NARANJAS ---
orange_contours, _ = cv2.findContours(orange_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
orange_points = []
for c in orange_contours:
    M = cv2.moments(c)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        orange_points.append((cx, cy))
if len(orange_points) < 4:
    raise Exception("No hay suficientes puntos naranjas para formar un cuadrilátero.")

# --- DETECCIÓN DEL PUNTO NARANJA ABAJO A LA DERECHA ---
# Se define como el punto con mayor Y y, en caso de empate, con mayor X.
bottom_right_orange_point = max(orange_points, key=lambda p: (p[1], p[0]))

# --- SELECCIONAR LOS 4 PUNTOS NARANJAS MÁS CERCANOS AL PUNTO VERDE ---
dists = [(p, distance.euclidean(p, leftmost_green_point)) for p in orange_points]
closest_4 = sorted(dists, key=lambda x: x[1])[:4]
closest_points = [p for p, _ in closest_4]

# Ordenar los 4 puntos para formar un polígono cerrado de forma coherente
cx_mean = sum([p[0] for p in closest_points]) / 4.0
cy_mean = sum([p[1] for p in closest_points]) / 4.0

def angle_from_center(point):
    return math.atan2(point[1] - cy_mean, point[0] - cx_mean)

closest_points = sorted(closest_points, key=angle_from_center)

# --- DIBUJAR EN LA IMAGEN ---
# 1) Dibujar el punto verde más a la izquierda
cv2.circle(image, leftmost_green_point, 5, (0, 255, 0), -1)
cv2.putText(image, "Green Leftmost", (leftmost_green_point[0] + 10, leftmost_green_point[1]),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 2) Dibujar los 4 puntos naranjas más cercanos al verde
for i, pt in enumerate(closest_points):
    cv2.circle(image, pt, 5, (0, 165, 255), -1)
    cv2.putText(image, f"P{i+1}", (pt[0] + 10, pt[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)

# 3) Dibujar el polígono (cuadrilátero) uniendo los 4 puntos
for i in range(4):
    p1 = closest_points[i]
    p2 = closest_points[(i+1) % 4]
    cv2.line(image, p1, p2, (255, 0, 0), 2)

# 4) Dibujar el punto naranja abajo a la derecha (origen del robot)
cv2.circle(image, bottom_right_orange_point, 5, (255, 255, 255), -1)
cv2.putText(image, "RobotOrigin (0,0)", (bottom_right_orange_point[0] - 10, bottom_right_orange_point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# Mostrar y guardar la imagen resultante
cv2.imshow("Detección", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("resultado_cuadrado.png", image)
print("Se guardó la imagen con el cuadrilátero en 'resultado_cuadrado.png'.")

# --- GUARDAR LAS COORDENADAS EN UN ARCHIVO TXT ---
# Se guardan:
# 1. GreenLeftmost, 2. RobotOrigin (0,0), y 3. Los 4 puntos naranjas más cercanos al punto verde.
coordinates_to_save = [
    ("GreenLeftmost", leftmost_green_point),
    ("RobotOrigin (0,0)", bottom_right_orange_point)
] + [(f"P{i+1}", closest_points[i]) for i in range(4)]

with open("coords.txt", "w") as f:
    for label, (x, y) in coordinates_to_save:
        f.write(f"{label}: ({x}, {y})\n")
print("Las coordenadas se han guardado en 'coords.txt'.")

# --- NORMALIZAR LAS COORDENADAS (0 a 1) Y GUARDAR SOLO LOS VALORES ---
normalized_coordinates = []
for label, (x, y) in coordinates_to_save:
    norm_x = x / width
    norm_y = y / height
    normalized_coordinates.append((norm_x, norm_y))

with open("coords_normalized.txt", "w") as f:
    for (nx, ny) in normalized_coordinates:
        f.write(f"({nx:.4f}, {ny:.4f})\n")
print("Las coordenadas normalizadas se han guardado en 'coords_normalized.txt'.")
