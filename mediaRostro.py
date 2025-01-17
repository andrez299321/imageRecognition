import cv2
import mediapipe as mp
import numpy as np

import torch


# Inicializar MediaPipe para la detección de rostros y puntos clave faciales
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Funciones para verificar los requisitos ICAO
def check_image_resolution(image, min_width=300, min_height=400):
    height, width = image.shape[:2]
    if width >= min_width and height >= min_height:
        return True
    else:
        return False

def check_head_orientation(landmarks):
    # Usamos los puntos de los ojos para comprobar la orientación
    left_eye = landmarks[36:42]  # Puntos del ojo izquierdo
    right_eye = landmarks[42:48]  # Puntos del ojo derecho

    left_eye_center = (left_eye[0][0] + left_eye[3][0]) / 2, (left_eye[0][1] + left_eye[3][1]) / 2
    right_eye_center = (right_eye[0][0] + right_eye[3][0]) / 2, (right_eye[0][1] + right_eye[3][1]) / 2

    delta_x = right_eye_center[0] - left_eye_center[0]
    delta_y = right_eye_center[1] - left_eye_center[1]
    angle = np.degrees(np.arctan2(delta_y, delta_x))

    # Si el ángulo de inclinación es mayor a un umbral, consideramos que la cabeza está inclinada
    if abs(angle) < 10:  # Ajusta el umbral según tus necesidades
        return True
    else:
        return False

def check_illumination(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    std_dev = np.std(hist)
    if std_dev > 30:
        return True
    else:
        return False

def check_background(image, threshold=100):
    height, width = image.shape[:2]
    top_left = image[0, 0]
    top_right = image[0, width-1]
    bottom_left = image[height-1, 0]
    bottom_right = image[height-1, width-1]
    
    corners_avg = np.mean([top_left, top_right, bottom_left, bottom_right])

    if np.mean(corners_avg) > threshold:
        return True
    else:
        return False

def check_for_glasses(rgb_image, confidence_threshold=0.5):
    model_path = r'C:\Code\python\liveness\modelos\glasses\best.pt'

    # Cargar el modelo entrenado
    #model = torch.load(model_path, map_location=torch.device('cpu'))['model'].float()
    
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=False)
    model.eval()  # Establecer el modelo en modo de evaluación



     # Realizar la predicción
    results = model(rgb_image)


    if results.xyxy[0].shape[0] > 0:  # Verifica si hay al menos una predicción
        for det in results.xyxy[0]:  # Recorre las predicciones
            x1, y1, x2, y2, conf, cls = det.tolist()  # Convierte el tensor a lista
            if conf > 0.5:  # Filtra por confianza (ajusta según sea necesario)
                # Aquí puedes aplicar lógica para verificar si el objeto es un "gafas"
                if int(cls) == 1:  # Clase 0 corresponde a personas, ajusta según la clase de gafas
                    print("Gafas detectadas")
                    return True  # Se detectaron gafas con una alta confianza
    return False  # No se detectaron gafas o la confianza es baja


def check_facial_expression(landmarks):
    mouth = landmarks[48:60]
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    
    mouth_width = np.linalg.norm(mouth[0] - mouth[6])
    eye_distance = np.linalg.norm(left_eye[0] - right_eye[3])
    
    if mouth_width > eye_distance * 1.2:
        return False
    return True
def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def draw_mouth_points(frame, landmarks):
    # Establecer el color para los puntos de la boca (por ejemplo, rojo)
    mouth_color = (10, 10, 205)  # Rojo en formato BGR
    
    # Los puntos de la boca están en el rango de 48 a 60
    for i in range(48, 60):
        # Obtener las coordenadas de los puntos de la boca
        x = int(landmarks[i].x * frame.shape[1])
        y = int(landmarks[i].y * frame.shape[0])
        
        # Dibujar un círculo en cada punto de la boca
        cv2.circle(frame, (x, y), 2, mouth_color, -1)  # Radio de 2, color rojo, relleno (-1)


# Función para verificar si la boca está abierta
def check_mouth_open(landmarks, threshold=0.03):
    # Los puntos clave de la boca están en el rango de 48-60 en FaceMesh
    # Usamos los puntos 62 (labio inferior) y 66 (labio superior) para medir la apertura
    top_lip = landmarks[62]  # Punto en la parte superior de la boca
    bottom_lip = landmarks[66]  # Punto en la parte inferior de la boca

    # Obtener las coordenadas de los puntos clave de la boca
    top_lip_coords = (top_lip.x, top_lip.y)
    bottom_lip_coords = (bottom_lip.x, bottom_lip.y)

    # Calcular la distancia entre el punto superior e inferior de la boca
    mouth_opening = euclidean_distance(top_lip_coords, bottom_lip_coords)

    # Si la distancia es mayor que el umbral, consideramos que la boca está abierta
    if mouth_opening > threshold:
        return True
    return False


# Función para dibujar el óvalo en el centro de la pantalla
def draw_center_oval(frame):
    height, width = frame.shape[:2]
    
    # Calcular el tamaño del óvalo como el 70% del ancho y la altura de la imagen
    oval_width = int(width * 0.4)
    oval_height = int(height * 0.7)
    
    # Calcular el centro de la imagen
    center = (width // 2, height // 2)

    # Dibujar el óvalo verde (color BGR, en este caso verde es (0, 255, 0))
    cv2.ellipse(frame, center, (oval_width // 2, oval_height // 2), 0, 0, 360, (0, 255, 0), 2)




# Función principal para detectar el rostro y validarlo
def validate_face(image):
    # Convertir la imagen a RGB para que MediaPipe la procese
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Obtener la detección de rostros
    results = face_mesh.process(rgb_image)

    if not results.multi_face_landmarks:
        print("No se detectó rostro.")
        return "No se detectó rostro."

    # Usamos los puntos clave de MediaPipe
    for landmarks in results.multi_face_landmarks:
        # Convertir los landmarks a una lista de coordenadas
        face_landmarks = [(int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])) for landmark in landmarks.landmark]

        # Dibujar los puntos clave en la imagen
        #mp_drawing.draw_landmarks(image, landmarks, mp_face_mesh.FACEMESH_TESSELATION)

   

        # Verificar los requisitos de ICAO para la imagen facial
        if not check_image_resolution(image):
            print("La imagen no tiene la resolución suficiente.")
            return "La imagen no tiene la resolución suficiente."
        if not check_head_orientation(face_landmarks):
            print("La cabeza no está orientada correctamente.")
            return "La cabeza no está orientada correctamente."
        if not check_illumination(image):
            print("La iluminación no es adecuada.")
            return "La iluminación no es adecuada."
        if check_for_glasses(rgb_image):
            print("tiene gafas.")
            return "tiene gafas."

        # Si todo pasa, se considera que la imagen facial es válida
        print("Rostro válido para los requisitos de ICAO.")
        return "OK"

# Captura de video desde la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Llamar a la función que valida la imagen facial
    valid = validate_face(frame)

    # Dibujar el óvalo verde en el centro de la imagen
    draw_center_oval(frame)

    # Mostrar la imagen con los resultados
    if valid == "OK":
        cv2.putText(frame, "Rostro Válido", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, valid, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    cv2.imshow("Verificación de Rostro ICAO", frame)

    # Salir del bucle si se presiona la tecla 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
