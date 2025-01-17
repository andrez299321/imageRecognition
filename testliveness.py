import cv2
from deepface import DeepFace

# Inicializa la cámara
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detecta rostros
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        # Recorta la cara detectada
        face = frame[y:y+h, x:x+w]
        
        # Llama a DeepFace para analizar la cara (puedes usar otras funciones aquí)
        result = DeepFace.analyze(face, actions=["age", "gender", "emotion"])
        print(result)
    
    # Muestra el video con la detección de rostros
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
