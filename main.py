import cv2
import mediapipe as mp
import math

# Realizar la captura de video
cap = cv2.VideoCapture(1)
cap.set(3, 1280)
cap.set(4, 720)

# Crear nuestra funcion dibujo
mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1) # Ajustar configuracion del dibujo

# Crear un objeto donde almacenarmos la malla facial
mp_face_mesh = mp.solutions.face_mesh # Llamar a la funcion de la malla facial
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1) # Crear el objeto (ctrl+click)

# Crear el while principal
while True:
    ret, frame = cap.read()
    # Correccion de color
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Observar los resultados
    results = face_mesh.process(frame_rgb)
    
    # Crear listas donde almacenar los puntos de la malla facial
    px = []
    py = []
    lista = []
    r = 5
    r = 3
    
    if results.multi_face_landmarks: # Si detecta la malla facial
        for face_landmarks in results.multi_face_landmarks: # Mostrar el rostro detectado
            mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION, drawing_spec, drawing_spec)
            
            # Extraer los puntos de la malla facial
            for id, point in enumerate(face_landmarks.landmark):
                h, w, c = frame.shape
                x, y = int(point.x * w), int(point.y * h)
                px.append(x)
                py.append(y)
                lista.append([id, x, y])
                
                if len(lista) == 468:
                    # Ceja Derecha
                    x1, y1 = lista[65][1:]
                    x2, y2 = lista[158][1:]
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    longitud1 = math.hypot(x2 - x1, y2 - y1)    
                    
                    # Ceja Izquierda
                    x3, y3 = lista[295][1:]
                    x4, y4 = lista[385][1:]
                    cx2, cy2 = (x3 + x4) // 2, (y3 + y4) // 2
                    longitud2 = math.hypot(x4 - x3, y4 - y3)
                    
                    # Boca Extremos
                    x5, y5 = lista[78][1:]
                    x6, y6 = lista[308][1:]
                    cx3, cy3 = (x5 + x6) // 2, (y5 + y6) // 2
                    longitud3 = math.hypot(x6 - x5, y6 - y5)
                    
                    # Boca Apertura
                    x7, y7 = lista[13][1:]
                    x8, y8 = lista[14][1:]
                    cx4, cy4 = (x7 + x8) // 2, (y7 + y8) // 2
                    longitud4 = math.hypot(x8 - x7, y8 - y7)
                    
                    # Clasificacion
                    # Persona Enojada
                    if longitud1 < 19 and longitud2 < 19 and longitud3 > 80 and longitud3 < 95 and longitud4 < 5:
                        cv2.putText(frame, "Persona Enojada", (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    # Persona Feliz
                    elif longitud1 > 20 and longitud1 < 30 and longitud2 > 20 and longitud2 < 30 and longitud3 > 109 and longitud4 > 10 and longitud4 < 20:
                        cv2.putText(frame, "Persona Feliz", (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                    # Persona Sorprendida
                    elif longitud1 > 35 and longitud2 > 35 and longitud3 > 85 and longitud3 < 90 and longitud4 > 20:
                        cv2.putText(frame, "Persona Sorprendida", (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)
                    # Persona Triste
                    elif longitud1 > 20 and longitud1 < 35 and longitud2 > 20 and longitud2 < 35 and longitud3 > 90 and longitud3 < 95 and longitud4 < 5:
                        cv2.putText(frame, "Persona Triste", (480, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)
                        
    cv2.imshow("Intensamente 2D", frame)
    t = cv2.waitKey(1)

    if t == 27:
        break
        
cap.release()
cv2.destroyAllWindows()