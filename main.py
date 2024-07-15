import cv2
import mediapipe as mp
import math
from PIL import Image
import os

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

#paths
angry = os.path.join("c:\\Users\\edujr\\OneDrive\\Documentos\\emotion\\facial-emotion", "angry.png")
happy = os.path.join("c:\\Users\\edujr\\OneDrive\\Documentos\\emotion\\facial-emotion", "happy.png")
waos = os.path.join("c:\\Users\\edujr\\OneDrive\\Documentos\\emotion\\facial-emotion", "asombro.png")
sad = os.path.join("c:\\Users\\edujr\\OneDrive\\Documentos\\emotion\\facial-emotion", "sad.png")

# Cargar las imágenes de las emociones
img_enojada = cv2.imread(angry, cv2.IMREAD_UNCHANGED)
img_feliz = cv2.imread(happy, cv2.IMREAD_UNCHANGED)
img_sorprendida = cv2.imread(waos, cv2.IMREAD_UNCHANGED)
img_triste = cv2.imread(sad, cv2.IMREAD_UNCHANGED)



def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """ Overlay img_overlay on top of img at (x, y) with alpha mask. """
    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    # Blend overlay within the determined ranges
    img_crop = img[y1:y2, x1:x2]

    # Ensure img_crop has 4 channels
    if img_crop.shape[2] == 3:
        img_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2BGRA)

    img_overlay_crop = img_overlay[y1o:y2o, x1o:x2o]
    alpha = alpha_mask[y1o:y2o, x1o:x2o, None] / 255.0
    alpha_inv = 1.0 - alpha

    img_crop[:] = alpha * img_overlay_crop + alpha_inv * img_crop

    # Convert img_crop back to 3 channels if needed
    img[y1:y2, x1:x2] = cv2.cvtColor(img_crop, cv2.COLOR_BGRA2BGR)

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
                        overlay_image_alpha(frame, img_enojada, 480, 80, img_enojada[:, :, 3])
                    # Persona Feliz
                    elif longitud1 > 20 and longitud1 < 30 and longitud2 > 20 and longitud2 < 30 and longitud3 > 109 and longitud4 > 10 and longitud4 < 20:
                        overlay_image_alpha(frame, img_feliz, 480, 80, img_feliz[:, :, 3])
                    # Persona Sorprendida
                    elif longitud1 > 35 and longitud2 > 35 and longitud3 > 85 and longitud3 < 90 and longitud4 > 20:
                        overlay_image_alpha(frame, img_sorprendida, 480, 80, img_sorprendida[:, :, 3])
                    # Persona Triste
                    elif longitud1 > 20 and longitud1 < 35 and longitud2 > 20 and longitud2 < 35 and longitud3 > 90 and longitud3 < 95 and longitud4 < 5:
                        overlay_image_alpha(frame, img_triste, 480, 80, img_triste[:, :, 3])
                        
    cv2.imshow("Intensamente 2D", frame)
    t = cv2.waitKey(1)

    if t == 27:
        break
        
cap.release()
cv2.destroyAllWindows()
