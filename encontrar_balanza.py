import cv2
import numpy as np

VIDEO_PATH = "videor.mp4"

# Frames donde se ve el camión con la balanza visible
# (basado en lo que vimos: t=35s → frame ~525)
FRAMES_A_VER = [520, 525, 530, 540, 600, 700]

cap = cv2.VideoCapture(VIDEO_PATH)

for frame_num in FRAMES_A_VER:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Dibujar grilla para identificar coordenadas
    h, w = frame.shape[:2]
    for x in range(0, w, 100):
        cv2.line(frame, (x, 0), (x, h), (50, 50, 50), 1)
        cv2.putText(frame, str(x), (x+2, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
    for y in range(0, h, 100):
        cv2.line(frame, (0, y), (w, y), (50, 50, 50), 1)
        cv2.putText(frame, str(y), (2, y+15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,0), 1)
    
    cv2.imwrite(f"frame_{frame_num}_grilla.jpg", frame)
    print(f"✅ Guardado: frame_{frame_num}_grilla.jpg")

cap.release()
print("\nAbre las imágenes y dime las coordenadas exactas del display rojo")