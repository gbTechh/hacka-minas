import cv2
from paddleocr import PaddleOCR
import numpy as np

VIDEO_PATH = "videor.mp4"
ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)

# ROI de la balanza — ajustada a lo que vimos en la grilla
X1, Y1, X2, Y2 = 780, 310, 920, 390

cap = cv2.VideoCapture(VIDEO_PATH)

# Probar en varios frames cercanos al 700
for frame_num in range(680, 750, 5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        continue

    # Recortar zona de balanza
    roi = frame[Y1:Y2, X1:X2]

    # Agrandar para que OCR lo lea mejor
    roi_grande = cv2.resize(roi, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    # Mejorar contraste — el display es rojo sobre fondo oscuro
    gris = cv2.cvtColor(roi_grande, cv2.COLOR_BGR2GRAY)
    _, binaria = cv2.threshold(gris, 80, 255, cv2.THRESH_BINARY)

    # OCR
    resultado = ocr.ocr(roi_grande, cls=False)
    texto = ""
    if resultado and resultado[0]:
        texto = " ".join([r[1][0] for r in resultado[0]])

    # Guardar recorte para verificar visualmente
    cv2.imwrite(f"roi_frame_{frame_num}.jpg", roi_grande)
    print(f"Frame {frame_num}: '{texto}'")

cap.release()