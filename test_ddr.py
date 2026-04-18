import cv2
import numpy as np
import ddddocr
import time

VIDEO_PATH = "videor.mp4"
X1, Y1, X2, Y2 = 600, 250, 1150, 480

ocr = ddddocr.DdddOcr(show_ad=False)
cap = cv2.VideoCapture(VIDEO_PATH)

# Probar con frames conocidos donde hay display
for frame_num in [395, 680, 960, 1290, 1560, 1875]:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        continue

    roi = frame[Y1:Y2, X1:X2]

    # Versión 1 — imagen en color directa
    _, enc1 = cv2.imencode('.jpg', roi)
    texto1 = ocr.classification(enc1.tobytes())

    # Versión 2 — solo canal rojo amplificado
    b, g, r = cv2.split(roi)
    solo_rojo = cv2.subtract(r, b)
    grande = cv2.resize(solo_rojo, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, enc2 = cv2.imencode('.jpg', grande)
    texto2 = ocr.classification(enc2.tobytes())

    # Versión 3 — binaria sin dilate
    _, binaria = cv2.threshold(solo_rojo, 70, 255, cv2.THRESH_BINARY)
    grande3 = cv2.resize(binaria, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
    _, enc3 = cv2.imencode('.jpg', grande3)
    texto3 = ocr.classification(enc3.tobytes())

    print(f"Frame {frame_num}: color='{texto1}' | rojo='{texto2}' | binaria='{texto3}'")

cap.release()