import cv2
import easyocr
import numpy as np

VIDEO_PATH = "videor.mp4"
reader = easyocr.Reader(['en'], gpu=False)

X1, Y1, X2, Y2 = 700, 280, 1050, 430

def leer_balanza(frame):
    roi = frame[Y1:Y2, X1:X2]
    b, g, r = cv2.split(roi)
    solo_rojo = cv2.subtract(r, b)
    _, binaria = cv2.threshold(solo_rojo, 80, 255, cv2.THRESH_BINARY)

    # Agrandar para OCR
    grande = cv2.resize(binaria, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)

    # Dilatar para unir segmentos del 7-segmentos
    kernel = np.ones((3,3), np.uint8)
    grande = cv2.dilate(grande, kernel, iterations=2)

    resultado = reader.readtext(grande, allowlist='0123456789', detail=1)
    if resultado:
        texto = "".join([r[1] for r in resultado])
        conf  = round(sum([r[2] for r in resultado]) / len(resultado), 2)
        return texto, conf
    return None, None

cap = cv2.VideoCapture(VIDEO_PATH)

print(f"{'Frame':>7} {'Tiempo':>8} {'Peso':>6} {'Conf':>6}")
print("-" * 35)

for frame_num in range(680, 750, 5):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        continue

    fps = 15.0
    seg = frame_num / fps
    tiempo = f"{int(seg//60):02d}:{int(seg%60):02d}"

    texto, conf = leer_balanza(frame)
    if texto:
        print(f"{frame_num:>7} {tiempo:>8} {texto:>6}t  {conf:>6}")
    else:
        print(f"{frame_num:>7} {tiempo:>8}  {'---':>6}")

cap.release()