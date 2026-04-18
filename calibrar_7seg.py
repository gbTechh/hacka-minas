import cv2
import numpy as np

VIDEO_PATH = "videor.mp4"
X1, Y1, X2, Y2 = 600, 250, 1150, 480

# Tomar el mejor frame conocido (conf=0.99)
cap = cv2.VideoCapture(VIDEO_PATH)
cap.set(cv2.CAP_PROP_POS_FRAMES, 1320)
ret, frame = cap.read()
cap.release()

roi = frame[Y1:Y2, X1:X2]
b, g, r = cv2.split(roi)
solo_rojo = cv2.subtract(r, b)
_, binaria = cv2.threshold(solo_rojo, 70, 255, cv2.THRESH_BINARY)

# Encontrar contornos de cada dígito
contornos, _ = cv2.findContours(binaria, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contornos = sorted(contornos, key=lambda c: cv2.boundingRect(c)[0])  # orden izq→der

print(f"Contornos encontrados: {len(contornos)}")
vis = cv2.cvtColor(binaria, cv2.COLOR_GRAY2BGR)

digitos_bbox = []
for i, c in enumerate(contornos):
    area = cv2.contourArea(c)
    if area < 100:
        continue
    x, y, w, h = cv2.boundingRect(c)
    print(f"  Contorno {i}: x={x} y={y} w={w} h={h} area={area:.0f}")
    cv2.rectangle(vis, (x,y), (x+w,y+h), (0,255,0), 2)
    cv2.putText(vis, str(i), (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    digitos_bbox.append((x, y, w, h))

grande = cv2.resize(vis, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
cv2.imwrite("calibracion_digitos.jpg", grande)
print("\n✅ Guardado: calibracion_digitos.jpg")
print("Sube la imagen para ver cómo detecta cada dígito")