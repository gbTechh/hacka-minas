import cv2
import numpy as np
import time

# Cargar un recorte de prueba
cap = cv2.VideoCapture("videor.mp4")
cap.set(cv2.CAP_PROP_POS_FRAMES, 1320)
ret, frame = cap.read()
cap.release()

X1, Y1, X2, Y2 = 600, 250, 1150, 480
roi = frame[Y1:Y2, X1:X2]
b, g, r = cv2.split(roi)
solo_rojo = cv2.subtract(r, b)
_, binaria = cv2.threshold(solo_rojo, 70, 255, cv2.THRESH_BINARY)
grande = cv2.resize(binaria, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
kernel = np.ones((3,3), np.uint8)
img = cv2.dilate(grande, kernel, iterations=2)
cv2.imwrite("test_img.jpg", img)

# ── Test EasyOCR ──
print("Probando EasyOCR...")
import easyocr
reader = easyocr.Reader(['en'], gpu=False, verbose=False)
t0 = time.time()
for _ in range(10):
    reader.readtext(img, allowlist='0123456789')
print(f"EasyOCR: {(time.time()-t0)/10:.2f}s por imagen")

# ── Test RapidOCR ──
print("\nProbando RapidOCR...")
from rapidocr_onnxruntime import RapidOCR
rapid = RapidOCR()
t0 = time.time()
for _ in range(10):
    rapid("test_img.jpg")
print(f"RapidOCR: {(time.time()-t0)/10:.2f}s por imagen")

# ── Test ddddocr ──
print("\nProbando ddddocr...")
import ddddocr
ocr = ddddocr.DdddOcr(show_ad=False)
with open("test_img.jpg", "rb") as f:
    img_bytes = f.read()
t0 = time.time()
for _ in range(10):
    ocr.classification(img_bytes)
print(f"ddddocr: {(time.time()-t0)/10:.2f}s por imagen")