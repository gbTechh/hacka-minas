import cv2
import numpy as np

VIDEO_PATH  = "videor.mp4"
X1, Y1, X2, Y2 = 600, 250, 1150, 480

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

frames_con_display = []
frame_num = 0

print("Fase 1: Escaneando rápido (sin OCR)...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1

    if frame_num % 5 != 0:  # 1 de cada 5
        continue

    roi = frame[Y1:Y2, X1:X2]
    b, g, r = cv2.split(roi)
    solo_rojo = cv2.subtract(r, b)
    pixeles = int(np.sum(solo_rojo > 70))

    if pixeles > 100:
        frames_con_display.append(frame_num)

    if frame_num % 2000 == 0:
        pct = frame_num/total*100
        print(f"  {pct:.0f}% — frames con display: {len(frames_con_display)}")

cap.release()
print(f"\n✅ Fase 1 completa: {len(frames_con_display)} frames con display")
print(f"Frames: {frames_con_display}")

# Guardar para la fase 2
np.save("frames_display.npy", frames_con_display)