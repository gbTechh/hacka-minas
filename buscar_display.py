import cv2
import numpy as np

VIDEO_PATH = "videor.mp4"
X1, Y1, X2, Y2 = 600, 250, 1150, 480

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

print("Buscando frames con display rojo visible...\n")

resultados = []
for frame_num in range(0, 2000, 15):  # primeros 2 minutos
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        continue

    roi = frame[Y1:Y2, X1:X2]
    b, g, r = cv2.split(roi)
    solo_rojo = cv2.subtract(r, b)
    pixeles = np.sum(solo_rojo > 70)

    seg    = frame_num / fps
    tiempo = f"{int(seg//60):02d}:{int(seg%60):02d}"

    if pixeles > 80:
        print(f"  Frame {frame_num:5d} ({tiempo}) → {pixeles} píxeles rojos ✅")
        resultados.append(frame_num)
    
cap.release()
print(f"\nTotal frames con display visible: {len(resultados)}")