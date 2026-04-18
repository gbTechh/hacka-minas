import cv2
import easyocr
import numpy as np
import pandas as pd

VIDEO_PATH  = "videor.mp4"
OUTPUT_CSV  = "pesos_detectados.csv"
reader = easyocr.Reader(['en'], gpu=False)

# ROI más amplia
X1, Y1, X2, Y2 = 600, 250, 1150, 480

def leer_balanza(frame):
    roi = frame[Y1:Y2, X1:X2]
    b, g, r = cv2.split(roi)
    solo_rojo = cv2.subtract(r, b)

    # Probar umbral 70 (más permisivo que 80)
    _, binaria = cv2.threshold(solo_rojo, 70, 255, cv2.THRESH_BINARY)

    # Solo continuar si hay suficientes píxeles rojos
    if np.sum(binaria > 0) < 100:
        return None, None

    grande = cv2.resize(binaria, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((3,3), np.uint8)
    grande = cv2.dilate(grande, kernel, iterations=2)

    resultado = reader.readtext(grande, allowlist='0123456789', detail=1)
    if resultado:
        # Filtrar solo resultados con al menos 1 dígito
        nums = [r for r in resultado if any(c.isdigit() for c in r[1])]
        if nums:
            texto = "".join([r[1] for r in nums])
            conf  = round(sum([r[2] for r in nums]) / len(nums), 2)
            return texto, conf
    return None, None

# ── Procesar todo el video ──
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

registros = []
frame_num = 0
ultimo_peso = None
ultimo_peso_conf = 0

print("Procesando video completo...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_num += 1
    if frame_num % 10 != 0:   # procesar 1 de cada 10 frames
        continue

    seg    = frame_num / fps
    tiempo = f"{int(seg//60):02d}:{int(seg%60):02d}"

    peso, conf = leer_balanza(frame)

    # Guardar solo cuando detecta algo
    if peso and len(peso) >= 2:   # mínimo 2 dígitos
        try:
            valor = int(peso)
            if 10 <= valor <= 500:   # rango realista de toneladas
                ultimo_peso      = valor
                ultimo_peso_conf = conf
                registros.append({
                    "frame":     frame_num,
                    "tiempo":    tiempo,
                    "tiempo_seg": round(seg, 1),
                    "peso_t":    valor,
                    "confianza": conf
                })
                print(f"  {tiempo} → {valor}t  (conf={conf})")
        except:
            pass

    if frame_num % 500 == 0:
        print(f"  [{frame_num}/{total}] último peso: {ultimo_peso}t")

cap.release()

if registros:
    df = pd.DataFrame(registros)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ {len(registros)} lecturas guardadas en {OUTPUT_CSV}")
    print(f"\nPesos detectados únicos: {sorted(df['peso_t'].unique())}")
    print(f"Rango: {df['peso_t'].min()}t — {df['peso_t'].max()}t")
else:
    print("❌ No se detectaron pesos")