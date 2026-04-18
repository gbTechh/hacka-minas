import cv2
import numpy as np
import pandas as pd
import pytesseract

VIDEO_PATH = "videor.mp4"
X1, Y1, X2, Y2 = 600, 250, 1150, 480

frames_display = np.load("frames_display.npy").tolist()

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

def leer_numero(frame):
    roi = frame[Y1:Y2, X1:X2]
    b, g, r = cv2.split(roi)
    solo_rojo = cv2.subtract(r, b)
    _, binaria = cv2.threshold(solo_rojo, 70, 255, cv2.THRESH_BINARY)
    grande = cv2.resize(binaria, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((3,3), np.uint8)
    grande = cv2.dilate(grande, kernel, iterations=2)

    # Tesseract — modo línea única, solo dígitos
    config = '--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789'
    texto = pytesseract.image_to_string(grande, config=config).strip()
    texto = ''.join(c for c in texto if c.isdigit())
    if len(texto) >= 2:
        try:
            val = int(texto)
            if 10 <= val <= 400:
                return val, 0.9
        except:
            pass
    return None, 0

print(f"Fase 2 (Tesseract): {len(frames_display)} frames...\n")
lecturas = []

for i, frame_num in enumerate(frames_display):
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        continue

    seg    = frame_num / fps
    tiempo = f"{int(seg//60):02d}:{int(seg%60):02d}"
    valor, conf = leer_numero(frame)

    if valor:
        lecturas.append({
            "frame":      frame_num,
            "tiempo":     tiempo,
            "tiempo_seg": round(seg, 1),
            "peso_t":     valor,
            "confianza":  conf
        })
        print(f"  Frame {frame_num:5d} ({tiempo}) → {valor}t")

    if (i+1) % 20 == 0:
        print(f"  [{i+1}/{len(frames_display)}] procesados...")

cap.release()

df = pd.DataFrame(lecturas)
if df.empty:
    print("❌ Sin lecturas")
    exit()

df = df.sort_values("frame").reset_index(drop=True)
grupos = []
grupo  = [df.iloc[0].to_dict()]
for i in range(1, len(df)):
    if df.iloc[i]["frame"] - grupo[-1]["frame"] < 200:
        grupo.append(df.iloc[i].to_dict())
    else:
        grupos.append(pd.DataFrame(grupo))
        grupo = [df.iloc[i].to_dict()]
grupos.append(pd.DataFrame(grupo))

print(f"\n{'='*50}")
print(f"{'#':>4} {'Tiempo':>8} {'Peso':>6} {'Delta':>7} {'Frames':>7}")
print(f"{'='*50}")

cargas = []
peso_prev = 0
for i, g in enumerate(grupos):
    from collections import Counter
    # Tomar el valor más frecuente del grupo (más robusto que el de mayor conf)
    conteo = Counter(int(x) for x in g["peso_t"])
    peso   = conteo.most_common(1)[0][0]
    mejor  = g[g["peso_t"] == peso].iloc[0]
    delta  = peso - peso_prev

    es_nuevo = peso_prev > 0 and peso < peso_prev - 50
    if es_nuevo:
        print(f"\n  🚛 NUEVO CAMIÓN\n")
        peso_prev = 0
        delta = peso

    cargas.append({
        "carga":       i + 1,
        "tiempo":      mejor["tiempo"],
        "tiempo_seg":  mejor["tiempo_seg"],
        "peso_t":      peso,
        "delta_t":     delta,
        "n_frames":    len(g),
        "nuevo_camion": es_nuevo
    })
    print(f"  #{i+1:>2}  {mejor['tiempo']:>6}  {peso:>4}t  +{delta:>3}t   {len(g)} frames")
    peso_prev = peso

df_cargas = pd.DataFrame(cargas)
df_cargas.to_csv("cargas_fase2.csv", index=False)
print(f"\nTotal: {len(cargas)} eventos | Peso máx: {df_cargas['peso_t'].max()}t")
print(f"✅ Guardado: cargas_fase2.csv")