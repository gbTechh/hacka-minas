import cv2
import numpy as np
import pandas as pd
import time
from rapidocr_onnxruntime import RapidOCR

VIDEO_PATH = "videor.mp4"
X1, Y1, X2, Y2 = 600, 250, 1150, 480

def leer_display(frame, ocr):
    roi = frame[Y1:Y2, X1:X2]
    b, g, r = cv2.split(roi)
    solo_rojo = cv2.subtract(r, b)
    _, binaria = cv2.threshold(solo_rojo, 70, 255, cv2.THRESH_BINARY)
    grande = cv2.resize(binaria, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((3,3), np.uint8)
    grande = cv2.dilate(grande, kernel, iterations=2)

    resultado, _ = ocr(grande)
    if not resultado:
        return None

    texto = "".join([r[1] for r in resultado])
    texto_limpio = ''.join(c for c in texto if c.isdigit())
    if len(texto_limpio) < 2:
        return None
    try:
        val = int(texto_limpio)
        if 10 <= val <= 400:
            return val
    except:
        pass
    return None

if __name__ == "__main__":
    frames_display = np.load("frames_display.npy").tolist()
    ocr = RapidOCR()
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)

    lecturas = []
    t0 = time.time()
    print(f"Procesando {len(frames_display)} frames con RapidOCR...")

    for frame_num in frames_display:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue

        val = leer_display(frame, ocr)
        if val:
            seg    = frame_num / fps
            tiempo = f"{int(seg//60):02d}:{int(seg%60):02d}"
            lecturas.append({"frame": frame_num, "tiempo": tiempo,
                             "tiempo_seg": round(seg,1), "peso_t": val})
            print(f"  Frame {frame_num} ({tiempo}) → {val}t")

    cap.release()
    print(f"\n✅ {len(lecturas)} lecturas en {time.time()-t0:.1f}s")

    df = pd.DataFrame(lecturas)
    if df.empty:
        print("❌ Sin lecturas")
        exit()

    df = df.sort_values("frame").reset_index(drop=True)
    grupos, grupo = [], [df.iloc[0].to_dict()]
    for i in range(1, len(df)):
        if df.iloc[i]["frame"] - grupo[-1]["frame"] < 200:
            grupo.append(df.iloc[i].to_dict())
        else:
            grupos.append(pd.DataFrame(grupo))
            grupo = [df.iloc[i].to_dict()]
    grupos.append(pd.DataFrame(grupo))

    cargas, peso_prev = [], 0
    for i, g in enumerate(grupos):
        gdf  = pd.DataFrame(g)
        peso = int(gdf["peso_t"].value_counts().idxmax())
        delta = peso - peso_prev
        es_nuevo = peso_prev > 0 and peso < peso_prev - 50
        if es_nuevo:
            peso_prev = 0
            delta = peso
        tiempo = gdf.iloc[0]["tiempo"]
        cargas.append({"carga": i+1, "tiempo": tiempo, "peso_t": peso,
                       "delta_t": delta, "nuevo_camion": es_nuevo})
        peso_prev = peso

    # Corrección automática
    cargas_corregidas = []
    peso_anterior = 0
    delta_promedio = 40
    for c in cargas:
        peso = c['peso_t']
        if peso < peso_anterior and not c.get('nuevo_camion'):
            peso_estimado = peso_anterior + delta_promedio
            print(f"  ⚠️  Corrigiendo {c['tiempo']}: {peso}t → ~{peso_estimado}t")
            c['peso_t'] = peso_estimado
            c['delta_t'] = delta_promedio
            c['fuente'] = 'ESTIMADO'
        else:
            c['fuente'] = 'OCR'
            if c['delta_t'] > 0:
                delta_promedio = int((delta_promedio + c['delta_t']) / 2)
        cargas_corregidas.append(c)
        peso_anterior = c['peso_t']

    print(f"\n{'='*50}")
    for c in cargas_corregidas:
        delta_str = f"+{int(c['delta_t'])}t" if c['delta_t'] >= 0 else f"{int(c['delta_t'])}t"
        print(f"  #{int(c['carga']):>2}  {c['tiempo']}  {int(c['peso_t'])}t  {delta_str}  {c['fuente']}")

    pd.DataFrame(cargas_corregidas).to_csv("cargas_final.csv", index=False)
    print(f"\n✅ Total: {time.time()-t0:.1f}s | Guardado: cargas_final.csv")