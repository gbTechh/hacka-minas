import cv2
import numpy as np
import pandas as pd
import time
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

VIDEO_PATH = "videor.mp4"
X1, Y1, X2, Y2 = 600, 250, 1150, 480

# ── Paso 1: extraer y guardar recortes como imágenes (rápido, un solo proceso) ──
def extraer_recortes():
    frames_display = np.load("frames_display.npy").tolist()
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    os.makedirs("recortes_tmp", exist_ok=True)

    info = []
    for frame_num in frames_display:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        if not ret:
            continue
        roi = frame[Y1:Y2, X1:X2]
        b, g, r = cv2.split(roi)
        solo_rojo = cv2.subtract(r, b)
        _, binaria = cv2.threshold(solo_rojo, 70, 255, cv2.THRESH_BINARY)
        grande = cv2.resize(binaria, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
        kernel = np.ones((3,3), np.uint8)
        grande = cv2.dilate(grande, kernel, iterations=2)

        path = f"recortes_tmp/frame_{frame_num}.jpg"
        cv2.imwrite(path, grande)

        seg = frame_num / fps
        info.append((frame_num, f"{int(seg//60):02d}:{int(seg%60):02d}", round(seg,1), path))

    cap.release()
    return info

# ── Paso 2: OCR en un worker (cada worker tiene su propio reader) ──
def ocr_worker(items):
    """items = lista de (frame_num, tiempo, seg, path)"""
    import easyocr
    reader = easyocr.Reader(['en'], gpu=False, verbose=False)
    resultados = []
    for frame_num, tiempo, seg, path in items:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        try:
            res = reader.readtext(img, allowlist='0123456789', detail=1)
            nums = [r for r in res if len(r[1]) >= 2]
            if nums:
                texto = "".join([n[1] for n in nums])
                conf  = round(sum([n[2] for n in nums]) / len(nums), 2)
                texto_limpio = ''.join(c for c in texto if c.isdigit())
                if len(texto_limpio) < 2:
                    continue
                val = int(texto_limpio)
                if 10 <= val <= 400:
                    resultados.append({
                        "frame":      frame_num,
                        "tiempo":     tiempo,
                        "tiempo_seg": seg,
                        "peso_t":     val,
                        "confianza":  conf
                    })
        except:
            pass
    return resultados

if __name__ == "__main__":
    print("Paso 1: Extrayendo recortes...")
    t0 = time.time()
    info = extraer_recortes()
    print(f"  {len(info)} recortes guardados en {time.time()-t0:.1f}s")

    # Dividir en chunks para cada worker
    N_WORKERS = 8
    chunk_size = max(1, len(info) // N_WORKERS)
    chunks = [info[i:i+chunk_size] for i in range(0, len(info), chunk_size)]
    print(f"\nPaso 2: OCR con {N_WORKERS} workers ({len(chunks)} chunks)...")

    t1 = time.time()
    todas_lecturas = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = {executor.submit(ocr_worker, chunk): i for i, chunk in enumerate(chunks)}
        for future in as_completed(futures):
            resultado = future.result()
            todas_lecturas.extend(resultado)
            print(f"  Worker {futures[future]+1} terminó → {len(resultado)} lecturas")

    elapsed = time.time() - t1
    print(f"\n✅ OCR completado en {elapsed:.1f}s")

    # Limpiar temporales
    import shutil
    shutil.rmtree("recortes_tmp")

    # ── Agrupar lecturas cercanas ──
    df = pd.DataFrame(todas_lecturas)
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

    # ── Construir lista de cargas ──
    print(f"\n{'='*50}")
    cargas, peso_prev = [], 0
    for i, g in enumerate(grupos):
        mejor = g.loc[g["confianza"].idxmax()]
        peso  = int(mejor["peso_t"])
        delta = peso - peso_prev
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
            "confianza":   mejor["confianza"],
            "nuevo_camion": es_nuevo
        })
        peso_prev = peso

    # ── Corrección automática de secuencia ──
    print("\n🔧 Corrigiendo secuencia de pesos...")
    cargas_corregidas = []
    peso_anterior  = 0
    delta_promedio = 40  # toneladas promedio por palada

    for c in cargas:
        peso = c['peso_t']

        # Si el peso bajó y NO es nuevo camión → corregir
        if peso < peso_anterior and not c.get('nuevo_camion'):
            peso_estimado = peso_anterior + delta_promedio
            print(f"  ⚠️  Corrigiendo {c['tiempo']}: {peso}t → ~{peso_estimado}t (estimado)")
            c['peso_t']   = peso_estimado
            c['delta_t']  = delta_promedio
            c['confianza'] = 0.0
            c['fuente']   = 'ESTIMADO'
        else:
            c['fuente'] = 'OCR'
            if c['delta_t'] > 0:
                # Actualizar promedio móvil del delta
                delta_promedio = int((delta_promedio + c['delta_t']) / 2)

        cargas_corregidas.append(c)
        peso_anterior = c['peso_t']

    cargas = cargas_corregidas

    # ── Mostrar resultado final ──
    print(f"\n{'='*50}")
    print(f"  {'#':>3}  {'Tiempo':>7}  {'Peso':>6}  {'Delta':>7}  {'Conf':>6}  {'Fuente'}")
    print(f"{'='*50}")
    for c in cargas:
        delta_str = f"+{int(c['delta_t'])}t" if c['delta_t'] >= 0 else f"{int(c['delta_t'])}t"
        fuente    = c.get('fuente', 'OCR')
        conf_str  = f"{c['confianza']:.0%}" if c['confianza'] > 0 else "estimado"
        print(f"  #{int(c['carga']):>2}  {c['tiempo']:>7}  {int(c['peso_t']):>4}t"
              f"  {delta_str:>7}  {conf_str:>6}  {fuente}")

    # ── Guardar CSV ──
    pd.DataFrame(cargas).to_csv("cargas_final.csv", index=False)
    print(f"\n✅ Total tiempo: {time.time()-t0:.1f}s")
    print(f"✅ Guardado: cargas_final.csv")