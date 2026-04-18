import cv2
import numpy as np
import pandas as pd
from rapidocr_onnxruntime import RapidOCR
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

VIDEO_PATH = "videor.mp4"
X1, Y1, X2, Y2 = 600, 250, 1150, 480

# ── Cargar señal de balanza ya calculada ──
senal  = np.load("senal_balanza.npy")
t_bal  = senal[:, 0]
px_bal = senal[:, 1]

# ── Cargar IMU para ciclos ──
imu      = np.load("imu.npy", allow_pickle=True)
t_imu    = (imu[:, 0] - imu[0, 0]) / 1e9
gyro_mag = np.sqrt(imu[:,4]**2 + imu[:,5]**2 + imu[:,6]**2)
gm_suave = uniform_filter1d(gyro_mag, size=8)
umbral   = gm_suave.mean() + gm_suave.std() * 1.2
picos, _ = find_peaks(gm_suave, height=umbral, distance=200, prominence=8)
t_ciclos = t_imu[picos]
print(f"Ciclos IMU: {len(t_ciclos)}")

# ── Detectar frames con display usando señal ya calculada ──
# Interpolar señal de balanza a cada frame del video
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Frames donde hay display (px > 100)
UMBRAL_PX = 100
frames_con_display = []
for frame_num in range(1, total+1):
    seg = frame_num / fps
    idx = np.argmin(np.abs(t_bal - seg))
    if px_bal[idx] > UMBRAL_PX:
        frames_con_display.append(frame_num)

print(f"Frames con display: {len(frames_con_display)}")

# ── Leer pesos con RapidOCR ──
ocr = RapidOCR()

def leer_display(frame):
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

print("Leyendo pesos...")
lecturas = []
for frame_num in frames_con_display:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    ret, frame = cap.read()
    if not ret:
        continue
    seg   = frame_num / fps
    valor = leer_display(frame)
    if valor:
        tiempo = f"{int(seg//60):02d}:{int(seg%60):02d}"
        lecturas.append({"frame": frame_num, "tiempo": tiempo,
                         "tiempo_seg": round(seg,1), "peso_t": valor})
        print(f"  {tiempo} → {valor}t")

cap.release()

# ── Agrupar lecturas cercanas (mismo evento) ──
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

# De cada grupo tomar el valor más frecuente
eventos = []
peso_prev = 0
for g in grupos:
    gdf  = pd.DataFrame(g)
    from collections import Counter
    conteo = Counter(int(x) for x in gdf["peso_t"])
    peso   = conteo.most_common(1)[0][0]
    mejor  = gdf[gdf["peso_t"] == peso].iloc[0]
    delta  = peso - peso_prev

    # Detectar nuevo camión: peso baja respecto al anterior
    es_nuevo = peso_prev > 0 and peso < peso_prev - 30
    if es_nuevo:
        peso_prev = 0
        delta = peso

    eventos.append({
        "tiempo":     mejor["tiempo"],
        "tiempo_seg": mejor["tiempo_seg"],
        "peso_t":     peso,
        "delta_t":    delta,
        "es_nuevo":   es_nuevo
    })
    peso_prev = peso

# ── Corrección secuencia creciente ──
DELTA_PROM = 38
eventos_corr = []
peso_ant = 0
delta_prom = DELTA_PROM
for e in eventos:
    if e["peso_t"] < peso_ant - 30 and not e["es_nuevo"]:
        # OCR leyó mal, estimar
        e["peso_t"]  = peso_ant + delta_prom
        e["delta_t"] = delta_prom
        e["fuente"]  = "EST"
    else:
        e["fuente"] = "OCR"
        if e["delta_t"] > 5:
            delta_prom = int((delta_prom + e["delta_t"]) / 2)
    eventos_corr.append(e)
    peso_ant = e["peso_t"]

# ── Separar por camiones ──
camiones = []
camion_actual = []
for e in eventos_corr:
    if e["es_nuevo"] and camion_actual:
        camiones.append(camion_actual)
        camion_actual = []
    camion_actual.append(e)
if camion_actual:
    camiones.append(camion_actual)

# ── Agregar ciclos IMU por camión ──
def ciclos_en_rango(t_ini, t_fin):
    return int(np.sum((t_ciclos >= t_ini) & (t_ciclos <= t_fin)))

# ── Reporte final ──
print(f"\n{'='*60}")
print(f"  REPORTE DE PESOS POR CAMIÓN")
print(f"{'='*60}")

resumen = []
for i, cam in enumerate(camiones):
    paladas   = [e["peso_t"] for e in cam]
    deltas    = [e["delta_t"] for e in cam if e["delta_t"] > 5]
    t_ini_seg = cam[0]["tiempo_seg"]
    t_fin_seg = cam[-1]["tiempo_seg"]
    n_ciclos  = ciclos_en_rango(t_ini_seg - 30, t_fin_seg + 30)

    # Último peso conocido
    peso_final_ocr = cam[-1]["peso_t"]

    # Estimar peso total: último OCR + 1 palada más (ciclo que no se ve)
    delta_est  = int(np.mean(deltas)) if deltas else DELTA_PROM
    peso_total = peso_final_ocr + delta_est  # estimado

    prom_delta = int(np.mean(deltas)) if deltas else DELTA_PROM

    print(f"\n🚛 CAMIÓN {i+1}  ({cam[0]['tiempo']} → {cam[-1]['tiempo']})")
    print(f"   Paladas visibles:   {len(paladas)}")
    print(f"   Ciclos IMU:         {n_ciclos}")
    print(f"   Pesos detectados:   {paladas}")
    print(f"   Promedio/palada:    ~{prom_delta}t")
    print(f"   Último peso (OCR):  {peso_final_ocr}t")
    print(f"   Peso total est.:    ~{peso_total}t (último OCR + 1 palada est.)")
    print(f"\n   {'Tiempo':>7} {'Peso':>6} {'Delta':>7}  Fuente")
    print(f"   {'-'*35}")
    for e in cam:
        d = f"+{int(e['delta_t'])}t" if e['delta_t'] > 0 else "---"
        print(f"   {e['tiempo']:>7}  {int(e['peso_t']):>4}t  {d:>6}  {e['fuente']}")

    resumen.append({
        "camion":          i + 1,
        "t_inicio":        cam[0]["tiempo"],
        "t_fin":           cam[-1]["tiempo"],
        "paladas_visibles": len(paladas),
        "ciclos_imu":      n_ciclos,
        "prom_delta_t":    prom_delta,
        "ultimo_peso_ocr": peso_final_ocr,
        "peso_total_est":  peso_total,
    })

print(f"\n{'='*60}")
print(f"  RESUMEN")
print(f"{'='*60}")
df_res = pd.DataFrame(resumen)
print(df_res.to_string(index=False))

df_res.to_csv("pesos_por_camion.csv", index=False)
print(f"\n✅ Guardado: pesos_por_camion.csv")