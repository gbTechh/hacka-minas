import cv2
import easyocr
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

VIDEO_PATH  = "videor.mp4"
IMU_PATH    = "imu.npy"
OUTPUT_CSV  = "reporte_adaptativo.csv"

reader = easyocr.Reader(['en'], gpu=False)
X1, Y1, X2, Y2 = 600, 250, 1150, 480

# ═══════════════════════════════════════
# PASO 1 — Leer IMU (fuente confiable)
# ═══════════════════════════════════════
imu = np.load(IMU_PATH, allow_pickle=True)
t_imu = (imu[:, 0] - imu[0, 0]) / 1e9
gyro_mag = np.sqrt(imu[:,4]**2 + imu[:,5]**2 + imu[:,6]**2)
gyro_suave = uniform_filter1d(gyro_mag, size=10)

umbral = gyro_suave.mean() + gyro_suave.std() * 1.2
picos_imu, _ = find_peaks(gyro_suave, height=umbral, distance=200, prominence=8)
ciclos_imu = t_imu[picos_imu]  # timestamps de cada ciclo en segundos

print(f"✅ IMU: {len(ciclos_imu)} ciclos detectados")

# ═══════════════════════════════════════
# PASO 2 — Escanear video completo
# ═══════════════════════════════════════
def tiene_display(frame):
    roi = frame[Y1:Y2, X1:X2]
    b, g, r = cv2.split(roi)
    solo_rojo = cv2.subtract(r, b)
    return np.sum(solo_rojo > 70), solo_rojo

def leer_numero(solo_rojo):
    _, binaria = cv2.threshold(solo_rojo, 70, 255, cv2.THRESH_BINARY)
    grande = cv2.resize(binaria, None, fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
    kernel = np.ones((3,3), np.uint8)
    grande = cv2.dilate(grande, kernel, iterations=2)
    resultado = reader.readtext(grande, allowlist='0123456789', detail=1)
    nums = [r for r in resultado if len(r[1]) >= 2]
    if not nums:
        return None, 0
    texto = "".join([n[1] for n in nums])
    conf  = round(sum([n[2] for n in nums]) / len(nums), 2)
    try:
        val = int(texto)
        if 10 <= val <= 400:
            return val, conf
    except:
        pass
    return None, 0

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Escaneo adaptativo:
# - Si hay display visible → leer cada frame
# - Si no hay display → saltar de a 5 frames
lecturas_raw = []
frame_num = 0
modo_lectura = False
frames_sin_display = 0

print("Escaneando video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1

    # Saltar frames cuando no hay display
    if not modo_lectura and frame_num % 5 != 0:
        continue

    pixeles, solo_rojo = tiene_display(frame)
    seg    = frame_num / fps
    tiempo = f"{int(seg//60):02d}:{int(seg%60):02d}"

    if pixeles > 100:
        modo_lectura = True
        frames_sin_display = 0
        valor, conf = leer_numero(solo_rojo)
        if valor:
            lecturas_raw.append({
                "frame":      frame_num,
                "tiempo":     tiempo,
                "tiempo_seg": round(seg, 1),
                "peso_t":     valor,
                "confianza":  conf,
                "pixeles":    pixeles
            })
    else:
        frames_sin_display += 1
        if frames_sin_display > 30:  # ~2 segundos sin display
            modo_lectura = False

    if frame_num % 1000 == 0:
        print(f"  [{frame_num}/{total}] modo={'LECTURA' if modo_lectura else 'scan'}")

cap.release()

df_raw = pd.DataFrame(lecturas_raw)
print(f"\n✅ {len(df_raw)} lecturas brutas encontradas")

# ═══════════════════════════════════════
# PASO 3 — Agrupar y detectar cambios
# ═══════════════════════════════════════
if df_raw.empty:
    print("❌ Sin lecturas")
    exit()

# Agrupar lecturas cercanas (mismo "evento" si están a <10s)
df_raw = df_raw.sort_values("tiempo_seg").reset_index(drop=True)
grupos = []
grupo  = [df_raw.iloc[0].to_dict()]

for i in range(1, len(df_raw)):
    row = df_raw.iloc[i]
    if row["tiempo_seg"] - grupo[-1]["tiempo_seg"] < 10:
        grupo.append(row.to_dict())
    else:
        grupos.append(grupo)
        grupo = [row.to_dict()]
grupos.append(grupo)

# De cada grupo → mejor lectura
eventos_video = []
for g in grupos:
    gdf  = pd.DataFrame(g)
    mejor = gdf.loc[gdf["confianza"].idxmax()]
    eventos_video.append({
        "tiempo":     mejor["tiempo"],
        "tiempo_seg": mejor["tiempo_seg"],
        "peso_t":     int(mejor["peso_t"]),
        "confianza":  mejor["confianza"],
        "n_lecturas": len(gdf)
    })

df_eventos = pd.DataFrame(eventos_video)
df_eventos = df_eventos.sort_values("tiempo_seg").reset_index(drop=True)

# ═══════════════════════════════════════
# PASO 4 — Cruzar con ciclos IMU
# ═══════════════════════════════════════
# Para cada ciclo IMU, buscar lectura OCR cercana (±45s)
# Si no hay lectura → inferir peso como último_conocido + promedio_delta

cargas = []
peso_anterior = 0
deltas_conocidos = []

for i, t_ciclo in enumerate(ciclos_imu):
    tiempo_str = f"{int(t_ciclo//60):02d}:{int(t_ciclo%60):02d}"

    # Buscar lectura OCR cercana
    cercanas = df_eventos[
        (df_eventos["tiempo_seg"] >= t_ciclo - 45) &
        (df_eventos["tiempo_seg"] <= t_ciclo + 45)
    ]

    if len(cercanas) > 0:
        mejor   = cercanas.loc[cercanas["confianza"].idxmax()]
        peso    = int(mejor["peso_t"])
        fuente  = f"OCR(conf={mejor['confianza']})"
        delta   = peso - peso_anterior
        if delta > 10:
            deltas_conocidos.append(delta)
    else:
        # Inferir usando promedio de deltas conocidos
        if deltas_conocidos:
            delta_estimado = int(np.mean(deltas_conocidos))
        else:
            delta_estimado = 40  # valor típico por defecto
        peso   = peso_anterior + delta_estimado
        delta  = delta_estimado
        fuente = f"ESTIMADO(~{delta_estimado}t/ciclo)"

    # Detectar nuevo camión (peso baja mucho)
    if peso_anterior > 0 and peso < peso_anterior - 50:
        cargas.append({
            "evento":    "NUEVO_CAMION",
            "ciclo_imu": i+1,
            "tiempo":    tiempo_str,
            "peso_t":    0,
            "delta_t":   0,
            "fuente":    "DETECTADO"
        })
        peso_anterior = 0
        deltas_conocidos = []
        continue

    if delta > 5:
        cargas.append({
            "evento":    "CARGA",
            "ciclo_imu": i+1,
            "tiempo":    tiempo_str,
            "peso_t":    peso,
            "delta_t":   delta,
            "fuente":    fuente
        })
        peso_anterior = peso

# ═══════════════════════════════════════
# PASO 5 — Reporte final
# ═══════════════════════════════════════
df_final = pd.DataFrame(cargas)
df_final.to_csv(OUTPUT_CSV, index=False)

print(f"\n{'='*65}")
print(f"{'Ciclo':>6} {'Tiempo':>8} {'Peso':>6} {'Delta':>7}  {'Fuente'}")
print(f"{'='*65}")
for _, r in df_final.iterrows():
    if r["evento"] == "NUEVO_CAMION":
        print(f"\n  🚛 NUEVO CAMIÓN en {r['tiempo']}\n")
        continue
    print(f"  #{int(r['ciclo_imu']):>3}   {r['tiempo']:>6}  "
          f"{int(r['peso_t']):>4}t  +{int(r['delta_t']):>3}t   {r['fuente']}")

cargas_reales = df_final[df_final["evento"] == "CARGA"]
print(f"\n{'='*65}")
print(f"TOTAL CICLOS:        {len(cargas_reales)}")
print(f"PESO FINAL:          {cargas_reales['peso_t'].max():.0f}t")
print(f"DELTA PROMEDIO:      {cargas_reales['delta_t'].mean():.1f}t por ciclo")
print(f"CICLOS CON OCR:      {cargas_reales['fuente'].str.startswith('OCR').sum()}")
print(f"CICLOS ESTIMADOS:    {cargas_reales['fuente'].str.startswith('ESTIMADO').sum()}")
print(f"\n✅ Guardado: {OUTPUT_CSV}")