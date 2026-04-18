import cv2
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

VIDEO_PATH  = "videor.mp4"
OUTPUT_PATH = "debug_overlay.avi"
X1, Y1, X2, Y2 = 600, 250, 1150, 480

# ── Cargar IMU ──
imu        = np.load("imu.npy", allow_pickle=True)
t_imu      = (imu[:, 0] - imu[0, 0]) / 1e9
gyro_mag   = np.sqrt(imu[:,4]**2 + imu[:,5]**2 + imu[:,6]**2)
gyro_suave = uniform_filter1d(gyro_mag, size=10)
umbral     = gyro_suave.mean() + gyro_suave.std() * 1.2
picos, _   = find_peaks(gyro_suave, height=umbral, distance=200, prominence=8)
t_ciclos   = t_imu[picos]

# ── Cargar pesos OCR ──
df_ocr = pd.read_csv("cargas_final.csv")
df_ocr['tiempo_seg'] = df_ocr['tiempo'].apply(
    lambda t: int(t.split(":")[0])*60 + int(t.split(":")[1])
)

# ── Cargar señal de balanza ──
senal   = np.load("senal_balanza.npy")
t_bal   = senal[:, 0]
px_bal  = senal[:, 1]

# ── Abrir video ──
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(OUTPUT_PATH,
      cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))

frame_num = 0
print("Generando video con overlay...")

# Estado actual
estado_actual    = "ESPERANDO"
ciclo_actual     = 0
peso_actual      = 0
ultimo_ciclo_t   = -999
VENTANA_CICLO    = 15  # segundos alrededor del pico IMU

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1
    seg = frame_num / fps

    # ── Detectar estado actual ──
    # 1. ¿Hay balanza activa?
    idx_bal = np.argmin(np.abs(t_bal - seg))
    hay_balanza = px_bal[idx_bal] > 100

    # 2. ¿Estamos cerca de un pico IMU? (excavando/desmontando)
    dist_ciclos = np.abs(t_ciclos - seg)
    ciclo_mas_cercano = np.argmin(dist_ciclos)
    dist_min = dist_ciclos[ciclo_mas_cercano]
    en_ciclo = dist_min < VENTANA_CICLO

    if en_ciclo:
        ciclo_actual = ciclo_mas_cercano + 1
        t_pico = t_ciclos[ciclo_mas_cercano]
        if seg < t_pico:
            estado_actual = "EXCAVANDO"
            color_estado  = (0, 165, 255)   # naranja
        else:
            estado_actual = "DESMONTANDO"
            color_estado  = (0, 0, 255)     # rojo
    else:
        estado_actual = "EN ESPERA"
        color_estado  = (200, 200, 200)     # gris

    # 3. ¿Hay peso visible?
    ocr_cerca = df_ocr[
        (df_ocr['tiempo_seg'] >= seg - 30) &
        (df_ocr['tiempo_seg'] <= seg + 5)
    ]
    if len(ocr_cerca) > 0:
        peso_actual = int(ocr_cerca.iloc[-1]['peso_t'])

    # ── Dibujar overlay ──
    # Panel semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (420, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    tiempo_str = f"{int(seg//60):02d}:{int(seg%60):02d}"
    cv2.putText(frame, f"Tiempo:  {tiempo_str}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Ciclo:   #{ciclo_actual}",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Estado:  {estado_actual}",
                (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_estado, 2)
    cv2.putText(frame, f"Peso:    {peso_actual}t",
                (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 255, 100) if hay_balanza else (150, 150, 150), 2)
    cv2.putText(frame, f"Balanza: {'ACTIVA' if hay_balanza else 'inactiva'}",
                (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (0, 255, 0) if hay_balanza else (100, 100, 100), 2)

    # Barra de gyro
    idx_imu  = np.argmin(np.abs(t_imu - seg))
    gyro_val = min(gyro_mag[idx_imu], 150)
    bar_w    = int((gyro_val / 150) * 300)
    bar_col  = (0, 0, 255) if gyro_val > umbral else (0, 200, 100)
    cv2.rectangle(frame, (10, 168), (10 + bar_w, 188), bar_col, -1)
    cv2.rectangle(frame, (10, 168), (310, 188), (180,180,180), 1)
    cv2.putText(frame, f"IMU: {gyro_val:.0f}deg/s",
                (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)

    # Marcar ROI de balanza
    cv2.rectangle(frame, (X1, Y1), (X2, Y2),
                  (0, 255, 0) if hay_balanza else (50, 50, 50), 2)

    # Flash rojo en borde cuando está desmontando
    if estado_actual == "DESMONTANDO":
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 0, 255), 6)
    elif estado_actual == "EXCAVANDO":
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (0, 165, 255), 4)

    out.write(frame)

    if frame_num % 500 == 0:
        print(f"  {frame_num}/{total} frames ({seg/60:.1f} min)...")

cap.release()
out.release()
print(f"\n✅ Video guardado: {OUTPUT_PATH}")
print(f"   Ábrelo con: vlc {OUTPUT_PATH}")