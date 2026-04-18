import cv2
import numpy as np
import pandas as pd

VIDEO_PATH  = "videor.mp4"   # ← cambia esto
IMU_PATH    = "imu.npy"
CICLOS_PATH = "ciclos_finales.csv"
OUTPUT_PATH = "video_con_imu.avi"

# ── Cargar datos ──
imu_data = np.load(IMU_PATH, allow_pickle=True)
df_ciclos = pd.read_csv(CICLOS_PATH)

timestamps = imu_data[:, 0]
t = (timestamps - timestamps[0]) / 1e9
gyro_mag  = np.sqrt(imu_data[:,4]**2 + imu_data[:,5]**2 + imu_data[:,6]**2)
accel_mag = np.sqrt(imu_data[:,1]**2 + imu_data[:,2]**2 + imu_data[:,3]**2)

# Índices de ciclos para marcarlos en el video
ciclo_indices = set()
for _, row in df_ciclos.iterrows():
    idx = np.argmin(np.abs(t - row["tiempo_seg"]))
    ciclo_indices.add(idx)

# ── Abrir video ──
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(OUTPUT_PATH,
      cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))

frame_idx = 0
print("Procesando video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Datos IMU de este frame exacto
    if frame_idx < len(imu_data):
        gyro  = gyro_mag[frame_idx]
        accel = accel_mag[frame_idx]
        t_seg = t[frame_idx]
        t_str = f"{int(t_seg//60):02d}:{int(t_seg%60):02d}"

        # ── Detectar si estamos cerca de un ciclo ──
        en_ciclo = any(abs(frame_idx - ci) < 30 for ci in ciclo_indices)

        # ── Barra de gyro (indicador de actividad) ──
        bar_max   = 200
        bar_w     = int((gyro / bar_max) * 200)
        bar_color = (0, 0, 255) if gyro > 45 else (0, 200, 100)
        cv2.rectangle(frame, (10, h-60), (10 + bar_w, h-40), bar_color, -1)
        cv2.rectangle(frame, (10, h-60), (210, h-40), (200,200,200), 1)

        # ── Panel de información ──
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (320, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        color_ciclo = (0, 100, 255) if en_ciclo else (200, 200, 200)
        estado      = ">>> CICLO DE CARGA <<<" if en_ciclo else "operando..."

        cv2.putText(frame, f"Tiempo:  {t_str}",        (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
        cv2.putText(frame, f"Gyro:    {gyro:.1f} deg/s", (10, 50),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, bar_color, 1)
        cv2.putText(frame, f"Accel:   {accel:.1f} m/s2", (10, 75),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
        cv2.putText(frame, estado,                       (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color_ciclo, 2)

        # ── Flash rojo en bordes cuando hay ciclo ──
        if en_ciclo:
            cv2.rectangle(frame, (0,0), (w-1, h-1), (0,0,255), 6)

    out.write(frame)
    frame_idx += 1

    if frame_idx % 300 == 0:
        print(f"  {frame_idx}/{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))} frames...")

cap.release()
out.release()
print(f"\n✅ Video guardado: {OUTPUT_PATH}")
print("Ábrelo con: vlc video_con_imu.mp4")