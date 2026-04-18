import numpy as np
import cv2

imu   = np.load("imu.npy", allow_pickle=True)
t_imu = (imu[:, 0] - imu[0, 0]) / 1e9

cap = cv2.VideoCapture("videor.mp4")
fps   = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()

print(f"=== VIDEO ===")
print(f"Frames:    {total}")
print(f"FPS:       {fps}")
print(f"Duración:  {total/fps:.2f}s")

print(f"\n=== IMU ===")
print(f"Muestras:  {len(imu)}")
print(f"Duración:  {t_imu[-1]:.2f}s")
print(f"Frecuencia real: {len(imu)/t_imu[-1]:.2f} Hz")

print(f"\n=== SINCRONIZACIÓN ===")
print(f"Frames == Muestras: {total == len(imu)}")

# Dos hipótesis de sincronización:
# H1: índice a índice (frame N = muestra N)
# H2: tiempo a tiempo (t_video → buscar muestra IMU más cercana)

print(f"\nH1 — Índice a índice:")
print(f"  Frame 60  (t_video={60/fps:.1f}s) → IMU muestra 60 (t_imu={t_imu[60]:.1f}s)")
print(f"  Frame 600 (t_video={600/fps:.1f}s) → IMU muestra 600 (t_imu={t_imu[600]:.1f}s)")

print(f"\nH2 — Tiempo a tiempo:")
for t_vid in [4, 7, 12, 21, 25]:
    idx = np.argmin(np.abs(t_imu - t_vid))
    print(f"  t_video={t_vid}s → IMU idx={idx} (t_imu={t_imu[idx]:.1f}s)")

# Factor de escala
factor = t_imu[-1] / (total/fps)
print(f"\nFactor de escala IMU/Video: {factor:.4f}")
print(f"  → Para t_video=4s, t_imu equivalente = {4*factor:.1f}s")
print(f"  → Para t_video=12s, t_imu equivalente = {12*factor:.1f}s")