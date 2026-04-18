import numpy as np
from scipy.ndimage import uniform_filter1d

imu      = np.load("imu.npy", allow_pickle=True)
t_imu    = (imu[:, 0] - imu[0, 0]) / 1e9
gyro_x   = imu[:, 4]
gyro_mag = np.sqrt(imu[:,4]**2 + imu[:,5]**2 + imu[:,6]**2)
accel_mag = np.sqrt(imu[:,1]**2 + imu[:,2]**2 + imu[:,3]**2)

gm_suave = uniform_filter1d(gyro_mag,  size=5)
gx_suave = uniform_filter1d(gyro_x,    size=5)
am_suave = uniform_filter1d(accel_mag, size=5)

# Eventos reales del video con su t_video
eventos = [
    ( 4, "DESCARGANDO"),
    ( 7, "GIRANDO→MONTICULO"),
    (12, "EXCAVANDO"),
    (21, "GIRANDO→CAMION"),
    (25, "DESCARGANDO"),
]

print(f"{'t_video':>8} {'Fase real':>20} {'gyro_mag':>10} {'gyro_x':>8} {'accel':>8}")
print("-" * 60)
for t_vid, fase_real in eventos:
    idx = np.argmin(np.abs(t_imu - t_vid))
    gm  = gm_suave[idx]
    gx  = gx_suave[idx]
    am  = am_suave[idx]
    print(f"{t_vid:>8}s {fase_real:>20}  {gm:>8.1f}°/s  {gx:>6.1f}°/s  {am:>6.1f}m/s²")