import numpy as np
from scipy.ndimage import uniform_filter1d

imu      = np.load("imu.npy", allow_pickle=True)
t_imu    = (imu[:, 0] - imu[0, 0]) / 1e9
gyro_mag = np.sqrt(imu[:,4]**2 + imu[:,5]**2 + imu[:,6]**2)
gyro_x   = imu[:, 4]

gm_suave = uniform_filter1d(gyro_mag, size=8)
gx_suave = uniform_filter1d(gyro_x,   size=8)

print("Señal IMU segundo a segundo (0-30s):")
print(f"{'t':>5} {'gm':>8} {'gx':>8}  fase_real")
print("-" * 45)

fases_reales = {
    range(0,4):   "GIRANDO→CAMION",
    range(4,7):   "DESCARGANDO",
    range(7,12):  "GIRANDO→MONTICULO",
    range(12,21): "EXCAVANDO",
    range(21,25): "GIRANDO→CAMION",
    range(25,30): "DESCARGANDO",
}

for t_s in range(0, 30):
    idx = np.argmin(np.abs(t_imu - t_s))
    gm  = gm_suave[idx]
    gx  = gx_suave[idx]
    fase = "?"
    for r, f in fases_reales.items():
        if t_s in r:
            fase = f
            break
    print(f"{t_s:>5}s  {gm:>7.1f}  {gx:>7.1f}  {fase}")