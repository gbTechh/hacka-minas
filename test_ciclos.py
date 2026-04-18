import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

imu      = np.load("imu.npy", allow_pickle=True)
t_imu    = (imu[:, 0] - imu[0, 0]) / 1e9
gyro_mag = np.sqrt(imu[:,4]**2 + imu[:,5]**2 + imu[:,6]**2)
gyro_suave = uniform_filter1d(gyro_mag, size=10)

# Probar distintos umbrales
for umbral_std in [1.0, 1.2, 1.5, 2.0]:
    umbral = gyro_suave.mean() + gyro_suave.std() * umbral_std
    picos, _ = find_peaks(gyro_suave, height=umbral, distance=200, prominence=8)
    print(f"  std={umbral_std}  umbral={umbral:.1f}°/s  ciclos={len(picos)}")