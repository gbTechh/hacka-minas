import numpy as np
from scipy.ndimage import uniform_filter1d

imu      = np.load("imu.npy", allow_pickle=True)
t_imu    = (imu[:, 0] - imu[0, 0]) / 1e9
gyro_mag = np.sqrt(imu[:,4]**2 + imu[:,5]**2 + imu[:,6]**2)
gyro_x   = imu[:, 4]
accel_mag = np.sqrt(imu[:,1]**2 + imu[:,2]**2 + imu[:,3]**2)

gm_suave = uniform_filter1d(gyro_mag,  size=8)
gx_suave = uniform_filter1d(gyro_x,    size=8)
am_suave = uniform_filter1d(accel_mag, size=8)

print("Valores IMU en los primeros 5 segundos:")
print(f"{'t':>6} {'gyro_mag':>10} {'gyro_x':>8} {'accel':>8}")
print("-" * 38)
for i in range(len(t_imu)):
    if t_imu[i] > 5:
        break
    if i % 5 == 0:
        print(f"{t_imu[i]:>6.1f}s  {gm_suave[i]:>8.1f}°/s  "
              f"{gx_suave[i]:>6.1f}°/s  {am_suave[i]:>6.1f}m/s²")