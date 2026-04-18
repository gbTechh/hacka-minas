import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

imu = np.load("imu.npy", allow_pickle=True)
t   = (imu[:, 0] - imu[0, 0]) / 1e9

accel_x = imu[:, 1]
accel_y = imu[:, 2]
accel_z = imu[:, 3]
gyro_x  = imu[:, 4]
gyro_y  = imu[:, 5]
gyro_z  = imu[:, 6]

accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)
gyro_mag  = np.sqrt(gyro_x**2  + gyro_y**2  + gyro_z**2)

# Zoom en el primer ciclo completo: 0 a 60 segundos
mask = (t >= 0) & (t <= 60)
t_z  = t[mask]
gx   = gyro_x[mask]
gy   = gyro_y[mask]
gz   = gyro_z[mask]
gm   = gyro_mag[mask]
am   = accel_mag[mask]

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

axes[0].plot(t_z, gx, color='orange', label='Gyro X (giro horizontal)', linewidth=1.2)
axes[0].axhline(0, color='white', linewidth=0.5)
axes[0].set_ylabel("Gyro X (°/s)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_title("Ciclo 1 — Señales IMU detalladas (0-60s)")

axes[1].plot(t_z, gz, color='cyan', label='Gyro Z (cabeceo)', linewidth=1.2)
axes[1].set_ylabel("Gyro Z (°/s)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(t_z, am, color='purple', label='Accel magnitud', linewidth=1.2)
axes[2].set_ylabel("Accel (m/s²)")
axes[2].set_xlabel("Tiempo (segundos)")
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ciclo1_detalle.png", dpi=150)
print("✅ Guardado: ciclo1_detalle.png")