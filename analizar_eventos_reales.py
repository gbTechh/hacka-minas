import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

imu    = np.load("imu.npy", allow_pickle=True)
t_imu  = (imu[:, 0] - imu[0, 0]) / 1e9
gyro_x = imu[:, 4]
gyro_y = imu[:, 5]
gyro_z = imu[:, 6]
gyro_mag = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
accel_x = imu[:, 1]
accel_y = imu[:, 2]
accel_z = imu[:, 3]
accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

# Zoom en los primeros 35 segundos
mask = (t_imu >= 0) & (t_imu <= 35)
t_z  = t_imu[mask]
gx_z = gyro_x[mask]
gm_z = gyro_mag[mask]
am_z = accel_mag[mask]

fig, axes = plt.subplots(3, 1, figsize=(16, 10), sharex=True)

# Gyro X
axes[0].plot(t_z, gx_z, color='orange', linewidth=1.2)
axes[0].set_ylabel("Gyro X (°/s)")
axes[0].grid(True, alpha=0.3)

# Gyro magnitud
axes[1].plot(t_z, gm_z, color='green', linewidth=1.2)
axes[1].set_ylabel("Gyro Mag (°/s)")
axes[1].grid(True, alpha=0.3)

# Accel magnitud
axes[2].plot(t_z, am_z, color='purple', linewidth=1.2)
axes[2].set_ylabel("Accel Mag (m/s²)")
axes[2].set_xlabel("Tiempo (s)")
axes[2].grid(True, alpha=0.3)

# Marcar los eventos conocidos
eventos = [
    (4,  "DESCARGA", "red"),
    (7,  "GIRA→MON", "blue"),
    (12, "EXCAVA",   "orange"),
    (21, "GIRA→CAM", "blue"),
    (25, "DESCARGA", "red"),
]
for t_ev, label, color in eventos:
    for ax in axes:
        ax.axvline(t_ev, color=color, linestyle='--', alpha=0.8)
    axes[0].annotate(label, (t_ev, axes[0].get_ylim()[1]*0.8),
                    ha='center', fontsize=8, color=color)

plt.suptitle("IMU vs eventos reales del video (primeros 35s)", fontsize=12)
plt.tight_layout()
plt.savefig("imu_vs_eventos.png", dpi=150)
print("✅ Guardado: imu_vs_eventos.png")