import numpy as np
from scipy.ndimage import uniform_filter1d

imu = np.load("imu.npy", allow_pickle=True)
t_imu = (imu[:, 0] - imu[0, 0]) / 1e9

# Cuaterniones = orientación absoluta
qw = imu[:, 7]
qx = imu[:, 8]
qy = imu[:, 9]
qz = imu[:, 10]

# Convertir cuaternión a ángulo de yaw (rotación horizontal)
# yaw = atan2(2*(qw*qz + qx*qy), 1 - 2*(qy^2 + qz^2))
yaw = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
yaw_deg = np.degrees(yaw)

yaw_suave = uniform_filter1d(yaw_deg, size=20)

print("Yaw (orientación horizontal) en eventos conocidos:")
print(f"{'t':>6} {'yaw':>8}  evento")
print("-" * 35)
for t_ev in [0, 30, 60, 90, 120, 150, 180, 210, 240, 300, 360, 420, 480, 540, 600, 660]:
    idx = np.argmin(np.abs(t_imu - t_ev))
    print(f"{t_ev:>6}s  {yaw_suave[idx]:>7.1f}°")

import matplotlib.pyplot as plt
plt.figure(figsize=(16,4))
plt.plot(t_imu/60, yaw_suave, color='blue', linewidth=1)
plt.xlabel("Tiempo (min)")
plt.ylabel("Yaw (°)")
plt.title("Orientación horizontal (yaw) — detectar lado de giro")
plt.grid(True, alpha=0.3)
plt.axvline(2.25, color='red', linestyle='--', label='cambio lado conocido')
plt.legend()
plt.savefig("yaw_orientacion.png", dpi=150)
print("\n✅ Guardado: yaw_orientacion.png")