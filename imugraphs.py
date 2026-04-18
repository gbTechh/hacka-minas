import numpy as np
import matplotlib.pyplot as plt

data = np.load("imu.npy", allow_pickle=True)

timestamps = data[:, 0]
t = (timestamps - timestamps[0]) / 1e9   # segundos desde inicio

accel_x = data[:, 1]
accel_y = data[:, 2]
accel_z = data[:, 3]
gyro_x  = data[:, 4]
gyro_y  = data[:, 5]
gyro_z  = data[:, 6]

# Magnitud total de aceleración (detecta vibración/impacto)
accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

# Magnitud total de giro (detecta rotación de la pala)
gyro_mag  = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)

fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
fig.suptitle("IMU - Operador de Pala (15 minutos)", fontsize=14)

axes[0].plot(t, accel_x, color="red",    alpha=0.7, label="Accel X")
axes[0].plot(t, accel_y, color="green",  alpha=0.7, label="Accel Y")
axes[0].plot(t, accel_z, color="blue",   alpha=0.7, label="Accel Z")
axes[0].set_ylabel("Aceleración (m/s²)")
axes[0].legend(loc="upper right")
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, accel_mag, color="purple", linewidth=1.2)
axes[1].set_ylabel("Magnitud Accel")
axes[1].set_title("← Picos altos = impacto/excavación")
axes[1].grid(True, alpha=0.3)

axes[2].plot(t, gyro_x, color="orange", alpha=0.7, label="Gyro X")
axes[2].plot(t, gyro_y, color="cyan",   alpha=0.7, label="Gyro Y")
axes[2].plot(t, gyro_z, color="brown",  alpha=0.7, label="Gyro Z")
axes[2].set_ylabel("Rotación (°/s)")
axes[2].legend(loc="upper right")
axes[2].grid(True, alpha=0.3)

axes[3].plot(t, gyro_mag, color="darkgreen", linewidth=1.2)
axes[3].set_ylabel("Magnitud Gyro")
axes[3].set_title("← Picos altos = giro de pala (ciclo de carga)")
axes[3].set_xlabel("Tiempo (segundos)")
axes[3].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("imu_analisis.png", dpi=150, bbox_inches="tight")
plt.show()
print("✅ Gráfica guardada como imu_analisis.png")