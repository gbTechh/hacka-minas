import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

imu = np.load("imu.npy", allow_pickle=True)
t   = (imu[:, 0] - imu[0, 0]) / 1e9

gyro_x    = imu[:, 4]
accel_mag = np.sqrt(imu[:,1]**2 + imu[:,2]**2 + imu[:,3]**2)

# Suavizar
gx_suave = uniform_filter1d(gyro_x,    size=5)
am_suave = uniform_filter1d(accel_mag, size=5)

# Detectar descargas: pico NEGATIVO fuerte de gyro_x
# (cuando la pala gira bruscamente al soltar tierra)
picos_neg, _ = find_peaks(-gx_suave, height=30, distance=80, prominence=20)
picos_pos, _ = find_peaks( gx_suave, height=30, distance=80, prominence=20)

print(f"Descargas detectadas (gyro negativo): {len(picos_neg)}")
print(f"Excavaciones detectadas (gyro positivo): {len(picos_pos)}")

mask = (t >= 0) & (t <= 120)
t_z  = t[mask]
gx_z = gx_suave[mask]
am_z = am_suave[mask]

fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

axes[0].plot(t_z, gx_z, color='orange', linewidth=1.2)
axes[0].axhline(0, color='white', linewidth=0.5, alpha=0.5)
axes[0].axhline( 30, color='green', linestyle='--', alpha=0.5, label='Umbral excavar (+30)')
axes[0].axhline(-30, color='red',   linestyle='--', alpha=0.5, label='Umbral descargar (-30)')

# Marcar descargas
for p in picos_neg:
    if t[p] <= 120:
        axes[0].axvline(t[p], color='red',   alpha=0.7, linewidth=2)
        axes[0].annotate('⬇ DESCARGA', (t[p], gx_suave[p]-5),
                        ha='center', fontsize=7, color='red')
# Marcar excavaciones
for p in picos_pos:
    if t[p] <= 120:
        axes[0].axvline(t[p], color='green', alpha=0.7, linewidth=2)
        axes[0].annotate('⬆ EXCAVA', (t[p], gx_suave[p]+2),
                        ha='center', fontsize=7, color='green')

axes[0].set_ylabel("Gyro X (°/s)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_title("Detección de fases: Excavar vs Descargar (primeros 2 minutos)")

axes[1].plot(t_z, am_z, color='purple', linewidth=1.2)
axes[1].set_ylabel("Accel magnitud (m/s²)")
axes[1].set_xlabel("Tiempo (segundos)")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("fases_detalle.png", dpi=150)
print("✅ Guardado: fases_detalle.png")