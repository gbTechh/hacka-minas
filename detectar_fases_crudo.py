import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

imu = np.load("imu.npy", allow_pickle=True)
t      = (imu[:, 0] - imu[0, 0]) / 1e9
gyro_x = imu[:, 4]  # señal cruda sin suavizar

# Detectar en señal CRUDA
picos_descarga, props = find_peaks(
    -gyro_x,          # invertir para buscar mínimos negativos
    height=40,        # umbral más bajo
    distance=50,      # mínimo 5s entre descargas
    prominence=30
)

picos_excavar, _ = find_peaks(
    gyro_x,
    height=30,
    distance=50,
    prominence=20
)

print(f"Descargas detectadas:   {len(picos_descarga)}")
print(f"Excavaciones detectadas: {len(picos_excavar)}")
print("\nTiempos de descarga:")
for i, p in enumerate(picos_descarga):
    print(f"  Descarga {i+1:>2}: t={t[p]:>6.1f}s  "
          f"({int(t[p]//60):02d}:{int(t[p]%60):02d})  "
          f"gyro_x={gyro_x[p]:.1f}°/s")

# Graficar primeros 120s con señal cruda
mask = (t >= 0) & (t <= 120)
t_z  = t[mask]
gx_z = gyro_x[mask]

fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(t_z, gx_z, color='orange', linewidth=0.8, alpha=0.8, label='Gyro X crudo')
ax.axhline( 30, color='green', linestyle='--', alpha=0.5)
ax.axhline(-40, color='red',   linestyle='--', alpha=0.5)

for p in picos_descarga:
    if t[p] <= 120:
        ax.axvline(t[p], color='red', alpha=0.8, linewidth=2)
        ax.annotate(f'D{np.where(picos_descarga==p)[0][0]+1}',
                   (t[p], gyro_x[p]-8), ha='center', fontsize=8, color='red')

for p in picos_excavar:
    if t[p] <= 120:
        ax.axvline(t[p], color='green', alpha=0.5, linewidth=1.5)

ax.set_ylabel("Gyro X (°/s)")
ax.set_xlabel("Tiempo (segundos)")
ax.set_title("Señal CRUDA — Detección de descargas (primeros 2 min)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fases_crudo.png", dpi=150)
print("✅ Guardado: fases_crudo.png")

# Guardar timestamps de descargas
np.save("timestamps_descargas.npy", t[picos_descarga])
np.save("timestamps_excavar.npy",   t[picos_excavar])