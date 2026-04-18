import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

data = np.load("imu.npy", allow_pickle=True)

timestamps = data[:, 0]
t = (timestamps - timestamps[0]) / 1e9

gyro_x  = data[:, 4]
gyro_y  = data[:, 5]
gyro_z  = data[:, 6]
accel_x = data[:, 1]
accel_y = data[:, 2]
accel_z = data[:, 3]

gyro_mag  = np.sqrt(gyro_x**2 + gyro_y**2 + gyro_z**2)
accel_mag = np.sqrt(accel_x**2 + accel_y**2 + accel_z**2)

# ── Suavizar señal para detectar eventos (ventana ~2 segundos) ──
ventana = 20   # 20 muestras * 0.1s = 2 segundos
gyro_suave  = uniform_filter1d(gyro_mag, size=ventana)
accel_suave = uniform_filter1d(accel_mag, size=ventana)

# ── Detectar picos de giro (ciclos de carga) ──
# umbral: picos que superen 30 °/s sostenido, separados por al menos 10 segundos
picos_gyro, props = find_peaks(
    gyro_suave,
    height=30,          # umbral mínimo °/s
    distance=100,       # al menos 100 muestras (~10s) entre picos
    prominence=15       # debe destacar sobre el ruido
)

# ── Detectar picos de impacto (excavación) ──
picos_accel, _ = find_peaks(
    accel_suave,
    height=15,
    distance=50,
    prominence=5
)

print(f"=== CICLOS DETECTADOS ===")
print(f"Eventos de giro (cargas): {len(picos_gyro)}")
print(f"Eventos de impacto:       {len(picos_accel)}")
print()
print("Tiempos de cada ciclo de carga:")
for i, p in enumerate(picos_gyro):
    seg = t[p]
    mm  = int(seg // 60)
    ss  = int(seg % 60)
    print(f"  Ciclo {i+1:2d} → t={seg:6.1f}s  ({mm:02d}:{ss:02d})  gyro={gyro_suave[p]:.1f} °/s")

# ── Calcular duración entre ciclos ──
if len(picos_gyro) > 1:
    tiempos_ciclo = np.diff(t[picos_gyro])
    print(f"\n=== DURACIÓN ENTRE CICLOS ===")
    print(f"  Promedio: {tiempos_ciclo.mean():.1f}s")
    print(f"  Mínimo:   {tiempos_ciclo.min():.1f}s")
    print(f"  Máximo:   {tiempos_ciclo.max():.1f}s")

# ── Graficar con ciclos marcados ──
fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

axes[0].plot(t, gyro_mag,   color="lightgreen", alpha=0.5, linewidth=0.8)
axes[0].plot(t, gyro_suave, color="darkgreen",  linewidth=1.5, label="Suavizado")
axes[0].scatter(t[picos_gyro], gyro_suave[picos_gyro],
                color="red", zorder=5, s=100, label=f"{len(picos_gyro)} ciclos")
for p in picos_gyro:
    axes[0].axvline(t[p], color="red", alpha=0.3, linewidth=1)
axes[0].set_ylabel("Magnitud Gyro (°/s)")
axes[0].set_title("Detección de Ciclos de Carga")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, accel_mag,   color="plum",   alpha=0.5, linewidth=0.8)
axes[1].plot(t, accel_suave, color="purple", linewidth=1.5, label="Suavizado")
axes[1].scatter(t[picos_accel], accel_suave[picos_accel],
                color="orange", zorder=5, s=80, label=f"{len(picos_accel)} impactos")
axes[1].set_ylabel("Magnitud Accel (m/s²)")
axes[1].set_title("Detección de Impactos (Excavación)")
axes[1].set_xlabel("Tiempo (segundos)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ciclos_detectados.png", dpi=150, bbox_inches="tight")
plt.show()
print("\n✅ Guardado: ciclos_detectados.png")