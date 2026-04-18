import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

# ── Suavizado más fino (1 segundo en vez de 2) ──
gyro_suave  = uniform_filter1d(gyro_mag,  size=10)
accel_suave = uniform_filter1d(accel_mag, size=5)   # menos suavizado para accel

# ── Umbral dinámico (media + 1 desviación estándar) ──
umbral_gyro  = gyro_suave.mean()  + gyro_suave.std()  * 1.2
umbral_accel = accel_mag.mean()   + accel_mag.std()   * 2.0

print(f"Umbral gyro:  {umbral_gyro:.1f} °/s")
print(f"Umbral accel: {umbral_accel:.1f} m/s²")

# ── Detectar ciclos de giro ──
picos_gyro, _ = find_peaks(
    gyro_suave,
    height=umbral_gyro,
    distance=80,        # mínimo 8 segundos entre ciclos
    prominence=8
)

# ── Detectar impactos en señal CRUDA (no suavizada) ──
picos_accel, _ = find_peaks(
    accel_mag,
    height=umbral_accel,
    distance=30,        # mínimo 3 segundos entre impactos
    prominence=5
)

print(f"\n=== CICLOS DETECTADOS ===")
print(f"Ciclos de giro:  {len(picos_gyro)}")
print(f"Impactos:        {len(picos_accel)}")

# ── Tabla de ciclos con duración ──
registros = []
for i, p in enumerate(picos_gyro):
    seg = t[p]
    dur = t[picos_gyro[i+1]] - seg if i < len(picos_gyro)-1 else None
    registros.append({
        "ciclo":       i + 1,
        "tiempo_seg":  round(seg, 1),
        "tiempo_str":  f"{int(seg//60):02d}:{int(seg%60):02d}",
        "gyro_peak":   round(gyro_suave[p], 1),
        "dur_hasta_sig": round(dur, 1) if dur else None
    })
    print(f"  Ciclo {i+1:2d} → {registros[-1]['tiempo_str']}  "
          f"gyro={registros[-1]['gyro_peak']:5.1f}°/s  "
          f"dur={registros[-1]['dur_hasta_sig']}s")

# ── Guardar CSV ──
df_ciclos = pd.DataFrame(registros)
df_ciclos.to_csv("ciclos_imu.csv", index=False)
print(f"\n✅ CSV guardado: ciclos_imu.csv")

# ── Estadísticas ──
durs = df_ciclos["dur_hasta_sig"].dropna()
print(f"\n=== ESTADÍSTICAS ===")
print(f"Total ciclos:     {len(picos_gyro)}")
print(f"Duración promedio: {durs.mean():.1f}s  ({durs.mean()/60:.1f} min)")
print(f"Más rápido:        {durs.min():.1f}s")
print(f"Más lento:         {durs.max():.1f}s")
print(f"Ciclos por hora:   {3600 / durs.mean():.0f}")

# ── Gráfica final ──
fig, axes = plt.subplots(2, 1, figsize=(18, 8), sharex=True)

axes[0].plot(t, gyro_mag,   color="lightgreen", alpha=0.4, linewidth=0.7)
axes[0].plot(t, gyro_suave, color="darkgreen",  linewidth=1.5)
axes[0].axhline(umbral_gyro, color="red", linestyle="--", alpha=0.6, label=f"Umbral {umbral_gyro:.0f}°/s")
axes[0].scatter(t[picos_gyro], gyro_suave[picos_gyro],
                color="red", zorder=5, s=120, label=f"{len(picos_gyro)} ciclos")
for i, p in enumerate(picos_gyro):
    axes[0].annotate(f"C{i+1}", (t[p], gyro_suave[p]+3), ha="center", fontsize=7, color="red")
axes[0].set_ylabel("Magnitud Gyro (°/s)")
axes[0].set_title("Ciclos de Carga Detectados")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, accel_mag,   color="plum",   alpha=0.4, linewidth=0.7)
axes[1].plot(t, accel_suave, color="purple", linewidth=1.2)
axes[1].axhline(umbral_accel, color="orange", linestyle="--", alpha=0.7, label=f"Umbral {umbral_accel:.0f} m/s²")
axes[1].scatter(t[picos_accel], accel_mag[picos_accel],
                color="orange", zorder=5, s=80, label=f"{len(picos_accel)} impactos")
axes[1].set_ylabel("Magnitud Accel (m/s²)")
axes[1].set_title("Impactos / Excavación")
axes[1].set_xlabel("Tiempo (segundos)")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ciclos_v2.png", dpi=150, bbox_inches="tight")
print("✅ Gráfica guardada: ciclos_v2.png")