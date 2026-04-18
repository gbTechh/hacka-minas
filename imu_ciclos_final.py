import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

data = np.load("imu.npy", allow_pickle=True)
timestamps = data[:, 0]
t = (timestamps - timestamps[0]) / 1e9

gyro_mag  = np.sqrt(data[:,4]**2 + data[:,5]**2 + data[:,6]**2)
accel_mag = np.sqrt(data[:,1]**2 + data[:,2]**2 + data[:,3]**2)

gyro_suave = uniform_filter1d(gyro_mag, size=10)

# ── Subir umbral y aumentar distancia mínima entre ciclos ──
# Un ciclo real dura >20s → distancia mínima = 200 muestras (20s * 10Hz)
picos, props = find_peaks(
    gyro_suave,
    height=28,        # subimos umbral (antes era 21)
    distance=200,     # mínimo 20 segundos entre ciclos reales
    prominence=12     # debe destacar claramente
)

print(f"Ciclos reales estimados: {len(picos)}")
print()

registros = []
for i, p in enumerate(picos):
    seg = t[p]
    dur = t[picos[i+1]] - seg if i < len(picos)-1 else None
    
    # Buscar impactos dentro de este ciclo (±15s del pico de giro)
    inicio = max(0, p - 150)
    fin    = min(len(t)-1, p + 150)
    impactos_ciclo = np.sum(accel_mag[inicio:fin] > 20)
    
    registros.append({
        "ciclo":           i + 1,
        "tiempo_seg":      round(seg, 1),
        "tiempo_str":      f"{int(seg//60):02d}:{int(seg%60):02d}",
        "gyro_peak":       round(gyro_suave[p], 1),
        "duracion_seg":    round(dur, 1) if dur else None,
        "impactos_nearby": int(impactos_ciclo),
        "intensidad":      "FUERTE" if gyro_suave[p] > 45 else "NORMAL"
    })

df = pd.DataFrame(registros)
df.to_csv("ciclos_finales.csv", index=False)

# ── Imprimir tabla ──
print(f"{'Ciclo':>6} {'Tiempo':>8} {'Gyro°/s':>8} {'Duración':>10} {'Intensidad':>10}")
print("-" * 50)
for _, r in df.iterrows():
    dur_str = f"{r['duracion_seg']}s" if r['duracion_seg'] else "---"
    print(f"  {int(r['ciclo']):>4}   {r['tiempo_str']:>6}   {r['gyro_peak']:>6.1f}   "
          f"{dur_str:>8}   {r['intensidad']:>10}")

durs = df["duracion_seg"].dropna()
print(f"\n=== RESUMEN ===")
print(f"Total ciclos reales:   {len(picos)}")
print(f"Duración promedio:     {durs.mean():.1f}s")
print(f"Ciclo más rápido:      {durs.min():.1f}s")
print(f"Ciclo más lento:       {durs.max():.1f}s")
print(f"Ciclos por hora:       {3600 / durs.mean():.0f}")
print(f"Tiempo total activo:   {durs.sum():.0f}s de {t[-1]:.0f}s totales")
print(f"Eficiencia operativa:  {100*durs.sum()/t[-1]:.1f}%")

# ── Gráfica limpia ──
fig, ax = plt.subplots(figsize=(18, 5))
ax.plot(t, gyro_mag,   color="lightgreen", alpha=0.3, linewidth=0.7)
ax.plot(t, gyro_suave, color="darkgreen",  linewidth=1.5)
ax.axhline(28, color="red", linestyle="--", alpha=0.5, label="Umbral 28°/s")

for i, p in enumerate(picos):
    color = "red" if gyro_suave[p] > 45 else "orange"
    ax.scatter(t[p], gyro_suave[p], color=color, s=150, zorder=5)
    ax.annotate(f"C{i+1}\n{int(t[p]//60):02d}:{int(t[p]%60):02d}",
                (t[p], gyro_suave[p]+2), ha="center", fontsize=7)

from matplotlib.lines import Line2D
leyenda = [
    Line2D([0],[0], marker='o', color='w', markerfacecolor='red',    markersize=10, label='Ciclo fuerte (>45°/s)'),
    Line2D([0],[0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Ciclo normal'),
]
ax.legend(handles=leyenda)
ax.set_xlabel("Tiempo (segundos)")
ax.set_ylabel("Magnitud Gyro (°/s)")
ax.set_title(f"Ciclos de Carga Reales — {len(picos)} ciclos en 15 minutos")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("ciclos_finales.png", dpi=150, bbox_inches="tight")
print("\n✅ Guardado: ciclos_finales.png y ciclos_finales.csv")