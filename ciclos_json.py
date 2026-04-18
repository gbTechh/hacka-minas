import numpy as np
import json
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

imu      = np.load("imu.npy", allow_pickle=True)
t_imu    = (imu[:, 0] - imu[0, 0]) / 1e9
gyro_mag = np.sqrt(imu[:,4]**2 + imu[:,5]**2 + imu[:,6]**2)
gm_suave = uniform_filter1d(gyro_mag, size=10)

umbral   = gm_suave.mean() + gm_suave.std() * 1.2
picos, _ = find_peaks(gm_suave, height=umbral, distance=200, prominence=8)
t_ciclos  = t_imu[picos]
gm_picos  = gm_suave[picos]
dur_ciclos = np.diff(t_ciclos, append=t_ciclos[-1] + 40)

# ── Detectar lado por señal de balanza ──
# Balanza visible = lado DERECHO
senal  = np.load("senal_balanza.npy")
t_bal  = senal[:, 0]
px_bal = senal[:, 1]

# Para cada ciclo, verificar si hay balanza activa en ±60s alrededor del pico
def lado_por_balanza(t_pico):
    mask    = (t_bal >= t_pico - 60) & (t_bal <= t_pico + 60)
    px_zona = px_bal[mask]
    if len(px_zona) == 0:
        return "DESCONOCIDO"
    pct_activo = np.mean(px_zona > 100)
    return "DERECHO" if pct_activo > 0.05 else "IZQUIERDO"

# ── Construir ciclos ──
ciclos = []
for i, (t, dur, gm) in enumerate(zip(t_ciclos, dur_ciclos, gm_picos)):
    lado = lado_por_balanza(t)
    ciclos.append({
        "ciclo":        i + 1,
        "tiempo":       f"{int(t//60):02d}:{int(t%60):02d}",
        "tiempo_seg":   round(float(t), 1),
        "duracion_seg": round(float(dur), 1),
        "gyro_peak":    round(float(gm), 1),
        "lado":         lado,
        "intensidad":   "FUERTE" if gm > 45 else "NORMAL"
    })

print("Ciclos por lado detectados:")
for c in ciclos:
    print(f"  C{c['ciclo']:>2}: {c['tiempo']}  {c['lado']:>10}  {c['duracion_seg']}s")

# ── Estadísticas ──
durs = np.array([c["duracion_seg"] for c in ciclos[:-1]])

for lado in ["DERECHO", "IZQUIERDO"]:
    sub = [c for c in ciclos if c["lado"] == lado]
    dur_sub = [c["duracion_seg"] for c in sub[:-1]] if len(sub) > 1 else \
              [c["duracion_seg"] for c in sub]
    print(f"\n{lado}: {len(sub)} ciclos, "
          f"prom={np.mean(dur_sub):.1f}s" if dur_sub else f"\n{lado}: {len(sub)} ciclos")

resultado = {
    "resumen": {
        "duracion_total_min":  round(float(t_imu[-1])/60, 1),
        "total_ciclos":        len(ciclos),
        "ciclos_por_hora":     round(3600/float(durs.mean()), 1),
        "duracion_prom_seg":   round(float(durs.mean()), 1),
        "duracion_min_seg":    round(float(durs.min()), 1),
        "duracion_max_seg":    round(float(durs.max()), 1),
        "ciclos_fuerte":       sum(1 for c in ciclos if c["intensidad"] == "FUERTE"),
        "ciclos_normal":       sum(1 for c in ciclos if c["intensidad"] == "NORMAL"),
    },
    "por_lado": {},
    "ciclos": ciclos
}

for lado in ["DERECHO", "IZQUIERDO"]:
    sub     = [c for c in ciclos if c["lado"] == lado]
    dur_sub = [c["duracion_seg"] for c in sub]
    resultado["por_lado"][lado] = {
        "total_ciclos":      len(sub),
        "duracion_prom_seg": round(float(np.mean(dur_sub)), 1) if dur_sub else 0,
        "duracion_min_seg":  round(float(np.min(dur_sub)),  1) if dur_sub else 0,
        "duracion_max_seg":  round(float(np.max(dur_sub)),  1) if dur_sub else 0,
    }

with open("ciclos_imu.json", "w") as f:
    json.dump(resultado, f, indent=2, ensure_ascii=False)

print("\n" + json.dumps(resultado, indent=2, ensure_ascii=False))
print("\n✅ Guardado: ciclos_imu.json")