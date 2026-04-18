import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

# ── Ciclos IMU ──
imu        = np.load("imu.npy", allow_pickle=True)
t_imu      = (imu[:, 0] - imu[0, 0]) / 1e9
gyro_mag   = np.sqrt(imu[:,4]**2 + imu[:,5]**2 + imu[:,6]**2)
gyro_suave = uniform_filter1d(gyro_mag, size=10)
umbral     = gyro_suave.mean() + gyro_suave.std() * 1.2
picos, _   = find_peaks(gyro_suave, height=umbral, distance=200, prominence=8)
t_ciclos   = t_imu[picos]
dur_ciclos = np.diff(t_ciclos, append=t_ciclos[-1] + 40)

# ── Detectar pausas entre ciclos ──
gaps = np.diff(t_ciclos)
print("Gaps entre ciclos consecutivos (segundos):")
for i, g in enumerate(gaps):
    t1 = f"{int(t_ciclos[i]//60):02d}:{int(t_ciclos[i]%60):02d}"
    t2 = f"{int(t_ciclos[i+1]//60):02d}:{int(t_ciclos[i+1]%60):02d}"
    marca = " ← PAUSA LARGA (posible cambio camión)" if g > 60 else ""
    print(f"  Ciclo {i+1:>2} → {i+2:>2}  ({t1} → {t2})  gap={g:.0f}s{marca}")

# ── Graficar gaps ──
plt.figure(figsize=(14, 4))
plt.bar(range(len(gaps)), gaps, color=['red' if g > 60 else 'steelblue' for g in gaps])
plt.axhline(60, color='orange', linestyle='--', label='60s (umbral cambio camión)')
plt.xlabel("Entre ciclo N y N+1")
plt.ylabel("Gap (segundos)")
plt.title("Gaps entre ciclos IMU — detectar cambio de camión")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("gaps_ciclos.png", dpi=150)
print("\n✅ Guardado: gaps_ciclos.png")