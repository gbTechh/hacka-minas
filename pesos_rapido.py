import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

# ── Cargar pesos ya calculados ──
df = pd.read_csv("cargas_final.csv")
print("Pesos ya calculados:")
print(df.to_string(index=False))

# ── Ciclos IMU ──
imu      = np.load("imu.npy", allow_pickle=True)
t_imu    = (imu[:, 0] - imu[0, 0]) / 1e9
gyro_mag = np.sqrt(imu[:,4]**2 + imu[:,5]**2 + imu[:,6]**2)
gm_suave = uniform_filter1d(gyro_mag, size=8)
umbral   = gm_suave.mean() + gm_suave.std() * 1.2
picos, _ = find_peaks(gm_suave, height=umbral, distance=200, prominence=8)
t_ciclos = t_imu[picos]

# ── Separar por camiones ──
df['tiempo_seg'] = df['tiempo'].apply(
    lambda t: int(t.split(":")[0])*60 + int(t.split(":")[1])
)

camiones = []
camion   = [df.iloc[0].to_dict()]
for i in range(1, len(df)):
    row = df.iloc[i]
    if row['nuevo_camion'] == True:
        camiones.append(camion)
        camion = []
    camion.append(row.to_dict())
camiones.append(camion)

DELTA_PROM = 38

print(f"\n{'='*55}")
print(f"  REPORTE DE PESOS POR CAMIÓN")
print(f"{'='*55}")

resumen = []
for i, cam in enumerate(camiones):
    gdf      = pd.DataFrame(cam)
    t_ini    = gdf['tiempo_seg'].min()
    t_fin    = gdf['tiempo_seg'].max()
    n_ciclos = int(np.sum((t_ciclos >= t_ini-30) & (t_ciclos <= t_fin+30)))
    deltas   = gdf[gdf['delta_t'] > 5]['delta_t'].tolist()
    paladas  = gdf['peso_t'].tolist()
    prom     = int(np.mean(deltas)) if deltas else DELTA_PROM
    p_final  = int(gdf['peso_t'].max())
    p_total  = p_final + prom  # último OCR + 1 palada estimada

    print(f"\n🚛 CAMIÓN {i+1}  ({gdf.iloc[0]['tiempo']} → {gdf.iloc[-1]['tiempo']})")
    print(f"   Paladas detectadas: {len(paladas)}")
    print(f"   Ciclos IMU:         {n_ciclos}")
    print(f"   Pesos:              {paladas}")
    print(f"   Promedio/palada:    ~{prom}t")
    print(f"   Último peso OCR:    {p_final}t")
    print(f"   Peso total est.:    ~{p_total}t")

    resumen.append({
        "camion": i+1, "paladas": len(paladas),
        "ciclos_imu": n_ciclos, "prom_palada_t": prom,
        "ultimo_ocr_t": p_final, "peso_total_est_t": p_total
    })

pd.DataFrame(resumen).to_csv("pesos_por_camion.csv", index=False)
print(f"\n{'='*55}")
print(f"✅ Guardado: pesos_por_camion.csv")