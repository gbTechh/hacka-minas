import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

PESO_MAXIMO_CAMION = 246  # toneladas aproximadas
CICLOS_POR_CAMION  = 7    # aproximado

# ── Cargar datos ──
df_ocr = pd.read_csv("cargas_final.csv")
df_ocr['tiempo_seg'] = df_ocr['tiempo'].apply(
    lambda t: int(t.split(":")[0])*60 + int(t.split(":")[1])
)

# ── Ciclos IMU ──
imu        = np.load("imu.npy", allow_pickle=True)
t_imu      = (imu[:, 0] - imu[0, 0]) / 1e9
gyro_mag   = np.sqrt(imu[:,4]**2 + imu[:,5]**2 + imu[:,6]**2)
gyro_suave = uniform_filter1d(gyro_mag, size=10)
umbral     = gyro_suave.mean() + gyro_suave.std() * 1.2
picos, _   = find_peaks(gyro_suave, height=umbral, distance=200, prominence=8)
t_ciclos   = t_imu[picos]
dur_ciclos = np.diff(t_ciclos, append=t_ciclos[-1] + 40)

print(f"Ciclos IMU detectados: {len(t_ciclos)}")
print(f"Cargas OCR disponibles: {len(df_ocr)}")

# ── Construir todos los ciclos ──
ciclos = []
for i, (t_c, dur) in enumerate(zip(t_ciclos, dur_ciclos)):
    tiempo_str = f"{int(t_c//60):02d}:{int(t_c%60):02d}"
    cercanas = df_ocr[
        (df_ocr['tiempo_seg'] >= t_c - 60) &
        (df_ocr['tiempo_seg'] <= t_c + 60)
    ]
    if len(cercanas) > 0:
        mejor  = cercanas.iloc[-1]
        peso   = int(mejor['peso_t'])
        delta  = int(mejor['delta_t']) if mejor['delta_t'] > 0 else 40
        fuente = 'OCR'
    else:
        peso   = None
        delta  = None
        fuente = 'SIN_DATO'

    ciclos.append({
        "ciclo":        i + 1,
        "tiempo":       tiempo_str,
        "tiempo_seg":   round(float(t_c), 1),
        "duracion_seg": round(float(dur), 1),
        "peso_acum_t":  peso,
        "delta_t":      delta,
        "fuente":       fuente,
    })

df = pd.DataFrame(ciclos)

# ── Asignar camiones ──
# Lógica: cuando el peso BAJA respecto al anterior con OCR = nuevo camión
# También: cada ~7 ciclos sin OCR = posible nuevo camión
df['camion'] = 0
camion_actual  = 1
peso_max_visto = 0
ciclos_sin_ocr = 0

for i, row in df.iterrows():
    p = row['peso_acum_t']

    if p is not None:
        ciclos_sin_ocr = 0
        if p < peso_max_visto * 0.3 and peso_max_visto > 50:
            # Peso bajó drásticamente = nuevo camión
            camion_actual += 1
            peso_max_visto = p
        elif p > peso_max_visto:
            peso_max_visto = p
    else:
        ciclos_sin_ocr += 1

    df.at[i, 'camion'] = camion_actual

# Mostrar asignación inicial
print("\nAsignación de ciclos a camiones (basada en OCR):")
for cam in df['camion'].unique():
    sub = df[df['camion'] == cam]
    con_ocr = sub[sub['fuente'] == 'OCR']
    print(f"  Camión {cam}: ciclos {sub['ciclo'].min()}-{sub['ciclo'].max()} "
          f"({len(sub)} ciclos, {len(con_ocr)} con OCR)")

# ── Estimar pesos faltantes por camión ──
for cam in df['camion'].unique():
    mask = df['camion'] == cam
    sub  = df[mask]

    # Delta promedio con datos reales
    deltas_reales = sub['delta_t'].dropna()
    deltas_reales = deltas_reales[deltas_reales > 5]
    delta_prom = int(deltas_reales.mean()) if len(deltas_reales) > 0 else 40

    peso_ant = 0
    for i, row in sub.iterrows():
        if row['peso_acum_t'] is not None and row['peso_acum_t'] > 0:
            peso_ant = row['peso_acum_t']
            if pd.isna(row['delta_t']):
                df.at[i, 'delta_t'] = delta_prom
        else:
            peso_est = min(peso_ant + delta_prom, PESO_MAXIMO_CAMION)
            df.at[i, 'peso_acum_t'] = peso_est
            df.at[i, 'delta_t']     = delta_prom
            df.at[i, 'fuente']      = f'EST({delta_prom}t)'
            peso_ant = peso_est

# Rellenar NaN en delta_t
df['delta_t'] = df['delta_t'].fillna(40)

# ── Reporte final ──
print("\n" + "=" * 65)
print("     REPORTE COMPLETO POR CAMIÓN")
print("=" * 65)

resumen = []
for cam in df['camion'].unique():
    sub        = df[df['camion'] == cam].copy()
    n_ciclos   = len(sub)
    dur_prom   = sub['duracion_seg'].mean()
    peso_final = sub['peso_acum_t'].max()
    n_ocr      = len(sub[sub['fuente'] == 'OCR'])
    delta_prom = sub[sub['delta_t'] > 5]['delta_t'].mean()
    t_inicio   = sub.iloc[0]['tiempo']
    t_fin      = sub.iloc[-1]['tiempo']
    t_total    = sub['duracion_seg'].sum() / 60

    print(f"\n🚛 CAMIÓN {cam}  ({t_inicio} → {t_fin})")
    print(f"   Ciclos totales:      {n_ciclos}")
    print(f"   Ciclos con OCR real: {n_ocr}  {'✅' if n_ocr > 0 else '⚠️ estimado'}")
    print(f"   Tiempo operación:    {t_total:.1f} min")
    print(f"   Duración prom/ciclo: {dur_prom:.1f}s")
    print(f"   Ciclos por hora:     {3600/dur_prom:.0f}")
    print(f"   Peso final:          ~{int(peso_final)}t")
    print(f"   Delta prom/palada:   ~{int(delta_prom) if not np.isnan(delta_prom) else 40}t")
    print(f"\n   {'#':>4} {'Tiempo':>7} {'Dur':>6} {'Peso':>7} {'Delta':>7}  Fuente")
    print(f"   {'-'*55}")
    for _, r in sub.iterrows():
        delta_s = f"+{int(r['delta_t'])}t"
        print(f"   #{int(r['ciclo']):>3}  {r['tiempo']:>6}  "
              f"{r['duracion_seg']:>5.1f}s  {int(r['peso_acum_t']):>5}t  "
              f"{delta_s:>6}  {r['fuente']}")

    resumen.append({
        "camion": cam, "ciclos": n_ciclos, "ocr_real": n_ocr,
        "peso_final_t": int(peso_final), "dur_prom_s": round(dur_prom,1),
        "ciclos_hora": int(3600/dur_prom)
    })

print(f"\n{'='*65}")
print(f"RESUMEN EJECUTIVO")
print(f"{'='*65}")
df_res = pd.DataFrame(resumen)
print(df_res.to_string(index=False))
print(f"\nTOTAL CICLOS:    {len(df)}")
print(f"TOTAL CAMIONES:  {df['camion'].nunique()}")
print(f"{'='*65}")

df.to_csv("reporte_por_camion.csv", index=False)
print(f"\n✅ Guardado: reporte_por_camion.csv")