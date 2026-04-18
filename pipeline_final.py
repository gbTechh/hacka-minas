import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

VIDEO_PATH = "videor.mp4"
X1, Y1, X2, Y2 = 600, 250, 1150, 480

# ── Cargar señal de balanza ──
senal = np.load("senal_balanza.npy")
tiempos = senal[:, 0]
pixeles = senal[:, 1]

# ── Cargar ciclos IMU ──
imu        = np.load("imu.npy", allow_pickle=True)
t_imu      = (imu[:, 0] - imu[0, 0]) / 1e9
gyro_mag   = np.sqrt(imu[:,4]**2 + imu[:,5]**2 + imu[:,6]**2)
gyro_suave = uniform_filter1d(gyro_mag, size=10)
umbral_imu = gyro_suave.mean() + gyro_suave.std() * 1.2
picos, _   = find_peaks(gyro_suave, height=umbral_imu, distance=200, prominence=8)
t_ciclos   = t_imu[picos]
dur_ciclos = np.diff(t_ciclos, append=t_ciclos[-1] + 40)

# ── Cargar pesos OCR ──
df_ocr = pd.read_csv("cargas_final.csv")
df_ocr['tiempo_seg'] = df_ocr['tiempo'].apply(
    lambda t: int(t.split(":")[0])*60 + int(t.split(":")[1])
)

# ── Detectar bloques de balanza activa (mín 3s de duración) ──
UMBRAL_PX  = 100
DURACION_MIN = 3   # segundos mínimos para ser bloque real
activo  = pixeles > UMBRAL_PX
cambios = np.diff(activo.astype(int))
inicios = np.where(cambios == 1)[0] + 1
fines   = np.where(cambios == -1)[0] + 1

# Ajustar si el video termina activo
if len(fines) < len(inicios):
    fines = np.append(fines, len(tiempos) - 1)

bloques = []
for ini, fin in zip(inicios, fines):
    dur = tiempos[fin] - tiempos[ini]
    if dur >= DURACION_MIN:
        bloques.append({
            "t_ini": tiempos[ini],
            "t_fin": tiempos[fin],
            "dur":   dur,
            "tiempo_str": f"{int(tiempos[ini]//60):02d}:{int(tiempos[ini]%60):02d}"
        })

print(f"Bloques válidos (≥{DURACION_MIN}s): {len(bloques)}")
for i, b in enumerate(bloques):
    print(f"  Bloque {i+1}: {b['tiempo_str']}  ({b['dur']:.0f}s)")

# ── Agrupar bloques en camiones ──
# Si hay un gap > 5 minutos entre bloques = nuevo camión
GAP_NUEVO_CAMION = 300  # segundos

camiones = []
camion_actual = {"id": 1, "bloques": [bloques[0]]}

for b in bloques[1:]:
    gap = b["t_ini"] - camion_actual["bloques"][-1]["t_fin"]
    if gap > GAP_NUEVO_CAMION:
        camiones.append(camion_actual)
        camion_actual = {"id": len(camiones) + 1, "bloques": [b]}
    else:
        camion_actual["bloques"].append(b)
camiones.append(camion_actual)

print(f"\nCamiones detectados: {len(camiones)}")
for c in camiones:
    t_ini = c["bloques"][0]["t_ini"]
    t_fin = c["bloques"][-1]["t_fin"]
    print(f"  Camión {c['id']}: {int(t_ini//60):02d}:{int(t_ini%60):02d} → "
          f"{int(t_fin//60):02d}:{int(t_fin%60):02d}  ({len(c['bloques'])} actualizaciones de peso)")

# ── Asignar ciclos IMU a cada camión ──
# Cada camión ocupa el tiempo entre su primera y última aparición
# Los ciclos IMU dentro de ese rango pertenecen a ese camión
print(f"\n{'='*65}")
print(f"     REPORTE POR CAMIÓN")
print(f"{'='*65}")

reporte_final = []

for c in camiones:
    t_cam_ini = c["bloques"][0]["t_ini"] - 60  # 1 min antes del primer peso
    t_cam_fin = c["bloques"][-1]["t_fin"] + 60  # 1 min después del último peso

    # Ciclos IMU de este camión
    mask_ciclos = (t_ciclos >= t_cam_ini) & (t_ciclos <= t_cam_fin)
    t_c_cam     = t_ciclos[mask_ciclos]
    dur_c_cam   = dur_ciclos[mask_ciclos]

    # Pesos OCR de este camión
    pesos_cam = df_ocr[
        (df_ocr['tiempo_seg'] >= t_cam_ini) &
        (df_ocr['tiempo_seg'] <= t_cam_fin)
    ].copy()

    # Estadísticas
    n_ciclos  = len(t_c_cam)
    dur_prom  = dur_c_cam.mean() if n_ciclos > 0 else 0
    peso_max  = pesos_cam['peso_t'].max() if len(pesos_cam) > 0 else None
    delta_prom = pesos_cam[pesos_cam['delta_t'] > 5]['delta_t'].mean() \
                 if len(pesos_cam) > 0 else 40

    t_ini_str = f"{int(t_cam_ini//60):02d}:{int(t_cam_ini%60):02d}"
    t_fin_str = f"{int(t_cam_fin//60):02d}:{int(t_cam_fin%60):02d}"

    print(f"\n🚛 CAMIÓN {c['id']}")
    print(f"   Período:             {t_ini_str} → {t_fin_str}")
    print(f"   Ciclos IMU:          {n_ciclos}")
    print(f"   Actualizaciones OCR: {len(pesos_cam)}")
    print(f"   Duración prom/ciclo: {dur_prom:.1f}s")
    if n_ciclos > 0:
        print(f"   Ciclos por hora:     {3600/dur_prom:.0f}")
    if peso_max:
        print(f"   Peso final (OCR):    {int(peso_max)}t")
        print(f"   Delta prom/palada:   ~{int(delta_prom)}t")
    else:
        delta_est = int(delta_prom) if not np.isnan(delta_prom) else 40
        peso_est  = n_ciclos * delta_est
        print(f"   Peso final (EST):    ~{peso_est}t  (estimado)")
        print(f"   Delta estimado:      ~{delta_est}t/ciclo")

    # Detalle de ciclos
    print(f"\n   {'#':>4} {'Tiempo':>7} {'Dur':>6} {'Peso':>7} {'Fuente'}")
    print(f"   {'-'*45}")

    peso_ant = 0
    for j, (t_c, dur) in enumerate(zip(t_c_cam, dur_c_cam)):
        tiempo_s = f"{int(t_c//60):02d}:{int(t_c%60):02d}"
        # Buscar peso OCR cercano
        ocr_cerca = pesos_cam[
            (pesos_cam['tiempo_seg'] >= t_c - 45) &
            (pesos_cam['tiempo_seg'] <= t_c + 45)
        ]
        if len(ocr_cerca) > 0:
            peso   = int(ocr_cerca.iloc[-1]['peso_t'])
            fuente = 'OCR'
        else:
            delta_est = int(delta_prom) if not np.isnan(delta_prom) else 40
            peso   = min(peso_ant + delta_est, 300)
            fuente = f'EST({delta_est}t)'
        print(f"   #{j+1:>3}  {tiempo_s:>6}  {dur:>5.1f}s  {peso:>5}t  {fuente}")
        peso_ant = peso
        reporte_final.append({
            "camion": c['id'], "ciclo_camion": j+1,
            "tiempo": tiempo_s, "tiempo_seg": round(float(t_c),1),
            "duracion_seg": round(float(dur),1),
            "peso_t": peso, "fuente": fuente
        })

print(f"\n{'='*65}")
df_rep = pd.DataFrame(reporte_final)
df_rep.to_csv("reporte_por_camion.csv", index=False)
print(f"✅ Guardado: reporte_por_camion.csv")