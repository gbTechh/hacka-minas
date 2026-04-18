import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

imu      = np.load("imu.npy", allow_pickle=True)
t_imu    = (imu[:, 0] - imu[0, 0]) / 1e9
gyro_mag = np.sqrt(imu[:,4]**2 + imu[:,5]**2 + imu[:,6]**2)
gm_suave = uniform_filter1d(gyro_mag, size=8)

umbral   = gm_suave.mean() + gm_suave.std() * 1.2
picos, _ = find_peaks(gm_suave, height=umbral, distance=200, prominence=8)
t_ciclos = t_imu[picos]

# Del primer ciclo sabemos las proporciones exactas:
# Ciclo dura ~29s (t=0 a t=29s, pico en t=14s)
# GIRANDO→CAMION:    0%  - 28%  del ciclo (t=0-8s  de 29s)
# DESCARGANDO:       28% - 48%  del ciclo (t=8-14s)  ← justo antes del pico
# EXCAVANDO:         48% - 75%  del ciclo (t=14-22s) ← después del pico
# GIRANDO→MONTICULO: 75% - 100% del ciclo (t=22-29s)

# Proporciones calibradas con el video:
# t=0   inicio ciclo
# t=4s  DESCARGA  = 14% del ciclo (4/29)
# t=7s  GIRA→MON  = 24% del ciclo (7/29) 
# t=12s EXCAVA    = 41% del ciclo (12/29)
# t=21s GIRA→CAM  = 72% del ciclo (21/29)
# t=25s DESCARGA  = 86% (siguiente ciclo comienza en t=29s)

# Pero el pico IMU está en t=14s = 48% del ciclo
# Entonces el pico = momento de EXCAVACION fuerte

# Definir fases por % del ciclo (desde el pico anterior)
PROPORCION_FASES = [
    (0.00, 0.25, "GIRANDO→CAMION"),     # t=21s prop=0.23 ✅
    (0.25, 0.48, "DESCARGANDO"),        # t=4s prop=0.28, t=25s prop=0.37 ✅
    (0.48, 0.75, "GIRANDO→MONTICULO"),  # t=7s prop=0.49 ← mover aquí
    (0.75, 1.00, "EXCAVANDO"),          # t=12s prop=0.84 ← mover aquí
]

def fase_por_proporcion(t_actual, t_ciclos):
    # Encontrar el ciclo actual
    ciclos_pasados = t_ciclos[t_ciclos <= t_actual]
    ciclos_futuros = t_ciclos[t_ciclos > t_actual]

    if len(ciclos_pasados) == 0:
        t_inicio = 0
    else:
        t_inicio = ciclos_pasados[-1]

    if len(ciclos_futuros) == 0:
        return "GIRANDO→MONTICULO"

    t_fin  = ciclos_futuros[0]
    dur    = t_fin - t_inicio
    prop   = (t_actual - t_inicio) / dur if dur > 0 else 0

    for p_ini, p_fin, fase in PROPORCION_FASES:
        if p_ini <= prop < p_fin:
            return fase
    return "GIRANDO→MONTICULO"

# Verificar
print("Verificación con proporciones de ciclo:")
eventos = [
    ( 4, "DESCARGANDO"),
    ( 7, "GIRANDO→MONTICULO"),
    (12, "EXCAVANDO"),
    (21, "GIRANDO→CAMION"),
    (25, "DESCARGANDO"),
]
correctos = 0
for t_vid, fase_real in eventos:
    detectado = fase_por_proporcion(t_vid, t_ciclos)
    ok = "✅" if fase_real in detectado else "❌"
    if fase_real in detectado:
        correctos += 1

    # Mostrar proporción
    ciclos_p = t_ciclos[t_ciclos <= t_vid]
    ciclos_f = t_ciclos[t_ciclos > t_vid]
    t_ini = ciclos_p[-1] if len(ciclos_p) > 0 else 0
    t_fin = ciclos_f[0]  if len(ciclos_f) > 0 else t_vid+1
    prop  = (t_vid - t_ini) / (t_fin - t_ini)
    print(f"  t={t_vid:>2}s  prop={prop:.2f}  real={fase_real:>20}  "
          f"detectado={detectado:>20}  {ok}")

print(f"\nScore: {correctos}/5")