import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

imu      = np.load("imu.npy", allow_pickle=True)
t_imu    = (imu[:, 0] - imu[0, 0]) / 1e9
gyro_mag = np.sqrt(imu[:,4]**2 + imu[:,5]**2 + imu[:,6]**2)
gyro_x   = imu[:, 4]
accel_mag = np.sqrt(imu[:,1]**2 + imu[:,2]**2 + imu[:,3]**2)

gm_suave = uniform_filter1d(gyro_mag,  size=8)
gx_suave = uniform_filter1d(gyro_x,    size=8)
am_suave = uniform_filter1d(accel_mag, size=8)

# ── Detectar estado inicial ──
inicio_mask = t_imu < 3
gm_inicio   = gm_suave[inicio_mask].mean()
gx_inicio   = gx_suave[inicio_mask].mean()

if gm_inicio > 25 and gx_inicio < -8:
    estado_inicial = "GIRANDO→MONTICULO"
elif gm_inicio > 12 and abs(gx_inicio) < 8:
    estado_inicial = "GIRANDO→CAMION"
elif gm_inicio < 8:
    estado_inicial = "DESCARGANDO"
else:
    estado_inicial = "EXCAVANDO"

print(f"Estado inicial: {estado_inicial}")

# ── Nueva máquina de estados basada en datos reales ──
# Observaciones clave:
# - EXCAVANDO tiene pico fuerte de gm (>30) con gx negativo
# - GIRANDO→CAMION (2do ciclo) tiene gm muy bajo (<6)
# - Las transiciones entre estados son graduales
# - Usamos ventana temporal para evitar cambios rápidos

FASES  = []
estado = estado_inicial
contador = 0  # frames en el estado actual

for i in range(len(t_imu)):
    gm = gm_suave[i]
    gx = gx_suave[i]
    am = am_suave[i]
    contador += 1

    # Mínimo 3 muestras (~0.3s) antes de cambiar de estado
    if contador < 3:
        FASES.append(estado)
        continue

    prev = estado

    if estado == "GIRANDO→CAMION":
        # Llegó al camión: gm baja mucho
        if gm < 6:
            estado = "DESCARGANDO"
        # Giro brusco hacia montículo: gm sube con gx negativo
        elif gm > 28 and gx < -15:
            estado = "GIRANDO→MONTICULO"

    elif estado == "DESCARGANDO":
        # Empieza a girar: gm sube con gx negativo
        if gm > 22 and gx < -5:
            estado = "GIRANDO→MONTICULO"
        # O sube sin dirección clara
        elif gm > 28:
            estado = "GIRANDO→MONTICULO"

    elif estado == "GIRANDO→MONTICULO":
        # Llegó al montículo: gm baja
        if gm < 12 and contador > 5:
            estado = "EXCAVANDO"

    elif estado == "EXCAVANDO":
        # Pico de impacto + luego gm baja mucho = terminó excavar, gira
        if gm < 6 and contador > 10:
            estado = "GIRANDO→CAMION"
        elif gm > 15 and gx > 0 and contador > 10:
            estado = "GIRANDO→CAMION"

    if estado != prev:
        contador = 0

    FASES.append(estado)

# ── Verificar ──
print("\nVerificación:")
eventos = [
    ( 4, "DESCARGANDO"),
    ( 7, "GIRANDO→MONTICULO"),
    (12, "EXCAVANDO"),
    (21, "GIRANDO→CAMION"),
    (25, "DESCARGANDO"),
]
correctos = 0
for t_vid, fase_real in eventos:
    idx       = np.argmin(np.abs(t_imu - t_vid))
    detectado = FASES[idx]
    ok = "✅" if fase_real in detectado else "❌"
    if fase_real in detectado:
        correctos += 1
    print(f"  t={t_vid:>2}s  real={fase_real:>20}  "
          f"detectado={detectado:>20}  {ok}")

print(f"\nScore: {correctos}/5")

# ── Ver transiciones detectadas en primeros 30s ──
print("\nTransiciones detectadas (0-30s):")
fase_prev = FASES[0]
for i, (t, f) in enumerate(zip(t_imu, FASES)):
    if t > 30:
        break
    if f != fase_prev:
        print(f"  t={t:.1f}s → {fase_prev} → {f}")
        fase_prev = f