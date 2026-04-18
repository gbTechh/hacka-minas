import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

VIDEO_PATH  = "videor.mp4"
OUTPUT_PATH = "overlay_ciclos2.avi"

# ── IMU ──
imu       = np.load("imu.npy", allow_pickle=True)
t_imu     = (imu[:, 0] - imu[0, 0]) / 1e9
gyro_mag  = np.sqrt(imu[:,4]**2 + imu[:,5]**2 + imu[:,6]**2)
gm_suave  = uniform_filter1d(gyro_mag, size=8)

# ── Ciclos ──
umbral    = gm_suave.mean() + gm_suave.std() * 1.2
picos, _  = find_peaks(gm_suave, height=umbral, distance=200, prominence=8)
t_ciclos  = t_imu[picos]
dur_ciclos = np.diff(t_ciclos, append=t_ciclos[-1] + 40)
print(f"Ciclos detectados: {len(t_ciclos)}")

# ── Proporciones de fases dentro de cada ciclo ──
# Calibradas con eventos reales del video:
# t=4s  prop=0.28 → DESCARGANDO
# t=7s  prop=0.49 → GIRANDO→MONTICULO
# t=12s prop=0.84 → EXCAVANDO
# t=21s prop=0.23 → GIRANDO→CAMION
# t=25s prop=0.37 → DESCARGANDO
PROPORCION_FASES = [
    (0.00, 0.25, "GIRANDO→CAMION"),
    (0.25, 0.48, "DESCARGANDO"),
    (0.48, 0.75, "GIRANDO→MONTICULO"),
    (0.75, 1.00, "EXCAVANDO"),
]

COLORES = {
    "GIRANDO→CAMION":    (0, 220, 100),   # verde
    "DESCARGANDO":       (0, 0, 255),     # rojo
    "GIRANDO→MONTICULO": (0, 220, 220),   # amarillo
    "EXCAVANDO":         (0, 165, 255),   # naranja
}

# ── Precomputar timestamps de fin de DESCARGANDO ──
# prop=0.48 = momento en que termina DESCARGANDO y empieza GIRANDO→MONTICULO
# Ese es el momento en que el ciclo se considera COMPLETADO
t_fin_descarga = []
for i in range(len(t_ciclos)):
    t_ini = t_ciclos[i-1] if i > 0 else 0
    t_fin = t_ciclos[i]
    dur   = t_fin - t_ini
    t_fd  = t_ini + 0.48 * dur
    t_fin_descarga.append(t_fd)
t_fin_descarga = np.array(t_fin_descarga)

print(f"Timestamps fin de DESCARGANDO:")
for i, t in enumerate(t_fin_descarga):
    print(f"  Ciclo {i+1}: {int(t//60):02d}:{int(t%60):02d}")

def get_fase(t_actual):
    """Retorna (fase, proporcion) para un tiempo dado."""
    ciclos_p = t_ciclos[t_ciclos <= t_actual]
    ciclos_f = t_ciclos[t_ciclos >  t_actual]

    t_ini = ciclos_p[-1] if len(ciclos_p) > 0 else 0
    if len(ciclos_f) == 0:
        return "GIRANDO→MONTICULO", 1.0

    t_fin = ciclos_f[0]
    dur   = t_fin - t_ini
    prop  = (t_actual - t_ini) / dur if dur > 0 else 0

    for p0, p1, fase in PROPORCION_FASES:
        if p0 <= prop < p1:
            return fase, prop
    return "GIRANDO→MONTICULO", prop

# ── Video ──
cap   = cv2.VideoCapture(VIDEO_PATH)
fps   = cap.get(cv2.CAP_PROP_FPS)
w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out   = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))

frame_num = 0
print("\nGenerando video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1
    seg = frame_num / fps

    # Fase actual
    fase, prop = get_fase(seg)
    color      = COLORES.get(fase, (180, 180, 180))

    # IMU actual
    idx_imu = np.argmin(np.abs(t_imu - seg))
    gm_val  = gm_suave[idx_imu]

    # ── Ciclos completados ──
    # Sube cuando termina DESCARGANDO (prop=0.48 del ciclo)
    ciclos_comp = int(np.sum(t_fin_descarga <= seg))

    # Duración del ciclo en curso (desde el último fin de DESCARGANDO)
    if ciclos_comp > 0:
        dur_en_curso = seg - t_fin_descarga[ciclos_comp - 1]
    else:
        dur_en_curso = seg

    # Duración promedio de ciclos completados
    if ciclos_comp > 1:
        durs = np.diff(t_fin_descarga[:ciclos_comp])
        dur_prom = float(np.mean(durs))
    elif ciclos_comp == 1:
        dur_prom = float(dur_ciclos[0])
    else:
        dur_prom = 0.0

    # ── Dibujar borde de color según fase ──
    cv2.rectangle(frame, (0, 0), (w-1, h-1), color, 5)

    # ── Panel semitransparente ──
    ov = frame.copy()
    cv2.rectangle(ov, (0, 0), (390, 262), (0, 0, 0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    # ── Textos ──
    t_str = f"{int(seg//60):02d}:{int(seg%60):02d}"
    cv2.putText(frame, f"Tiempo:    {t_str}",
                (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Ciclos:    {ciclos_comp} / {len(t_ciclos)}",
                (10, 62),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"En curso:  {dur_en_curso:.1f}s",
                (10, 94),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    if dur_prom > 0:
        cv2.putText(frame, f"Prom ciclo:{dur_prom:.1f}s",
                    (10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)
    cv2.putText(frame, f"Fase:      {fase}",
                (10, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Prop:      {prop:.2f}",
                (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
    cv2.putText(frame, f"Gyro:      {gm_val:.1f} deg/s",
                (10, 207), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    # ── Barra gyro ──
    bw = int(min(gm_val / 150, 1.0) * 300)
    bc = (0,0,255) if gm_val > 25 else (0,200,100) if gm_val > 12 else (100,100,100)
    cv2.rectangle(frame, (10, 222), (10+bw, 239), bc, -1)
    cv2.rectangle(frame, (10, 222), (310, 239), (150,150,150), 1)
    cv2.putText(frame, f"IMU: {gm_val:.0f} deg/s",
                (10, 257), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)

    # ── Timeline inferior ──
    tl_y = h - 20
    tl_w = w - 40
    cv2.rectangle(frame, (20, tl_y-8), (20+tl_w, tl_y+8), (40,40,40), -1)

    # Posición actual
    px = int(20 + (seg / t_imu[-1]) * tl_w)
    cv2.line(frame, (px, tl_y-14), (px, tl_y+14), (255,255,255), 2)

    # Marcar fin de cada DESCARGANDO (= cuando sube el contador)
    for t_fd in t_fin_descarga:
        cx = int(20 + (t_fd / t_imu[-1]) * tl_w)
        c  = (0, 220, 0) if t_fd <= seg else (120, 120, 120)
        cv2.circle(frame, (cx, tl_y), 5, c, -1)

    out.write(frame)

    if frame_num % 500 == 0:
        pct = frame_num / total * 100
        print(f"  {pct:.0f}%  t={seg/60:.1f}min  "
              f"ciclos={ciclos_comp}  fase={fase}  prop={prop:.2f}")

cap.release()
out.release()
print(f"\n✅ Guardado: {OUTPUT_PATH}")
print(f"   Abre con: vlc {OUTPUT_PATH}")