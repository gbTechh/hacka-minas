import cv2
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

VIDEO_PATH  = "videor.mp4"
OUTPUT_PATH = "overlay_ciclos.avi"

imu       = np.load("imu.npy", allow_pickle=True)
t_imu     = (imu[:, 0] - imu[0, 0]) / 1e9
gyro_mag  = np.sqrt(imu[:,4]**2 + imu[:,5]**2 + imu[:,6]**2)
gm_suave  = uniform_filter1d(gyro_mag, size=8)

umbral    = gm_suave.mean() + gm_suave.std() * 1.2
picos, _  = find_peaks(gm_suave, height=umbral, distance=200, prominence=8)
t_ciclos  = t_imu[picos]
dur_ciclos = np.diff(t_ciclos, append=t_ciclos[-1] + 40)
print(f"Ciclos: {len(t_ciclos)}")

PROPORCION_FASES = [
    (0.00, 0.25, "GIRANDO→CAMION"),
    (0.25, 0.48, "DESCARGANDO"),
    (0.48, 0.75, "GIRANDO→MONTICULO"),
    (0.75, 1.00, "EXCAVANDO"),
]

COLORES = {
    "GIRANDO→CAMION":      (0, 220, 100),
    "DESCARGANDO":         (0, 0, 255),
    "GIRANDO→MONTICULO":   (0, 220, 220),
    "EXCAVANDO":           (0, 165, 255),
}

def get_fase(t_actual):
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

cap   = cv2.VideoCapture(VIDEO_PATH)
fps   = cap.get(cv2.CAP_PROP_FPS)
w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out   = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"XVID"), fps, (w,h))

frame_num = 0
print("Generando video...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1
    seg = frame_num / fps

    fase, prop    = get_fase(seg)
    color         = COLORES.get(fase, (180,180,180))
    idx_imu       = np.argmin(np.abs(t_imu - seg))
    gm_val        = gm_suave[idx_imu]

    ciclos_comp   = int(np.sum(t_ciclos <= seg))
    dur_en_curso  = seg - t_ciclos[ciclos_comp-1] if ciclos_comp > 0 else seg
    dur_prom      = float(np.mean(np.diff(t_ciclos[:ciclos_comp]))) \
                    if ciclos_comp > 1 else 0.0

    # Borde
    cv2.rectangle(frame, (0,0), (w-1,h-1), color, 5)

    # Panel
    ov = frame.copy()
    cv2.rectangle(ov, (0,0), (390,255), (0,0,0), -1)
    cv2.addWeighted(ov, 0.55, frame, 0.45, 0, frame)

    t_str = f"{int(seg//60):02d}:{int(seg%60):02d}"
    cv2.putText(frame, f"Tiempo:    {t_str}",
                (10,30),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"Ciclos:    {ciclos_comp} / {len(t_ciclos)}",
                (10,62),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(frame, f"En curso:  {dur_en_curso:.1f}s",
                (10,94),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    if dur_prom > 0:
        cv2.putText(frame, f"Prom ciclo:{dur_prom:.1f}s",
                    (10,122), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200,200,200), 1)
    cv2.putText(frame, f"Fase:      {fase}",
                (10,152), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, f"Prop:      {prop:.2f}",
                (10,180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
    cv2.putText(frame, f"Gyro:      {gm_val:.1f} deg/s",
                (10,205), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    # Barra gyro
    bw = int(min(gm_val/150,1.0)*300)
    bc = (0,0,255) if gm_val>25 else (0,200,100) if gm_val>12 else (100,100,100)
    cv2.rectangle(frame, (10,220), (10+bw,237), bc, -1)
    cv2.rectangle(frame, (10,220), (310,237), (150,150,150), 1)
    cv2.putText(frame, f"IMU: {gm_val:.0f} deg/s",
                (10,252), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,180,180), 1)

    # Timeline
    tl_y = h-20
    tl_w = w-40
    cv2.rectangle(frame, (20,tl_y-8), (20+tl_w,tl_y+8), (40,40,40), -1)
    px = int(20 + (seg/t_imu[-1])*tl_w)
    cv2.line(frame, (px,tl_y-14), (px,tl_y+14), (255,255,255), 2)
    for t_c in t_ciclos:
        cx = int(20 + (t_c/t_imu[-1])*tl_w)
        c  = (0,220,0) if t_c<=seg else (120,120,120)
        cv2.circle(frame, (cx,tl_y), 4, c, -1)

    out.write(frame)
    if frame_num % 500 == 0:
        print(f"  {frame_num/total*100:.0f}%  fase={fase}  prop={prop:.2f}")

cap.release()
out.release()
print(f"\n✅ {OUTPUT_PATH}  —  vlc overlay_ciclos.avi")