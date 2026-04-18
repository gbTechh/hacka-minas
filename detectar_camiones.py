import cv2
import numpy as np
import matplotlib.pyplot as plt

VIDEO_PATH  = "videor.mp4"
X1, Y1, X2, Y2 = 600, 250, 1150, 480

cap   = cv2.VideoCapture(VIDEO_PATH)
fps   = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

tiempos  = []
pixeles  = []
frame_num = 0

print("Escaneando señal de balanza completa...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_num += 1

    if frame_num % 15 != 0:  # ~1 muestra por segundo
        continue

    roi = frame[Y1:Y2, X1:X2]
    b, g, r = cv2.split(roi)
    solo_rojo = cv2.subtract(r, b)
    n_pix = int(np.sum(solo_rojo > 70))

    seg = frame_num / fps
    tiempos.append(seg)
    pixeles.append(n_pix)

    if frame_num % 1500 == 0:
        print(f"  {seg/60:.1f} min procesado...")

cap.release()

tiempos = np.array(tiempos)
pixeles = np.array(pixeles)

# Detectar bloques donde hay balanza activa
UMBRAL = 100
activo = pixeles > UMBRAL

# Encontrar transiciones
cambios = np.diff(activo.astype(int))
inicios = np.where(cambios == 1)[0] + 1   # 0→1
fines   = np.where(cambios == -1)[0] + 1  # 1→0

print(f"\nBloques de balanza activa detectados: {len(inicios)}")
for i, (ini, fin) in enumerate(zip(inicios, fines if len(fines) == len(inicios) else np.append(fines, len(tiempos)-1))):
    t_ini = tiempos[ini]
    t_fin = tiempos[fin]
    dur   = t_fin - t_ini
    print(f"  Bloque {i+1}: {int(t_ini//60):02d}:{int(t_ini%60):02d} → "
          f"{int(t_fin//60):02d}:{int(t_fin%60):02d}  ({dur:.0f}s)")

# Guardar señal para graficar
np.save("senal_balanza.npy", np.column_stack([tiempos, pixeles]))

# Graficar
plt.figure(figsize=(18, 4))
plt.plot(tiempos/60, pixeles, color='red', linewidth=0.8, alpha=0.7)
plt.axhline(UMBRAL, color='orange', linestyle='--', label=f'Umbral ({UMBRAL}px)')
for ini, fin in zip(inicios, fines if len(fines) == len(inicios) else np.append(fines, len(tiempos)-1)):
    plt.axvspan(tiempos[ini]/60, tiempos[fin]/60, alpha=0.2, color='green')
plt.xlabel("Tiempo (minutos)")
plt.ylabel("Píxeles rojos")
plt.title("Señal de balanza — presencia de camión")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("senal_balanza.png", dpi=150)
print("\n✅ Guardado: senal_balanza.png")