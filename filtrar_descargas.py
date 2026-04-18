import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d

imu    = np.load("imu.npy", allow_pickle=True)
t_imu  = (imu[:, 0] - imu[0, 0]) / 1e9
gyro_x = imu[:, 4]

df_ocr = pd.read_csv("cargas_final.csv")
df_ocr['tiempo_seg'] = df_ocr['tiempo'].apply(
    lambda t: int(t.split(":")[0])*60 + int(t.split(":")[1])
)

# Detectar con umbral bajo
picos_desc, props = find_peaks(
    -gyro_x,
    height=30,
    distance=40,
    prominence=15
)
t_desc  = t_imu[picos_desc]
gx_desc = gyro_x[picos_desc]
amp_desc = -gyro_x[picos_desc]  # amplitud positiva

# ── Filtro: dentro de cada ventana de 20s, quedarse solo con el pico más fuerte ──
VENTANA = 20  # segundos
t_filtradas = []
gx_filtradas = []

i = 0
while i < len(t_desc):
    # Tomar todos los picos en la ventana actual
    ventana_mask = (t_desc >= t_desc[i]) & (t_desc < t_desc[i] + VENTANA)
    idx_ventana  = np.where(ventana_mask)[0]
    # Quedarse con el más fuerte
    idx_max = idx_ventana[np.argmax(amp_desc[ventana_mask])]
    t_filtradas.append(t_desc[idx_max])
    gx_filtradas.append(gx_desc[idx_max])
    i = idx_ventana[-1] + 1

t_filtradas  = np.array(t_filtradas)
gx_filtradas = np.array(gx_filtradas)

print(f"Descargas antes del filtro: {len(t_desc)}")
print(f"Descargas después del filtro: {len(t_filtradas)}")

# Verificar cobertura OCR
print("\nVerificando cobertura OCR:")
for _, row in df_ocr.iterrows():
    t_ocr    = row['tiempo_seg']
    cercanas = t_filtradas[(t_filtradas >= t_ocr-15) & (t_filtradas <= t_ocr+15)]
    estado   = f"✅ {len(cercanas)} det." if len(cercanas) > 0 else "❌ PERDIDA"
    print(f"  {row['tiempo']}  {int(row['peso_t']):>4}t  → {estado}")

print(f"\nDescargas filtradas por tiempo:")
for i, (t, gx) in enumerate(zip(t_filtradas, gx_filtradas)):
    t_str = f"{int(t//60):02d}:{int(t%60):02d}"
    ocr   = df_ocr[(df_ocr['tiempo_seg'] >= t-15) & (df_ocr['tiempo_seg'] <= t+15)]
    peso_str = f"OCR={int(ocr.iloc[-1]['peso_t'])}t" if len(ocr) > 0 else "EST"
    print(f"  D{i+1:>2}: {t_str}  {gx:>8.1f}°/s  {peso_str}")

np.save("descargas_filtradas.npy", np.column_stack([t_filtradas, gx_filtradas]))
print(f"\n✅ Guardado: descargas_filtradas.npy")