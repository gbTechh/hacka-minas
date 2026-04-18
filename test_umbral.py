import numpy as np
import pandas as pd
from scipy.signal import find_peaks

imu    = np.load("imu.npy", allow_pickle=True)
t_imu  = (imu[:, 0] - imu[0, 0]) / 1e9
gyro_x = imu[:, 4]

# Umbral más bajo para capturar todas las descargas OCR confirmadas
picos_desc, _ = find_peaks(-gyro_x, height=30, distance=40, prominence=15)
t_desc  = t_imu[picos_desc]
gx_desc = gyro_x[picos_desc]

df_ocr = pd.read_csv("cargas_final.csv")
df_ocr['tiempo_seg'] = df_ocr['tiempo'].apply(
    lambda t: int(t.split(":")[0])*60 + int(t.split(":")[1])
)

print(f"Total descargas detectadas con umbral -30°/s: {len(t_desc)}")

# Verificar que ahora captura los momentos OCR
print("\nVerificando cobertura de momentos OCR:")
for _, row in df_ocr.iterrows():
    t_ocr = row['tiempo_seg']
    cercanas = t_desc[(t_desc >= t_ocr-15) & (t_desc <= t_ocr+15)]
    estado = f"✅ {len(cercanas)} detecciones" if len(cercanas) > 0 else "❌ NO DETECTADO"
    print(f"  {row['tiempo']}  {int(row['peso_t']):>4}t  → {estado}")