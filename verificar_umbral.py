import numpy as np
import pandas as pd
from scipy.signal import find_peaks

imu    = np.load("imu.npy", allow_pickle=True)
t_imu  = (imu[:, 0] - imu[0, 0]) / 1e9
gyro_x = imu[:, 4]

df_ocr = pd.read_csv("cargas_final.csv")
df_ocr['tiempo_seg'] = df_ocr['tiempo'].apply(
    lambda t: int(t.split(":")[0])*60 + int(t.split(":")[1])
)

# Ver señal de gyro_x en los momentos donde OCR confirma peso
print("Gyro_x en momentos de actualización OCR confirmada:")
print(f"{'Tiempo':>8} {'Peso OCR':>10} {'Gyro_x min en ±10s':>20}")
print("-" * 45)
for _, row in df_ocr.iterrows():
    t_ocr = row['tiempo_seg']
    mask  = (t_imu >= t_ocr - 10) & (t_imu <= t_ocr + 10)
    if mask.sum() > 0:
        gx_min = gyro_x[mask].min()
        gx_max = gyro_x[mask].max()
        print(f"{row['tiempo']:>8}  {int(row['peso_t']):>6}t  "
              f"  min={gx_min:>7.1f}°/s  max={gx_max:>6.1f}°/s")