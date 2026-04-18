import numpy as np

data = np.load("imu.npy", allow_pickle=True)

print("Forma del array:", data.shape)
print("Tipo de dato:", data.dtype)
print("\nPrimeras 5 filas:")
print(data[:5])
print("\nÚltimas 5 filas:")
print(data[-5:])
print("\nValores mínimos:", data.min(axis=0) if data.ndim > 1 else data.min())
print("Valores máximos:", data.max(axis=0) if data.ndim > 1 else data.max())


timestamps = data[:, 0]

# Convertir nanosegundos a segundos
ts_seg = (timestamps - timestamps[0]) / 1e9

duracion = ts_seg[-1]
frecuencia = len(data) / duracion

print(f"Duración total: {duracion:.2f} segundos ({duracion/60:.1f} minutos)")
print(f"Frecuencia de muestreo: {frecuencia:.1f} Hz")
print(f"Tiempo inicio: {timestamps[0]}")
print(f"Tiempo fin:    {timestamps[-1]}")