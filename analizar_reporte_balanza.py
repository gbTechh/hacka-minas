import numpy as np
import pandas as pd

senal  = np.load("senal_balanza.npy")
t_bal  = senal[:, 0]
px_bal = senal[:, 1]

df_ocr = pd.read_csv("cargas_final.csv")
df_ocr['tiempo_seg'] = df_ocr['tiempo'].apply(
    lambda t: int(t.split(":")[0])*60 + int(t.split(":")[1])
)

# Detectar encendidos y apagados de balanza
UMBRAL = 100
activo  = px_bal > UMBRAL
cambios = np.diff(activo.astype(int))
t_encendidos = t_bal[np.where(cambios ==  1)[0] + 1]
t_apagados   = t_bal[np.where(cambios == -1)[0] + 1]

print("=== EVENTOS DE BALANZA ===\n")
print(f"{'Evento':>8} {'Tiempo':>8} {'Acción':>12} {'Peso OCR cercano':>18}")
print("-" * 55)

todos_eventos = []
for t in t_encendidos:
    todos_eventos.append((t, 'ENCENDIDO'))
for t in t_apagados:
    todos_eventos.append((t, 'APAGADO'))
todos_eventos.sort()

for t, accion in todos_eventos:
    t_str = f"{int(t//60):02d}:{int(t%60):02d}"
    # Buscar peso OCR más cercano
    ocr_cerca = df_ocr[abs(df_ocr['tiempo_seg'] - t) < 30]
    if len(ocr_cerca) > 0:
        peso_str = f"{int(ocr_cerca.iloc[0]['peso_t'])}t"
    else:
        peso_str = "---"
    print(f"  {t_str:>8}  {accion:>12}  {peso_str:>16}")

print(f"\n=== INTERPRETACIÓN ===")
print(f"Cada ENCENDIDO = actualización de peso (después de una palada)")
print(f"Gap largo entre APAGADO y ENCENDIDO = camión se fue, llegó nuevo")

# Detectar gaps entre apagado y encendido
print(f"\n=== GAPS ENTRE SESIONES ===")
for i in range(len(t_apagados)):
    if i < len(t_encendidos) - 1:
        # Próximo encendido después de este apagado
        sig_enc = t_encendidos[t_encendidos > t_apagados[i]]
        if len(sig_enc) > 0:
            gap = sig_enc[0] - t_apagados[i]
            t1  = f"{int(t_apagados[i]//60):02d}:{int(t_apagados[i]%60):02d}"
            t2  = f"{int(sig_enc[0]//60):02d}:{int(sig_enc[0]%60):02d}"
            marca = " ← NUEVO CAMIÓN" if gap > 30 else ""
            print(f"  Apagado {t1} → Encendido {t2}  gap={gap:.0f}s{marca}")