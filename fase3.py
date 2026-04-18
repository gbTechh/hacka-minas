import pandas as pd
import numpy as np

df_cargas = pd.read_csv("cargas_final.csv")
df_ciclos = pd.read_csv("ciclos_finales.csv")

print("=== REPORTE FINAL ===\n")
print(f"Ciclos IMU detectados:  {len(df_ciclos)}")
print(f"Lecturas OCR válidas:   {len(df_cargas[df_cargas['confianza'] > 0.7])}")
print(f"Peso máximo detectado:  {df_cargas['peso_t'].max()}t")
print(f"Ciclos por hora:        {3600 / df_ciclos['duracion_seg'].mean():.0f}")
print(f"Duración promedio:      {df_ciclos['duracion_seg'].mean():.1f}s")
print(f"\nCargas confirmadas con peso:")
for _, r in df_cargas[df_cargas['confianza'] > 0.7].iterrows():
    print(f"  {r['tiempo']}  {int(r['peso_t'])}t  (+{int(r['delta_t'])}t)")