import pandas as pd
import numpy as np

df = pd.read_csv("pesos_detectados.csv")

# ── Filtrar por confianza mínima ──
df = df[df["confianza"] >= 0.2].copy()

# ── Agrupar lecturas cercanas en el tiempo (ventana de 3 segundos) ──
df = df.sort_values("tiempo_seg").reset_index(drop=True)

grupos = []
grupo_actual = [df.iloc[0]]

for i in range(1, len(df)):
    row = df.iloc[i]
    ultimo = grupo_actual[-1]
    if row["tiempo_seg"] - ultimo["tiempo_seg"] < 5:
        grupo_actual.append(row)
    else:
        grupos.append(grupo_actual)
        grupo_actual = [row]
grupos.append(grupo_actual)

# ── De cada grupo tomar la lectura de mayor confianza ──
lecturas_limpias = []
for g in grupos:
    g_df = pd.DataFrame(g)
    mejor = g_df.loc[g_df["confianza"].idxmax()]
    lecturas_limpias.append(mejor)

df_limpio = pd.DataFrame(lecturas_limpias).reset_index(drop=True)

# ── Detectar secuencias crecientes (carga progresiva) ──
# Un camión se llena = pesos van subiendo
df_limpio["delta"] = df_limpio["peso_t"].diff()
df_limpio["tipo"] = df_limpio["delta"].apply(
    lambda d: "CARGA" if d > 10 else ("NUEVO_CAMION" if d < -30 else "ESTABLE")
)

print("=== LECTURAS LIMPIAS ===")
print(f"{'Tiempo':>8} {'Peso':>6} {'Delta':>7} {'Tipo':>14} {'Conf':>6}")
print("-" * 50)
for _, r in df_limpio.iterrows():
    delta_str = f"+{int(r['delta'])}t" if r['delta'] > 0 else f"{int(r['delta'])}t" if not np.isnan(r['delta']) else "---"
    print(f"{r['tiempo']:>8} {int(r['peso_t']):>5}t {delta_str:>7}  {r['tipo']:>14}  {r['confianza']:.2f}")

# ── Guardar ──
df_limpio.to_csv("pesos_limpios.csv", index=False)
print(f"\n✅ {len(df_limpio)} lecturas limpias guardadas")

# ── Identificar cargas por camión ──
print("\n=== CARGAS POR CAMIÓN ===")
camion = 1
inicio = df_limpio.iloc[0]["tiempo"]
peso_inicial = df_limpio.iloc[0]["peso_t"]

for _, r in df_limpio.iterrows():
    if r["tipo"] == "NUEVO_CAMION":
        camion += 1
        inicio = r["tiempo"]
        peso_inicial = r["peso_t"]
    if r["tipo"] in ["CARGA", "NUEVO_CAMION"]:
        print(f"  Camión {camion} | {r['tiempo']} | {int(r['peso_t'])}t")