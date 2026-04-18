import numpy as np
import pandas as pd

datos = np.load("descargas_filtradas.npy")
t_desc  = datos[:, 0]
gx_desc = datos[:, 1]

df_ocr = pd.read_csv("cargas_final.csv")
df_ocr['tiempo_seg'] = df_ocr['tiempo'].apply(
    lambda t: int(t.split(":")[0])*60 + int(t.split(":")[1])
)

DELTA_PROM = 36
PESO_MAX   = 216

periodos = [
    {"lado": "DERECHO",   "t_ini":   0, "t_fin": 135, "balanza": True},
    {"lado": "IZQUIERDO", "t_ini": 135, "t_fin": 575, "balanza": False},
    {"lado": "DERECHO",   "t_ini": 575, "t_fin": 900, "balanza": True},
]

camion_global = 0
reporte_total = []

print("=" * 65)
print("     REPORTE FINAL DE OPERACIÓN — PALA MINERA")
print("=" * 65)

for periodo in periodos:
    mask = (t_desc >= periodo["t_ini"]) & (t_desc < periodo["t_fin"])
    t_p  = t_desc[mask]
    gx_p = gx_desc[mask]

    if len(t_p) == 0:
        continue

    # Separar en camiones: peso acumulado → si baja o llega al máximo = nuevo camión
    camiones_periodo = []
    camion_actual    = []
    peso_acum        = 0

    for t_d, gx in zip(t_p, gx_p):
        # Buscar OCR más cercano DESPUÉS de esta descarga (el peso se actualiza después)
        ocr_sig = df_ocr[
            (df_ocr['tiempo_seg'] >= t_d) &
            (df_ocr['tiempo_seg'] <= t_d + 25)
        ]
        ocr_ant = df_ocr[
            (df_ocr['tiempo_seg'] >= t_d - 25) &
            (df_ocr['tiempo_seg'] < t_d)
        ]

        if len(ocr_sig) > 0:
            nuevo_peso = int(ocr_sig.iloc[0]['peso_t'])
            if nuevo_peso < peso_acum - 30:
                # Peso bajó = nuevo camión
                camiones_periodo.append(camion_actual)
                camion_actual = []
                peso_acum = nuevo_peso
            else:
                peso_acum = nuevo_peso
            fuente = 'OCR'
        elif len(ocr_ant) > 0:
            nuevo_peso = int(ocr_ant.iloc[-1]['peso_t'])
            if nuevo_peso > peso_acum:
                peso_acum = nuevo_peso
            else:
                peso_acum = min(peso_acum + DELTA_PROM, PESO_MAX)
            fuente = 'OCR~'
        else:
            peso_acum = min(peso_acum + DELTA_PROM, PESO_MAX)
            fuente = 'EST'

        camion_actual.append({
            "t": t_d, "gx": gx,
            "peso": peso_acum, "fuente": fuente
        })

        # Si llegó al máximo → próximo es nuevo camión
        if peso_acum >= PESO_MAX - DELTA_PROM//2:
            camiones_periodo.append(camion_actual)
            camion_actual = []
            peso_acum = 0

    if camion_actual:
        camiones_periodo.append(camion_actual)

    t1_p = f"{int(periodo['t_ini']//60):02d}:{int(periodo['t_ini']%60):02d}"
    t2_p = f"{int(periodo['t_fin']//60):02d}:{int(periodo['t_fin']%60):02d}"
    print(f"\n📍 Lado {periodo['lado']}  ({t1_p}→{t2_p})"
          f"  {'[balanza visible]' if periodo['balanza'] else '[sin balanza]'}")

    for cam in camiones_periodo:
        if not cam:
            continue
        camion_global += 1
        n     = len(cam)
        p_max = cam[-1]['peso']
        t1    = f"{int(cam[0]['t']//60):02d}:{int(cam[0]['t']%60):02d}"
        t2    = f"{int(cam[-1]['t']//60):02d}:{int(cam[-1]['t']%60):02d}"
        dur   = cam[-1]['t'] - cam[0]['t']
        n_ocr = sum(1 for c in cam if 'OCR' in c['fuente'])
        ciclos_hora = 3600/(dur/n) if n > 1 and dur > 0 else 0

        print(f"\n  🚛 CAMIÓN {camion_global}  ({t1}→{t2}  {dur/60:.1f}min)")
        print(f"     Descargas:    {n}  |  OCR: {n_ocr}  |  "
              f"Peso: ~{int(p_max)}t {'✅' if n_ocr>0 else '⚠️'}")
        if ciclos_hora > 0:
            print(f"     Ciclos/hora:  {ciclos_hora:.0f}")

        print(f"\n     {'#':>3} {'Tiempo':>7} {'Gyro':>8} {'Peso':>7}  Fuente")
        print(f"     {'-'*40}")
        for j, c in enumerate(cam):
            t_s = f"{int(c['t']//60):02d}:{int(c['t']%60):02d}"
            print(f"     #{j+1:>2}  {t_s:>6}  {c['gx']:>7.1f}°/s"
                  f"  {int(c['peso']):>5}t  {c['fuente']}")

        reporte_total.extend([{
            "camion": camion_global, "descarga": j+1,
            "tiempo": f"{int(c['t']//60):02d}:{int(c['t']%60):02d}",
            "tiempo_seg": round(float(c['t']),1),
            "gyro_x": round(float(c['gx']),1),
            "peso_t": int(c['peso']),
            "fuente": c['fuente'],
            "lado": periodo['lado'],
            "balanza_visible": periodo['balanza']
        } for j, c in enumerate(cam)])

# ── Resumen ──
df_rep = pd.DataFrame(reporte_total)
print(f"\n{'='*65}")
print(f"  RESUMEN EJECUTIVO")
print(f"{'='*65}")
print(f"  Duración analizada:     15 minutos")
print(f"  Total camiones:         {camion_global}")
print(f"  Total descargas:        {len(df_rep)}")
n_ocr_total = len(df_rep[df_rep['fuente'].str.contains('OCR')])
print(f"  Descargas con OCR:      {n_ocr_total}")
print(f"  Descargas estimadas:    {len(df_rep)-n_ocr_total}")
print(f"  Peso prom/camión:       ~{int(df_rep.groupby('camion')['peso_t'].max().mean())}t")
print(f"  Descargas prom/camión:  {len(df_rep)/camion_global:.1f}")
print(f"{'='*65}")

df_rep.to_csv("reporte_final_camiones.csv", index=False)
print(f"\n✅ Guardado: reporte_final_camiones.csv")