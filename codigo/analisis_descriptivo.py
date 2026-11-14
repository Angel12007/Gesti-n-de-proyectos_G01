import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from inicio import cargar

df, PREDICTORES, DESENLACES = cargar()
df["tumor"] = df["origin_of_tumor"].map({1: "Sólido", 0: "Hematológico"})
df["ingreso"] = df["type_of_admission"].map({1: "Médico", 2: "Quirúrgico"})
df["periodo"] = df["particion_temporal"].str.capitalize()

# ---------- 1. Estadísticas básicas y segmentación (técnicas 5 y 7) ----------
def med_ric(s):
    s = s.dropna(); q = s.quantile([.25, .5, .75])
    return f"{q[.5]:.0f} ({q[.25]:.0f}–{q[.75]:.0f})"

def n_pct(s):
    s = s.dropna(); return f"{int(s.sum())} ({100 * s.mean():.1f})".replace(".", ",")

def fmt_p(p):
    return "< 0,001" if p < 0.001 else f"{p:.3f}".replace(".", ",")

FILAS = [("Edad (años), mediana (RIC)", "age", "num"),
         ("SAPS III, mediana (RIC)", "sapsiii", "num"),
         ("APACHE II, mediana (RIC)", "apacheii", "num"),
         ("SOFA, mediana (RIC)", "sofa", "num"),
         ("Soporte ventilatorio, n (%)", "ventilatory_support", "bin"),
         ("Días de estancia, mediana (RIC)", "dias_estancia", "num"),
         ("Estancia ≥ 7 d, n (%)", "estancia_prolongada", "bin"),
         ("Muerte en UCI, n (%)", "mortality", "bin")]

def tabla_segmento(col, orden):
    g1, g2 = (df[df[col] == k] for k in orden)
    filas = []
    for etiqueta, var, tipo in FILAS:
        f = med_ric if tipo == "num" else n_pct
        if tipo == "num":   
            p = stats.mannwhitneyu(g1[var].dropna(), g2[var].dropna()).pvalue
        else:               
            p = stats.chi2_contingency(pd.crosstab(df[col], df[var]))[1]
        filas.append([etiqueta, f(g1[var]), f(g2[var]), f(df[var]), fmt_p(p)])
    cab = ["Variable", f"{orden[0]} (n = {len(g1)})", f"{orden[1]} (n = {len(g2)})",
           f"Total (n = {len(df)})", "Valor p"]
    return pd.DataFrame(filas, columns=cab)

t_tumor = tabla_segmento("tumor", ["Sólido", "Hematológico"])
t_ingreso = tabla_segmento("ingreso", ["Médico", "Quirúrgico"])
t_periodo = tabla_segmento("periodo", ["Entrenamiento", "Prueba"])
for n, t in [("tumor", t_tumor), ("ingreso", t_ingreso), ("periodo", t_periodo)]:
    t.to_csv(f"tabla1_por_{n}.csv", index=False, encoding="utf-8-sig")
    print(t.to_string(index=False), "\n")

# ---------- 2. Adición de información: costo del día-cama (técnica 6) ----------
TARIFA_UCI_2020 = 2_458_848      
TARIFA_INT_2020 = 1_921_669      
IPC = {"2021": 5.62, "2022": 13.12, "2023": 9.28, "2024": 5.20, "2025": 5.10,
       "2026 (ene–oct)": 5.94}   # DANE, variación anual a diciembre y año corrido a octubre de 2026
factor = np.prod([1 + v / 100 for v in IPC.values()])
tarifa_2026 = TARIFA_UCI_2020 * factor
grupos = {"Total de la cohorte": df,
          "Estancias ≥ 7 d": df[df.estancia_prolongada == 1],
          "Estancias < 7 d": df[df.estancia_prolongada == 0]}
costos = []
for n, g in grupos.items():
    dc = g["dias_estancia"].sum()
    costos.append([n, len(g.dropna(subset=["dias_estancia"])), int(dc),
                   round(100 * dc / df.dias_estancia.sum(), 1),
                   dc * TARIFA_UCI_2020, dc * tarifa_2026])
t_costos = pd.DataFrame(costos, columns=["Grupo", "Ingresos", "Días-cama", "% días-cama",
                                         "Costo pesos 2020", "Costo pesos 2026"])
t_costos.to_csv("tabla2_costos.csv", index=False, encoding="utf-8-sig")
print(f"Factor IPC 2020→oct 2026 = {factor:.4f}; tarifa UCI 2026 = {tarifa_2026:,.0f}")
print(t_costos.to_string(index=False), "\n")

# ---------- 3. O/E de SAPS III ----------
def oe(g):
    g = g.dropna(subset=["mort_esperada_saps3_general"])
    O = g.mortality.sum(); E = (g.mort_esperada_saps3_general / 100).sum()
    return [len(g), int(O), round(E, 1), round(O / E, 2)]
t_oe = pd.DataFrame([["Sólido"] + oe(df[df.tumor == "Sólido"]),
                     ["Hematológico"] + oe(df[df.tumor == "Hematológico"]),
                     ["Total"] + oe(df),
                     ["Médico"] + oe(df[df.ingreso == "Médico"]),
                     ["Quirúrgico"] + oe(df[df.ingreso == "Quirúrgico"]),
                     ["Entrenamiento"] + oe(df[df.periodo == "Entrenamiento"]),
                     ["Prueba"] + oe(df[df.periodo == "Prueba"])],
                    columns=["Grupo", "Ingresos con SAPS III", "Muertes observadas",
                             "Muertes esperadas", "O/E"])
t_oe.to_csv("tabla3_oe_saps3.csv", index=False, encoding="utf-8-sig")
print(t_oe.to_string(index=False), "\n")

# ---------- 4. Concentración de días-cama ----------
c = (df.dropna(subset=["sapsiii", "dias_estancia"])
       .sort_values("sapsiii", ascending=False, kind="mergesort").reset_index(drop=True))
n = len(c); k = int(round(0.2 * n))
x = np.arange(1, n + 1) / n * 100
y = c.dias_estancia.cumsum() / c.dias_estancia.sum() * 100
y_oraculo = c.dias_estancia.sort_values(ascending=False).cumsum().values / c.dias_estancia.sum() * 100
linea_base = y.iloc[k - 1]; techo = y_oraculo[k - 1]
# sensibilidad a empates de SAPS III
sims = []
for s in range(200):
    cc = c.sample(frac=1, random_state=s).sort_values("sapsiii", ascending=False, kind="mergesort")
    sims.append(cc.dias_estancia.iloc[:k].sum() / cc.dias_estancia.sum() * 100)
top = c.iloc[:k]
res = {"n": n, "k": k, "corte_saps": float(c.sapsiii.iloc[k - 1]), "linea_base": linea_base,
       "empates_min": min(sims), "empates_max": max(sims), "techo_oraculo": techo,
       "captura_prolongadas": top.estancia_prolongada.sum() / c.estancia_prolongada.sum() * 100,
       "captura_muertes": top.mortality.sum() / c.mortality.sum() * 100}
for p in ["Entrenamiento", "Prueba"]:
    cp = c[c.periodo == p]; kp = int(round(0.2 * len(cp)))
    res[f"linea_base_{p}"] = cp.dias_estancia.iloc[:kp].sum() / cp.dias_estancia.sum() * 100
print(json.dumps(res, indent=2))

fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)
ax.plot(x, y, color="#1f4e79", lw=2.2, label="Orden por SAPS III (línea base)")
ax.plot(x, y_oraculo, color="#9e9e9e", lw=1.4, ls=":", label="Orden ideal por días reales (techo)")
ax.plot([0, 100], [0, 100], color="#bdbdbd", lw=1, ls="--", label="Orden al azar")
ax.axvline(20, color="#c00000", lw=1.2)
ax.axhline(45, color="#e69500", lw=1.2, ls="--", label="Meta Guía 2 (≥ 45 %)")
ax.scatter([20], [linea_base], color="#c00000", zorder=5)
ax.annotate(f"{linea_base:.1f} %".replace(".", ","), (20, linea_base), xytext=(24, linea_base - 8),
            color="#c00000", fontsize=10, fontweight="bold")
ax.set_xlabel("Pacientes ordenados de mayor a menor SAPS III (% acumulado)")
ax.set_ylabel("Días-cama acumulados (%)")
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.grid(alpha=.25)
ax.legend(loc="lower right", fontsize=8, frameon=False)
fig.tight_layout(); fig.savefig("figura_concentracion_dias_cama.png")
print("Listo: tablas CSV y figura_concentracion_dias_cama.png")
