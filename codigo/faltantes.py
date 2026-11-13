"""Perfil de faltantes por subgrupo (insumo para decidir MAR/MNAR, riesgo DA1)."""
import pandas as pd
from pathlib import Path
RAIZ = Path(__file__).resolve().parent.parent
df = pd.read_csv(RAIZ / "datos" / "dataset_limpio_v1.csv", parse_dates=["fecha_ingreso"])
df["anio"] = df.fecha_ingreso.dt.year
df["origen"] = df.origin_of_tumor.map({0: "hematológico", 1: "sólido"})
df["ingreso"] = df.type_of_admission.map({1: "médico", 2: "quirúrgico"})
df["muerte_uci"] = df.mortality.map({0: "vivo", 1: "muerto"})
df["estancia_7d"] = df.estancia_prolongada.map({0: "<7 d", 1: ">=7 d"})
vars_ = ["sofa", "ecog", "lactate", "pafi", "calcium", "pt", "ast", "total_bilirubin", "staging_risk"]
filas = []
for g in ["anio", "origen", "ingreso", "muerte_uci", "estancia_7d", "ventilatory_support"]:
    t = df.groupby(g)[vars_].apply(lambda x: x.isna().mean() * 100).round(1)
    t.insert(0, "n", df.groupby(g).size())
    t.index = [f"{g} = {i}" for i in t.index]
    filas.append(t)
out = pd.concat(filas)
out.loc["TOTAL"] = [len(df)] + (df[vars_].isna().mean() * 100).round(1).tolist()
out.to_csv(RAIZ / "anexos" / "E_faltantes_por_subgrupo.csv", encoding="utf-8-sig")
print(out.to_string())
