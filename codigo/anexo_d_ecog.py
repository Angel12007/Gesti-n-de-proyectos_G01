"""Anexo D: cruce ECOG (código del EDC) x Karnofsky en ingresos únicos."""
from pathlib import Path
import pandas as pd
RAIZ = Path(__file__).resolve().parent.parent
r = pd.read_csv(RAIZ / "datos" / "entrada" / "datos_cargados.csv", low_memory=False).drop_duplicates("record_id")
t = pd.crosstab(r["karnofsky"].fillna(-1), r["ecog"].fillna(-1))
t.index = ["Sin dato" if i == -1 else int(i) for i in t.index]
t.columns = ["Sin dato" if c == -1 else f"Código {int(c)}" for c in t.columns]
t.index.name = "Karnofsky"
t.to_csv(RAIZ / "anexos" / "D_ecog_karnofsky.csv", encoding="utf-8-sig")
print(t)
