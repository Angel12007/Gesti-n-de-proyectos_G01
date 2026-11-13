"""Punto de partida para todos. Copie estas líneas al inicio de su notebook.

    from inicio import cargar
    df, PREDICTORES, DESENLACES = cargar()
"""
import pandas as pd
from pathlib import Path
RAIZ = Path(__file__).resolve().parent.parent

DESENLACES = {
    "mortality": "Muerte en UCI (0 = vivo, 1 = muerto)",
    "estancia_prolongada": "Estancia de 7 días o más (0 = no, 1 = sí)",
}
# Registradas durante la estancia: NO se usan para predecir (fuga de información).
NO_USAR = ["sedatives_use", "painkillers_use", "methylene_blue_use", "support renal",
           "delirium", "oncological_treatment"]
# Desenlaces, identificadores y columnas auxiliares: tampoco son predictores.
NO_PREDICTORES = NO_USAR + list(DESENLACES) + [
    "record_id", "id_paciente_aprox", "fecha_ingreso", "particion_temporal", "version_limpieza",
    "dias_estancia", "death_30d", "muerte_o_estancia_prolongada", "muerte_uci_o_30d_post_egreso",
    "mort_esperada_saps3", "mort_esperada_saps3_general", "mort_esperada_apache"]


def cargar(ruta=RAIZ / "datos" / "dataset_limpio_v1.csv"):
    df = pd.read_csv(ruta, parse_dates=["fecha_ingreso"])
    predictores = [c for c in df.columns if c not in NO_PREDICTORES]
    return df, predictores, DESENLACES


if __name__ == "__main__":
    df, P, D = cargar()
    print(f"{len(df)} ingresos | {len(P)} predictores")
    print(P)
    print(df["particion_temporal"].value_counts().to_dict())
    for d in D:
        print(d, f"{df[d].mean():.1%}")
