"""Guía 4 – Limpieza de datos (Parte II). Base limpia v1.

Entrada : dataset_modelado.csv (salida de procesar_datos.py del repo del grupo)
          datos_cargados.csv   (solo para peso, talla y sitio del cáncer, usados
                                en el identificador aproximado de paciente)
Salidas : dataset_limpio_v1.csv, bitacora_limpieza.csv, queries_fuente.csv

Uso en otro script o notebook:
    from limpieza import limpiar
    df_limpio, bitacora = limpiar(df_modelado, df_cargado)

Principios:
  1. Se corrige solo cuando hay UNA lectura plausible del dato (p. ej. IMC
     calculado con talla en cm, potasio 40.5 -> 4.05).
  2. Si el valor es imposible y admite varias lecturas, pasa a faltante y se
     genera una query para verificar contra el EDC.
  3. Los extremos clínicamente posibles se conservan y se marcan.
  4. No se imputa ni se eliminan pacientes: la imputación va dentro de cada
     partición de entrenamiento.
Cada cambio queda en la bitácora con record_id, valor original y nuevo.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

AQUI = Path(__file__).resolve().parent
RAIZ = AQUI.parent
ENTRADA = RAIZ / "datos" / "entrada"  # base procesada y base cargada del pipeline del grupo

VERSION = "v1.1"

# Registradas durante la estancia (confirmado por el equipo clínico, 3 nov 2026):
# no se usan como predictores al ingreso porque filtran información (riesgo DA4).
VARIABLES_FUGA = ["sedatives_use", "painkillers_use", "methylene_blue_use",
                  "support renal", "delirium", "oncological_treatment"]


class Bitacora:
    def __init__(self):
        self.filas = []

    def registrar(self, df, mascara, var, nuevo, regla, tipo, justificacion, query=False):
        """Aplica el cambio en df[var] donde mascara es True y lo registra."""
        idx = df.index[mascara.fillna(False)]
        for i in idx:
            original = df.at[i, var]
            valor = nuevo(original) if callable(nuevo) else nuevo
            self.filas.append({
                "regla": regla, "variable": var, "record_id": int(df.at[i, "record_id"]),
                "valor_original": original, "valor_nuevo": valor, "tipo": tipo,
                "justificacion": justificacion, "requiere_query": query,
            })
            df.at[i, var] = valor
        return len(idx)

    def marcar(self, df, mascara, var, regla, justificacion, query=False, tipo="conservado (extremo plausible)"):
        """Registra sin cambiar (extremo plausible que se conserva)."""
        for i in df.index[mascara.fillna(False)]:
            self.filas.append({
                "regla": regla, "variable": var, "record_id": int(df.at[i, "record_id"]),
                "valor_original": df.at[i, var], "valor_nuevo": df.at[i, var],
                "tipo": tipo, "justificacion": justificacion,
                "requiere_query": query,
            })

    def tabla(self):
        return pd.DataFrame(self.filas)


def id_paciente_aproximado(df, cargado):
    """Agrupa ingresos que probablemente son del mismo paciente.

    La base no trae documento ni ID de paciente: cada record_id es un ingreso.
    Se agrupan registros con igual sexo, peso, talla y sitio del cáncer y edad
    que difiere en <= 2 años. Es una aproximación para auditar la partición.
    """
    aux = (cargado.drop_duplicates("record_id")[["record_id", "peso", "talla", "sitiocancer"]])
    d = df[["record_id", "sex", "age", "fecha_ingreso"]].merge(aux, on="record_id", how="left")
    d = d.sort_values(["sex", "peso", "talla", "sitiocancer", "age"])
    pid, ultimo, actual = {}, None, 0
    for _, r in d.iterrows():
        clave = (r.sex, r.peso, r.talla, r.sitiocancer)
        completo = not any(pd.isna(x) for x in clave)
        if completo and ultimo is not None and clave == ultimo[0] and r.age - ultimo[1] <= 2:
            pass
        else:
            actual += 1
        pid[r.record_id] = actual
        ultimo = (clave, r.age) if completo else None
    return df["record_id"].map(pid)


def limpiar(df_modelado, df_cargado=None):
    df = df_modelado.copy()
    b = Bitacora()
    J = {}  # textos de justificación reutilizados

    # ---------- A. Codificación y nombres ----------
    # A1. AST/ALT: tgoing = TGO = AST; tgping = TGP = ALT. El código heredado los invirtió.
    df = df.rename(columns={"alt": "ast", "ast": "alt"})
    b.filas.append({"regla": "A1", "variable": "ast/alt", "record_id": None, "valor_original": "alt<-tgoing; ast<-tgping",
                    "valor_nuevo": "ast<-tgoing; alt<-tgping", "tipo": "corrección de etiqueta",
                    "justificacion": "TGO = AST y TGP = ALT; el renombrado heredado estaba invertido.", "requiere_query": False})

    # A2. ECOG: el EDC codifica 1-5 = ECOG 0-4 (verificado contra Karnofsky: 315/337 con KPS 100 tienen código 1).
    df["ecog"] = df["ecog"] - 1
    b.filas.append({"regla": "A2", "variable": "ecog", "record_id": None, "valor_original": "códigos 1-5",
                    "valor_nuevo": "ECOG 0-4 (código - 1)", "tipo": "recodificación",
                    "justificacion": "Cruce con Karnofsky: KPS 100 -> código 1 en 315/337 ingresos únicos; KPS 90 -> código 2 en 84/121. Equivale a ECOG 0-4 + 1.",
                    "requiere_query": False})

    # A3. Binarias 1 = no / 2 = sí -> 0/1
    for var in ["shock", "sepsis", "delirium", "support renal"]:
        df[var] = df[var].map({1: 0, 2: 1})
    b.filas.append({"regla": "A3", "variable": "shock, sepsis, delirium, support renal", "record_id": None,
                    "valor_original": "1 = no, 2 = sí", "valor_nuevo": "0 = no, 1 = sí", "tipo": "recodificación",
                    "justificacion": "Unificar con el resto de binarias (vasopresores, soporte ventilatorio) y con los desenlaces.",
                    "requiere_query": False})

    # A4. Estadio/riesgo global: 1 = temprano/bajo, 2 = avanzado/alto -> 0/1
    df["staging_risk"] = df["staging_risk"].map({1: 0, 2: 1})
    b.filas.append({"regla": "A4", "variable": "staging_risk", "record_id": None,
                    "valor_original": "1 / 2", "valor_nuevo": "0 = sólido temprano, leucemia de riesgo bajo, linfoma o mieloma temprano; 1 = sólido avanzado, leucemia de riesgo alto, linfoma o mieloma avanzado",
                    "tipo": "recodificación", "justificacion": "Codificación confirmada por el equipo clínico.", "requiere_query": False})

    # A5. Variables registradas durante la estancia: se conservan para describir, no para predecir.
    b.filas.append({"regla": "A5", "variable": ", ".join(VARIABLES_FUGA), "record_id": None,
                    "valor_original": "predictores", "valor_nuevo": "excluidas como predictores (lista VARIABLES_FUGA)",
                    "tipo": "exclusión por fuga de información",
                    "justificacion": "Se registran durante la estancia, no en las primeras 24 h (DA4). Vasopresores, choque y sepsis sí se registran al ingreso.",
                    "requiere_query": False})

    # ---------- B. Errores de unidad o de digitación con una sola lectura posible ----------
    m = df["bmi"] < 1
    b.registrar(df, m, "bmi", lambda v: round(v * 10000, 2), "B1", "corrección determinística",
                "IMC calculado con talla en cm (kg/cm²); x10.000 lo lleva a kg/m².")

    m = (df["leukocytes"] < 100) & ((df["neuthrophils"] > df["leukocytes"]) | (df["lymphocytes"] > df["leukocytes"]))
    filas = df.index[m]
    b.registrar(df, m, "leukocytes", lambda v: v * 1000, "B2", "corrección determinística",
                "Leucocitos en x10³/µL mientras neutrófilos y linfocitos están en /µL (diferencial > total).")
    m2 = df.index.isin(filas) & (df["neuthrophils"] < 100) & (df["neuthrophils"] * 1000 <= df["leukocytes"])
    b.registrar(df, pd.Series(m2, index=df.index), "neuthrophils", lambda v: v * 1000, "B2", "corrección determinística",
                "Mismo registro: neutrófilos también en x10³/µL.")

    m = (df["platelet"] > 0) & (df["platelet"] < 1000)
    b.registrar(df, m, "platelet", lambda v: v * 1000, "B3", "corrección determinística",
                "Plaquetas en x10³/µL (190, 220, 10); el resto de la base está en /µL.")

    m = df["potassium"] > 15
    b.registrar(df, m, "potassium", lambda v: round(v / 10, 2), "B4", "corrección determinística",
                "Punto decimal corrido: 40.5 solo tiene la lectura 4.05 mmol/L.")

    # ---------- C. Valores imposibles con más de una lectura -> faltante + query ----------
    C = [
        ("lactate", (df["lactate"] > 30) | (df["lactate"] <= 0), "C1",
         "Lactato fuera de 0,1-30 mmol/L (231, 53, 0). 231 puede ser 2,31 o 23,1: no hay una sola lectura."),
        ("bicarbonate", (df["bicarbonate"] < 0) | ((df["bicarbonate"] - df["excess base"].abs()).abs() < 0.05)
         & (df["bicarbonate"] < 5), "C2",
         "HCO3 negativo o igual al valor absoluto del exceso de base: el campo se llenó con el BE."),
        ("hemoglobin", df["hemoglobin"] > 25, "C3",
         "Hemoglobina > 25 g/dL (4370; 42,5 parece hematocrito)."),
        ("lymphocytes", df["lymphocytes"] > df["leukocytes"], "C4",
         "Linfocitos mayores que leucocitos totales después de corregir unidades."),
        ("platelet", df["platelet"] == 0, "C10", "Plaquetas = 0 /µL es incompatible con la vida; dato no registrado."),
        ("pt", df["pt"] < 8, "C5", "TP de 1,07 s: parece INR digitado en el campo de TP."),
        ("creatinine", df["creatinine"] > 20, "C6",
         "Creatinina 68 mg/dL es incompatible con BUN 18,6; probablemente µmol/L (0,77 mg/dL). Se verifica antes de convertir."),
        ("sodium", (df["sodium"] < 100), "C7",
         "Sodio 4,5 mmol/L: idéntico al potasio del mismo registro, campo copiado."),
        ("calcium", df["calcium"] < 5, "C8",
         "Calcio total < 5 mg/dL (3,58-4,9): compatible con calcio iónico en mg/dL registrado en el campo de calcio total."),
        ("sapsiii", df["sapsiii"] < 1, "C9",
         "SAPS III de 0,85: el puntaje es entero (0-217); parece una probabilidad registrada en el campo del puntaje."),
    ]
    for var, mask, regla, just in C:
        b.registrar(df, mask, var, np.nan, regla, "a faltante + query", just, query=True)

    # ---------- D. Extremos posibles que se conservan ----------
    D = [
        ("bmi", (df["bmi"] < 13) | (df["bmi"] > 40), "D1", "IMC < 13 (caquexia) o > 40: posible."),
        ("lactate", df["lactate"] > 15, "D2", "Lactato 16-23 mmol/L: posible en choque."),
        ("bicarbonate", df["bicarbonate"] < 5, "D3", "HCO3 3-4 con BE -22 a -24: acidosis metabólica grave, coherente."),
        ("leukocytes", df["leukocytes"] < 500, "D4", "Leucopenia profunda; 94 % son tumores hematológicos (aplasia)."),
        ("leukocytes", df["leukocytes"] > 100000, "D5", "Hiperleucocitosis: posible en leucemias."),
        ("platelet", df["platelet"] > 1_000_000, "D6", "Trombocitosis > 1 millón: posible."),
        ("calcium", df["calcium"] > 15, "D7", "Hipercalcemia > 15 mg/dL: posible en cáncer."),
        ("potassium", df["potassium"] < 2, "D8", "Hipopotasemia < 2: posible."),
        ("sapsiii", (df["sapsiii"] >= 1) & (df["sapsiii"] < 16), "D10",
         "SAPS III < 16: posible (rango teórico 0-217; Moreno 2005), pero bajo y no coherente con la mortalidad esperada registrada. Se conserva y se verifica."),
        ("sapsiii", (df["sapsiii"] >= 1) & df["sapsiii"].notna() & ((df["sapsiii"] % 1) != 0), "D11",
         "SAPS III con decimales (17,3; 18,8; 24,13): el puntaje es entero. Valor posible en magnitud; se conserva y se verifica."),
        ("apacheii", df["apacheii"] == 0, "D9",
         "APACHE II = 0: valor posible; confirmado por el equipo clínico. Se conserva."),
    ]
    for var, mask, regla, just in D:
        b.marcar(df, mask, var, regla, just, query=(regla in ("D10", "D11")))

    # ---------- E. Faltantes estructurales y variables derivadas ----------
    # E1. La muerte a 30 días se cuenta desde el EGRESO de la UCI (confirmado 3 nov 2026):
    # solo aplica a quienes salen vivos. Los faltantes en muertes en UCI son "no aplica", no datos perdidos.
    m = df["death_30d"].isna() & (df["mortality"] == 1)
    b.marcar(df, m, "death_30d", "E1",
             "No aplica: murió en la UCI y el seguimiento de 30 días empieza en el egreso vivo. No se imputa.",
             tipo="no aplica (faltante estructural)")
    m = df["death_30d"].notna() & (df["mortality"] == 1)
    b.registrar(df, m, "death_30d", np.nan, "E1b", "a faltante + query",
                "Muerte en UCI con dato de muerte a 30 días post-egreso: contradictorio.", query=True)
    df["muerte_uci_o_30d_post_egreso"] = np.where(
        df["mortality"] == 1, 1, df["death_30d"]).astype("float")
    b.filas.append({"regla": "E1c", "variable": "muerte_uci_o_30d_post_egreso", "record_id": None,
                    "valor_original": "-", "valor_nuevo": "1 si murió en UCI o en los 30 días tras el egreso; 0 si sobrevivió",
                    "tipo": "variable derivada",
                    "justificacion": "Desenlace definido para toda la cohorte; la muerte a 30 d solo existe para los egresos vivos.",
                    "requiere_query": False})

    lg = -32.6659 + np.log(df["sapsiii"] + 20.5958) * 7.3068
    df["mort_esperada_saps3_general"] = (100 / (1 + np.exp(-lg))).round(2)
    b.filas.append({"regla": "E2", "variable": "mort_esperada_saps3_general", "record_id": None,
                    "valor_original": "mortalidad esperada registrada (12,4 % faltante; mezcla de ecuaciones)",
                    "valor_nuevo": "recalculada con la ecuación general de SAPS III para todos", "tipo": "variable derivada",
                    "justificacion": "La registrada coincide con la ecuación general en ~80 % y con la de Centro/Suramérica en otros casos; 96 tienen SAPS III sin probabilidad. Una sola ecuación hace comparable el O/E.",
                    "requiere_query": False})

    # ---------- F. Granularidad: ingresos del mismo paciente ----------
    if df_cargado is not None:
        df["id_paciente_aprox"] = id_paciente_aproximado(df, df_cargado)
        b.filas.append({"regla": "F1", "variable": "id_paciente_aprox", "record_id": None,
                        "valor_original": "-", "valor_nuevo": "identificador aproximado (sexo, peso, talla, sitio, edad ±2)",
                        "tipo": "variable derivada",
                        "justificacion": "Cada record_id es un ingreso. Hay pacientes con varios ingresos; sirve para validación agrupada y auditar la partición temporal.",
                        "requiere_query": False})

    df["version_limpieza"] = VERSION
    return df, b.tabla()


def main():
    modelado = pd.read_csv(ENTRADA / "dataset_modelado.csv")
    cargado = pd.read_csv(ENTRADA / "datos_cargados.csv", low_memory=False)
    df, bit = limpiar(modelado, cargado)

    df.to_csv(RAIZ / "datos" / "dataset_limpio_v1.csv", index=False)
    bit.to_csv(RAIZ / "anexos" / "A_bitacora_limpieza.csv", index=False, encoding="utf-8-sig")
    q = bit[bit["requiere_query"] == True]
    q.to_csv(RAIZ / "anexos" / "B_verificaciones_edc.csv", index=False, encoding="utf-8-sig")

    print(f"Registros: {len(df)} | columnas: {df.shape[1]}")
    print(bit.groupby(["regla", "variable", "tipo"]).size().to_string())
    print(f"Queries a la fuente: {len(q)}")


if __name__ == "__main__":
    main()
