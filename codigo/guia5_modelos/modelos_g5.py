"""Guía 5 – Selección de modelos con validación cruzada en entrenamiento y
evaluación única en la prueba temporal.

Uso: python -I modelos_g5.py <dataset_limpio_v1.csv> <carpeta_salida>
"""
import sys, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
SEED = 2026
RNG = np.random.default_rng(SEED)
N_BOOT = 1000

ruta, salida = Path(sys.argv[1]), Path(sys.argv[2])
salida.mkdir(parents=True, exist_ok=True)
d = pd.read_csv(ruta)

FUGA = ["sedatives_use", "painkillers_use", "methylene_blue_use", "support renal",
        "delirium", "oncological_treatment"]
P37 = ["sex", "age", "bmi", "type_of_admission", "ecog", "shock", "sepsis", "lactate",
       "excess base", "bicarbonate", "pafi", "hemoglobin", "leukocytes", "neuthrophils",
       "lymphocytes", "platelet", "pt", "aptt", "ast", "alt", "total_bilirubin",
       "direct_bilirubin", "indirect_bilirubin", "creatinine", "bun", "potassium", "sodium",
       "calcium", "apacheii", "sapsiii", "sofa", "vasopressors_use", "Sequ_mol_fish_study",
       "icu_readmission", "origin_of_tumor", "staging_risk", "ventilatory_support"]
P14 = ["age", "type_of_admission", "ecog", "shock", "sepsis", "lactate", "pafi", "platelet",
       "apacheii", "sapsiii", "sofa", "vasopressors_use", "origin_of_tumor", "ventilatory_support"]
assert len(P37) == 37 and not set(FUGA) & set(P37)

tr = d[d.particion_temporal == "entrenamiento"].copy()
te = d[d.particion_temporal == "prueba"].copy()


def imputador(indicador):
    return [("esc", StandardScaler()), ("imp", KNNImputer(n_neighbors=5, add_indicator=indicador))]


def modelos():
    return {
        "Regresión logística (L2)": [LogisticRegression(C=c, max_iter=5000) for c in (0.01, 0.1, 1.0)],
        "Random Forest": [RandomForestClassifier(n_estimators=500, max_depth=m, min_samples_leaf=3,
                                                 random_state=SEED, n_jobs=-1) for m in (4, 8)],
        "Extra Trees": [ExtraTreesClassifier(n_estimators=500, max_depth=m, min_samples_leaf=3,
                                             random_state=SEED, n_jobs=-1) for m in (6, None)],
        "HistGradientBoosting": [HistGradientBoostingClassifier(max_depth=3, learning_rate=lr,
                                                                max_iter=200, random_state=SEED)
                                 for lr in (0.03, 0.1)],
        "XGBoost": [XGBClassifier(n_estimators=300, max_depth=m, learning_rate=0.03, subsample=0.8,
                                  colsample_bytree=0.8, random_state=SEED, n_jobs=2, verbosity=0)
                    for m in (2, 4)],
        "CatBoost": [CatBoostClassifier(iterations=500, depth=dep, learning_rate=0.03,
                                        random_seed=SEED, verbose=0, thread_count=2)
                     for dep in (4, 6)],
    }


def boot_auc(y, p, n=N_BOOT):
    y, p = np.asarray(y), np.asarray(p)
    v = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        v.append(roc_auc_score(y[i], p[i]))
    return np.percentile(v, [2.5, 97.5])


def boot_diff(y, p1, p2, n=N_BOOT):
    y, p1, p2 = map(np.asarray, (y, p1, p2))
    v = []
    for _ in range(n):
        i = RNG.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        v.append(roc_auc_score(y[i], p1[i]) - roc_auc_score(y[i], p2[i]))
    return np.percentile(v, [2.5, 97.5])


def cal_int_slope(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p)).reshape(-1, 1)
    m = LogisticRegression(C=1e6, max_iter=1000).fit(lp, y)
    slope = m.coef_[0][0]
    # intercepto de calibración con pendiente fija en 1 (offset)
    import statsmodels.api as sm
    off = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp.ravel()).fit()
    return float(off.params[0]), float(slope)


def concentracion(df_eval, score, frac=0.2):
    s = df_eval.assign(_s=score).dropna(subset=["dias_estancia"]).sort_values("_s", ascending=False)
    k = int(round(frac * len(s)))
    top = s.head(k)
    return (100 * top.dias_estancia.sum() / s.dias_estancia.sum(),
            100 * top.estancia_prolongada.sum() / s.estancia_prolongada.sum(), k)


cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
resultados, oof_guardado, finales = [], {}, {}

for y_col in ["estancia_prolongada", "mortality"]:
    trn = tr.dropna(subset=[y_col]); y = trn[y_col].astype(int).values
    for conj_nombre, cols in [("37 variables", P37), ("14 variables (formulario)", P14)]:
        for ind in (False, True):
            for nombre, variantes in modelos().items():
                mejor = None
                for est in variantes:
                    pipe = Pipeline(imputador(ind) + [("mod", est)])
                    p = cross_val_predict(pipe, trn[cols], y, cv=cv, method="predict_proba")[:, 1]
                    auc = roc_auc_score(y, p)
                    if mejor is None or auc > mejor[0]:
                        mejor = (auc, brier_score_loss(y, p), est, p)
                clave = (y_col, conj_nombre, ind, nombre)
                resultados.append(dict(desenlace=y_col, variables=conj_nombre,
                                       imputacion="KNN + indicador" if ind else "KNN",
                                       modelo=nombre, auroc_cv=mejor[0], brier_cv=mejor[1],
                                       hiper=str(mejor[2].get_params().get("C", mejor[2].get_params().get("max_depth", mejor[2].get_params().get("depth", mejor[2].get_params().get("learning_rate")))))))
                oof_guardado[clave] = (mejor[2], mejor[3])
                print(clave, round(mejor[0], 3), flush=True)

cvdf = pd.DataFrame(resultados)
cvdf.to_csv(salida / "seleccion_cv_entrenamiento.csv", index=False)

# Regla de selección, declarada antes de mirar la prueba:
# 1) candidatos con AUROC CV a <= 0,01 del mejor; 2) menor Brier CV;
# 3) a igualdad (<= 0,002 de Brier) se prefiere el conjunto de 14 variables y sin indicador.
seleccion = {}
for y_col, g in cvdf.groupby("desenlace"):
    cand = g[g.auroc_cv >= g.auroc_cv.max() - 0.01].copy()
    cand["simple"] = (cand.variables.str.startswith("14")).astype(int) * 2 + (cand.imputacion == "KNN").astype(int)
    cand = cand.sort_values(["brier_cv", "simple"], ascending=[True, False])
    best_b = cand.brier_cv.min()
    empate = cand[cand.brier_cv <= best_b + 0.002].sort_values(["simple", "brier_cv"], ascending=[False, True])
    seleccion[y_col] = empate.iloc[0].to_dict()
json.dump(seleccion, open(salida / "seleccion_previa.json", "w"), indent=2, ensure_ascii=False, default=str)
print("SELECCION", seleccion)

# Evaluación en prueba temporal (una vez) de la combinación elegida y, para transparencia,
# del mejor de cada algoritmo con su conjunto/imputación elegidos.
filas, pred_test = [], {}
for y_col, sel in seleccion.items():
    trn = tr.dropna(subset=[y_col]); tst = te.dropna(subset=[y_col])
    y, yt = trn[y_col].astype(int).values, tst[y_col].astype(int).values
    cols = P37 if sel["variables"].startswith("37") else P14
    ind = sel["imputacion"] != "KNN"
    for nombre in cvdf.modelo.unique():
        est, oof = oof_guardado[(y_col, sel["variables"], ind, nombre)]
        pipe = Pipeline(imputador(ind) + [("mod", est)]).fit(trn[cols], y)
        pt = pipe.predict_proba(tst[cols])[:, 1]
        # recalibración logística con predicciones fuera de pliegue del entrenamiento
        lo = lambda q: np.log(np.clip(q, 1e-6, 1 - 1e-6) / (1 - np.clip(q, 1e-6, 1 - 1e-6)))
        rc = LogisticRegression(C=1e6).fit(lo(oof).reshape(-1, 1), y)
        ptr = rc.predict_proba(lo(pt).reshape(-1, 1))[:, 1]
        lo_ci, hi_ci = boot_auc(yt, pt)
        ci_, sl_ = cal_int_slope(yt, pt); cir, slr = cal_int_slope(yt, ptr)
        fila = dict(desenlace=y_col, modelo=nombre, elegido=(nombre == sel["modelo"]),
                    variables=sel["variables"], imputacion=sel["imputacion"],
                    auroc_cv=float(cvdf[(cvdf.desenlace == y_col) & (cvdf.variables == sel["variables"]) & (cvdf.imputacion == sel["imputacion"]) & (cvdf.modelo == nombre)].auroc_cv.iloc[0]),
                    auroc_prueba=roc_auc_score(yt, pt), ic_inf=lo_ci, ic_sup=hi_ci,
                    auprc=average_precision_score(yt, pt), brier=brier_score_loss(yt, pt),
                    brier_recal=brier_score_loss(yt, ptr),
                    o_e=yt.sum() / pt.sum(), o_e_recal=yt.sum() / ptr.sum(),
                    intercepto=ci_, pendiente=sl_, intercepto_recal=cir, pendiente_recal=slr,
                    observados=int(yt.sum()), n=len(yt))
        if y_col == "estancia_prolongada":
            for f in (0.1, 0.2, 0.3):
                dc, ep, k = concentracion(tst, pt, f)
                fila[f"dias_cama_top{int(f*100)}"] = dc; fila[f"estancias_top{int(f*100)}"] = ep
        filas.append(fila)
        if nombre == sel["modelo"]:
            pred_test[y_col] = (tst, pt, ptr, oof, trn)
        print(y_col, nombre, round(fila["auroc_prueba"], 3), flush=True)

prueba = pd.DataFrame(filas)
prueba.to_csv(salida / "evaluacion_prueba_temporal.csv", index=False)

# Comparación con SAPS III y APACHE II en los mismos pacientes, subgrupos y sensibilidad
extra = []
for y_col, (tst, pt, ptr, oof, trn) in pred_test.items():
    yt = tst[y_col].astype(int).values
    for esc in ["sapsiii", "apacheii"]:
        m = tst[esc].notna().values
        a_e = roc_auc_score(yt[m], tst[esc].values[m]); a_m = roc_auc_score(yt[m], pt[m])
        dl, dh = boot_diff(yt[m], pt[m], tst[esc].values[m])
        el, eh = boot_auc(yt[m], tst[esc].values[m])
        extra.append(dict(desenlace=y_col, analisis=f"vs {esc}", n=int(m.sum()), auroc_modelo=a_m,
                          auroc_escala=a_e, escala_ic_inf=el, escala_ic_sup=eh,
                          diferencia=a_m - a_e, dif_ic_inf=dl, dif_ic_sup=dh))
    if y_col == "mortality":
        m = tst.mort_esperada_saps3_general.notna().values
        extra.append(dict(desenlace=y_col, analisis="O/E SAPS III ecuación general (prueba)",
                          n=int(m.sum()), o_e=yt[m].sum() / (tst.mort_esperada_saps3_general.values[m] / 100).sum()))
    if y_col == "estancia_prolongada":
        for f in (0.1, 0.2, 0.3):
            dc, ep, k = concentracion(tst, tst.sapsiii.fillna(-1).values, f)
            extra.append(dict(desenlace=y_col, analisis=f"SAPS III top {int(f*100)} %", n=k,
                              dias_cama=dc, estancias=ep))
    for col, etiquetas in [("origin_of_tumor", {0: "hematológico", 1: "sólido"}),
                           ("type_of_admission", {1: "médico", 2: "quirúrgico"})]:
        for v, et in etiquetas.items():
            m = (tst[col] == v).values
            if yt[m].sum() >= 3 and yt[m].sum() < m.sum():
                l, h = boot_auc(yt[m], pt[m])
                extra.append(dict(desenlace=y_col, analisis=f"subgrupo {et}", n=int(m.sum()),
                                  eventos=int(yt[m].sum()), auroc_modelo=roc_auc_score(yt[m], pt[m]),
                                  ic_inf=l, ic_sup=h, o_e=yt[m].sum() / pt[m].sum(),
                                  o_e_recal=yt[m].sum() / ptr[m].sum()))
            else:
                extra.append(dict(desenlace=y_col, analisis=f"subgrupo {et}", n=int(m.sum()),
                                  eventos=int(yt[m].sum()), nota="eventos insuficientes"))
    ids_tr = set(trn.id_paciente_aprox)
    m = (~tst.id_paciente_aprox.isin(ids_tr)).values
    extra.append(dict(desenlace=y_col, analisis="sin pacientes repetidos en entrenamiento",
                      n=int(m.sum()), eventos=int(yt[m].sum()), auroc_modelo=roc_auc_score(yt[m], pt[m])))
    pd.DataFrame(dict(record_id=tst.record_id.values, y=yt, p=pt, p_recal=ptr,
                      dias=tst.dias_estancia.values, sapsiii=tst.sapsiii.values)).to_csv(
        salida / f"predicciones_prueba_{y_col}.csv", index=False)

pd.DataFrame(extra).to_csv(salida / "comparaciones_subgrupos.csv", index=False)
print("FIN")
