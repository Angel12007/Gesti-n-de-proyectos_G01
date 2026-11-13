"""Genera diccionario_variables.xlsx a partir de dataset_limpio_v1.csv."""
import pandas as pd
from pathlib import Path
RAIZ = Path(__file__).resolve().parent.parent
df = pd.read_csv(RAIZ / "datos" / "dataset_limpio_v1.csv")
I, H24, EST, EGR, ANT = "Ingreso", "Primeras 24 h", "Durante la estancia (confirmado)", "Egreso / desenlace", "Antecedente al ingreso"
T = set(["age","origin_of_tumor","type_of_admission","ecog","apacheii","sapsiii","sofa","lactate","pafi","platelet",
         "ventilatory_support","vasopressors_use","shock","sepsis"])
# variable: (campo EDC, descripción, unidad, codificación, rango plausible, momento)
D = {
 "record_id": ("record_id","Identificador del ingreso","-","entero","-",I),
 "id_paciente_aprox": ("derivada","Paciente aproximado (sexo, peso, talla, sitio, edad ±2)","-","entero","-","-"),
 "sex": ("sexo","Sexo","-","1 hombre, 2 mujer","1-2",ANT),
 "age": ("edad","Edad","años","continua","18-100",ANT),
 "bmi": ("imc","Índice de masa corporal","kg/m²","continua","10-70",I),
 "type_of_admission": ("categoria","Tipo de ingreso","-","1 médico, 2 quirúrgico","1-2",I),
 "ecog": ("ecog","Estado funcional ECOG","-","0-4 (recodificado desde 1-5)","0-4",ANT),
 "origin_of_tumor": ("ufc","Origen del tumor","-","0 hematológico, 1 sólido","0-1",ANT),
 "staging_risk": ("estadio_cat_final / L-MM / leucemias","Estadio o riesgo global","-","0 sólido temprano / leucemia riesgo bajo / linfoma o mieloma temprano; 1 sólido avanzado / leucemia riesgo alto / linfoma o mieloma avanzado","0-1",ANT),
 "Sequ_mol_fish_study": ("result_secuenc_final","Resultado de secuenciación / FISH","-","1 / 2","1-2",ANT),
 "icu_readmission": ("nro_ingresos_cat","Ingresos previos a UCI","-","0, 1, 2+","0-2",ANT),
 "shock": ("choque","Choque","-","0 no, 1 sí","0-1",I+" (confirmado)"),
 "sepsis": ("sepsis","Sepsis","-","0 no, 1 sí","0-1",I+" (confirmado)"),
 "lactate": ("lactatoing","Lactato","mmol/L","continua","0,1-30",I),
 "excess base": ("being","Exceso de base","mmol/L","continua","-30 a 30",I),
 "bicarbonate": ("hco3ing","Bicarbonato","mmol/L","continua","2-50",I),
 "pafi": ("pafiing","PaO2/FiO2","mmHg","continua","20-700",I),
 "hemoglobin": ("hbing","Hemoglobina","g/dL","continua","2-25",I),
 "leukocytes": ("leucosingr","Leucocitos","/µL","continua","10-500.000",I),
 "neuthrophils": ("neutingr","Neutrófilos","/µL","continua","0-leucocitos",I),
 "lymphocytes": ("linfingre","Linfocitos","/µL","continua","0-leucocitos",I),
 "platelet": ("plaqingr","Plaquetas","/µL","continua","1.000-2.000.000",I),
 "pt": ("pting","Tiempo de protrombina","s","continua","8-100",I),
 "aptt": ("pttingr","Tiempo parcial de tromboplastina","s","continua","15-150",I),
 "ast": ("tgoing","AST (TGO)","U/L","continua","1-10.000",I),
 "alt": ("tgping","ALT (TGP)","U/L","continua","1-10.000",I),
 "total_bilirubin": ("bilitorring","Bilirrubina total","mg/dL","continua","0-50",I),
 "direct_bilirubin": ("bding","Bilirrubina directa","mg/dL","continua","0-total",I),
 "indirect_bilirubin": ("biing","Bilirrubina indirecta","mg/dL","continua","0-total",I),
 "creatinine": ("creing","Creatinina","mg/dL","continua","0,1-20",I),
 "bun": ("buning","Nitrógeno ureico","mg/dL","continua","1-200",I),
 "potassium": ("potasioing","Potasio","mmol/L","continua","1,5-10",I),
 "sodium": ("sodioing","Sodio","mmol/L","continua","100-180",I),
 "calcium": ("calciing","Calcio total","mg/dL","continua","5-20",I),
 "apacheii": ("apacheii","APACHE II","puntos","continua","0-71",H24),
 "sapsiii": ("saps_3","SAPS III","puntos","entero","0-217",H24),
 "sofa": ("sofa_24h","SOFA","puntos","continua","0-24",H24),
 "support renal": ("soportere","Terapia de reemplazo renal","-","0 no, 1 sí","0-1",EST),
 "vasopressors_use": ("medicamento_uci___1","Vasopresores","-","0 no, 1 sí","0-1",I+" (confirmado)"),
 "sedatives_use": ("medicamento_uci___3","Sedantes","-","0 no, 1 sí","0-1",EST),
 "painkillers_use": ("medicamento_uci___4","Analgésicos","-","0 no, 1 sí","0-1",EST),
 "methylene_blue_use": ("medicamento_uci___8","Azul de metileno","-","0 no, 1 sí","0-1",EST),
 "delirium": ("delirium","Delirium","-","0 no, 1 sí","0-1",EST),
 "ventilatory_support": ("vmni/cafo/vmi","Soporte ventilatorio (VMNI, CNAF o VMI)","-","0 no, 1 sí","0-1",H24+" (confirmado)"),
 "oncological_treatment": ("quimio/inmuno/terapia_dirigida/hormonal/medicamento_uci___11","Tratamiento oncológico en UCI","-","0 no, 1 sí","0-1",EST),
 "fecha_ingreso": ("fecha_ing","Fecha de ingreso a UCI","fecha","AAAA-MM-DD","2022-07 a 2025-08",I),
 "dias_estancia": ("diastot","Días de estancia en UCI","días","entero","0-365",EGR),
 "mortality": ("mortalidad","Muerte en UCI","-","0 vivo, 1 muerto","0-1",EGR),
 "death_30d": ("sobrevida30","Muerte en los 30 días tras el egreso vivo de la UCI (no aplica si murió en UCI)","-","0 vivo, 1 muerto","0-1",EGR),
 "muerte_uci_o_30d_post_egreso": ("derivada","Muerte en UCI o en los 30 días tras el egreso","-","0/1","0-1",EGR),
 "estancia_prolongada": ("derivada","Estancia >= 7 días","-","0/1","0-1",EGR),
 "muerte_o_estancia_prolongada": ("derivada","Compuesto","-","0/1","0-1",EGR),
 "mort_esperada_saps3": ("mortesper_saps3","Mortalidad esperada SAPS III (registrada)","%","continua","0-100",H24),
 "mort_esperada_saps3_general": ("derivada","Mortalidad esperada SAPS III, ecuación general","%","continua","0-100",H24),
 "mort_esperada_apache": ("mortaespe","Mortalidad esperada APACHE II","%","continua","0-100",H24),
 "particion_temporal": ("derivada","Partición temporal (corte 2024-07-01)","-","entrenamiento / prueba","-","-"),
 "version_limpieza": ("derivada","Versión de la limpieza","-","texto","-","-"),
}
rows = []
for c in df.columns:
    campo, desc, uni, cod, rango, mom = D.get(c, ("?","?","?","?","?","?"))
    rol = "desenlace" if mom == EGR else ("auxiliar" if mom == "-" or c.startswith("mort_esperada") else "predictor")
    if c in ("record_id", "fecha_ingreso"):
        rol = "identificador"
    rows.append({"variable": c, "campo_EDC": campo, "descripcion": desc, "unidad": uni, "codificacion": cod,
                 "rango_plausible": rango, "momento_registro": mom, "rol": rol,
                 "en_formulario_tablero": "sí" if c in T else "",
                 "riesgo_fuga_DA4": "excluir como predictor" if mom == EST and rol == "predictor" else "",
                 "pct_faltante": round(df[c].isna().mean()*100, 1),
                 "n_valores_distintos": df[c].nunique()})
out = pd.DataFrame(rows)
with pd.ExcelWriter(RAIZ / "anexos" / "C_diccionario_variables.xlsx") as w:
    out.to_excel(w, sheet_name="diccionario", index=False)
    pd.read_csv(RAIZ / "anexos" / "A_bitacora_limpieza.csv").to_excel(w, sheet_name="bitacora", index=False)
    pd.read_csv(RAIZ / "anexos" / "B_verificaciones_edc.csv").to_excel(w, sheet_name="queries", index=False)
print(out[["variable","momento_registro","rol","riesgo_fuga_DA4","pct_faltante"]].to_string())
