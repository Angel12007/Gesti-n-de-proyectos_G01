# Gestión de Proyectos de Analytics – Grupo G01

Repositorio del anteproyecto del curso Gerencia de Proyectos de Analytics (MIAD, Universidad de los Andes). Reúne los anexos técnicos, el código y los datos que soportan el documento *Anteproyecto_Grupo01* (noviembre de 2026).

**Proyecto.** Dos modelos que estiman, con 14 variables registradas en las primeras 24 horas, el riesgo de estancia en UCI de 7 días o más y de muerte en UCI en adultos con cáncer atendidos en un hospital de cuarto nivel. El objetivo de negocio es anticipar el uso de camas y los costos de referencia asociados.

**Equipo.** Adriana C. Aya, Jeison Castelblanco, Ángel Hernández, Alejandro Ardila.

## Organización

```
anexos/                     Tablas de soporte (A–E, K, L)
codigo/                     Limpieza y utilidades (Anexo F)
codigo/parte3_descriptivo/  Técnicas descriptivas y de negocio (Anexo G)
codigo/parte3_modelado/     Modelado exploratorio de la Guía 4: PCA, SHAP, rangos de la API (Anexo H)
codigo/guia5_modelos/       Selección y evaluación de modelos de la Guía 5 (Anexo J)
datos/                      Bases de datos usadas en el proyecto
```

## Anexos del documento

| Anexo | Contenido | Ubicación |
|---|---|---|
| A | Reglas de limpieza y bitácora por ingreso | [`anexos/A_bitacora_limpieza.csv`](anexos/A_bitacora_limpieza.csv) |
| B | Verificaciones pendientes en el formulario de captura (EDC) | [`anexos/B_verificaciones_edc.csv`](anexos/B_verificaciones_edc.csv) |
| C | Diccionario de variables | [`anexos/C_diccionario_variables.xlsx`](anexos/C_diccionario_variables.xlsx) |
| D | Verificación de la codificación del ECOG (cruce con Karnofsky) | [`anexos/D_ecog_karnofsky.csv`](anexos/D_ecog_karnofsky.csv) |
| E | Porcentaje de datos faltantes por subgrupo | [`anexos/E_faltantes_por_subgrupo.csv`](anexos/E_faltantes_por_subgrupo.csv) |
| F | Código de limpieza y carga de la base limpia | [`codigo/limpieza.py`](codigo/limpieza.py), [`codigo/inicio.py`](codigo/inicio.py) |
| G | Perfil por origen tumoral y periodo, costos de referencia, O/E de SAPS III y curva de concentración de días-cama | [`codigo/parte3_descriptivo/`](codigo/parte3_descriptivo/) |
| H | Modelado exploratorio de la Guía 4 (PCA, SHAP, rangos de validación de la API) | [`codigo/parte3_modelado/`](codigo/parte3_modelado/) |
| I | Glosario | En el documento del anteproyecto |
| J | Selección con validación cruzada y evaluación en la prueba temporal (Guía 5) | [`codigo/guia5_modelos/`](codigo/guia5_modelos/) |
| K | RBS y AMEF completos (49 riesgos, 9 factores) | [`anexos/K_rbs_amef.xlsx`](anexos/K_rbs_amef.xlsx) |
| L | Carta proyecto diligenciada | [`anexos/L_carta_proyecto.xlsx`](anexos/L_carta_proyecto.xlsx) |

## Cómo reproducir

Python 3.10 o superior, desde la raíz del repositorio:

```bash
pip install -r requirements.txt
python codigo/limpieza.py      # base limpia + anexos A y B
python codigo/diccionario.py   # anexo C
python codigo/anexo_d_ecog.py  # anexo D
python codigo/faltantes.py     # anexo E
python -I codigo/guia5_modelos/modelos_g5.py datos/dataset_limpio_v1.csv codigo/guia5_modelos/resultados   # anexo J
```

El anexo J necesita además `xgboost`, `catboost` y `statsmodels` (ver [`codigo/guia5_modelos/README.md`](codigo/guia5_modelos/README.md)).

Para trabajar con la base limpia (`datos/dataset_limpio_v1.csv`, 997 ingresos):

```python
import sys; sys.path.append("codigo")
from inicio import cargar
df, PREDICTORES, DESENLACES = cargar()   # 37 predictores; desenlaces: mortality y estancia_prolongada
```

## Modelos elegidos (Guía 5)

| Desenlace | Modelo | AUROC prueba temporal (IC 95 %) | O/E |
|---|---|---|---|
| Estancia ≥ 7 días | Extra Trees, 14 variables | 0,898 (0,849–0,940) | 0,91 |
| Muerte en UCI | Regresión logística L2, 14 variables | 0,933 (0,874–0,974) | 0,83 |

Entrenamiento: 816 ingresos (julio de 2022 a junio de 2024). Prueba temporal: 181 ingresos (julio de 2024 a agosto de 2025), usada una sola vez.

## Datos

Base anonimizada proveniente del repositorio previo del equipo investigador (licencia CC BY 4.0). No contiene identificadores directos de pacientes ni el nombre de la institución.
