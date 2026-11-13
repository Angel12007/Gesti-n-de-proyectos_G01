# Gestión de proyectos de analytics – Grupo G01

Anexos técnicos del anteproyecto (Gerencia de Proyectos de Analytics, MIAD, Universidad de los Andes).

**Proyecto:** predecir, con la información de las primeras 24 horas, la muerte en UCI y la estancia de 7 días o más en adultos con cáncer atendidos en la UCI de un hospital de cuarto nivel.

**Equipo:** Adriana C. Aya, Jeison Castelblanco, Ángel Hernández, Alejandro Ardila.

## Anexos de la Guía 4

| Anexo | Parte del informe | Contenido | Ubicación | Estado |
|---|---|---|---|---|
| A | II. Limpieza | Reglas de limpieza y bitácora por ingreso | [`anexos/A_bitacora_limpieza.csv`](anexos/A_bitacora_limpieza.csv) | Disponible |
| B | II. Limpieza | Verificaciones pendientes en el EDC (33) | [`anexos/B_verificaciones_edc.csv`](anexos/B_verificaciones_edc.csv) | Disponible |
| C | I y II | Diccionario de variables (incluye bitácora y verificaciones en pestañas) | [`anexos/C_diccionario_variables.xlsx`](anexos/C_diccionario_variables.xlsx) | Disponible |
| D | II. Limpieza | Verificación de la codificación del ECOG (cruce con Karnofsky) | [`anexos/D_ecog_karnofsky.csv`](anexos/D_ecog_karnofsky.csv) | Disponible |
| E | II. Limpieza | Porcentaje de datos faltantes por subgrupo | [`anexos/E_faltantes_por_subgrupo.csv`](anexos/E_faltantes_por_subgrupo.csv) | Disponible |
| F | II. Limpieza | Código de limpieza y carga de la base limpia | [`codigo/`](codigo/) | Disponible |
| G | III. Técnicas descriptivas y de negocio | Código y figuras | `codigo/parte3_descriptivo/` | Pendiente |
| H | III. Técnicas de modelado | Notebook | `codigo/parte3_modelado/` | Pendiente |

## Estructura

```
anexos/                    Tablas de soporte del informe (A–E)
codigo/
  limpieza.py              Reglas de limpieza: genera la base limpia y los anexos A y B
  diccionario.py           Anexo C
  anexo_d_ecog.py          Anexo D
  faltantes.py             Anexo E
  inicio.py                Carga la base limpia y separa predictores y desenlaces
  parte3_descriptivo/      Anexo G (pendiente)
  parte3_modelado/         Anexo H (pendiente)
datos/
  entrada/                 Base procesada y base cargada (punto de partida)
  dataset_limpio_v1.csv    Base limpia: 997 ingresos, una fila por ingreso
```

## Cómo reproducir

Requiere Python 3.10 o superior.

```bash
pip install -r requirements.txt
python codigo/limpieza.py      # base limpia + anexos A y B
python codigo/diccionario.py   # anexo C
python codigo/anexo_d_ecog.py  # anexo D
python codigo/faltantes.py     # anexo E
```

Para analizar la base limpia:

```python
import sys; sys.path.append("codigo")
from inicio import cargar
df, PREDICTORES, DESENLACES = cargar()   # 37 predictores; desenlaces: mortality y estancia_prolongada
```

## Criterios de limpieza (Parte II)

1. Un valor se corrige solo si tiene una única lectura plausible.
2. Un valor imposible con varias lecturas pasa a faltante y se envía a verificación en el EDC.
3. Los extremos clínicamente posibles se conservan.
4. No se eliminan ingresos ni se imputa sobre la base completa: la imputación se ajusta solo con la partición de entrenamiento.

Seis variables registradas durante la estancia (sedantes, analgésicos, azul de metileno, terapia de reemplazo renal, delirium y tratamiento oncológico) no se usan como predictores: `NO_USAR` en `codigo/inicio.py`.

## Datos

Base anonimizada proveniente del repositorio previo del equipo investigador (licencia CC BY 4.0). No contiene identificadores directos de pacientes ni el nombre de la institución.
