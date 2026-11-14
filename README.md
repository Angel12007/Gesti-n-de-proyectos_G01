# Gestión de Proyectos de Analytics – Grupo G01

Repositorio del anteproyecto del curso Gerencia de Proyectos de Analytics (MIAD, Universidad de los Andes). Reúne los anexos técnicos, el código y los datos que soportan los informes del proyecto.

**Proyecto.** Herramienta que estima, con la información de las primeras 24 horas, el riesgo de muerte en UCI y de estancia de 7 días o más en adultos con cáncer atendidos en la UCI de un hospital de cuarto nivel. El objetivo de negocio es anticipar el uso de camas y los costos asociados.

**Equipo.** Adriana C. Aya, Jeison Castelblanco, Ángel Hernández, Alejandro Ardila.

## Organización

```
anexos/     Tablas de soporte de los informes
codigo/     Scripts y notebooks del análisis
datos/      Bases de datos usadas en el proyecto
```

## Anexos de limpieza de datos (Guía 4)

| Anexo | Contenido | Ubicación |
|---|---|---|
| A | Reglas de limpieza y bitácora por ingreso | [`anexos/A_bitacora_limpieza.csv`](anexos/A_bitacora_limpieza.csv) |
| B | Verificaciones pendientes en el formulario de captura (EDC) | [`anexos/B_verificaciones_edc.csv`](anexos/B_verificaciones_edc.csv) |
| C | Diccionario de variables, con la bitácora y las verificaciones en pestañas | [`anexos/C_diccionario_variables.xlsx`](anexos/C_diccionario_variables.xlsx) |
| D | Verificación de la codificación del ECOG (cruce con Karnofsky) | [`anexos/D_ecog_karnofsky.csv`](anexos/D_ecog_karnofsky.csv) |
| E | Porcentaje de datos faltantes por subgrupo | [`anexos/E_faltantes_por_subgrupo.csv`](anexos/E_faltantes_por_subgrupo.csv) |
| F | Código de limpieza y carga de la base limpia | [`codigo/`](codigo/) |

Para reproducirlos (Python 3.10 o superior, desde la raíz del repositorio):

```bash
pip install -r requirements.txt
python codigo/limpieza.py      # base limpia + anexos A y B
python codigo/diccionario.py   # anexo C
python codigo/anexo_d_ecog.py  # anexo D
python codigo/faltantes.py     # anexo E
```

Para trabajar con la base limpia (`datos/dataset_limpio_v1.csv`, 997 ingresos):

```python
import sys; sys.path.append("codigo")
from inicio import cargar
df, PREDICTORES, DESENLACES = cargar()   # 37 predictores; desenlaces: mortality y estancia_prolongada
```

## Datos

Base anonimizada proveniente del repositorio previo del equipo investigador (licencia CC BY 4.0). No contiene identificadores directos de pacientes ni el nombre de la institución.
