# Anexo J. Selección y evaluación de modelos (Guía 5)

`modelos_g5.py` compara 6 algoritmos (regresión logística L2, Random Forest, Extra Trees,
HistGradientBoosting, XGBoost, CatBoost) × 2 conjuntos de variables (37 y 14 del formulario)
× 2 imputaciones (KNN, KNN + indicador de ausencia), con validación cruzada estratificada de
5 pliegues solo en el entrenamiento (816 ingresos, jul 2022–jun 2024).

La regla de selección se fija antes de mirar la prueba: AUROC de validación cruzada a ≤ 0,01 del
mejor → menor Brier → a igualdad (≤ 0,002), 14 variables y sin indicador. El modelo elegido se
evalúa una sola vez en la prueba temporal (181 ingresos, jul 2024–ago 2025).

Ejecución (desde la raíz del repositorio):

    python -I codigo/guia5_modelos/modelos_g5.py datos/dataset_limpio_v1.csv codigo/guia5_modelos/resultados

Requiere scikit-learn, xgboost, catboost, statsmodels, pandas y numpy. Semilla fija: 2026.

## Resultados (carpeta `resultados/`)

| Archivo | Contenido |
| --- | --- |
| seleccion_cv_entrenamiento.csv | AUROC y Brier de validación cruzada de las 24 configuraciones por desenlace |
| seleccion_previa.json | Configuración elegida por la regla, antes de evaluar en prueba |
| evaluacion_prueba_temporal.csv | Tabla J.1: AUROC (IC 95 % bootstrap), AUPRC, Brier, O/E, calibración, concentración de días-cama |
| comparaciones_subgrupos.csv | Comparación pareada con SAPS III y APACHE II, subgrupos y sensibilidad sin pacientes repetidos |
| predicciones_prueba_*.csv | Probabilidades por ingreso de la prueba temporal |
