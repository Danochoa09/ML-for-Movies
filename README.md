# IMDb 100k — Aprendizaje Maquinal

Proyecto de la asignatura **Sistemas Inteligentes** (UNAL). Aplicamos **tres
técnicas de aprendizaje maquinal** sobre un dataset de **+100.000 registros**
para extraer conocimiento y comparar los métodos en un informe IEEE de doble
columna.

## Dataset

[IMDb 100000 Movies/TV Shows](https://www.kaggle.com/datasets/kurtnakasato/imdb-100000-moviestvshows)
(Kaggle, 101.606 títulos). Tres archivos:

| Archivo | Contenido |
|---|---|
| `contentDataPrime.csv` | Tabla principal: título, tipo, duración, año, votos, rating, recaudación, certificado, descripción |
| `contentDataGenre.csv` | `dataId → genre` (formato largo, 30 géneros) |
| `contentDataRegion.csv` | `dataId → region` (formato largo, 194 regiones) |

> Valores faltantes codificados como `-1` (numéricos) o vacío (texto).

## Idea del proyecto

El dataset es de hace ~3 años. La meta es **predecir qué películas/series le
gustarían a la gente** (según género, región, duración, etc.) y **contrastarlo
con lo que realmente les gustó** (rating / votos reales).

## Estructura

```
IMDb/
├── data/raw/              # CSV descargados (no se versionan)
├── src/
│   ├── download_data.py        # descarga el dataset desde Kaggle
│   ├── data_utils.py           # carga + limpieza + tabla unida (reutilizable)
│   ├── eda.py                  # Análisis Exploratorio de Datos
│   ├── run_clasificacion.py    # orquesta la comparación de los 3 modelos
│   └── clasificacion/          # paquete de clasificación
│       ├── config.py           # umbrales de clases, semillas, rutas
│       ├── data.py             # construcción del dataset + preprocesador
│       ├── models.py           # los 3 clasificadores (LogReg, RF, XGBoost)
│       ├── evaluate.py         # entrenamiento, métricas y gráficas
│       └── deliverable.py      # top-3 géneros por región (mejor modelo)
├── outputs/
│   ├── figures/                # gráficas (EDA + clasificación)
│   ├── eda_summary.md
│   └── clf_summary.md          # resumen de la comparación de modelos
├── requirements.txt
└── README.md
```

## Las tres técnicas (todas de clasificación)

Mismo problema, tres clasificadores comparados para predecir la **clase de éxito**
de un título a partir de género, región, tipo, duración y año:

| Clase | Regla (rating IMDb) | Interpretación |
|---|---|---|
| `exito`    | rating ≥ 8     | posible éxito |
| `mediocre` | 5 ≤ rating < 8 | sin pena ni gloria |
| `fracaso`  | rating < 5     | posible fracaso |

1. **Regresión Logística** — modelo lineal base/simple.
2. **Random Forest** — ensamble de árboles (bagging).
3. **XGBoost** — boosting de árboles (modelo avanzado).

Entregable de negocio: **top-3 géneros con mayor P(éxito) por región** (mejor modelo).

## Cómo ejecutar

```bash
pip install -r requirements.txt
python src/download_data.py       # descarga CSV a data/raw/
python src/eda.py                 # EDA -> figuras + resumen
python src/run_clasificacion.py   # compara los 3 modelos + entregable
```

## Estado / Roadmap

- [x] **Fase 1 — EDA** (limpieza, distribuciones, correlaciones, géneros/regiones)
- [x] **Fase 2 — Clasificación (3 modelos comparados).** Regresión Logística,
  Random Forest y XGBoost sobre la clase de éxito. F1-macro: LogReg 0.46,
  RF 0.53, XGBoost 0.53 (`src/run_clasificacion.py`).
- [ ] **Fase 3 — Informe IEEE** doble columna (máx. 6 páginas)
- [ ] **Fase 4 — Video** presentación (máx. 6 min)
