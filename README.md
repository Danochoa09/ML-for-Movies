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
│   ├── download_data.py   # descarga el dataset desde Kaggle
│   ├── data_utils.py      # carga + limpieza + tabla unida (reutilizable)
│   └── eda.py             # Análisis Exploratorio de Datos
├── outputs/
│   ├── figures/           # gráficas del EDA
│   └── eda_summary.md     # resumen del EDA
├── requirements.txt
└── README.md
```

## Cómo ejecutar

```bash
pip install -r requirements.txt
python src/download_data.py   # descarga CSV a data/raw/
python src/eda.py             # genera figuras + resumen en outputs/
```

## Estado / Roadmap

- [x] **Fase 1 — EDA** (limpieza, distribuciones, correlaciones, géneros/regiones)
- [x] **Fase 2 — Técnica 1: Clasificación.** Random Forest (vs Reg. Logística)
  predice clase de éxito por rating (`éxito` ≥8, `mediocre` 5–8, `fracaso` <5).
  Entregable: top-3 géneros con mayor P(éxito) por región
  (`src/tecnica1_clasificacion.py`). F1-macro RF ≈ 0.53.
- [ ] **Fase 3 — Técnica 2:** Agrupación (clustering de perfiles de contenido)
- [ ] **Fase 4 — Técnica 3:** (regresión / otra técnica a definir)
- [ ] **Fase 5 — Informe IEEE** doble columna (máx. 6 páginas)
- [ ] **Fase 6 — Video** presentación (máx. 6 min)
