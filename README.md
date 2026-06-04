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
│       └── deliverable.py      # ranking de géneros + mejores regiones
├── outputs/
│   ├── figures/                # gráficas (EDA + clasificación)
│   ├── eda_summary.md
│   └── clf_summary.md          # resumen de la comparación de modelos
├── requirements.txt
└── README.md
```

## Las tres técnicas (todas de clasificación)

Mismo problema, tres clasificadores comparados para predecir el **nivel de éxito**
de un título (`fracaso` / `mediocre` / `exito`) a partir de género, región, tipo y
duración.

**Definición de éxito — IEP (Índice de Rendimiento Económico):** un puntaje
continuo en [0,1] que combina calidad, alcance y rentabilidad. Como `votes` y
`gross` siguen ley de potencias, se les aplica **log** antes de normalizar
(Min-Max `N(·)`):

> Con `gross`:  `IEP = 0.2·N(rating) + 0.3·N(log votes) + 0.5·N(log gross)`
> Sin `gross` (series de TV / cine indie, ~85%):  `IEP = 0.2·N(rating) + 0.8·N(log votes)`

El `gross` falta en el 85% de los títulos (las series no tienen taquilla); por eso
**su peso (0.5) se transfiere a los votos** cuando no existe, sin perder registros.
El IEP se corta en **3 clases por terciles**. Esto corrige el sesgo de usar solo
rating (documentales de nicho como "mayores éxitos") al pesar el alcance comercial.

**Validación temporal:** se entrena con títulos **anteriores a 2020** y se prueba
con títulos de **2020 en adelante** (simula predecir el futuro). Normalización,
terciles y agrupado de regiones se calculan solo con train (sin fuga de datos). El
año **no** se usa como variable predictora (es el eje de la partición).

1. **Regresión Logística** — modelo lineal base/simple.
2. **Random Forest** — ensamble de árboles (bagging).
3. **XGBoost** — boosting de árboles (modelo avanzado).

Entregable de negocio: **ranking de géneros por P(éxito)** (para los 3 modelos),
clasificados en 3 niveles (posible éxito / sin pena ni gloria / posible fracaso),
con las **3 regiones de origen que mejor producen** cada género. Solo se
recomiendan **regiones activas** (que aún producen tras 2020), excluyendo regiones
desaparecidas (Soviet Union, West Germany).

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
  Random Forest y XGBoost sobre el nivel de éxito (IEP en 3 clases), con
  validación temporal (<2020 train / ≥2020 test). F1-macro: LogReg 0.50,
  RF 0.51, XGBoost 0.52 (mejor) (`src/run_clasificacion.py`).
- [ ] **Fase 3 — Informe IEEE** doble columna (máx. 6 páginas)
- [ ] **Fase 4 — Video** presentación (máx. 6 min)
