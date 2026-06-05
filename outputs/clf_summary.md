# Clasificacion del nivel de exito (IEP) - 3 modelos

**Objetivo (IEP):** combina rating (0.2) + log-votos + log-gross. Con gross: 0.2/0.3/0.5; sin gross el 0.5 pasa a votos -> 0.2/0.8. Normalizado Min-Max y cortado en 3 clases por terciles.

- Titulos con gross: 15.3% (el resto usa la formula sin gross). Cortes IEP (train): q1=0.260, q2=0.399
- Validacion temporal: train anio < 2020 (105,935 filas), test anio >= 2020 (12,863 filas)
- Distribucion clases train: {'mediocre': 35312, 'exito': 35312, 'fracaso': 35311}
- Distribucion clases test:  {'mediocre': 4954, 'fracaso': 4508, 'exito': 3401}
- Regiones modeladas (>= 500 en train): 30 + 'Other'
- Generos (one-hot): 27
- Predictoras: ['length', 'is_movie', 'region_grp'] + generos (el anio NO se usa: es el eje del split)

## Resultados en test temporal (>= 2020)

| Modelo | F1-macro | ROC-AUC | Accuracy |
|--------|----------|---------|----------|
| Regresion Logistica | 0.499 | 0.688 | 0.506 |
| Random Forest | 0.510 | 0.711 | 0.514 |
| XGBoost | 0.516 | 0.712 | 0.518 |

### Regresion Logistica
*Como funciona:* Modelo lineal base: aprende un peso por variable y combina todo en una probabilidad. Simple e interpretable; solo capta relaciones lineales.

*Resultado:* F1-macro=0.499, ROC-AUC=0.688, accuracy=0.506. Detecta el 53% de los 'exito' reales del test.

```
              precision    recall  f1-score   support

     fracaso      0.510     0.668     0.578      4508
    mediocre      0.546     0.346     0.424      4954
       exito      0.468     0.526     0.495      3401

    accuracy                          0.506     12863
   macro avg      0.508     0.513     0.499     12863
weighted avg      0.513     0.506     0.497     12863

```

### Random Forest
*Como funciona:* Ensamble (bagging): cientos de arboles entrenados con muestras distintas que votan. Capta no linealidad e interacciones y entrega importancia de variables.

*Resultado:* F1-macro=0.510, ROC-AUC=0.711, accuracy=0.514. Detecta el 57% de los 'exito' reales del test.

```
              precision    recall  f1-score   support

     fracaso      0.546     0.647     0.592      4508
    mediocre      0.515     0.355     0.420      4954
       exito      0.472     0.570     0.517      3401

    accuracy                          0.514     12863
   macro avg      0.511     0.524     0.510     12863
weighted avg      0.514     0.514     0.506     12863

```

### XGBoost
*Como funciona:* Boosting: arboles en serie donde cada uno corrige los errores del anterior. Modelo avanzado, suele rendir muy bien en datos tabulares.

*Resultado:* F1-macro=0.516, ROC-AUC=0.712, accuracy=0.518. Detecta el 59% de los 'exito' reales del test.

```
              precision    recall  f1-score   support

     fracaso      0.558     0.622     0.588      4508
    mediocre      0.506     0.376     0.431      4954
       exito      0.479     0.587     0.528      3401

    accuracy                          0.518     12863
   macro avg      0.514     0.528     0.516     12863
weighted avg      0.517     0.518     0.512     12863

```

**Mejor modelo (F1-macro): XGBoost**

## Ranking de generos por probabilidad de exito (por modelo)

P(exito) promedio entre **regiones activas** (las que aun producen, >= 2020). La region es el origen de produccion; se listan las 3 mejores por genero. Regiones activas: 28.

> Excluidas por no producir desde 2020 (region desaparecida): Soviet Union, West Germany.

### Regresion Logistica

| Genero | P(exito) | Categoria | 3 mejores regiones productoras |
|--------|----------|-----------|--------------------------------|
| Biography | 0.209 | posible exito | United States (0.38), United Kingdom (0.35), Canada (0.32) |
| Drama | 0.198 | posible exito | United States (0.36), United Kingdom (0.33), Canada (0.31) |
| Mystery | 0.158 | posible exito | United States (0.30), United Kingdom (0.27), Canada (0.25) |
| Action | 0.150 | posible exito | United States (0.29), United Kingdom (0.26), Canada (0.24) |
| Fantasy | 0.149 | posible exito | United States (0.29), United Kingdom (0.26), Canada (0.23) |
| Adventure | 0.146 | posible exito | United States (0.28), United Kingdom (0.25), Canada (0.23) |
| Comedy | 0.145 | posible exito | United States (0.28), United Kingdom (0.25), Canada (0.23) |
| Thriller | 0.144 | posible exito | United States (0.28), United Kingdom (0.25), Canada (0.23) |
| Crime | 0.141 | posible exito | United States (0.27), United Kingdom (0.25), Canada (0.22) |
| Romance | 0.140 | posible exito | United States (0.27), United Kingdom (0.25), Canada (0.22) |
| War | 0.138 | sin pena ni gloria | United States (0.27), United Kingdom (0.24), Canada (0.22) |
| Sci-Fi | 0.135 | sin pena ni gloria | United States (0.26), United Kingdom (0.24), Canada (0.21) |
| Music | 0.134 | sin pena ni gloria | United States (0.26), United Kingdom (0.24), Canada (0.21) |
| Documentary | 0.133 | sin pena ni gloria | United States (0.26), United Kingdom (0.23), Canada (0.21) |
| History | 0.125 | sin pena ni gloria | United States (0.25), United Kingdom (0.22), Canada (0.20) |
| Sport | 0.121 | sin pena ni gloria | United States (0.24), United Kingdom (0.22), Canada (0.19) |
| Western | 0.108 | sin pena ni gloria | United States (0.22), United Kingdom (0.19), Canada (0.18) |
| Horror | 0.107 | sin pena ni gloria | United States (0.22), United Kingdom (0.19), Canada (0.17) |
| Musical | 0.099 | sin pena ni gloria | United States (0.20), United Kingdom (0.18), Canada (0.16) |
| Animation | 0.098 | sin pena ni gloria | United States (0.19), United Kingdom (0.18), Canada (0.16) |
| Family | 0.066 | posible fracaso | United States (0.14), United Kingdom (0.12), Canada (0.11) |
| Film-Noir | 0.063 | posible fracaso | United States (0.13), United Kingdom (0.12), Canada (0.11) |
| News | 0.050 | posible fracaso | United States (0.11), United Kingdom (0.09), Canada (0.08) |
| Game-Show | 0.043 | posible fracaso | United States (0.09), United Kingdom (0.08), Canada (0.07) |
| Talk-Show | 0.030 | posible fracaso | United States (0.06), United Kingdom (0.06), Canada (0.05) |
| Short | 0.024 | posible fracaso | United States (0.05), United Kingdom (0.05), Canada (0.04) |
| Reality-TV | 0.021 | posible fracaso | United States (0.05), United Kingdom (0.04), Canada (0.04) |

### Random Forest

| Genero | P(exito) | Categoria | 3 mejores regiones productoras |
|--------|----------|-----------|--------------------------------|
| Biography | 0.273 | posible exito | United States (0.45), United Kingdom (0.44), Canada (0.32) |
| Mystery | 0.265 | posible exito | United States (0.39), Canada (0.38), United Kingdom (0.34) |
| Music | 0.264 | posible exito | United States (0.34), United Kingdom (0.34), Canada (0.31) |
| Action | 0.242 | posible exito | United Kingdom (0.38), United States (0.37), Germany (0.37) |
| Crime | 0.229 | posible exito | United States (0.46), Canada (0.36), Germany (0.35) |
| War | 0.219 | posible exito | United States (0.44), United Kingdom (0.33), Canada (0.33) |
| Documentary | 0.219 | posible exito | United States (0.30), Denmark (0.30), China (0.27) |
| Drama | 0.216 | posible exito | United States (0.52), United Kingdom (0.42), Canada (0.36) |
| History | 0.208 | posible exito | United States (0.36), United Kingdom (0.29), Canada (0.28) |
| Fantasy | 0.204 | posible exito | United States (0.38), United Kingdom (0.31), Canada (0.27) |
| Musical | 0.195 | sin pena ni gloria | United States (0.38), Canada (0.36), United Kingdom (0.35) |
| Thriller | 0.194 | sin pena ni gloria | United States (0.32), United Kingdom (0.31), Canada (0.29) |
| Romance | 0.176 | sin pena ni gloria | United States (0.39), United Kingdom (0.37), Canada (0.35) |
| Adventure | 0.174 | sin pena ni gloria | United States (0.31), United Kingdom (0.26), Canada (0.24) |
| Sci-Fi | 0.169 | sin pena ni gloria | United Kingdom (0.26), United States (0.24), Germany (0.23) |
| Sport | 0.167 | sin pena ni gloria | United States (0.33), United Kingdom (0.28), Canada (0.25) |
| Western | 0.151 | sin pena ni gloria | United States (0.25), United Kingdom (0.24), Canada (0.23) |
| Horror | 0.144 | sin pena ni gloria | United Kingdom (0.25), Australia (0.18), Germany (0.18) |
| Comedy | 0.135 | sin pena ni gloria | United States (0.41), Canada (0.28), United Kingdom (0.27) |
| Film-Noir | 0.124 | posible fracaso | United States (0.19), United Kingdom (0.17), Canada (0.17) |
| Animation | 0.116 | posible fracaso | France (0.15), United Kingdom (0.15), United States (0.14) |
| Family | 0.107 | posible fracaso | United States (0.19), Canada (0.17), United Kingdom (0.16) |
| News | 0.069 | posible fracaso | United Kingdom (0.10), South Korea (0.10), India (0.09) |
| Short | 0.061 | posible fracaso | Spain (0.11), Canada (0.09), Poland (0.08) |
| Game-Show | 0.048 | posible fracaso | India (0.08), United Kingdom (0.08), South Korea (0.08) |
| Talk-Show | 0.037 | posible fracaso | India (0.10), South Korea (0.06), United States (0.05) |
| Reality-TV | 0.032 | posible fracaso | South Korea (0.06), Japan (0.05), India (0.04) |

### XGBoost

| Genero | P(exito) | Categoria | 3 mejores regiones productoras |
|--------|----------|-----------|--------------------------------|
| Drama | 0.217 | posible exito | United States (0.44), Canada (0.36), United Kingdom (0.35) |
| Film-Noir | 0.197 | posible exito | France (0.28), Canada (0.27), Australia (0.27) |
| Documentary | 0.180 | posible exito | United States (0.29), China (0.27), Denmark (0.25) |
| Biography | 0.151 | posible exito | United States (0.38), United Kingdom (0.33), Mexico (0.24) |
| Action | 0.140 | posible exito | Australia (0.27), Germany (0.26), United States (0.25) |
| Mystery | 0.138 | posible exito | United States (0.28), Germany (0.26), Canada (0.25) |
| Fantasy | 0.133 | posible exito | United States (0.29), United Kingdom (0.27), Canada (0.21) |
| Music | 0.132 | posible exito | United States (0.24), United Kingdom (0.22), Canada (0.22) |
| Thriller | 0.132 | posible exito | United States (0.24), United Kingdom (0.23), Australia (0.23) |
| Comedy | 0.128 | sin pena ni gloria | United States (0.40), United Kingdom (0.27), Canada (0.21) |
| Romance | 0.122 | sin pena ni gloria | Canada (0.29), United States (0.29), United Kingdom (0.27) |
| War | 0.113 | sin pena ni gloria | United States (0.25), Canada (0.21), Australia (0.18) |
| Western | 0.107 | sin pena ni gloria | Mexico (0.30), Australia (0.23), United States (0.17) |
| Horror | 0.105 | sin pena ni gloria | Netherlands (0.19), Belgium (0.19), Canada (0.17) |
| Crime | 0.105 | sin pena ni gloria | United States (0.26), United Kingdom (0.20), Germany (0.18) |
| Adventure | 0.102 | sin pena ni gloria | United States (0.20), United Kingdom (0.18), Canada (0.18) |
| Musical | 0.101 | sin pena ni gloria | Canada (0.24), United States (0.21), United Kingdom (0.20) |
| History | 0.091 | sin pena ni gloria | United States (0.24), United Kingdom (0.18), Mexico (0.17) |
| Sci-Fi | 0.087 | sin pena ni gloria | United States (0.16), United Kingdom (0.16), Germany (0.14) |
| Sport | 0.087 | sin pena ni gloria | United States (0.25), United Kingdom (0.17), Mexico (0.13) |
| Animation | 0.073 | posible fracaso | Belgium (0.13), Netherlands (0.12), Australia (0.11) |
| Family | 0.052 | posible fracaso | United States (0.11), France (0.10), United Kingdom (0.09) |
| News | 0.027 | posible fracaso | India (0.05), United Kingdom (0.05), United States (0.04) |
| Short | 0.026 | posible fracaso | Brazil (0.05), China (0.04), Germany (0.04) |
| Game-Show | 0.025 | posible fracaso | India (0.08), South Korea (0.05), United Kingdom (0.04) |
| Talk-Show | 0.021 | posible fracaso | India (0.08), Iran (0.04), United States (0.03) |
| Reality-TV | 0.015 | posible fracaso | India (0.05), South Korea (0.03), United States (0.03) |

## Matriz P(exito) por region y genero (los 3 modelos)

Matriz completa (region activa x genero) por modelo en `outputs/clf_matriz_region_genero.csv` y figuras `13/14/15_clf_heatmap_*.png` (ejes en el mismo orden para comparar).
> En Regresion Logistica el orden de regiones es igual en todos los generos: el modelo lineal no captura interaccion region-genero. En Random Forest y XGBoost varia por genero.
