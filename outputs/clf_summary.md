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
| Random Forest | 0.509 | 0.712 | 0.513 |
| XGBoost | 0.519 | 0.712 | 0.521 |

### Regresion Logistica
*Como funciona:* Modelo lineal base: aprende un peso por variable y combina todo en una probabilidad. Simple e interpretable; solo capta relaciones lineales.

*Resultado:* F1-macro=0.499, ROC-AUC=0.688, accuracy=0.506. Detecta el 53% de los 'exito' reales del test.

```
              precision    recall  f1-score   support

     fracaso      0.511     0.668     0.579      4508
    mediocre      0.545     0.345     0.423      4954
       exito      0.467     0.526     0.495      3401

    accuracy                          0.506     12863
   macro avg      0.508     0.513     0.499     12863
weighted avg      0.513     0.506     0.497     12863

```

### Random Forest
*Como funciona:* Ensamble (bagging): cientos de arboles entrenados con muestras distintas que votan. Capta no linealidad e interacciones y entrega importancia de variables.

*Resultado:* F1-macro=0.509, ROC-AUC=0.712, accuracy=0.513. Detecta el 57% de los 'exito' reales del test.

```
              precision    recall  f1-score   support

     fracaso      0.543     0.648     0.591      4508
    mediocre      0.517     0.355     0.421      4954
       exito      0.471     0.566     0.514      3401

    accuracy                          0.513     12863
   macro avg      0.510     0.523     0.509     12863
weighted avg      0.514     0.513     0.505     12863

```

### XGBoost
*Como funciona:* Boosting: arboles en serie donde cada uno corrige los errores del anterior. Modelo avanzado, suele rendir muy bien en datos tabulares.

*Resultado:* F1-macro=0.519, ROC-AUC=0.712, accuracy=0.521. Detecta el 59% de los 'exito' reales del test.

```
              precision    recall  f1-score   support

     fracaso      0.557     0.619     0.587      4508
    mediocre      0.513     0.385     0.439      4954
       exito      0.484     0.589     0.531      3401

    accuracy                          0.521     12863
   macro avg      0.518     0.531     0.519     12863
weighted avg      0.521     0.521     0.515     12863

```

**Mejor modelo (F1-macro): XGBoost**

## Ranking de generos por probabilidad de exito (por modelo)

P(exito) promedio entre **regiones activas** (las que aun producen, >= 2020). La region es el origen de produccion; se listan las 3 mejores por genero. Regiones activas: 28.

> Excluidas por no producir desde 2020 (region desaparecida): Soviet Union, West Germany.

### Regresion Logistica

| Genero | P(exito) | Categoria | 3 mejores regiones productoras |
|--------|----------|-----------|--------------------------------|
| Biography | 0.207 | posible exito | United States (0.38), United Kingdom (0.35), Canada (0.32) |
| Drama | 0.198 | posible exito | United States (0.36), United Kingdom (0.33), Canada (0.31) |
| Mystery | 0.158 | posible exito | United States (0.30), United Kingdom (0.27), Canada (0.25) |
| Action | 0.150 | posible exito | United States (0.29), United Kingdom (0.26), Canada (0.24) |
| Fantasy | 0.148 | posible exito | United States (0.28), United Kingdom (0.26), Canada (0.23) |
| Adventure | 0.147 | posible exito | United States (0.28), United Kingdom (0.25), Canada (0.23) |
| Comedy | 0.145 | posible exito | United States (0.28), United Kingdom (0.25), Canada (0.23) |
| Thriller | 0.143 | posible exito | United States (0.28), United Kingdom (0.25), Canada (0.23) |
| Crime | 0.141 | posible exito | United States (0.27), United Kingdom (0.25), Canada (0.22) |
| Romance | 0.140 | sin pena ni gloria | United States (0.27), United Kingdom (0.25), Canada (0.22) |
| War | 0.137 | sin pena ni gloria | United States (0.27), United Kingdom (0.24), Canada (0.22) |
| Sci-Fi | 0.135 | sin pena ni gloria | United States (0.26), United Kingdom (0.24), Canada (0.21) |
| Music | 0.134 | sin pena ni gloria | United States (0.26), United Kingdom (0.24), Canada (0.21) |
| Documentary | 0.133 | sin pena ni gloria | United States (0.26), United Kingdom (0.23), Canada (0.21) |
| History | 0.126 | sin pena ni gloria | United States (0.25), United Kingdom (0.22), Canada (0.20) |
| Sport | 0.122 | sin pena ni gloria | United States (0.24), United Kingdom (0.22), Canada (0.20) |
| Western | 0.108 | sin pena ni gloria | United States (0.22), United Kingdom (0.19), Canada (0.18) |
| Horror | 0.107 | sin pena ni gloria | United States (0.22), United Kingdom (0.19), Canada (0.17) |
| Musical | 0.099 | posible fracaso | United States (0.20), United Kingdom (0.18), Canada (0.16) |
| Animation | 0.096 | posible fracaso | United States (0.19), United Kingdom (0.17), Canada (0.15) |
| Family | 0.066 | posible fracaso | United States (0.14), United Kingdom (0.12), Canada (0.11) |
| Film-Noir | 0.064 | posible fracaso | United States (0.14), United Kingdom (0.12), Canada (0.11) |
| News | 0.049 | posible fracaso | United States (0.10), United Kingdom (0.09), Canada (0.08) |
| Game-Show | 0.045 | posible fracaso | United States (0.10), United Kingdom (0.09), Canada (0.07) |
| Talk-Show | 0.029 | posible fracaso | United States (0.06), United Kingdom (0.06), Canada (0.05) |
| Short | 0.024 | posible fracaso | United States (0.05), United Kingdom (0.05), Canada (0.04) |
| Reality-TV | 0.021 | posible fracaso | United States (0.05), United Kingdom (0.04), Canada (0.04) |

### Random Forest

| Genero | P(exito) | Categoria | 3 mejores regiones productoras |
|--------|----------|-----------|--------------------------------|
| Mystery | 0.279 | posible exito | Canada (0.41), United States (0.40), United Kingdom (0.38) |
| Music | 0.274 | posible exito | United Kingdom (0.37), United States (0.35), Canada (0.33) |
| Biography | 0.268 | posible exito | United States (0.45), United Kingdom (0.43), France (0.33) |
| Action | 0.243 | posible exito | United Kingdom (0.37), United States (0.36), Germany (0.35) |
| War | 0.224 | posible exito | United States (0.41), Canada (0.34), United Kingdom (0.33) |
| Documentary | 0.218 | posible exito | United States (0.31), Denmark (0.31), China (0.26) |
| Drama | 0.215 | posible exito | United States (0.52), United Kingdom (0.42), Canada (0.35) |
| Crime | 0.215 | posible exito | United States (0.46), Canada (0.34), Germany (0.32) |
| Fantasy | 0.212 | posible exito | United States (0.40), United Kingdom (0.35), Canada (0.32) |
| History | 0.197 | sin pena ni gloria | United States (0.34), United Kingdom (0.29), Canada (0.28) |
| Musical | 0.192 | sin pena ni gloria | United States (0.37), Canada (0.36), United Kingdom (0.34) |
| Thriller | 0.183 | sin pena ni gloria | United States (0.35), United Kingdom (0.32), Canada (0.31) |
| Sci-Fi | 0.181 | sin pena ni gloria | United Kingdom (0.29), United States (0.28), Canada (0.24) |
| Adventure | 0.177 | sin pena ni gloria | United States (0.35), Canada (0.28), United Kingdom (0.27) |
| Sport | 0.175 | sin pena ni gloria | United States (0.36), United Kingdom (0.30), Canada (0.27) |
| Romance | 0.170 | sin pena ni gloria | United States (0.37), United Kingdom (0.37), Canada (0.33) |
| Western | 0.158 | sin pena ni gloria | United States (0.29), United Kingdom (0.29), Canada (0.26) |
| Horror | 0.152 | sin pena ni gloria | United Kingdom (0.24), Australia (0.20), Canada (0.19) |
| Comedy | 0.137 | posible fracaso | United States (0.41), Canada (0.30), United Kingdom (0.28) |
| Film-Noir | 0.125 | posible fracaso | United States (0.23), United Kingdom (0.18), Canada (0.17) |
| Animation | 0.111 | posible fracaso | United States (0.17), United Kingdom (0.15), France (0.15) |
| Family | 0.091 | posible fracaso | United States (0.18), United Kingdom (0.14), France (0.13) |
| News | 0.074 | posible fracaso | United Kingdom (0.12), India (0.10), France (0.10) |
| Game-Show | 0.061 | posible fracaso | India (0.11), United Kingdom (0.09), France (0.08) |
| Short | 0.060 | posible fracaso | Spain (0.09), Canada (0.09), Poland (0.09) |
| Reality-TV | 0.050 | posible fracaso | United States (0.07), Japan (0.07), United Kingdom (0.06) |
| Talk-Show | 0.042 | posible fracaso | India (0.11), United States (0.06), France (0.06) |

### XGBoost

| Genero | P(exito) | Categoria | 3 mejores regiones productoras |
|--------|----------|-----------|--------------------------------|
| Drama | 0.216 | posible exito | United States (0.44), Canada (0.36), United Kingdom (0.35) |
| Documentary | 0.186 | posible exito | China (0.28), United States (0.28), France (0.27) |
| Film-Noir | 0.176 | posible exito | France (0.25), Canada (0.25), China (0.24) |
| Biography | 0.164 | posible exito | United States (0.38), United Kingdom (0.34), Canada (0.25) |
| Mystery | 0.156 | posible exito | United States (0.29), Germany (0.29), Canada (0.29) |
| Action | 0.145 | posible exito | Australia (0.28), Germany (0.27), United States (0.26) |
| Music | 0.135 | posible exito | United States (0.25), Belgium (0.24), United Kingdom (0.22) |
| Thriller | 0.132 | posible exito | United States (0.25), United Kingdom (0.23), Australia (0.22) |
| Comedy | 0.130 | posible exito | United States (0.40), United Kingdom (0.27), Canada (0.23) |
| Fantasy | 0.129 | sin pena ni gloria | United States (0.30), United Kingdom (0.27), Canada (0.21) |
| Romance | 0.123 | sin pena ni gloria | Canada (0.32), United States (0.30), United Kingdom (0.27) |
| War | 0.116 | sin pena ni gloria | United States (0.27), Canada (0.24), Australia (0.20) |
| Western | 0.108 | sin pena ni gloria | Mexico (0.27), Australia (0.19), United States (0.18) |
| Horror | 0.107 | sin pena ni gloria | Belgium (0.20), Netherlands (0.19), Canada (0.18) |
| Crime | 0.104 | sin pena ni gloria | United States (0.27), United Kingdom (0.19), Germany (0.18) |
| Adventure | 0.103 | sin pena ni gloria | United States (0.21), Canada (0.18), United Kingdom (0.18) |
| Musical | 0.101 | sin pena ni gloria | Canada (0.24), United States (0.23), United Kingdom (0.22) |
| Sci-Fi | 0.094 | sin pena ni gloria | United States (0.18), United Kingdom (0.17), Mexico (0.15) |
| History | 0.087 | posible fracaso | United States (0.23), United Kingdom (0.17), Mexico (0.16) |
| Sport | 0.084 | posible fracaso | United States (0.23), United Kingdom (0.16), Australia (0.13) |
| Animation | 0.075 | posible fracaso | Belgium (0.13), United States (0.13), Australia (0.12) |
| Family | 0.046 | posible fracaso | United States (0.10), France (0.10), United Kingdom (0.08) |
| Game-Show | 0.040 | posible fracaso | India (0.11), United Kingdom (0.07), South Korea (0.06) |
| News | 0.035 | posible fracaso | India (0.06), United Kingdom (0.06), United States (0.06) |
| Reality-TV | 0.035 | posible fracaso | India (0.09), South Korea (0.06), United Kingdom (0.05) |
| Talk-Show | 0.029 | posible fracaso | India (0.14), United States (0.04), United Kingdom (0.04) |
| Short | 0.028 | posible fracaso | Brazil (0.05), Germany (0.04), Turkey (0.04) |

## Matriz P(exito) por region y genero (los 3 modelos)

Matriz completa (region activa x genero) por modelo en `outputs/clf_matriz_region_genero.csv` y figuras `13/14/15_clf_heatmap_*.png` (ejes en el mismo orden para comparar).
> En Regresion Logistica el orden de regiones es igual en todos los generos: el modelo lineal no captura interaccion region-genero. En Random Forest y XGBoost varia por genero.
