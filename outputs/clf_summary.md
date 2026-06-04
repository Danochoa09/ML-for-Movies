# Clasificacion del nivel de exito (IEP) - 3 modelos

**Objetivo (IEP):** combina rating (0.2) + log-votos + log-gross. Con gross: 0.2/0.3/0.5; sin gross el 0.5 pasa a votos -> 0.2/0.8. Normalizado Min-Max y cortado en 3 clases por terciles.

- Titulos con gross: 15.3% (el resto usa la formula sin gross). Cortes IEP (train): q1=0.253, q2=0.376
- Validacion temporal: train anio < 2020 (105,935 filas), test anio >= 2020 (12,863 filas)
- Distribucion clases train: {'exito': 39426, 'mediocre': 34167, 'fracaso': 32342}
- Distribucion clases test:  {'mediocre': 4605, 'fracaso': 4183, 'exito': 4075}
- Regiones modeladas (>= 500 en train): 30 + 'Other'
- Generos (one-hot): 27
- Predictoras: ['length', 'is_movie', 'region_grp'] + generos (el anio NO se usa: es el eje del split)

## Resultados en test temporal (>= 2020)

| Modelo | F1-macro | ROC-AUC | Accuracy |
|--------|----------|---------|----------|
| Regresion Logistica | 0.500 | 0.688 | 0.506 |
| Random Forest | 0.509 | 0.711 | 0.514 |
| XGBoost | 0.523 | 0.714 | 0.526 |

### Regresion Logistica
*Como funciona:* Modelo lineal base: aprende un peso por variable y combina todo en una probabilidad. Simple e interpretable; solo capta relaciones lineales.

*Resultado:* F1-macro=0.500, ROC-AUC=0.688, accuracy=0.506. Detecta el 51% de los 'exito' reales del test.

```
              precision    recall  f1-score   support

     fracaso      0.486     0.670     0.563      4183
    mediocre      0.521     0.355     0.422      4605
       exito      0.522     0.507     0.514      4075

    accuracy                          0.506     12863
   macro avg      0.510     0.511     0.500     12863
weighted avg      0.510     0.506     0.497     12863

```

### Random Forest
*Como funciona:* Ensamble (bagging): cientos de arboles entrenados con muestras distintas que votan. Capta no linealidad e interacciones y entrega importancia de variables.

*Resultado:* F1-macro=0.509, ROC-AUC=0.711, accuracy=0.514. Detecta el 55% de los 'exito' reales del test.

```
              precision    recall  f1-score   support

     fracaso      0.512     0.662     0.577      4183
    mediocre      0.495     0.344     0.406      4605
       exito      0.532     0.555     0.543      4075

    accuracy                          0.514     12863
   macro avg      0.513     0.520     0.509     12863
weighted avg      0.512     0.514     0.505     12863

```

### XGBoost
*Como funciona:* Boosting: arboles en serie donde cada uno corrige los errores del anterior. Modelo avanzado, suele rendir muy bien en datos tabulares.

*Resultado:* F1-macro=0.523, ROC-AUC=0.714, accuracy=0.526. Detecta el 58% de los 'exito' reales del test.

```
              precision    recall  f1-score   support

     fracaso      0.539     0.638     0.584      4183
    mediocre      0.496     0.374     0.426      4605
       exito      0.536     0.584     0.559      4075

    accuracy                          0.526     12863
   macro avg      0.524     0.532     0.523     12863
weighted avg      0.523     0.526     0.520     12863

```

**Mejor modelo (F1-macro): XGBoost**

## Ranking de generos por probabilidad de exito (por modelo)

P(exito) promedio entre **regiones activas** (las que aun producen, >= 2020). La region es el origen de produccion; se listan las 3 mejores por genero. Regiones activas: 28.

> Excluidas por no producir desde 2020 (region desaparecida): Soviet Union, West Germany.

### Regresion Logistica

| Genero | P(exito) | Categoria | 3 mejores regiones productoras |
|--------|----------|-----------|--------------------------------|
| Biography | 0.207 | posible exito | United States (0.36), United Kingdom (0.33), Canada (0.30) |
| Drama | 0.201 | posible exito | United States (0.35), United Kingdom (0.33), Canada (0.29) |
| Mystery | 0.167 | posible exito | United States (0.30), United Kingdom (0.28), Canada (0.25) |
| Comedy | 0.155 | posible exito | United States (0.28), United Kingdom (0.26), Canada (0.23) |
| Adventure | 0.155 | posible exito | United States (0.28), United Kingdom (0.26), Canada (0.23) |
| Action | 0.154 | posible exito | United States (0.28), United Kingdom (0.26), Canada (0.23) |
| Fantasy | 0.154 | posible exito | United States (0.28), United Kingdom (0.26), Canada (0.23) |
| Thriller | 0.147 | posible exito | United States (0.27), United Kingdom (0.25), Canada (0.22) |
| Crime | 0.147 | posible exito | United States (0.27), United Kingdom (0.25), Canada (0.22) |
| Romance | 0.143 | sin pena ni gloria | United States (0.26), United Kingdom (0.24), Canada (0.21) |
| Documentary | 0.142 | sin pena ni gloria | United States (0.26), United Kingdom (0.24), Canada (0.21) |
| War | 0.141 | sin pena ni gloria | United States (0.26), United Kingdom (0.24), Canada (0.21) |
| Music | 0.140 | sin pena ni gloria | United States (0.26), United Kingdom (0.24), Canada (0.21) |
| Sci-Fi | 0.139 | sin pena ni gloria | United States (0.25), United Kingdom (0.24), Canada (0.21) |
| History | 0.130 | sin pena ni gloria | United States (0.24), United Kingdom (0.22), Canada (0.20) |
| Sport | 0.129 | sin pena ni gloria | United States (0.24), United Kingdom (0.22), Canada (0.19) |
| Western | 0.119 | sin pena ni gloria | United States (0.22), United Kingdom (0.20), Canada (0.18) |
| Horror | 0.119 | sin pena ni gloria | United States (0.22), United Kingdom (0.20), Canada (0.18) |
| Animation | 0.118 | posible fracaso | United States (0.22), United Kingdom (0.20), Canada (0.18) |
| Musical | 0.101 | posible fracaso | United States (0.19), United Kingdom (0.18), Canada (0.15) |
| Film-Noir | 0.077 | posible fracaso | United States (0.15), United Kingdom (0.14), Canada (0.12) |
| Family | 0.076 | posible fracaso | United States (0.15), United Kingdom (0.13), Canada (0.12) |
| Game-Show | 0.061 | posible fracaso | United States (0.12), United Kingdom (0.11), Canada (0.09) |
| News | 0.058 | posible fracaso | United States (0.11), United Kingdom (0.11), Canada (0.09) |
| Short | 0.042 | posible fracaso | United States (0.08), United Kingdom (0.08), Canada (0.07) |
| Talk-Show | 0.036 | posible fracaso | United States (0.07), United Kingdom (0.07), Canada (0.06) |
| Reality-TV | 0.032 | posible fracaso | United States (0.06), United Kingdom (0.06), Canada (0.05) |

### Random Forest

| Genero | P(exito) | Categoria | 3 mejores regiones productoras |
|--------|----------|-----------|--------------------------------|
| Biography | 0.218 | posible exito | United Kingdom (0.29), United States (0.28), France (0.27) |
| Music | 0.205 | posible exito | United States (0.26), United Kingdom (0.25), France (0.24) |
| Documentary | 0.199 | posible exito | United States (0.31), Denmark (0.26), China (0.23) |
| Horror | 0.190 | posible exito | United Kingdom (0.24), Canada (0.24), Australia (0.23) |
| Thriller | 0.185 | posible exito | United Kingdom (0.25), United States (0.23), China (0.23) |
| Crime | 0.179 | posible exito | United States (0.28), Canada (0.24), Germany (0.24) |
| Mystery | 0.173 | posible exito | United States (0.26), United Kingdom (0.24), Canada (0.23) |
| Drama | 0.168 | posible exito | United States (0.31), United Kingdom (0.27), South Korea (0.25) |
| Film-Noir | 0.165 | posible exito | United States (0.30), United Kingdom (0.23), France (0.21) |
| Action | 0.156 | sin pena ni gloria | France (0.23), United Kingdom (0.21), Germany (0.20) |
| History | 0.155 | sin pena ni gloria | United States (0.23), United Kingdom (0.20), France (0.19) |
| Sport | 0.146 | sin pena ni gloria | United States (0.24), United Kingdom (0.20), France (0.18) |
| Fantasy | 0.144 | sin pena ni gloria | United States (0.27), United Kingdom (0.22), France (0.20) |
| Adventure | 0.143 | sin pena ni gloria | United Kingdom (0.20), United States (0.20), France (0.19) |
| Musical | 0.143 | sin pena ni gloria | United Kingdom (0.22), United States (0.21), France (0.20) |
| Sci-Fi | 0.141 | sin pena ni gloria | United Kingdom (0.21), France (0.19), Germany (0.18) |
| War | 0.138 | sin pena ni gloria | United States (0.22), United Kingdom (0.19), France (0.18) |
| Western | 0.128 | sin pena ni gloria | United Kingdom (0.20), United States (0.19), France (0.17) |
| Comedy | 0.128 | posible fracaso | United States (0.33), United Kingdom (0.21), China (0.20) |
| Romance | 0.124 | posible fracaso | United Kingdom (0.25), United States (0.22), France (0.20) |
| Animation | 0.116 | posible fracaso | United States (0.17), United Kingdom (0.16), France (0.15) |
| Short | 0.110 | posible fracaso | France (0.15), Spain (0.14), Canada (0.14) |
| Family | 0.086 | posible fracaso | United Kingdom (0.15), United States (0.15), France (0.14) |
| News | 0.061 | posible fracaso | United Kingdom (0.09), Iran (0.09), France (0.09) |
| Game-Show | 0.029 | posible fracaso | Iran (0.07), India (0.05), South Korea (0.04) |
| Talk-Show | 0.029 | posible fracaso | Iran (0.07), India (0.06), Canada (0.04) |
| Reality-TV | 0.022 | posible fracaso | Japan (0.04), South Korea (0.04), Iran (0.04) |

### XGBoost

| Genero | P(exito) | Categoria | 3 mejores regiones productoras |
|--------|----------|-----------|--------------------------------|
| Film-Noir | 0.219 | posible exito | United States (0.34), China (0.33), Canada (0.30) |
| Documentary | 0.182 | posible exito | United States (0.34), China (0.28), Denmark (0.25) |
| Horror | 0.146 | posible exito | Netherlands (0.35), Belgium (0.28), Canada (0.22) |
| Drama | 0.137 | posible exito | United States (0.24), United Kingdom (0.22), China (0.20) |
| Comedy | 0.110 | posible exito | United States (0.31), United Kingdom (0.21), Canada (0.17) |
| Western | 0.102 | posible exito | Australia (0.19), Mexico (0.17), United States (0.16) |
| Thriller | 0.101 | posible exito | United States (0.17), China (0.17), United Kingdom (0.16) |
| Music | 0.099 | posible exito | Belgium (0.21), United States (0.15), United Kingdom (0.15) |
| Crime | 0.085 | posible exito | United States (0.15), Australia (0.13), United Kingdom (0.13) |
| Action | 0.084 | sin pena ni gloria | Australia (0.16), France (0.14), United States (0.14) |
| Fantasy | 0.080 | sin pena ni gloria | United States (0.21), United Kingdom (0.14), China (0.13) |
| Romance | 0.078 | sin pena ni gloria | United Kingdom (0.17), France (0.16), United States (0.14) |
| Sport | 0.075 | sin pena ni gloria | United States (0.15), United Kingdom (0.14), Germany (0.11) |
| Mystery | 0.074 | sin pena ni gloria | United States (0.15), Canada (0.13), United Kingdom (0.13) |
| War | 0.073 | sin pena ni gloria | United States (0.12), Germany (0.11), China (0.11) |
| Biography | 0.071 | sin pena ni gloria | United States (0.14), United Kingdom (0.12), China (0.12) |
| Animation | 0.070 | sin pena ni gloria | United States (0.14), Belgium (0.11), United Kingdom (0.11) |
| Musical | 0.069 | sin pena ni gloria | United Kingdom (0.12), United States (0.12), Canada (0.12) |
| History | 0.068 | posible fracaso | United States (0.11), Norway (0.10), United Kingdom (0.10) |
| Adventure | 0.067 | posible fracaso | United States (0.12), United Kingdom (0.11), Mexico (0.11) |
| Sci-Fi | 0.066 | posible fracaso | United Kingdom (0.12), United States (0.12), China (0.11) |
| Short | 0.050 | posible fracaso | France (0.08), United States (0.08), China (0.08) |
| Family | 0.042 | posible fracaso | France (0.09), United States (0.09), United Kingdom (0.08) |
| News | 0.018 | posible fracaso | United Kingdom (0.03), United States (0.03), China (0.03) |
| Talk-Show | 0.007 | posible fracaso | India (0.03), United Kingdom (0.01), Iran (0.01) |
| Reality-TV | 0.007 | posible fracaso | Japan (0.02), United Kingdom (0.01), Iran (0.01) |
| Game-Show | 0.006 | posible fracaso | South Korea (0.02), United Kingdom (0.01), India (0.01) |

## Matriz P(exito) por region y genero (los 3 modelos)

Matriz completa (region activa x genero) por modelo en `outputs/clf_matriz_region_genero.csv` y figuras `13/14/15_clf_heatmap_*.png` (ejes en el mismo orden para comparar).
> En Regresion Logistica el orden de regiones es igual en todos los generos: el modelo lineal no captura interaccion region-genero. En Random Forest y XGBoost varia por genero.
