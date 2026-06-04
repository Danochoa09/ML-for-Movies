# Clasificacion de EXITO de taquilla/publico - 3 modelos

**Definicion de exito:** rating >= 6.5 Y (votes > mediana O gross > mediana)

- Umbrales (solo train): mediana votos = 973, mediana gross = 1950000
- Validacion temporal: train anio < 2020 (105,935 filas), test anio >= 2020 (12,863 filas)
- Tasa de exito: train 30.8% | test 29.8%
- Regiones modeladas (>= 500 en train): 30 + 'Other'
- Generos (one-hot): 27
- Predictoras: ['length', 'is_movie', 'region_grp'] + generos (el anio NO se usa: es el eje del split)

## Resultados en test temporal (>= 2020)

| Modelo | F1 (exito) | ROC-AUC | Accuracy |
|--------|-----------|---------|----------|
| Regresion Logistica | 0.513 | 0.691 | 0.629 |
| Random Forest | 0.554 | 0.742 | 0.665 |
| XGBoost | 0.555 | 0.738 | 0.663 |

### Regresion Logistica
*Como funciona:* Modelo lineal base: aprende un peso por variable y combina todo en una probabilidad. Simple e interpretable; solo capta relaciones lineales.

*Resultado:* F1(exito)=0.513, ROC-AUC=0.691, accuracy=0.629. Detecta el 66% de los exitos reales del test.

```
              precision    recall  f1-score   support

    no_exito      0.810     0.617     0.700      9035
       exito      0.421     0.658     0.513      3828

    accuracy                          0.629     12863
   macro avg      0.615     0.637     0.607     12863
weighted avg      0.694     0.629     0.645     12863

```

### Random Forest
*Como funciona:* Ensamble (bagging): cientos de arboles entrenados con muestras distintas que votan. Capta no linealidad e interacciones y entrega importancia de variables.

*Resultado:* F1(exito)=0.554, ROC-AUC=0.742, accuracy=0.665. Detecta el 70% de los exitos reales del test.

```
              precision    recall  f1-score   support

    no_exito      0.836     0.651     0.732      9035
       exito      0.459     0.699     0.554      3828

    accuracy                          0.665     12863
   macro avg      0.647     0.675     0.643     12863
weighted avg      0.724     0.665     0.679     12863

```

### XGBoost
*Como funciona:* Boosting: arboles en serie donde cada uno corrige los errores del anterior. Modelo avanzado, suele rendir muy bien en datos tabulares.

*Resultado:* F1(exito)=0.555, ROC-AUC=0.738, accuracy=0.663. Detecta el 70% de los exitos reales del test.

```
              precision    recall  f1-score   support

    no_exito      0.838     0.646     0.729      9035
       exito      0.457     0.705     0.555      3828

    accuracy                          0.663     12863
   macro avg      0.647     0.675     0.642     12863
weighted avg      0.724     0.663     0.677     12863

```

**Mejor modelo (ROC-AUC): Random Forest**

## Ranking de generos por probabilidad de exito (por modelo)

P(exito) promedio entre **regiones activas** (las que aun producen, >= 2020). La region es el origen de produccion; se listan las 3 mejores por genero. Regiones activas: 28.

> Excluidas por no producir desde 2020 (region desaparecida): Soviet Union, West Germany.

### Regresion Logistica

| Genero | P(exito) | Categoria | 3 mejores regiones productoras |
|--------|----------|-----------|--------------------------------|
| Drama | 0.464 | posible exito | South Korea (0.55), France (0.54), United Kingdom (0.54) |
| Film-Noir | 0.463 | posible exito | South Korea (0.55), France (0.54), United Kingdom (0.54) |
| Biography | 0.418 | posible exito | South Korea (0.51), France (0.49), United Kingdom (0.49) |
| Documentary | 0.409 | posible exito | South Korea (0.50), France (0.48), United Kingdom (0.48) |
| Animation | 0.379 | posible exito | South Korea (0.47), France (0.45), United Kingdom (0.45) |
| War | 0.378 | posible exito | South Korea (0.46), France (0.45), United Kingdom (0.45) |
| Mystery | 0.339 | posible exito | South Korea (0.42), France (0.41), United Kingdom (0.40) |
| Music | 0.337 | posible exito | South Korea (0.42), France (0.41), United Kingdom (0.40) |
| Western | 0.336 | posible exito | South Korea (0.42), France (0.41), United Kingdom (0.40) |
| Crime | 0.328 | sin pena ni gloria | South Korea (0.41), France (0.40), United Kingdom (0.39) |
| Adventure | 0.327 | sin pena ni gloria | South Korea (0.41), France (0.40), United Kingdom (0.39) |
| Comedy | 0.315 | sin pena ni gloria | South Korea (0.40), France (0.38), United Kingdom (0.38) |
| History | 0.312 | sin pena ni gloria | South Korea (0.39), France (0.38), United Kingdom (0.37) |
| Short | 0.307 | sin pena ni gloria | South Korea (0.39), France (0.37), United Kingdom (0.37) |
| Game-Show | 0.299 | sin pena ni gloria | South Korea (0.38), France (0.37), United Kingdom (0.36) |
| Musical | 0.299 | sin pena ni gloria | South Korea (0.38), France (0.36), United Kingdom (0.36) |
| Fantasy | 0.292 | sin pena ni gloria | South Korea (0.37), France (0.36), United Kingdom (0.35) |
| Romance | 0.278 | sin pena ni gloria | South Korea (0.35), France (0.34), United Kingdom (0.34) |
| Sport | 0.275 | posible fracaso | South Korea (0.35), France (0.34), United Kingdom (0.33) |
| Action | 0.265 | posible fracaso | South Korea (0.34), France (0.33), United Kingdom (0.32) |
| Thriller | 0.264 | posible fracaso | South Korea (0.34), France (0.33), United Kingdom (0.32) |
| Sci-Fi | 0.253 | posible fracaso | South Korea (0.32), France (0.31), United Kingdom (0.31) |
| News | 0.228 | posible fracaso | South Korea (0.29), France (0.28), United Kingdom (0.28) |
| Family | 0.205 | posible fracaso | South Korea (0.27), France (0.26), United Kingdom (0.25) |
| Talk-Show | 0.205 | posible fracaso | South Korea (0.27), France (0.26), United Kingdom (0.25) |
| Reality-TV | 0.190 | posible fracaso | South Korea (0.25), France (0.24), United Kingdom (0.24) |
| Horror | 0.143 | posible fracaso | South Korea (0.19), France (0.18), United Kingdom (0.18) |

### Random Forest

| Genero | P(exito) | Categoria | 3 mejores regiones productoras |
|--------|----------|-----------|--------------------------------|
| Film-Noir | 0.437 | posible exito | United States (0.49), France (0.46), South Korea (0.46) |
| Documentary | 0.395 | posible exito | Denmark (0.47), Australia (0.45), United States (0.45) |
| Biography | 0.374 | posible exito | France (0.45), United Kingdom (0.44), Japan (0.41) |
| Drama | 0.358 | posible exito | South Korea (0.49), Japan (0.47), France (0.46) |
| War | 0.314 | posible exito | France (0.42), Japan (0.36), United Kingdom (0.35) |
| Music | 0.304 | posible exito | Japan (0.36), France (0.35), United States (0.33) |
| History | 0.303 | posible exito | France (0.39), South Korea (0.33), Iran (0.33) |
| Western | 0.265 | posible exito | United States (0.38), France (0.33), Japan (0.30) |
| Comedy | 0.228 | posible exito | Poland (0.35), China (0.33), United Kingdom (0.33) |
| Crime | 0.227 | sin pena ni gloria | Japan (0.34), France (0.30), Turkey (0.27) |
| Short | 0.225 | sin pena ni gloria | France (0.33), Japan (0.25), South Korea (0.25) |
| Mystery | 0.220 | sin pena ni gloria | France (0.30), United Kingdom (0.28), Japan (0.27) |
| Animation | 0.218 | sin pena ni gloria | Japan (0.30), France (0.27), United States (0.25) |
| Adventure | 0.200 | sin pena ni gloria | France (0.32), Japan (0.31), Mexico (0.22) |
| Romance | 0.198 | sin pena ni gloria | France (0.32), Japan (0.25), United Kingdom (0.24) |
| Sport | 0.197 | sin pena ni gloria | France (0.28), United Kingdom (0.25), Japan (0.25) |
| Musical | 0.196 | sin pena ni gloria | France (0.30), Japan (0.25), United Kingdom (0.24) |
| Fantasy | 0.181 | sin pena ni gloria | France (0.26), Japan (0.23), United Kingdom (0.23) |
| Thriller | 0.174 | posible fracaso | India (0.26), Japan (0.25), United Kingdom (0.23) |
| Action | 0.157 | posible fracaso | Japan (0.27), France (0.23), South Korea (0.22) |
| Horror | 0.148 | posible fracaso | France (0.24), Belgium (0.24), Japan (0.20) |
| Sci-Fi | 0.142 | posible fracaso | France (0.26), Japan (0.22), India (0.17) |
| Family | 0.138 | posible fracaso | France (0.24), Iran (0.20), United Kingdom (0.19) |
| Talk-Show | 0.120 | posible fracaso | India (0.27), France (0.17), South Korea (0.17) |
| News | 0.119 | posible fracaso | France (0.20), South Korea (0.17), India (0.17) |
| Game-Show | 0.108 | posible fracaso | South Korea (0.21), France (0.18), India (0.18) |
| Reality-TV | 0.092 | posible fracaso | Japan (0.17), South Korea (0.15), France (0.13) |

### XGBoost

| Genero | P(exito) | Categoria | 3 mejores regiones productoras |
|--------|----------|-----------|--------------------------------|
| Film-Noir | 0.689 | posible exito | Japan (0.81), Denmark (0.76), Norway (0.76) |
| Documentary | 0.364 | posible exito | Japan (0.47), United States (0.45), South Korea (0.45) |
| Drama | 0.343 | posible exito | Japan (0.44), Iran (0.44), Denmark (0.42) |
| Western | 0.275 | posible exito | Japan (0.39), United States (0.37), France (0.34) |
| Comedy | 0.217 | posible exito | United Kingdom (0.31), Norway (0.31), India (0.27) |
| Animation | 0.181 | posible exito | Netherlands (0.29), Iran (0.28), Japan (0.25) |
| War | 0.165 | posible exito | France (0.25), Japan (0.25), Denmark (0.21) |
| Crime | 0.163 | posible exito | Japan (0.26), Denmark (0.23), India (0.23) |
| Music | 0.163 | posible exito | Japan (0.24), France (0.21), Germany (0.21) |
| History | 0.161 | sin pena ni gloria | Iran (0.30), Japan (0.23), Turkey (0.20) |
| Mystery | 0.154 | sin pena ni gloria | Japan (0.23), United Kingdom (0.21), France (0.20) |
| Biography | 0.153 | sin pena ni gloria | Japan (0.24), United Kingdom (0.21), France (0.20) |
| Romance | 0.147 | sin pena ni gloria | France (0.24), Japan (0.21), South Korea (0.20) |
| Short | 0.141 | sin pena ni gloria | France (0.29), Iran (0.23), Japan (0.19) |
| Thriller | 0.133 | sin pena ni gloria | Norway (0.21), Japan (0.20), United Kingdom (0.19) |
| Fantasy | 0.132 | sin pena ni gloria | Japan (0.20), United Kingdom (0.19), Norway (0.18) |
| Musical | 0.132 | sin pena ni gloria | France (0.23), Japan (0.21), Denmark (0.17) |
| Adventure | 0.128 | sin pena ni gloria | Japan (0.23), Mexico (0.20), United Kingdom (0.19) |
| Sport | 0.127 | posible fracaso | India (0.24), Japan (0.18), United Kingdom (0.18) |
| Action | 0.102 | posible fracaso | Denmark (0.19), Japan (0.19), Hong Kong (0.14) |
| Sci-Fi | 0.085 | posible fracaso | Japan (0.13), France (0.13), United Kingdom (0.13) |
| Horror | 0.085 | posible fracaso | Japan (0.14), France (0.13), Italy (0.12) |
| Family | 0.084 | posible fracaso | Iran (0.25), Japan (0.11), Norway (0.11) |
| Talk-Show | 0.054 | posible fracaso | India (0.43), United States (0.09), Iran (0.06) |
| News | 0.048 | posible fracaso | United Kingdom (0.08), India (0.07), Japan (0.07) |
| Reality-TV | 0.043 | posible fracaso | Japan (0.12), United Kingdom (0.07), Iran (0.07) |
| Game-Show | 0.041 | posible fracaso | South Korea (0.23), India (0.07), United Kingdom (0.06) |

## Matriz P(exito) por region y genero (los 3 modelos)

Matriz completa (region activa x genero) por modelo en `outputs/clf_matriz_region_genero.csv` y figuras `13/14/15_clf_heatmap_*.png` (ejes en el mismo orden para comparar).
> En Regresion Logistica el orden de regiones es igual en todos los generos: el modelo lineal no captura interaccion region-genero. En Random Forest y XGBoost varia por genero.
