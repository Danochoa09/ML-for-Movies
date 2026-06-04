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

## Ranking de generos por probabilidad de exito

(P(exito) promedio entre regiones, segun Random Forest. La region es el origen de produccion; se listan las 3 mejores por genero.)

| Genero | P(exito) | Categoria | 3 mejores regiones productoras |
|--------|----------|-----------|--------------------------------|
| Documentary | 0.502 | posible exito | West Germany (0.65), Denmark (0.61), Australia (0.60) |
| Film-Noir | 0.440 | posible exito | Soviet Union (0.50), United States (0.49), West Germany (0.46) |
| Drama | 0.408 | posible exito | South Korea (0.56), Japan (0.55), China (0.52) |
| Biography | 0.397 | posible exito | France (0.47), United Kingdom (0.46), Japan (0.44) |
| History | 0.360 | posible exito | France (0.44), Soviet Union (0.41), Japan (0.39) |
| Music | 0.337 | posible exito | Japan (0.39), Soviet Union (0.39), France (0.38) |
| War | 0.331 | posible exito | France (0.44), Soviet Union (0.41), Japan (0.37) |
| Animation | 0.310 | posible exito | Japan (0.43), France (0.37), Soviet Union (0.37) |
| Western | 0.277 | posible exito | United States (0.39), Soviet Union (0.36), France (0.34) |
| Crime | 0.258 | sin pena ni gloria | Japan (0.39), France (0.34), Soviet Union (0.34) |
| Mystery | 0.249 | sin pena ni gloria | Soviet Union (0.34), France (0.33), Japan (0.31) |
| Comedy | 0.244 | sin pena ni gloria | Soviet Union (0.44), Poland (0.41), China (0.38) |
| Short | 0.241 | sin pena ni gloria | France (0.35), Soviet Union (0.32), Japan (0.27) |
| Adventure | 0.227 | sin pena ni gloria | Soviet Union (0.38), France (0.36), Japan (0.35) |
| Romance | 0.222 | sin pena ni gloria | Soviet Union (0.38), France (0.34), Japan (0.27) |
| Sport | 0.213 | sin pena ni gloria | Soviet Union (0.32), France (0.29), Japan (0.27) |
| Musical | 0.209 | sin pena ni gloria | Soviet Union (0.33), France (0.31), Japan (0.26) |
| Fantasy | 0.202 | sin pena ni gloria | Soviet Union (0.31), France (0.28), Japan (0.26) |
| News | 0.197 | posible fracaso | Soviet Union (0.31), France (0.29), Japan (0.26) |
| Game-Show | 0.190 | posible fracaso | Soviet Union (0.30), France (0.29), Japan (0.26) |
| Talk-Show | 0.183 | posible fracaso | Soviet Union (0.28), France (0.27), Japan (0.25) |
| Thriller | 0.181 | posible fracaso | India (0.27), Japan (0.26), Soviet Union (0.24) |
| Family | 0.174 | posible fracaso | Soviet Union (0.32), France (0.29), Japan (0.23) |
| Sci-Fi | 0.151 | posible fracaso | France (0.28), Japan (0.24), Soviet Union (0.23) |
| Action | 0.151 | posible fracaso | Japan (0.29), Soviet Union (0.28), France (0.23) |
| Reality-TV | 0.149 | posible fracaso | Japan (0.26), Soviet Union (0.23), France (0.22) |
| Horror | 0.146 | posible fracaso | France (0.24), Belgium (0.24), Japan (0.20) |
