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

### Reporte - Regresion Logistica
```
              precision    recall  f1-score   support

    no_exito      0.810     0.617     0.700      9035
       exito      0.421     0.658     0.513      3828

    accuracy                          0.629     12863
   macro avg      0.615     0.637     0.607     12863
weighted avg      0.694     0.629     0.645     12863

```

### Reporte - Random Forest
```
              precision    recall  f1-score   support

    no_exito      0.836     0.651     0.732      9035
       exito      0.459     0.699     0.554      3828

    accuracy                          0.665     12863
   macro avg      0.647     0.675     0.643     12863
weighted avg      0.724     0.665     0.679     12863

```

### Reporte - XGBoost
```
              precision    recall  f1-score   support

    no_exito      0.838     0.646     0.729      9035
       exito      0.457     0.705     0.555      3828

    accuracy                          0.663     12863
   macro avg      0.647     0.675     0.642     12863
weighted avg      0.724     0.663     0.677     12863

```

**Mejor modelo (ROC-AUC): Random Forest**

## Top 3 generos con mayor P(exito) por region

(probabilidad predicha por Random Forest)

- **Argentina**: Film-Noir (0.44), Documentary (0.42), Biography (0.39)
- **Australia**: Documentary (0.60), Film-Noir (0.43), Biography (0.39)
- **Belgium**: Documentary (0.53), Film-Noir (0.43), Biography (0.39)
- **Brazil**: Film-Noir (0.42), Documentary (0.36), History (0.33)
- **Canada**: Documentary (0.48), Film-Noir (0.38), Biography (0.37)
- **China**: Drama (0.52), Documentary (0.47), Film-Noir (0.44)
- **Denmark**: Documentary (0.61), Film-Noir (0.45), Drama (0.45)
- **Egypt**: Documentary (0.49), Film-Noir (0.41), Biography (0.39)
- **Finland**: Documentary (0.50), Film-Noir (0.42), Biography (0.39)
- **France**: Documentary (0.51), Drama (0.50), Biography (0.47)
- **Germany**: Documentary (0.52), Film-Noir (0.44), Drama (0.42)
- **Greece**: Documentary (0.46), Film-Noir (0.44), Drama (0.40)
- **Hong Kong**: Documentary (0.54), Drama (0.46), Film-Noir (0.43)
- **India**: Documentary (0.49), Film-Noir (0.44), Biography (0.40)
- **Iran**: Drama (0.50), Documentary (0.47), Film-Noir (0.43)
- **Italy**: Documentary (0.45), Film-Noir (0.43), Drama (0.41)
- **Japan**: Drama (0.55), Documentary (0.48), Film-Noir (0.45)
- **Mexico**: Film-Noir (0.45), Drama (0.42), Biography (0.40)
- **Netherlands**: Documentary (0.42), Film-Noir (0.41), Biography (0.37)
- **Norway**: Documentary (0.54), Film-Noir (0.45), Biography (0.40)
- **Poland**: Drama (0.46), Documentary (0.46), Film-Noir (0.45)
- **Russia**: Documentary (0.49), Film-Noir (0.42), Biography (0.37)
- **South Korea**: Drama (0.56), Documentary (0.54), Film-Noir (0.46)
- **Soviet Union**: Drama (0.52), Film-Noir (0.50), Documentary (0.47)
- **Spain**: Documentary (0.54), Film-Noir (0.44), Drama (0.40)
- **Sweden**: Documentary (0.56), Film-Noir (0.45), Biography (0.41)
- **Turkey**: Documentary (0.54), Drama (0.45), Film-Noir (0.44)
- **United Kingdom**: Documentary (0.53), Biography (0.46), Film-Noir (0.45)
- **United States**: Documentary (0.57), Film-Noir (0.49), Biography (0.39)
- **West Germany**: Documentary (0.65), Drama (0.48), Film-Noir (0.46)
