# Clasificacion de exito - comparacion de 3 modelos

- Filas (titulo, region): **126,276**
- Regiones modeladas (>= 500 titulos): 33 + 'Other'
- Generos: 27
- Distribucion de clases: {'mediocre': 100378, 'fracaso': 16162, 'exito': 9736}

## Resultados en test (20%)

| Modelo | F1-macro | Accuracy |
|--------|----------|----------|
| Regresion Logistica | 0.462 | 0.539 |
| Random Forest | 0.530 | 0.628 |
| XGBoost | 0.529 | 0.621 |

### Reporte - Regresion Logistica
```
              precision    recall  f1-score   support

     fracaso      0.314     0.681     0.430      3233
    mediocre      0.917     0.496     0.644     20076
       exito      0.197     0.744     0.311      1947

    accuracy                          0.539     25256
   macro avg      0.476     0.641     0.462     25256
weighted avg      0.784     0.539     0.591     25256

```

### Reporte - Random Forest
```
              precision    recall  f1-score   support

     fracaso      0.384     0.708     0.498      3233
    mediocre      0.924     0.603     0.730     20076
       exito      0.237     0.755     0.361      1947

    accuracy                          0.628     25256
   macro avg      0.515     0.689     0.530     25256
weighted avg      0.802     0.628     0.672     25256

```

### Reporte - XGBoost
```
              precision    recall  f1-score   support

     fracaso      0.373     0.740     0.496      3233
    mediocre      0.930     0.588     0.721     20076
       exito      0.242     0.766     0.368      1947

    accuracy                          0.621     25256
   macro avg      0.515     0.698     0.529     25256
weighted avg      0.806     0.621     0.665     25256

```

**Mejor modelo (F1-macro): Random Forest**

## Top 3 generos con mayor P(exito) por region

(probabilidad predicha por Random Forest)

- **Argentina**: Documentary (0.45), Short (0.43), History (0.20)
- **Australia**: Documentary (0.47), Short (0.41), History (0.20)
- **Belgium**: Documentary (0.50), Short (0.39), History (0.19)
- **Brazil**: Documentary (0.54), Short (0.42), Music (0.19)
- **Canada**: Documentary (0.52), Short (0.38), Music (0.22)
- **China**: Documentary (0.47), Short (0.42), History (0.19)
- **Denmark**: Short (0.40), Documentary (0.39), History (0.19)
- **Egypt**: Documentary (0.53), Short (0.41), History (0.20)
- **Finland**: Short (0.42), Documentary (0.38), History (0.19)
- **France**: Documentary (0.53), Short (0.37), History (0.19)
- **Germany**: Documentary (0.47), Short (0.37), History (0.17)
- **Greece**: Documentary (0.57), Short (0.45), History (0.25)
- **Hong Kong**: Documentary (0.51), Short (0.39), Music (0.18)
- **Hungary**: Documentary (0.53), Short (0.42), History (0.20)
- **India**: Documentary (0.61), Biography (0.51), History (0.51)
- **Iran**: Documentary (0.51), Short (0.43), History (0.19)
- **Ireland**: Documentary (0.55), Short (0.45), History (0.20)
- **Italy**: Documentary (0.38), Short (0.36), Music (0.16)
- **Japan**: Short (0.35), Documentary (0.31), Drama (0.22)
- **Mexico**: Documentary (0.49), Short (0.42), Music (0.19)
- **Netherlands**: Documentary (0.42), Short (0.37), History (0.17)
- **Norway**: Documentary (0.48), Short (0.42), History (0.19)
- **Poland**: Documentary (0.48), Short (0.39), History (0.18)
- **Russia**: Documentary (0.47), Short (0.40), History (0.17)
- **South Korea**: Documentary (0.54), Short (0.42), Music (0.25)
- **Soviet Union**: Short (0.53), Documentary (0.53), Comedy (0.45)
- **Spain**: Documentary (0.44), Short (0.40), History (0.18)
- **Sweden**: Short (0.39), Documentary (0.32), Music (0.18)
- **Switzerland**: Short (0.41), Documentary (0.29), History (0.19)
- **Turkey**: Documentary (0.65), Short (0.38), Music (0.18)
- **United Kingdom**: Documentary (0.69), Short (0.40), Music (0.20)
- **United States**: Documentary (0.58), Short (0.38), Animation (0.20)
- **West Germany**: Short (0.42), Documentary (0.36), Music (0.20)
