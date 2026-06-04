# Tecnica 1 - Clasificacion de exito (rating)

- Filas (titulo, region): **126,276**
- Regiones modeladas (>= 500 titulos): 33 + 'Other'
- Generos: 27
- Distribucion de clases: {'mediocre': 100378, 'fracaso': 16162, 'exito': 9736}

## Desempenio en test (20%)

- Random Forest   : F1-macro = 0.530
- Reg. Logistica  : F1-macro = 0.462

### Reporte Random Forest
```
              precision    recall  f1-score   support

       exito      0.238     0.760     0.363      1947
     fracaso      0.385     0.706     0.498      3233
    mediocre      0.924     0.604     0.730     20076

    accuracy                          0.629     25256
   macro avg      0.516     0.690     0.530     25256
weighted avg      0.802     0.629     0.672     25256

```

## Top 3 generos con mayor P(exito) por region

(probabilidad predicha por Random Forest)

- **Argentina**: Documentary (0.47), Short (0.40), History (0.22)
- **Australia**: Documentary (0.49), Short (0.37), History (0.20)
- **Belgium**: Documentary (0.52), Short (0.35), History (0.21)
- **Brazil**: Documentary (0.56), Short (0.40), History (0.20)
- **Canada**: Documentary (0.54), Short (0.36), Music (0.23)
- **China**: Documentary (0.49), Short (0.40), History (0.21)
- **Denmark**: Documentary (0.41), Short (0.37), History (0.20)
- **Egypt**: Documentary (0.55), Short (0.39), History (0.22)
- **Finland**: Documentary (0.40), Short (0.39), History (0.21)
- **France**: Documentary (0.54), Short (0.36), History (0.20)
- **Germany**: Documentary (0.47), Short (0.33), Music (0.18)
- **Greece**: Documentary (0.57), Short (0.42), History (0.28)
- **Hong Kong**: Documentary (0.53), Short (0.36), History (0.19)
- **Hungary**: Documentary (0.55), Short (0.39), History (0.21)
- **India**: Documentary (0.61), Biography (0.54), History (0.52)
- **Iran**: Documentary (0.51), Short (0.40), History (0.21)
- **Ireland**: Documentary (0.56), Short (0.42), History (0.22)
- **Italy**: Documentary (0.39), Short (0.35), History (0.17)
- **Japan**: Documentary (0.35), Short (0.34), Drama (0.23)
- **Mexico**: Documentary (0.50), Short (0.39), History (0.21)
- **Netherlands**: Documentary (0.43), Short (0.34), History (0.19)
- **Norway**: Documentary (0.50), Short (0.39), History (0.21)
- **Poland**: Documentary (0.50), Short (0.37), History (0.19)
- **Russia**: Documentary (0.49), Short (0.37), History (0.19)
- **South Korea**: Documentary (0.56), Short (0.41), Music (0.26)
- **Soviet Union**: Documentary (0.54), Short (0.51), Comedy (0.43)
- **Spain**: Documentary (0.45), Short (0.37), History (0.19)
- **Sweden**: Short (0.36), Documentary (0.35), History (0.20)
- **Switzerland**: Short (0.39), Documentary (0.30), History (0.21)
- **Turkey**: Documentary (0.66), Short (0.36), History (0.20)
- **United Kingdom**: Documentary (0.71), Short (0.39), Music (0.21)
- **United States**: Documentary (0.58), Short (0.36), Animation (0.21)
- **West Germany**: Short (0.39), Documentary (0.38), History (0.21)
