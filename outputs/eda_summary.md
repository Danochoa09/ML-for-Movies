# EDA - IMDb 100k Movies/TV Shows

- Registros: **101,606**
- Columnas: 11 -> ['dataId', 'contentType', 'title', 'length', 'releaseYear', 'endYear', 'votes', 'rating', 'gross', 'certificate', 'description']
- Tipos de contenido: {'movie': 72912, 'tvSeries': 28694}
- Generos distintos: 27
- Regiones distintas: 191
- Rango de anios: 1874 - 2023

## Valores faltantes (% por columna)

- endYear: 89.9%
- gross: 85.6%
- length: 7.6%
- releaseYear: 6.1%

## Estadisticos descriptivos

```
         length  releaseYear       votes     rating         gross
count  93869.00     95417.00   101605.00  101606.00  1.465000e+04
mean      88.92      1999.60    11348.78       6.36  2.086566e+07
std       50.23        23.28    61437.97       1.32  5.039984e+07
min        1.00      1874.00      102.00       1.00  0.000000e+00
25%       61.00      1991.00      455.00       5.70  1.600000e+05
50%       91.00      2008.00      952.00       6.50  1.980000e+06
75%      106.00      2016.00     3097.00       7.30  1.882750e+07
max      990.00      2023.00  2715939.00      10.00  9.366600e+08
```

## Top 15 generos

- Drama: 46,809
- Comedy: 34,946
- Romance: 14,314
- Action: 14,285
- Crime: 13,030
- Thriller: 10,432
- Adventure: 10,030
- Documentary: 8,666
- Horror: 8,664
- Animation: 8,531
- Mystery: 6,507
- Family: 6,415
- Fantasy: 5,020
- Short: 4,800
- Sci-Fi: 4,127

## Rating promedio por genero (mejores y peores)

Mejores:
- Documentary: 7.26
- History: 7.05
- Biography: 6.94
- Animation: 6.84
- War: 6.79
Peores:
- Fantasy: 6.17
- Action: 6.01
- Thriller: 5.75
- Sci-Fi: 5.51
- Horror: 5.16

## Correlacion (variables numericas)

```
             length  releaseYear  votes  rating  gross
length         1.00         0.08   0.08   -0.01   0.17
releaseYear    0.08         1.00   0.03   -0.07   0.09
votes          0.08         0.03   1.00    0.11   0.66
rating        -0.01        -0.07   0.11    1.00   0.13
gross          0.17         0.09   0.66    0.13   1.00
```
