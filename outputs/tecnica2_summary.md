# Tecnica 2 - Agrupacion (K-Means)

- Titulos: **101,606**
- Variables: 5 numericas + 27 generos

## Seleccion de k

| k | inercia | silueta |
|---|---------|---------|
| 2 | 3,081,678 | 0.095 |
| 3 | 2,958,067 | 0.075 |
| 4 | 2,842,937 | 0.104 |
| 5 | 2,722,674 | 0.110 |
| 6 | 2,630,290 | 0.111 |
| 7 | 2,526,350 | 0.132 |  <- elegido
| 8 | 2,438,294 | 0.121 |
| 9 | 2,331,151 | 0.132 |
| 10 | 2,227,146 | 0.132 |

k elegido (parsimonia, silueta a <= 0.005 del maximo): **7**
> Nota: la silueta es baja (~0.13); los generos binarios y la alta dimension hacen que los grupos se solapen. Aun asi los perfiles son interpretables.

## Perfil de los clusters

### Cluster 0  (n=1,494, 1.5%)
- Rating medio: 6.43 | votos mediana: 957
- Duracion media: 105 min | anio medio: 1981 | % pelicula: 90%
- Generos dominantes: Musical, Comedy, Drama

### Cluster 1  (n=3,733, 3.7%)
- Rating medio: 7.06 | votos mediana: 1,099
- Duracion media: 117 min | anio medio: 2002 | % pelicula: 65%
- Generos dominantes: History, Drama, Documentary

### Cluster 2  (n=12,228, 12.0%)
- Rating medio: 6.41 | votos mediana: 1,104
- Duracion media: 75 min | anio medio: 1999 | % pelicula: 56%
- Generos dominantes: Adventure, Animation, Action

### Cluster 3  (n=10,554, 10.4%)
- Rating medio: 6.93 | votos mediana: 517
- Duracion media: 83 min | anio medio: 2010 | % pelicula: 44%
- Generos dominantes: Documentary, Reality-TV, Comedy

### Cluster 4  (n=50,034, 49.2%)
- Rating medio: 6.41 | votos mediana: 969
- Duracion media: 95 min | anio medio: 2000 | % pelicula: 75%
- Generos dominantes: Drama, Comedy, Romance

### Cluster 5  (n=18,816, 18.5%)
- Rating medio: 5.63 | votos mediana: 1,442
- Duracion media: 96 min | anio medio: 2003 | % pelicula: 86%
- Generos dominantes: Thriller, Horror, Drama

### Cluster 6  (n=4,747, 4.7%)
- Rating medio: 6.79 | votos mediana: 657
- Duracion media: 17 min | anio medio: 1977 | % pelicula: 86%
- Generos dominantes: Short, Animation, Comedy

Varianza explicada PCA 2D: 7% + 6% = 13%
