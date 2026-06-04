"""Configuracion compartida del experimento de clasificacion.

EXITO via IEP (Indice de Rendimiento Economico): combina calidad, alcance y
rentabilidad en un puntaje continuo en [0, 1].

  log-escalado: votes y gross siguen ley de potencias -> se aplica log antes de
  normalizar. N(x) = normalizacion Min-Max a [0, 1] (min/max calculados en train).

  Si el titulo TIENE gross (peliculas comerciales):
      IEP = 0.2*N(rating) + 0.3*N(log votes) + 0.5*N(log gross)

  Si NO tiene gross (series de TV, cine independiente, ~85% de los datos):
      el 50% del peso del dinero se transfiere a los votos:
      IEP = 0.2*N(rating) + 0.8*N(log votes)

Asi no se pierden registros (mantenemos > 100.000) y el sesgo "solo rating"
(documentales de nicho como mayores exitos) se corrige al pesar el alcance
comercial (votos/recaudo).

El IEP se discretiza en 3 clases por TERCILES (calculados en train):
  tercio bajo -> 'fracaso', medio -> 'mediocre', alto -> 'exito'.

Validacion temporal: train con titulos < 2020, test con titulos >= 2020.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

# Pesos del IEP (deben sumar 1.0)
W_RATING = 0.2
W_VOTES = 0.3
W_GROSS = 0.5
# Sin gross: el peso del gross pasa a votos
W_VOTES_SIN_GROSS = W_VOTES + W_GROSS   # 0.8

SPLIT_YEAR = 2020          # train: anio < 2020 ; test: anio >= 2020

# Clasificacion multiclase (orden ordinal: fracaso < mediocre < exito)
CLASS_ORDER = ["fracaso", "mediocre", "exito"]

MIN_REGION = 500           # regiones con al menos este numero de titulos (en train)
# Para el ENTREGABLE solo se recomiendan regiones aun activas (>= MIN_ACTIVE
# titulos en test, >= 2020): excluye regiones desaparecidas (Soviet Union, etc.).
MIN_ACTIVE = 20
RANDOM_STATE = 42
