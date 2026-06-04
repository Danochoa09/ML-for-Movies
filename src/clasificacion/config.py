"""Configuracion compartida del experimento de clasificacion.

Definicion de EXITO (clasificacion binaria): un titulo es 'exito' si combina
buena aceptacion con traccion masiva:

    rating >= 6.5  Y  ( votes > mediana(votes)  O  gross > mediana(gross) )

es decir, no basta con gustar a la critica: tambien debe haber sido visto
masivamente (votos) o haber recaudado (gross). Asi los modelos aprenden a
predecir "exito de taquilla y publico", no solo calidad percibida.

Validacion temporal: se entrena con titulos anteriores a 2020 y se prueba con
titulos de 2020 en adelante (simula predecir el futuro). Los umbrales (medianas)
y el agrupado de regiones se calculan SOLO con train para evitar fuga de datos.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

RATING_MIN_EXITO = 6.5     # aceptacion minima
SPLIT_YEAR = 2020          # train: anio < 2020 ; test: anio >= 2020

# Clasificacion binaria. Positiva = 'exito'.
CLASS_ORDER = ["no_exito", "exito"]
POS_LABEL = "exito"

MIN_REGION = 500           # regiones con al menos este numero de titulos (en train)
# Para el ENTREGABLE solo se recomiendan regiones aun activas: con al menos esta
# cantidad de titulos en el periodo de test (>= 2020). Asi se excluyen regiones
# desaparecidas (Soviet Union, West Germany) que no producen en el presente.
MIN_ACTIVE = 20
RANDOM_STATE = 42
