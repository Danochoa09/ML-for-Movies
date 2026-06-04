"""Configuracion compartida del experimento de clasificacion."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "outputs"
FIG_DIR = OUT_DIR / "figures"

# Clases de exito definidas sobre el rating IMDb:
#   rating >= 8       -> 'exito'    (posible exito)
#   5 <= rating < 8   -> 'mediocre' (sin pena ni gloria)
#   rating < 5        -> 'fracaso'  (posible fracaso)
EXITO_MIN = 8.0
FRACASO_MAX = 5.0
# Orden ordinal: 0 fracaso < 1 mediocre < 2 exito
CLASS_ORDER = ["fracaso", "mediocre", "exito"]

MIN_REGION = 500     # regiones con al menos este numero de titulos
TEST_SIZE = 0.2
RANDOM_STATE = 42
