"""Utilidades de carga y limpieza del dataset IMDb 100k.

Tres CSV:
  - contentDataPrime.csv  : tabla principal (1 fila por contenido)
  - contentDataGenre.csv  : dataId -> genre  (formato largo, varios por id)
  - contentDataRegion.csv : dataId -> region (formato largo, varios por id)

Los valores faltantes vienen codificados como -1 (numericos) o NaN (texto).
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Columnas numericas donde -1 significa "faltante"
_SENTINEL_COLS = ["length", "releaseYear", "endYear", "votes", "rating", "gross"]


def load_prime(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Carga la tabla principal y reemplaza los centinelas -1 por NaN."""
    df = pd.read_csv(raw_dir / "contentDataPrime.csv")

    # length llega como texto; forzar numerico
    df["length"] = pd.to_numeric(df["length"], errors="coerce")

    for col in _SENTINEL_COLS:
        df[col] = df[col].replace(-1, np.nan)

    # certificate: normalizar vacios
    df["certificate"] = df["certificate"].fillna("Unknown")

    return df


def load_genres(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    g = pd.read_csv(raw_dir / "contentDataGenre.csv")
    # Eliminar artefactos de parseo (generos espurios de 1 letra: 'l', 'n', 'u')
    g = g[g["genre"].str.len() > 1]
    return g.reset_index(drop=True)


def load_regions(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    r = pd.read_csv(raw_dir / "contentDataRegion.csv")
    # Eliminar artefactos de parseo (regiones espurias 'l', 'n', 'u')
    r = r[r["region"].str.len() > 2]
    return r.reset_index(drop=True)


def genres_wide(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """One-hot de generos: una columna por genero, indexado por dataId."""
    g = load_genres(raw_dir)
    return (
        pd.crosstab(g["dataId"], g["genre"])
        .clip(upper=1)
        .add_prefix("genre_")
    )


def primary_region(raw_dir: Path = RAW_DIR) -> pd.Series:
    """Primera region listada por dataId (proxy de region principal)."""
    r = load_regions(raw_dir)
    return r.groupby("dataId")["region"].first().rename("primary_region")


def load_merged(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Tabla principal + one-hot de generos + region principal."""
    df = load_prime(raw_dir).set_index("dataId")
    df = df.join(genres_wide(raw_dir))
    df = df.join(primary_region(raw_dir))
    genre_cols = [c for c in df.columns if c.startswith("genre_")]
    df[genre_cols] = df[genre_cols].fillna(0).astype(int)
    return df.reset_index()
