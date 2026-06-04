"""Construccion del conjunto de datos para clasificacion.

Una fila por (titulo, region). El objetivo es la clase de exito derivada del
rating. Las variables predictoras son tipo de contenido, duracion, anio, region
y generos (one-hot).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_utils import load_prime, load_genres, load_regions

from .config import EXITO_MIN, FRACASO_MAX, MIN_REGION


def success_class(rating: float) -> str:
    if rating >= EXITO_MIN:
        return "exito"
    if rating < FRACASO_MAX:
        return "fracaso"
    return "mediocre"


class Dataset:
    """Contenedor del conjunto de datos y metadatos de columnas."""

    def __init__(self, X, y, num_cols, cat_cols, genre_cols, regions_kept):
        self.X = X
        self.y = y
        self.num_cols = num_cols
        self.cat_cols = cat_cols
        self.genre_cols = genre_cols
        self.genre_names = [c.replace("genre_", "") for c in genre_cols]
        self.regions_kept = regions_kept


def build_dataset() -> Dataset:
    prime = load_prime()
    genres = load_genres()
    regions = load_regions()

    # one-hot de generos por titulo
    gw = (
        pd.crosstab(genres["dataId"], genres["genre"])
        .clip(upper=1)
        .add_prefix("genre_")
    )
    genre_cols = list(gw.columns)

    base = prime.set_index("dataId").join(gw)
    base[genre_cols] = base[genre_cols].fillna(0).astype(int)

    # una fila por (titulo, region)
    df = regions.merge(base.reset_index(), on="dataId", how="inner")

    # agrupar regiones poco frecuentes
    counts = df["region"].value_counts()
    keep = sorted(counts[counts >= MIN_REGION].index)
    df["region_grp"] = np.where(df["region"].isin(set(keep)), df["region"], "Other")

    # objetivo y variables
    df = df.dropna(subset=["rating"]).copy()
    df["clase"] = df["rating"].apply(success_class)
    df["is_movie"] = (df["contentType"] == "movie").astype(int)
    df["length"] = df["length"].fillna(df["length"].median())
    df["releaseYear"] = df["releaseYear"].fillna(df["releaseYear"].median())

    num_cols = ["length", "releaseYear", "is_movie"]
    cat_cols = ["region_grp"]
    feat_cols = num_cols + cat_cols + genre_cols

    return Dataset(
        X=df[feat_cols].copy(),
        y=df["clase"].copy(),
        num_cols=num_cols,
        cat_cols=cat_cols,
        genre_cols=genre_cols,
        regions_kept=keep,
    )


def make_preprocessor(num_cols, cat_cols, genre_cols) -> ColumnTransformer:
    """Escala numericas, one-hot de region, deja pasar los generos binarios."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("gen", "passthrough", genre_cols),
        ]
    )
