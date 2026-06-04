"""Construccion del conjunto de datos para clasificacion binaria de EXITO,
con validacion temporal (train < 2020, test >= 2020).

Una fila por (titulo, region). Variables predictoras: tipo de contenido,
duracion, region y generos (one-hot). NO se usa el anio como predictor: es el
eje de la particion temporal y, al estar el test en anios no vistos, usarlo
seria extrapolar. Los umbrales (medianas de votes/gross) y el agrupado de
regiones se calculan unicamente con train.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_utils import load_prime, load_genres, load_regions

from .config import MIN_REGION, RATING_MIN_EXITO, SPLIT_YEAR


@dataclass
class Dataset:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    num_cols: list
    cat_cols: list
    genre_cols: list
    regions_kept: list
    info: dict = field(default_factory=dict)

    @property
    def genre_names(self):
        return [c.replace("genre_", "") for c in self.genre_cols]


def _label_exito(df: pd.DataFrame, vmed: float, gmed: float) -> pd.Series:
    masivo = (df["votes"] > vmed) | (df["gross"].notna() & (df["gross"] > gmed))
    es_exito = (df["rating"] >= RATING_MIN_EXITO) & masivo
    return np.where(es_exito, "exito", "no_exito")


def build_dataset() -> Dataset:
    prime = load_prime()
    genres = load_genres()
    regions = load_regions()

    # Se requiere anio conocido para la particion temporal
    prime = prime.dropna(subset=["releaseYear"]).copy()
    is_train = prime["releaseYear"] < SPLIT_YEAR

    # ---- umbrales calculados SOLO con train (sin fuga) ----
    vmed = float(prime.loc[is_train, "votes"].median())
    gmed = float(prime.loc[is_train, "gross"].median())
    len_med = float(prime.loc[is_train, "length"].median())

    # etiqueta de exito (a nivel titulo)
    prime["clase"] = _label_exito(prime, vmed, gmed)

    # variables a nivel titulo
    prime["is_movie"] = (prime["contentType"] == "movie").astype(int)
    prime["length"] = prime["length"].fillna(len_med)

    # one-hot de generos
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

    # agrupar regiones poco frecuentes (conteo solo en train)
    train_counts = df.loc[df["releaseYear"] < SPLIT_YEAR, "region"].value_counts()
    keep = sorted(train_counts[train_counts >= MIN_REGION].index)
    df["region_grp"] = np.where(df["region"].isin(set(keep)), df["region"], "Other")

    num_cols = ["length", "is_movie"]
    cat_cols = ["region_grp"]
    feat_cols = num_cols + cat_cols + genre_cols

    train = df[df["releaseYear"] < SPLIT_YEAR]
    test = df[df["releaseYear"] >= SPLIT_YEAR]

    info = {
        "median_votes": vmed,
        "median_gross": gmed,
        "n_train": len(train),
        "n_test": len(test),
        "exito_rate_train": float((train["clase"] == "exito").mean()),
        "exito_rate_test": float((test["clase"] == "exito").mean()),
    }

    return Dataset(
        X_train=train[feat_cols].copy(),
        y_train=train["clase"].copy(),
        X_test=test[feat_cols].copy(),
        y_test=test["clase"].copy(),
        num_cols=num_cols,
        cat_cols=cat_cols,
        genre_cols=genre_cols,
        regions_kept=keep,
        info=info,
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
