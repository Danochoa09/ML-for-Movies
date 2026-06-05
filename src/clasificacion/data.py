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

from .config import (
    MIN_ACTIVE, MIN_REGION, SPLIT_YEAR,
    W_RATING, W_VOTES, W_GROSS, W_VOTES_SIN_GROSS,
)


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
    active_regions: list          # regiones aun activas (>= MIN_ACTIVE en test)
    info: dict = field(default_factory=dict)

    @property
    def genre_names(self):
        return [c.replace("genre_", "") for c in self.genre_cols]


def _minmax_params(s: pd.Series) -> tuple[float, float]:
    return float(s.min()), float(s.max())


def _norm(s: pd.Series, lo: float, hi: float) -> pd.Series:
    """Min-Max a [0,1] con parametros (lo,hi) de train; clip para test."""
    if hi <= lo:
        return pd.Series(0.0, index=s.index)
    return ((s - lo) / (hi - lo)).clip(0, 1)


def _impute_length(base: pd.DataFrame, genre_cols, train_mask) -> pd.Series:
    """Imputa la duracion faltante con la mediana por genero (de train), no con la
    mediana global. Asi un Short/News sin dato recibe una duracion tipica de su
    nicho (coherente con _median_length de deliverable.py). Respaldos: mediana por
    tipo (pelicula/serie) -> mediana global."""
    length = base["length"].copy()
    tr_ok = train_mask & length.notna()
    global_med = float(length[tr_ok].median())
    type_med = base.loc[tr_ok].groupby("is_movie")["length"].median()
    # mediana de duracion por genero (en train, con dato)
    gmed = {}
    for g in genre_cols:
        sel = tr_ok & (base[g] == 1)
        gmed[g] = float(base.loc[sel, "length"].median()) if sel.any() else np.nan

    miss = length.isna()
    if miss.any():
        gmat = base.loc[miss, genre_cols].to_numpy(dtype=float)  # (k, n_gen)
        gvec = np.array([gmed[g] for g in genre_cols], dtype=float)
        valid = ~np.isnan(gvec)
        num = np.nansum(gmat[:, valid] * gvec[valid], axis=1)
        den = gmat[:, valid].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            imp = num / den  # promedio de medianas de sus generos
        # respaldo por tipo y global donde no haya generos con dato
        type_fallback = base.loc[miss, "is_movie"].map(type_med).to_numpy(dtype=float)
        imp = np.where(den > 0, imp, type_fallback)
        imp = np.where(np.isnan(imp), global_med, imp)
        length.loc[miss] = imp
    return length


def _compute_iep(df: pd.DataFrame, params: dict) -> pd.Series:
    """IEP por titulo. Normaliza con parametros de train. Redistribuye el peso
    del gross a votos cuando no hay recaudo."""
    r_n = _norm(df["rating"], *params["rating"])
    v_n = _norm(np.log10(df["votes"].clip(lower=1)), *params["logvotes"])
    g_log = np.log10(df["gross"].clip(lower=1))
    g_n = _norm(g_log, *params["loggross"])

    has_g = df["gross"].notna()
    iep = pd.Series(0.0, index=df.index)
    # con gross
    iep[has_g] = (W_RATING * r_n[has_g] + W_VOTES * v_n[has_g]
                  + W_GROSS * g_n[has_g])
    # sin gross: peso del gross -> votos
    iep[~has_g] = W_RATING * r_n[~has_g] + W_VOTES_SIN_GROSS * v_n[~has_g]
    return iep


def build_dataset() -> Dataset:
    prime = load_prime()
    genres = load_genres()
    regions = load_regions()

    # Se requiere anio conocido para la particion temporal
    prime = prime.dropna(subset=["releaseYear"]).copy()
    is_train = prime["releaseYear"] < SPLIT_YEAR

    # ---- parametros de normalizacion calculados SOLO con train (sin fuga) ----
    tr = prime.loc[is_train]
    params = {
        "rating": _minmax_params(tr["rating"]),
        "logvotes": _minmax_params(np.log10(tr["votes"].clip(lower=1))),
        # gross se normaliza solo sobre titulos con recaudo (train)
        "loggross": _minmax_params(np.log10(tr.loc[tr["gross"].notna(), "gross"].clip(lower=1))),
    }
    # IEP por titulo (el IEP es una propiedad del titulo, no de la region)
    prime["iep"] = _compute_iep(prime, params)
    prime["is_movie"] = (prime["contentType"] == "movie").astype(int)

    # one-hot de generos
    gw = (
        pd.crosstab(genres["dataId"], genres["genre"])
        .clip(upper=1)
        .add_prefix("genre_")
    )
    genre_cols = list(gw.columns)

    base = prime.set_index("dataId").join(gw)
    base[genre_cols] = base[genre_cols].fillna(0).astype(int)

    # imputar duracion faltante por genero (no con la mediana global)
    base["length"] = _impute_length(base, genre_cols, base["releaseYear"] < SPLIT_YEAR)

    # una fila por (titulo, region)
    df = regions.merge(base.reset_index(), on="dataId", how="inner")

    # Terciles calculados SOBRE LAS FILAS DE ENTRENAMIENTO YA EXPLOTADAS por region
    # (no a nivel titulo): asi las 3 clases quedan balanceadas en el dataset que el
    # modelo realmente ve. Si se calcularan antes del merge, los titulos con muchas
    # regiones (tipicamente exitos comerciales) inflarian su clase tras la explosion.
    is_train_row = df["releaseYear"] < SPLIT_YEAR
    q1, q2 = df.loc[is_train_row, "iep"].quantile([1 / 3, 2 / 3])

    def _clase(v: float) -> str:
        if v >= q2:
            return "exito"
        if v >= q1:
            return "mediocre"
        return "fracaso"

    df["clase"] = df["iep"].map(_clase)

    # agrupar regiones poco frecuentes (conteo solo en train)
    train_counts = df.loc[is_train_row, "region"].value_counts()
    keep = sorted(train_counts[train_counts >= MIN_REGION].index)
    df["region_grp"] = np.where(df["region"].isin(set(keep)), df["region"], "Other")

    # regiones aun activas: de las modeladas, las que siguen produciendo (>= 2020)
    test_counts = df.loc[df["releaseYear"] >= SPLIT_YEAR, "region"].value_counts()
    active = sorted(reg for reg in keep if test_counts.get(reg, 0) >= MIN_ACTIVE)

    num_cols = ["length", "is_movie"]
    cat_cols = ["region_grp"]
    feat_cols = num_cols + cat_cols + genre_cols

    train = df[df["releaseYear"] < SPLIT_YEAR]
    test = df[df["releaseYear"] >= SPLIT_YEAR]

    info = {
        "iep_q1": float(q1),
        "iep_q2": float(q2),
        "pct_con_gross": float(prime["gross"].notna().mean()),
        "n_train": len(train),
        "n_test": len(test),
        "dist_train": train["clase"].value_counts().to_dict(),
        "dist_test": test["clase"].value_counts().to_dict(),
        "regions_excluded": sorted(set(keep) - set(active)),
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
        active_regions=active,
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
