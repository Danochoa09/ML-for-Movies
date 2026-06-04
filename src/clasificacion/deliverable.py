"""Entregable de negocio: top-3 generos con mayor P(exito) por region.

Se usa el mejor clasificador. Para cada (region, genero) se arma un titulo
sintetico (solo ese genero activo, duracion/anio medianos, formato pelicula)
y se predice la probabilidad de la clase 'exito'.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import FIG_DIR, OUT_DIR
from .data import Dataset
from .evaluate import CLASS_TO_INT, savefig


def _exito_col(model) -> int:
    """Indice de la columna 'exito' en predict_proba."""
    classes = list(model.named_steps["clf"].classes_)
    return classes.index(CLASS_TO_INT["exito"])


def top3_por_region(model, ds: Dataset) -> pd.DataFrame:
    col = _exito_col(model)
    med_len = float(ds.X["length"].median())
    med_year = float(ds.X["releaseYear"].median())

    rows = []
    for region in ds.regions_kept:
        synth = []
        for gcol in ds.genre_cols:
            r = {c: 0 for c in ds.genre_cols}
            r[gcol] = 1
            r["length"] = med_len
            r["releaseYear"] = med_year
            r["is_movie"] = 1
            r["region_grp"] = region
            synth.append(r)
        sx = pd.DataFrame(synth)[ds.num_cols + ds.cat_cols + ds.genre_cols]
        proba = model.predict_proba(sx)[:, col]
        for gname, p in zip(ds.genre_names, proba):
            rows.append({"region": region, "genre": gname, "p_exito": float(p)})

    pred = pd.DataFrame(rows)
    top3 = (
        pred.sort_values(["region", "p_exito"], ascending=[True, False])
        .groupby("region")
        .head(3)
        .reset_index(drop=True)
    )
    top3.to_csv(OUT_DIR / "clf_top3_generos_por_region.csv", index=False)

    # heatmap region x genero
    pivot = pred.pivot(index="region", columns="genre", values="p_exito")
    order = pivot.max(axis=1).sort_values(ascending=False).index[:20]
    plt.figure(figsize=(12, 7))
    sns.heatmap(pivot.loc[order], cmap="viridis", cbar_kws={"label": "P(exito)"})
    plt.title("Probabilidad de exito por region y genero (mejor modelo)")
    plt.xlabel("Genero"); plt.ylabel("Region")
    savefig("13_clf_heatmap_region_genero.png")

    return top3


def plot_feature_importance(model, ds: Dataset, fname: str) -> None:
    """Importancia de variables para modelos de arboles (RF / XGBoost)."""
    clf = model.named_steps["clf"]
    if not hasattr(clf, "feature_importances_"):
        return
    pre = model.named_steps["pre"]
    feat_names = (
        ds.num_cols
        + list(pre.named_transformers_["cat"].get_feature_names_out(ds.cat_cols))
        + ds.genre_names
    )
    imp = (
        pd.Series(clf.feature_importances_, index=feat_names)
        .sort_values(ascending=False)
        .head(15)
    )
    plt.figure(figsize=(7, 5))
    imp.sort_values().plot(kind="barh", color="#4682B4")
    plt.xlabel("Importancia")
    plt.title("Top 15 variables (mejor modelo de arboles)")
    savefig(fname)
