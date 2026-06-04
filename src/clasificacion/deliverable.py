"""Entregable de negocio: ranking de GENEROS por probabilidad de exito.

Para cada (region de origen, genero) se arma un titulo sintetico (solo ese genero
activo, duracion mediana, formato pelicula) y se predice P(exito) con el mejor
modelo. Luego, por genero:

  - se promedia P(exito) entre regiones -> puntaje del genero,
  - se clasifica en 3 niveles: posible exito / sin pena ni gloria / posible
    fracaso (por terciles del puntaje),
  - se listan las 3 regiones que mejor producen ese genero (mayor P(exito)).

La region se interpreta como el ORIGEN de produccion del titulo.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import FIG_DIR, OUT_DIR
from .data import Dataset
from .evaluate import CLASS_TO_INT, savefig

NIVELES = ["posible fracaso", "sin pena ni gloria", "posible exito"]
COLORES = {
    "posible exito": "#2ca02c",
    "sin pena ni gloria": "#ff7f0e",
    "posible fracaso": "#d62728",
}


def _exito_col(model) -> int:
    """Indice de la columna 'exito' en predict_proba."""
    classes = list(model.named_steps["clf"].classes_)
    return classes.index(CLASS_TO_INT["exito"])


def _predict_region_genre(model, ds: Dataset) -> pd.DataFrame:
    """P(exito) para cada combinacion (region, genero)."""
    col = _exito_col(model)
    med_len = float(ds.X_train["length"].median())

    rows = []
    for region in ds.regions_kept:
        synth = []
        for gcol in ds.genre_cols:
            r = {c: 0 for c in ds.genre_cols}
            r[gcol] = 1
            r["length"] = med_len
            r["is_movie"] = 1
            r["region_grp"] = region
            synth.append(r)
        sx = pd.DataFrame(synth)[ds.num_cols + ds.cat_cols + ds.genre_cols]
        proba = model.predict_proba(sx)[:, col]
        for gname, p in zip(ds.genre_names, proba):
            rows.append({"region": region, "genre": gname, "p_exito": float(p)})
    return pd.DataFrame(rows)


def ranking_generos(model, ds: Dataset, top_n_regiones: int = 3) -> pd.DataFrame:
    pred = _predict_region_genre(model, ds)

    # puntaje por genero = P(exito) promedio entre regiones
    score = pred.groupby("genre")["p_exito"].mean().sort_values(ascending=False)

    # 3 niveles por terciles del puntaje
    q1, q2 = score.quantile([1 / 3, 2 / 3])

    def nivel(p: float) -> str:
        if p >= q2:
            return "posible exito"
        if p >= q1:
            return "sin pena ni gloria"
        return "posible fracaso"

    # top regiones por genero (mayor P(exito))
    top_reg = (
        pred.sort_values(["genre", "p_exito"], ascending=[True, False])
        .groupby("genre")
        .head(top_n_regiones)
        .groupby("genre")
        .apply(lambda d: ", ".join(
            f"{r.region} ({r.p_exito:.2f})" for r in d.itertuples()
        ), include_groups=False)
    )

    out = (
        pd.DataFrame({"p_exito_promedio": score.round(4)})
        .assign(categoria=lambda d: d["p_exito_promedio"].map(nivel))
        .assign(top_regiones=top_reg)
        .reset_index()
        .rename(columns={"index": "genre"})
    )
    out.to_csv(OUT_DIR / "clf_ranking_generos.csv", index=False)

    # figura: barras de generos por P(exito), coloreadas por nivel
    plt.figure(figsize=(8, 8))
    colors = out["categoria"].map(COLORES)
    plt.barh(out["genre"], out["p_exito_promedio"], color=colors)
    plt.gca().invert_yaxis()  # mayor arriba
    plt.xlabel("P(exito) promedio entre regiones")
    plt.title("Generos por probabilidad de exito (mejor modelo)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORES[n]) for n in reversed(NIVELES)]
    plt.legend(handles, list(reversed(NIVELES)), loc="lower right")
    savefig("13_clf_ranking_generos.png")

    return out


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
