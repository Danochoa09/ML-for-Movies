"""Entregable de negocio: ranking de GENEROS por probabilidad de exito,
calculado para los TRES modelos.

Para cada (region de origen activa, genero) se predice P(exito) con un titulo
sintetico. Para no inventar un formato arbitrario, se predice tanto como pelicula
(is_movie=1) como serie (is_movie=0) y se combinan ponderando por la proporcion
real de cada formato en ese genero (asi un genero de TV como Reality-TV no se
evalua como si fuera pelicula).

Solo se recomiendan REGIONES ACTIVAS (que siguen produciendo: >= MIN_ACTIVE
titulos desde 2020). Esto excluye regiones desaparecidas como Soviet Union o
West Germany, que tenian exito inflado por survivorship y hoy no producen.

Por genero:
  - puntaje = P(exito) promedio entre regiones activas,
  - 3 niveles por terciles: posible exito / sin pena ni gloria / posible fracaso,
  - las 3 regiones activas con mayor P(exito) para ese genero.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from .config import OUT_DIR
from .data import Dataset
from .evaluate import CLASS_TO_INT, savefig

NIVELES = ["posible fracaso", "sin pena ni gloria", "posible exito"]
COLORES = {
    "posible exito": "#2ca02c",
    "sin pena ni gloria": "#ff7f0e",
    "posible fracaso": "#d62728",
}


def _exito_col(model) -> int:
    classes = list(model.named_steps["clf"].classes_)
    return classes.index(CLASS_TO_INT["exito"])


def _movie_share(ds: Dataset) -> dict:
    """Proporcion de peliculas (vs series) en train para cada genero."""
    shares = {}
    for gcol, gname in zip(ds.genre_cols, ds.genre_names):
        mask = ds.X_train[gcol] == 1
        shares[gname] = float(ds.X_train.loc[mask, "is_movie"].mean()) if mask.any() else 1.0
    return shares


def _predict_region_genre(model, ds: Dataset, regions, shares) -> pd.DataFrame:
    """P(exito) por (region, genero), promediando formatos segun su peso real."""
    col = _exito_col(model)
    med_len = float(ds.X_train["length"].median())

    synth, recs = [], []
    for region in regions:
        for gcol, gname in zip(ds.genre_cols, ds.genre_names):
            for fmt in (0, 1):
                r = {c: 0 for c in ds.genre_cols}
                r[gcol] = 1
                r["length"] = med_len
                r["is_movie"] = fmt
                r["region_grp"] = region
                synth.append(r)
                recs.append((region, gname, fmt))

    sx = pd.DataFrame(synth)[ds.num_cols + ds.cat_cols + ds.genre_cols]
    proba = model.predict_proba(sx)[:, col]

    tmp = pd.DataFrame(recs, columns=["region", "genre", "fmt"])
    tmp["p"] = proba
    piv = tmp.pivot_table(index=["region", "genre"], columns="fmt", values="p").reset_index()
    piv["p_exito"] = piv.apply(
        lambda r: shares[r["genre"]] * r[1] + (1 - shares[r["genre"]]) * r[0], axis=1
    )
    return piv[["region", "genre", "p_exito"]]


def _ranking_one(pred: pd.DataFrame) -> pd.DataFrame:
    score = pred.groupby("genre")["p_exito"].mean().sort_values(ascending=False)
    q1, q2 = score.quantile([1 / 3, 2 / 3])

    def nivel(p: float) -> str:
        if p >= q2:
            return "posible exito"
        if p >= q1:
            return "sin pena ni gloria"
        return "posible fracaso"

    top_reg = (
        pred.sort_values(["genre", "p_exito"], ascending=[True, False])
        .groupby("genre")
        .head(3)
        .groupby("genre")
        .apply(lambda d: ", ".join(
            f"{r.region} ({r.p_exito:.2f})" for r in d.itertuples()
        ), include_groups=False)
    )

    return (
        pd.DataFrame({"p_exito_promedio": score.round(4)})
        .assign(categoria=lambda d: d["p_exito_promedio"].map(nivel))
        .assign(top_regiones=top_reg)
        .reset_index()
    )


def rankings_all_models(results, ds: Dataset) -> dict:
    """Ranking de generos para cada modelo. Guarda CSV (largo) y una figura
    comparativa de 3 paneles. Devuelve {modelo: DataFrame}."""
    shares = _movie_share(ds)
    regions = ds.active_regions

    rankings, long_rows = {}, []
    for name, res in results.items():
        pred = _predict_region_genre(res.model, ds, regions, shares)
        rk = _ranking_one(pred)
        rankings[name] = rk
        for x in rk.itertuples():
            long_rows.append({
                "modelo": name,
                "genre": x.genre,
                "p_exito_promedio": x.p_exito_promedio,
                "categoria": x.categoria,
                "top_regiones": x.top_regiones,
            })

    pd.DataFrame(long_rows).to_csv(OUT_DIR / "clf_ranking_generos.csv", index=False)

    # figura: 3 paneles (uno por modelo)
    n = len(rankings)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 8))
    if n == 1:
        axes = [axes]
    for ax, (name, rk) in zip(axes, rankings.items()):
        ax.barh(rk["genre"], rk["p_exito_promedio"], color=rk["categoria"].map(COLORES))
        ax.invert_yaxis()
        ax.set_title(name)
        ax.set_xlabel("P(exito) promedio")
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORES[x]) for x in reversed(NIVELES)]
    fig.legend(handles, list(reversed(NIVELES)), loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Generos por probabilidad de exito (regiones activas) - por modelo")
    savefig("12_clf_ranking_generos.png")

    return rankings


def _slug(s: str) -> str:
    return s.lower().replace(" ", "_").replace(".", "")


def heatmaps_all_models(results, ds: Dataset, best_name: str) -> None:
    """Matriz P(exito) region (activa) x genero para CADA modelo: una figura por
    modelo (con ejes en el mismo orden para poder compararlas) + un CSV largo.

    Nota: en Regresion Logistica el orden de regiones es identico en todos los
    generos (modelo lineal sin interaccion region-genero); en los modelos de
    arboles varia por genero (capturan la interaccion).
    """
    shares = _movie_share(ds)
    preds = {
        name: _predict_region_genre(res.model, ds, ds.active_regions, shares)
        for name, res in results.items()
    }

    # orden compartido de ejes, definido por el mejor modelo
    ref = preds[best_name].pivot(index="region", columns="genre", values="p_exito")
    row_order = ref.mean(axis=1).sort_values(ascending=False).index
    col_order = ref.mean(axis=0).sort_values(ascending=False).index

    long_rows = []
    fig_idx = 13
    for name, pred in preds.items():
        mat = (
            pred.pivot(index="region", columns="genre", values="p_exito")
            .loc[row_order, col_order]
        )
        for r in pred.itertuples():
            long_rows.append({"modelo": name, "region": r.region,
                              "genre": r.genre, "p_exito": round(r.p_exito, 4)})

        plt.figure(figsize=(13, 8))
        sns.heatmap(mat, cmap="RdYlGn", center=float(mat.values.mean()),
                    cbar_kws={"label": "P(exito)"}, linewidths=0.3, linecolor="white")
        plt.title(f"P(exito) por region de origen y genero - {name}")
        plt.xlabel("Genero"); plt.ylabel("Region (activa)")
        savefig(f"{fig_idx}_clf_heatmap_{_slug(name)}.png")
        fig_idx += 1

    pd.DataFrame(long_rows).to_csv(
        OUT_DIR / "clf_matriz_region_genero.csv", index=False
    )
