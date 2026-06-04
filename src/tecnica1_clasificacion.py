"""Tecnica 1 - CLASIFICACION.

Objetivo: predecir la clase de exito de un titulo a partir de su genero,
region, tipo, duracion y anio. La clase se define por el rating IMDb:

    rating >= 8         -> 'exito'      (posible exito)
    5 <= rating < 8     -> 'mediocre'   (sin pena ni gloria)
    rating < 5          -> 'fracaso'    (posible fracaso)

Entregable de negocio: para cada region, los 3 generos con mayor probabilidad
predicha de 'exito'.

Modelo: Random Forest (maneja no linealidad + da importancia de variables).
Se compara contra una Regresion Logistica como linea base.

Uso:
    python src/tecnica1_clasificacion.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_utils import load_prime, load_genres, load_regions

sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
OUT_DIR = ROOT / "outputs"
SUMMARY = OUT_DIR / "tecnica1_summary.md"

CLASSES = ["fracaso", "mediocre", "exito"]
MIN_REGION = 500       # regiones con al menos este nro de titulos
RANDOM_STATE = 42


def success_class(rating: float) -> str:
    if rating >= 8:
        return "exito"
    if rating < 5:
        return "fracaso"
    return "mediocre"


def savefig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=120)
    plt.close()
    print(f"  figura -> {name}")


def build_dataset():
    """Una fila por (titulo, region). Devuelve X (DataFrame), y, listas de cols."""
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

    base = prime.set_index("dataId")
    base = base.join(gw)
    base[genre_cols] = base[genre_cols].fillna(0).astype(int)

    # explotar por region (1 fila por titulo-region)
    df = regions.merge(base.reset_index(), on="dataId", how="inner")

    # agrupar regiones poco frecuentes en 'Other'
    counts = df["region"].value_counts()
    keep = set(counts[counts >= MIN_REGION].index)
    df["region_grp"] = np.where(df["region"].isin(keep), df["region"], "Other")

    # objetivo
    df = df.dropna(subset=["rating"]).copy()
    df["clase"] = df["rating"].apply(success_class)

    # features
    df["is_movie"] = (df["contentType"] == "movie").astype(int)
    df["length"] = df["length"].fillna(df["length"].median())
    df["releaseYear"] = df["releaseYear"].fillna(df["releaseYear"].median())

    num_cols = ["length", "releaseYear", "is_movie"]
    cat_cols = ["region_grp"]
    feat_cols = num_cols + cat_cols + genre_cols

    X = df[feat_cols].copy()
    y = df["clase"].copy()
    return X, y, num_cols, cat_cols, genre_cols, sorted(keep)


def make_pipeline(model, num_cols, cat_cols, genre_cols) -> Pipeline:
    pre = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("gen", "passthrough", genre_cols),
        ]
    )
    return Pipeline([("pre", pre), ("clf", model)])


def main() -> None:
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    X, y, num_cols, cat_cols, genre_cols, regions_kept = build_dataset()
    genre_names = [c.replace("genre_", "") for c in genre_cols]

    log("# Tecnica 1 - Clasificacion de exito (rating)\n")
    log(f"- Filas (titulo, region): **{len(X):,}**")
    log(f"- Regiones modeladas (>= {MIN_REGION} titulos): {len(regions_kept)} + 'Other'")
    log(f"- Generos: {len(genre_cols)}")
    dist = y.value_counts()
    log(f"- Distribucion de clases: {dist.to_dict()}")
    log("")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # -------------------------------------------------------------- #
    # Modelo principal: Random Forest
    # -------------------------------------------------------------- #
    rf = make_pipeline(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=5,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        num_cols, cat_cols, genre_cols,
    )
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)
    rf_f1 = f1_score(y_te, rf_pred, average="macro")

    # -------------------------------------------------------------- #
    # Linea base: Regresion Logistica
    # -------------------------------------------------------------- #
    lr = make_pipeline(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        num_cols, cat_cols, genre_cols,
    )
    lr.fit(X_tr, y_tr)
    lr_pred = lr.predict(X_te)
    lr_f1 = f1_score(y_te, lr_pred, average="macro")

    log("## Desempenio en test (20%)\n")
    log(f"- Random Forest   : F1-macro = {rf_f1:.3f}")
    log(f"- Reg. Logistica  : F1-macro = {lr_f1:.3f}")
    log("")
    log("### Reporte Random Forest")
    log("```")
    log(classification_report(y_te, rf_pred, digits=3))
    log("```")
    log("")

    # matriz de confusion RF
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_te, rf_pred, labels=CLASSES, cmap="Blues", ax=ax, colorbar=False
    )
    ax.set_title("Matriz de confusion - Random Forest")
    savefig("10_t1_confusion_rf.png")

    # comparacion de modelos
    plt.figure(figsize=(5, 4))
    sns.barplot(x=["Random Forest", "Reg. Logistica"], y=[rf_f1, lr_f1],
                palette="Set2")
    plt.ylabel("F1-macro")
    plt.ylim(0, 1)
    plt.title("Comparacion de clasificadores")
    savefig("11_t1_model_compare.png")

    # -------------------------------------------------------------- #
    # Importancia de variables (RF)
    # -------------------------------------------------------------- #
    pre = rf.named_steps["pre"]
    feat_names = (
        num_cols
        + list(pre.named_transformers_["cat"].get_feature_names_out(cat_cols))
        + genre_names
    )
    importances = rf.named_steps["clf"].feature_importances_
    imp = pd.Series(importances, index=feat_names).sort_values(ascending=False).head(15)
    plt.figure(figsize=(7, 5))
    imp.sort_values().plot(kind="barh", color="#4682B4")
    plt.xlabel("Importancia")
    plt.title("Top 15 variables - Random Forest")
    savefig("12_t1_feature_importance.png")

    # -------------------------------------------------------------- #
    # Entregable: top 3 generos por region con mayor P(exito)
    # -------------------------------------------------------------- #
    exito_idx = list(rf.named_steps["clf"].classes_).index("exito")
    med_len = float(X["length"].median())
    med_year = float(X["releaseYear"].median())

    rows = []
    for region in regions_kept:
        synth = []
        for gcol in genre_cols:
            r = {c: 0 for c in genre_cols}
            r[gcol] = 1
            r["length"] = med_len
            r["releaseYear"] = med_year
            r["is_movie"] = 1
            r["region_grp"] = region
            synth.append(r)
        sx = pd.DataFrame(synth)[num_cols + cat_cols + genre_cols]
        proba = rf.predict_proba(sx)[:, exito_idx]
        for gname, p in zip(genre_names, proba):
            rows.append({"region": region, "genre": gname, "p_exito": p})

    pred = pd.DataFrame(rows)
    top3 = (
        pred.sort_values(["region", "p_exito"], ascending=[True, False])
        .groupby("region")
        .head(3)
        .reset_index(drop=True)
    )
    top3.to_csv(OUT_DIR / "tecnica1_top3_generos_por_region.csv", index=False)

    log("## Top 3 generos con mayor P(exito) por region\n")
    log("(probabilidad predicha por Random Forest)\n")
    for region in regions_kept:
        sub = top3[top3["region"] == region]
        items = ", ".join(
            f"{r.genre} ({r.p_exito:.2f})" for r in sub.itertuples()
        )
        log(f"- **{region}**: {items}")
    log("")

    # heatmap region x genero (P exito) para las regiones top por volumen
    pivot = pred.pivot(index="region", columns="genre", values="p_exito")
    # ordenar regiones por su mejor P(exito)
    order = pivot.max(axis=1).sort_values(ascending=False).index[:20]
    plt.figure(figsize=(12, 7))
    sns.heatmap(pivot.loc[order], cmap="viridis", cbar_kws={"label": "P(exito)"})
    plt.title("Probabilidad de exito por region y genero")
    plt.xlabel("Genero")
    plt.ylabel("Region")
    savefig("13_t1_heatmap_region_genero.png")

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResumen -> {SUMMARY}")
    print(f"Top3 CSV -> {OUT_DIR / 'tecnica1_top3_generos_por_region.csv'}")


if __name__ == "__main__":
    main()
