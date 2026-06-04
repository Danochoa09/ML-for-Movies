"""Analisis Exploratorio de Datos (EDA) del dataset IMDb 100k.

Genera figuras en outputs/figures/ y un resumen en outputs/eda_summary.md.

Uso:
    python src/eda.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana, solo guarda archivos
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from data_utils import load_prime, load_genres, load_regions

sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
SUMMARY = ROOT / "outputs" / "eda_summary.md"


def savefig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=120)
    plt.close()
    print(f"  figura -> {name}")


def main() -> None:
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    df = load_prime()
    genres = load_genres()
    regions = load_regions()

    # ------------------------------------------------------------------ #
    # 1. Vista general
    # ------------------------------------------------------------------ #
    log("# EDA - IMDb 100k Movies/TV Shows\n")
    log(f"- Registros: **{len(df):,}**")
    log(f"- Columnas: {len(df.columns)} -> {list(df.columns)}")
    log(f"- Tipos de contenido: {df['contentType'].value_counts().to_dict()}")
    log(f"- Generos distintos: {genres['genre'].nunique()}")
    log(f"- Regiones distintas: {regions['region'].nunique()}")
    log(f"- Rango de anios: {int(df['releaseYear'].min())} - {int(df['releaseYear'].max())}")
    log("")

    # ------------------------------------------------------------------ #
    # 2. Valores faltantes
    # ------------------------------------------------------------------ #
    miss = df.isna().mean().mul(100).round(1).sort_values(ascending=False)
    miss = miss[miss > 0]
    log("## Valores faltantes (% por columna)\n")
    for col, pct in miss.items():
        log(f"- {col}: {pct}%")
    log("")

    plt.figure(figsize=(7, 4))
    miss.plot(kind="barh", color="#d9534f")
    plt.xlabel("% faltante")
    plt.title("Valores faltantes por columna")
    savefig("01_missing_values.png")

    # ------------------------------------------------------------------ #
    # 3. Estadisticos descriptivos numericos
    # ------------------------------------------------------------------ #
    num_cols = ["length", "releaseYear", "votes", "rating", "gross"]
    desc = df[num_cols].describe().round(2)
    log("## Estadisticos descriptivos\n")
    log("```")
    log(desc.to_string())
    log("```")
    log("")

    # ------------------------------------------------------------------ #
    # 4. Distribucion de rating
    # ------------------------------------------------------------------ #
    plt.figure(figsize=(7, 4))
    sns.histplot(df["rating"].dropna(), bins=40, kde=True, color="#5bc0de")
    plt.xlabel("Rating IMDb")
    plt.title("Distribucion de calificaciones")
    savefig("02_rating_dist.png")

    # ------------------------------------------------------------------ #
    # 5. Rating por tipo de contenido
    # ------------------------------------------------------------------ #
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="contentType", y="rating", palette="Set2")
    plt.title("Rating por tipo de contenido")
    savefig("03_rating_by_type.png")

    # ------------------------------------------------------------------ #
    # 6. Estrenos por anio (ultimas decadas)
    # ------------------------------------------------------------------ #
    plt.figure(figsize=(8, 4))
    yr = df[df["releaseYear"].between(1920, 2025)]["releaseYear"]
    sns.histplot(yr, bins=int(yr.max() - yr.min()), color="#9370DB")
    plt.xlabel("Anio de estreno")
    plt.title("Cantidad de estrenos por anio")
    savefig("04_releases_per_year.png")

    # ------------------------------------------------------------------ #
    # 7. Top generos
    # ------------------------------------------------------------------ #
    top_g = genres["genre"].value_counts().head(15)
    log("## Top 15 generos\n")
    for g, c in top_g.items():
        log(f"- {g}: {c:,}")
    log("")

    plt.figure(figsize=(7, 5))
    top_g.sort_values().plot(kind="barh", color="#5cb85c")
    plt.xlabel("Cantidad de titulos")
    plt.title("Top 15 generos")
    savefig("05_top_genres.png")

    # ------------------------------------------------------------------ #
    # 8. Top regiones
    # ------------------------------------------------------------------ #
    top_r = regions["region"].value_counts().head(15)
    plt.figure(figsize=(7, 5))
    top_r.sort_values().plot(kind="barh", color="#f0ad4e")
    plt.xlabel("Cantidad de titulos")
    plt.title("Top 15 regiones")
    savefig("06_top_regions.png")

    # ------------------------------------------------------------------ #
    # 9. Rating promedio por genero
    # ------------------------------------------------------------------ #
    gj = genres.merge(df[["dataId", "rating"]], on="dataId")
    rating_by_genre = (
        gj.groupby("genre")["rating"].mean().sort_values(ascending=False).round(2)
    )
    log("## Rating promedio por genero (mejores y peores)\n")
    log("Mejores:")
    for g, v in rating_by_genre.head(5).items():
        log(f"- {g}: {v}")
    log("Peores:")
    for g, v in rating_by_genre.tail(5).items():
        log(f"- {g}: {v}")
    log("")

    plt.figure(figsize=(7, 6))
    rating_by_genre.sort_values().plot(kind="barh", color="#4682B4")
    plt.xlabel("Rating promedio")
    plt.title("Rating promedio por genero")
    savefig("07_rating_by_genre.png")

    # ------------------------------------------------------------------ #
    # 10. Correlacion numerica
    # ------------------------------------------------------------------ #
    plt.figure(figsize=(6, 5))
    corr = df[num_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", center=0)
    plt.title("Correlacion entre variables numericas")
    savefig("08_correlation.png")
    log("## Correlacion (variables numericas)\n")
    log("```")
    log(corr.round(2).to_string())
    log("```")
    log("")

    # ------------------------------------------------------------------ #
    # 11. Popularidad (votos) vs rating
    # ------------------------------------------------------------------ #
    plt.figure(figsize=(7, 5))
    sample = df.dropna(subset=["votes", "rating"]).sample(
        min(8000, len(df)), random_state=42
    )
    sns.scatterplot(
        data=sample, x="votes", y="rating", alpha=0.3, s=10, color="#c0392b"
    )
    plt.xscale("log")
    plt.xlabel("Votos (escala log)")
    plt.ylabel("Rating")
    plt.title("Popularidad vs Calificacion")
    savefig("09_votes_vs_rating.png")

    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResumen -> {SUMMARY}")
    print(f"Figuras -> {FIG_DIR}")


if __name__ == "__main__":
    main()
