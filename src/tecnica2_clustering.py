"""Tecnica 2 - AGRUPACION (clustering).

Objetivo: descubrir "perfiles" naturales de contenido (peliculas/series) sin
usar etiquetas. Agrupamos titulos por sus caracteristicas (rating, popularidad,
duracion, epoca, tipo y generos) con K-Means y describimos cada grupo.

Esto responde a la idea del proyecto: que tipos de contenido existen y cuales
tienden a gustar (alto rating / muchos votos).

Seleccion de k: metodo del codo (inercia) + coeficiente de silueta.

Uso:
    python src/tecnica2_clustering.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from data_utils import load_prime, load_genres

sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "outputs" / "figures"
OUT_DIR = ROOT / "outputs"
SUMMARY = OUT_DIR / "tecnica2_summary.md"

K_RANGE = range(2, 11)
RANDOM_STATE = 42
SIL_SAMPLE = 10000   # muestra para silueta (costosa en 100k filas)


def savefig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=120)
    plt.close()
    print(f"  figura -> {name}")


def build_features():
    """Una fila por titulo. Numericas + one-hot de generos."""
    prime = load_prime().set_index("dataId")
    genres = load_genres()
    gw = (
        pd.crosstab(genres["dataId"], genres["genre"])
        .clip(upper=1)
        .add_prefix("genre_")
    )
    genre_cols = list(gw.columns)

    df = prime.join(gw)
    df[genre_cols] = df[genre_cols].fillna(0).astype(int)

    # imputar numericas
    df["length"] = df["length"].fillna(df["length"].median())
    df["releaseYear"] = df["releaseYear"].fillna(df["releaseYear"].median())
    df["votes"] = df["votes"].fillna(df["votes"].median())
    df["log_votes"] = np.log1p(df["votes"])
    df["is_movie"] = (df["contentType"] == "movie").astype(int)

    num_cols = ["rating", "log_votes", "length", "releaseYear", "is_movie"]
    feat_cols = num_cols + genre_cols
    X = df[feat_cols].copy()
    return df, X, num_cols, genre_cols


def main() -> None:
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    df, X, num_cols, genre_cols = build_features()
    genre_names = [c.replace("genre_", "") for c in genre_cols]

    log("# Tecnica 2 - Agrupacion (K-Means)\n")
    log(f"- Titulos: **{len(X):,}**")
    log(f"- Variables: {len(num_cols)} numericas + {len(genre_cols)} generos")
    log("")

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    # -------------------------------------------------------------- #
    # Seleccion de k: codo + silueta
    # -------------------------------------------------------------- #
    rng = np.random.RandomState(RANDOM_STATE)
    sil_idx = rng.choice(len(Xs), size=min(SIL_SAMPLE, len(Xs)), replace=False)

    inertias, sils = [], []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(Xs)
        inertias.append(km.inertia_)
        sils.append(silhouette_score(Xs[sil_idx], labels[sil_idx]))

    # Parsimonia: el k mas pequeno cuya silueta este dentro de una tolerancia
    # del maximo (evita anadir clusters por mejoras despreciables).
    SIL_TOL = 0.005
    sil_max = max(sils)
    best_k = next(
        k for k, s in zip(K_RANGE, sils) if s >= sil_max - SIL_TOL
    )
    log("## Seleccion de k\n")
    log("| k | inercia | silueta |")
    log("|---|---------|---------|")
    for k, inr, s in zip(K_RANGE, inertias, sils):
        mark = "  <- elegido" if k == best_k else ""
        log(f"| {k} | {inr:,.0f} | {s:.3f} |{mark}")
    log(f"\nk elegido (parsimonia, silueta a <= {SIL_TOL} del maximo): **{best_k}**")
    log(f"> Nota: la silueta es baja (~{sil_max:.2f}); los generos binarios y la "
        f"alta dimension hacen que los grupos se solapen. Aun asi los perfiles "
        f"son interpretables.\n")

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(list(K_RANGE), inertias, "o-", color="#4682B4")
    ax[0].set(xlabel="k", ylabel="Inercia", title="Metodo del codo")
    ax[1].plot(list(K_RANGE), sils, "o-", color="#5cb85c")
    ax[1].axvline(best_k, ls="--", color="#d9534f")
    ax[1].set(xlabel="k", ylabel="Silueta", title="Coeficiente de silueta")
    savefig("14_t2_k_selection.png")

    # -------------------------------------------------------------- #
    # K-Means final
    # -------------------------------------------------------------- #
    km = KMeans(n_clusters=best_k, n_init=10, random_state=RANDOM_STATE)
    df["cluster"] = km.fit_predict(Xs)

    # -------------------------------------------------------------- #
    # Perfil de cada cluster
    # -------------------------------------------------------------- #
    log("## Perfil de los clusters\n")
    profiles = []
    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]
        top_g = (
            sub[genre_cols].mean().sort_values(ascending=False).head(3)
            .index.str.replace("genre_", "").tolist()
        )
        prof = {
            "cluster": c,
            "n": len(sub),
            "rating": round(sub["rating"].mean(), 2),
            "votes_med": int(sub["votes"].median()),
            "length": round(sub["length"].mean(), 0),
            "year": round(sub["releaseYear"].mean(), 0),
            "pct_movie": round(sub["is_movie"].mean() * 100, 0),
            "top_generos": ", ".join(top_g),
        }
        profiles.append(prof)
        log(f"### Cluster {c}  (n={len(sub):,}, {len(sub)/len(df)*100:.1f}%)")
        log(f"- Rating medio: {prof['rating']} | votos mediana: {prof['votes_med']:,}")
        log(f"- Duracion media: {prof['length']:.0f} min | anio medio: {prof['year']:.0f}"
            f" | % pelicula: {prof['pct_movie']:.0f}%")
        log(f"- Generos dominantes: {prof['top_generos']}")
        log("")

    pd.DataFrame(profiles).to_csv(OUT_DIR / "tecnica2_perfiles_cluster.csv", index=False)
    df[["title", "contentType", "rating", "votes", "cluster"]].to_csv(
        OUT_DIR / "tecnica2_asignaciones.csv", index=False
    )

    # -------------------------------------------------------------- #
    # Figuras de perfil
    # -------------------------------------------------------------- #
    # heatmap: media de generos por cluster (z-score para resaltar)
    gmean = df.groupby("cluster")[genre_cols].mean()
    gmean.columns = genre_names
    z = (gmean - gmean.mean()) / (gmean.std() + 1e-9)
    plt.figure(figsize=(12, 5))
    sns.heatmap(z, cmap="coolwarm", center=0, cbar_kws={"label": "z-score"})
    plt.title("Perfil de generos por cluster (z-score)")
    plt.xlabel("Genero"); plt.ylabel("Cluster")
    savefig("15_t2_cluster_genres.png")

    # numericas por cluster
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    sns.boxplot(data=df, x="cluster", y="rating", ax=ax[0], palette="Set2")
    ax[0].set_title("Rating por cluster")
    sns.boxplot(data=df, x="cluster", y="log_votes", ax=ax[1], palette="Set2")
    ax[1].set_title("log(votos) por cluster")
    sns.boxplot(data=df[df["releaseYear"] >= 1950], x="cluster", y="releaseYear",
                ax=ax[2], palette="Set2")
    ax[2].set_title("Anio por cluster")
    savefig("16_t2_cluster_numeric.png")

    # PCA 2D
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(Xs)
    samp = rng.choice(len(coords), size=min(8000, len(coords)), replace=False)
    plt.figure(figsize=(7, 6))
    sns.scatterplot(
        x=coords[samp, 0], y=coords[samp, 1],
        hue=df["cluster"].values[samp], palette="tab10", s=12, alpha=0.5,
        legend="full",
    )
    var = pca.explained_variance_ratio_ * 100
    plt.xlabel(f"PC1 ({var[0]:.0f}%)"); plt.ylabel(f"PC2 ({var[1]:.0f}%)")
    plt.title("Clusters proyectados en 2D (PCA)")
    savefig("17_t2_pca_scatter.png")

    log(f"Varianza explicada PCA 2D: {var[0]:.0f}% + {var[1]:.0f}% = {var.sum():.0f}%")
    log("")

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResumen -> {SUMMARY}")
    print(f"Perfiles -> {OUT_DIR / 'tecnica2_perfiles_cluster.csv'}")


if __name__ == "__main__":
    main()
