"""Experimento de CLASIFICACION: compara tres clasificadores sobre la clase de
exito de un titulo (derivada del rating IMDb).

    Regresion Logistica  -> modelo base / lineal
    Random Forest        -> ensamble de arboles (bagging)
    XGBoost              -> boosting de arboles (modelo avanzado)

Clases:
    rating >= 8       -> 'exito'    (posible exito)
    5 <= rating < 8   -> 'mediocre' (sin pena ni gloria)
    rating < 5        -> 'fracaso'  (posible fracaso)

Entregable: top-3 generos con mayor P(exito) por region (mejor modelo).

Uso:
    python src/run_clasificacion.py
"""
from pathlib import Path

from sklearn.model_selection import train_test_split

from clasificacion.config import CLASS_ORDER, MIN_REGION, OUT_DIR, RANDOM_STATE, TEST_SIZE
from clasificacion.data import build_dataset
from clasificacion.models import build_models
from clasificacion.evaluate import evaluate_all, plot_comparison, plot_confusions
from clasificacion.deliverable import top3_por_region, plot_feature_importance

SUMMARY = OUT_DIR / "clf_summary.md"


def main() -> None:
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    # ----------------------------------------------------------------- #
    # 1. Datos
    # ----------------------------------------------------------------- #
    ds = build_dataset()
    log("# Clasificacion de exito - comparacion de 3 modelos\n")
    log(f"- Filas (titulo, region): **{len(ds.X):,}**")
    log(f"- Regiones modeladas (>= {MIN_REGION} titulos): {len(ds.regions_kept)} + 'Other'")
    log(f"- Generos: {len(ds.genre_cols)}")
    log(f"- Distribucion de clases: {ds.y.value_counts().to_dict()}")
    log("")

    X_tr, X_te, y_tr, y_te = train_test_split(
        ds.X, ds.y, test_size=TEST_SIZE, stratify=ds.y, random_state=RANDOM_STATE
    )

    # ----------------------------------------------------------------- #
    # 2. Entrenar y evaluar los 3 modelos
    # ----------------------------------------------------------------- #
    models = build_models(ds.num_cols, ds.cat_cols, ds.genre_cols)
    results = evaluate_all(models, X_tr, y_tr, X_te, y_te)

    log("## Resultados en test (20%)\n")
    log("| Modelo | F1-macro | Accuracy |")
    log("|--------|----------|----------|")
    for r in results.values():
        log(f"| {r.name} | {r.f1_macro:.3f} | {r.accuracy:.3f} |")
    log("")

    for r in results.values():
        log(f"### Reporte - {r.name}")
        log("```")
        log(r.report)
        log("```")
        log("")

    comp = plot_comparison(results, "10_clf_model_compare.png")
    plot_confusions(results, "11_clf_confusion.png")

    # ----------------------------------------------------------------- #
    # 3. Mejor modelo -> importancia + entregable
    # ----------------------------------------------------------------- #
    best_name = max(results, key=lambda n: results[n].f1_macro)
    best = results[best_name].model
    log(f"**Mejor modelo (F1-macro): {best_name}**\n")

    plot_feature_importance(best, ds, "12_clf_feature_importance.png")

    top3 = top3_por_region(best, ds)
    log("## Top 3 generos con mayor P(exito) por region\n")
    log(f"(probabilidad predicha por {best_name})\n")
    for region in ds.regions_kept:
        sub = top3[top3["region"] == region]
        items = ", ".join(f"{x.genre} ({x.p_exito:.2f})" for x in sub.itertuples())
        log(f"- **{region}**: {items}")
    log("")

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResumen -> {SUMMARY}")
    print(f"Top3 CSV -> {OUT_DIR / 'clf_top3_generos_por_region.csv'}")


if __name__ == "__main__":
    main()
