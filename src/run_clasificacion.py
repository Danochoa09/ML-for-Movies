"""Experimento de CLASIFICACION BINARIA: predecir si un titulo sera un EXITO de
taquilla y publico, con validacion temporal.

EXITO  =  rating >= 6.5  Y  ( votes > mediana  O  gross > mediana )

Se comparan tres clasificadores:
    Regresion Logistica  -> modelo base / lineal
    Random Forest        -> ensamble de arboles (bagging)
    XGBoost              -> boosting de arboles (modelo avanzado)

Validacion temporal: train con titulos < 2020, test con titulos >= 2020.
Entregable: top-3 generos con mayor P(exito) por region (mejor modelo).

Uso:
    python src/run_clasificacion.py
"""
from clasificacion.config import MIN_REGION, OUT_DIR, SPLIT_YEAR
from clasificacion.data import build_dataset
from clasificacion.models import build_models
from clasificacion.evaluate import evaluate_all, plot_comparison, plot_confusions
from clasificacion.deliverable import rankings_all_models, heatmaps_all_models

from sklearn.metrics import recall_score

from clasificacion.evaluate import POS

SUMMARY = OUT_DIR / "clf_summary.md"


def _recall_exito(r) -> float:
    return recall_score(r.y_true, r.y_pred, pos_label=POS)

# Breve explicacion de cada tecnica (para el informe / lectura rapida)
COMO_FUNCIONA = {
    "Regresion Logistica": "Modelo lineal base: aprende un peso por variable y "
    "combina todo en una probabilidad. Simple e interpretable; solo capta "
    "relaciones lineales.",
    "Random Forest": "Ensamble (bagging): cientos de arboles entrenados con "
    "muestras distintas que votan. Capta no linealidad e interacciones y entrega "
    "importancia de variables.",
    "XGBoost": "Boosting: arboles en serie donde cada uno corrige los errores del "
    "anterior. Modelo avanzado, suele rendir muy bien en datos tabulares.",
}


def main() -> None:
    lines: list[str] = []

    def log(msg: str = "") -> None:
        print(msg)
        lines.append(msg)

    # ----------------------------------------------------------------- #
    # 1. Datos (con particion temporal y umbrales calculados en train)
    # ----------------------------------------------------------------- #
    ds = build_dataset()
    i = ds.info
    log("# Clasificacion de EXITO de taquilla/publico - 3 modelos\n")
    log("**Definicion de exito:** rating >= 6.5 Y (votes > mediana O gross > mediana)\n")
    log(f"- Umbrales (solo train): mediana votos = {i['median_votes']:.0f}, "
        f"mediana gross = {i['median_gross']:.0f}")
    log(f"- Validacion temporal: train anio < {SPLIT_YEAR} ({i['n_train']:,} filas), "
        f"test anio >= {SPLIT_YEAR} ({i['n_test']:,} filas)")
    log(f"- Tasa de exito: train {i['exito_rate_train']*100:.1f}% | "
        f"test {i['exito_rate_test']*100:.1f}%")
    log(f"- Regiones modeladas (>= {MIN_REGION} en train): {len(ds.regions_kept)} + 'Other'")
    log(f"- Generos (one-hot): {len(ds.genre_cols)}")
    log(f"- Predictoras: {ds.num_cols + ds.cat_cols} + generos "
        f"(el anio NO se usa: es el eje del split)")
    log("")

    # ----------------------------------------------------------------- #
    # 2. Entrenar y evaluar los 3 modelos
    # ----------------------------------------------------------------- #
    models = build_models(ds.num_cols, ds.cat_cols, ds.genre_cols)
    results = evaluate_all(
        models, ds.X_train, ds.y_train, ds.X_test, ds.y_test
    )

    log("## Resultados en test temporal (>= 2020)\n")
    log("| Modelo | F1 (exito) | ROC-AUC | Accuracy |")
    log("|--------|-----------|---------|----------|")
    for r in results.values():
        log(f"| {r.name} | {r.f1_exito:.3f} | {r.roc_auc:.3f} | {r.accuracy:.3f} |")
    log("")

    for r in results.values():
        log(f"### {r.name}")
        log(f"*Como funciona:* {COMO_FUNCIONA.get(r.name, '')}\n")
        log(f"*Resultado:* F1(exito)={r.f1_exito:.3f}, ROC-AUC={r.roc_auc:.3f}, "
            f"accuracy={r.accuracy:.3f}. Detecta el "
            f"{_recall_exito(r)*100:.0f}% de los exitos reales del test.\n")
        log("```")
        log(r.report)
        log("```")
        log("")

    plot_comparison(results, "10_clf_model_compare.png")
    plot_confusions(results, "11_clf_confusion.png")

    best_name = max(results, key=lambda n: results[n].roc_auc)
    log(f"**Mejor modelo (ROC-AUC): {best_name}**\n")

    # ----------------------------------------------------------------- #
    # 3. Entregable: ranking de generos por P(exito) para los 3 modelos
    # ----------------------------------------------------------------- #
    rankings = rankings_all_models(results, ds)
    excl = ds.info.get("regions_excluded", [])
    log("## Ranking de generos por probabilidad de exito (por modelo)\n")
    log(f"P(exito) promedio entre **regiones activas** (las que aun producen, "
        f">= 2020). La region es el origen de produccion; se listan las 3 mejores "
        f"por genero. Regiones activas: {len(ds.active_regions)}.")
    if excl:
        log(f"\n> Excluidas por no producir desde 2020 (region desaparecida): "
            f"{', '.join(excl)}.")
    log("")
    for name, rk in rankings.items():
        log(f"### {name}\n")
        log("| Genero | P(exito) | Categoria | 3 mejores regiones productoras |")
        log("|--------|----------|-----------|--------------------------------|")
        for x in rk.itertuples():
            log(f"| {x.genre} | {x.p_exito_promedio:.3f} | {x.categoria} | {x.top_regiones} |")
        log("")

    # matriz completa P(exito) region x genero, para los 3 modelos
    heatmaps_all_models(results, ds, best_name)
    log("## Matriz P(exito) por region y genero (los 3 modelos)\n")
    log("Matriz completa (region activa x genero) por modelo en "
        "`outputs/clf_matriz_region_genero.csv` y figuras "
        "`13/14/15_clf_heatmap_*.png` (ejes en el mismo orden para comparar).")
    log("> En Regresion Logistica el orden de regiones es igual en todos los "
        "generos: el modelo lineal no captura interaccion region-genero. En "
        "Random Forest y XGBoost varia por genero.\n")

    SUMMARY.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nResumen -> {SUMMARY}")
    print(f"Ranking CSV (3 modelos) -> {OUT_DIR / 'clf_ranking_generos.csv'}")


if __name__ == "__main__":
    main()
