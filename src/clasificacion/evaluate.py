"""Entrenamiento, evaluacion y graficas comparativas de los clasificadores."""
from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.utils.class_weight import compute_sample_weight

from .config import CLASS_ORDER, FIG_DIR
from .models import NEEDS_SAMPLE_WEIGHT

sns.set_theme(style="whitegrid")

# mapa clase -> entero ordinal (fracaso=0, mediocre=1, exito=2)
CLASS_TO_INT = {c: i for i, c in enumerate(CLASS_ORDER)}


def encode(y) -> np.ndarray:
    return y.map(CLASS_TO_INT).to_numpy()


@dataclass
class Result:
    name: str
    model: object
    y_true: np.ndarray
    y_pred: np.ndarray
    f1_macro: float
    accuracy: float
    report: str


def fit_one(name, model, X_tr, y_tr_enc):
    if name in NEEDS_SAMPLE_WEIGHT:
        sw = compute_sample_weight(class_weight="balanced", y=y_tr_enc)
        model.fit(X_tr, y_tr_enc, clf__sample_weight=sw)
    else:
        model.fit(X_tr, y_tr_enc)
    return model


def evaluate_all(models, X_tr, y_tr, X_te, y_te) -> dict[str, Result]:
    y_tr_enc = encode(y_tr)
    y_te_enc = encode(y_te)
    results: dict[str, Result] = {}
    for name, model in models.items():
        print(f"  entrenando {name} ...")
        fit_one(name, model, X_tr, y_tr_enc)
        pred = model.predict(X_te)
        results[name] = Result(
            name=name,
            model=model,
            y_true=y_te_enc,
            y_pred=pred,
            f1_macro=f1_score(y_te_enc, pred, average="macro"),
            accuracy=accuracy_score(y_te_enc, pred),
            report=classification_report(
                y_te_enc, pred, target_names=CLASS_ORDER, digits=3
            ),
        )
    return results


def savefig(name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIG_DIR / name, dpi=120)
    plt.close()
    print(f"  figura -> {name}")


def plot_comparison(results: dict[str, Result], fname: str) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "Modelo": list(results),
            "F1-macro": [r.f1_macro for r in results.values()],
            "Accuracy": [r.accuracy for r in results.values()],
        }
    )
    m = df.melt(id_vars="Modelo", var_name="Metrica", value_name="Valor")
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(data=m, x="Modelo", y="Valor", hue="Metrica", palette="Set2")
    for c in ax.containers:
        ax.bar_label(c, fmt="%.3f", fontsize=8)
    plt.ylim(0, 1)
    plt.title("Comparacion de clasificadores")
    savefig(fname)
    return df


def plot_confusions(results: dict[str, Result], fname: str) -> None:
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, results.values()):
        ConfusionMatrixDisplay.from_predictions(
            r.y_true, r.y_pred,
            display_labels=CLASS_ORDER, cmap="Blues", ax=ax, colorbar=False,
        )
        ax.set_title(f"{r.name}\nF1-macro={r.f1_macro:.3f}")
    savefig(fname)
