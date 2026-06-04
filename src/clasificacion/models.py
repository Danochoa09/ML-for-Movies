"""Definicion de los tres clasificadores a comparar.

1. Regresion Logistica  -> modelo lineal base/simple.
2. Random Forest         -> ensamble de arboles (bagging).
3. XGBoost               -> boosting de arboles (modelo avanzado).

Todos comparten el mismo preprocesamiento dentro de un Pipeline de sklearn.
El desbalance de clases se maneja con class_weight (logistica / RF) y con
sample_weight balanceado (XGBoost, ver evaluate.py).
"""
from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import RANDOM_STATE
from .data import make_preprocessor

# Los modelos que usan sample_weight en vez de class_weight (XGBoost)
NEEDS_SAMPLE_WEIGHT = {"XGBoost"}


def _pipe(preprocessor, clf) -> Pipeline:
    return Pipeline([("pre", preprocessor), ("clf", clf)])


def build_models(num_cols, cat_cols, genre_cols) -> dict[str, Pipeline]:
    """Devuelve {nombre: pipeline} para los tres clasificadores."""

    def pre():
        return make_preprocessor(num_cols, cat_cols, genre_cols)

    models = {
        "Regresion Logistica": _pipe(
            pre(),
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                n_jobs=-1,
            ),
        ),
        "Random Forest": _pipe(
            pre(),
            RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                class_weight="balanced",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
        "XGBoost": _pipe(
            pre(),
            XGBClassifier(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.9,
                colsample_bytree=0.9,
                objective="multi:softprob",
                num_class=3,
                eval_metric="mlogloss",
                tree_method="hist",
                n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
        ),
    }
    return models
