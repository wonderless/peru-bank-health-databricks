"""Entrenamiento y evaluación de modelos de pronóstico de morosidad."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .features import feature_columns


class DeltaRegressor(BaseEstimator, RegressorMixin):
    """Pronostica el *cambio* de la morosidad y se lo suma al valor actual.

    La morosidad es muy persistente: predecir el nivel directamente hace que el modelo
    compita contra el baseline ingenuo con desventaja. Modelando solo el cambio, el
    punto de partida es el baseline y el modelo aprende a corregirlo.
    """

    def __init__(self, estimator=None, base_col: str = "morosidad"):
        self.estimator = estimator
        self.base_col = base_col

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(X, np.asarray(y) - X[self.base_col].to_numpy())
        return self

    def predict(self, X):
        return X[self.base_col].to_numpy() + self.estimator_.predict(X)


def candidate_models(seed: int = 42) -> dict:
    return {
        "ridge": DeltaRegressor(make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
        "gbr": DeltaRegressor(GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.03,
                                                        subsample=0.8, min_samples_leaf=20,
                                                        loss="huber", random_state=seed)),
    }


def _metrics(y_true, y_pred) -> dict:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def evaluate(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Compara contra un baseline ingenuo (la morosidad no cambia) y devuelve métricas y modelos."""
    feats = feature_columns(train)
    tr = train.dropna(subset=feats + ["target"])
    te = test.dropna(subset=feats + ["target"])

    results = {"baseline_naive": {**_metrics(te["target"], te["morosidad"]), "model": None}}
    for name, model in candidate_models().items():
        model.fit(tr[feats], tr["target"])
        pred = model.predict(te[feats])
        results[name] = {**_metrics(te["target"], pred), "model": model}
    results["_n_train"], results["_n_test"], results["_features"] = len(tr), len(te), feats
    return results
