"""Monitoreo de drift con Population Stability Index (PSI)."""
from __future__ import annotations

import numpy as np


def psi(expected, actual, bins: int = 10) -> float:
    """PSI entre la distribución de entrenamiento (expected) y la reciente (actual).
    Regla usual: < 0.1 estable · 0.1–0.25 cambio moderado · > 0.25 cambio fuerte."""
    expected = np.asarray(expected, dtype=float)
    actual = np.asarray(actual, dtype=float)
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    if len(expected) == 0 or len(actual) == 0:
        return float("nan")
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def psi_label(value: float) -> str:
    if np.isnan(value):
        return "sin datos"
    return "estable" if value < 0.1 else ("moderado" if value < 0.25 else "fuerte")
