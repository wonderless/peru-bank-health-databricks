"""Construcción de variables para pronosticar la morosidad por banco."""
from __future__ import annotations

import pandas as pd

MACRO_COLS = ["tasa_referencia", "inflacion_12m", "pbi_var_12m", "tipo_cambio"]


def build_features(bank: pd.DataFrame, macro: pd.DataFrame, horizon: int = 3) -> pd.DataFrame:
    """Arma un panel banco-mes con rezagos y el objetivo a `horizon` meses.

    bank:  columnas [bank, period, morosidad]   (period = primer día del mes)
    macro: columnas [period, tasa_referencia, inflacion_12m, pbi_var_12m, tipo_cambio]
    """
    df = bank.sort_values(["bank", "period"]).copy()
    g = df.groupby("bank")["morosidad"]
    for lag in (1, 3, 6, 12):
        df[f"mor_lag{lag}"] = g.shift(lag)
    df["mor_diff3"] = df["morosidad"] - df["mor_lag3"]
    df["mor_ma6"] = g.transform(lambda s: s.rolling(6, min_periods=6).mean())
    df["target"] = g.shift(-horizon)
    df["target_date"] = df["period"] + pd.DateOffset(months=horizon)

    m = macro.sort_values("period").copy()
    for c in [c for c in MACRO_COLS if c in m.columns]:
        m[f"{c}_lag3"] = m[c].shift(3)  # la macro se publica con rezago
    df = df.merge(m, on="period", how="left")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    base = ["morosidad", "mor_lag1", "mor_lag3", "mor_lag6", "mor_lag12", "mor_diff3", "mor_ma6"]
    # Solo la macro rezagada: el dato del mes actual (p. ej. el PBI) aún no está publicado
    # al momento de pronosticar, y usarlo sería filtrar información del futuro.
    macro = [f"{m}_lag3" for m in MACRO_COLS if f"{m}_lag3" in df.columns]
    return base + macro


def time_split(df: pd.DataFrame, test_months: int = 12):
    """Corte temporal: los últimos `test_months` meses con objetivo conocido van a test."""
    known = df.dropna(subset=["target"])
    cutoff = known["period"].max() - pd.DateOffset(months=test_months - 1)
    return known[known["period"] < cutoff], known[known["period"] >= cutoff], cutoff
