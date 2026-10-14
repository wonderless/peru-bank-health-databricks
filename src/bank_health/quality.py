"""Reglas de calidad de datos para las series del BCRP."""
from __future__ import annotations

from datetime import date

import pandas as pd

# Hallazgo del EDA: hasta dic-2017 la serie "Cartera atrasada neta / Colocaciones netas"
# viene neta de provisiones (valores negativos, porque las provisiones superan a la cartera
# atrasada). Desde ene-2018 la serie se publica como ratio de morosidad (valores ~2-9%).
# Mezclar ambos tramos rompería cualquier modelo, así que el tramo anterior se marca y se excluye.
MOROSIDAD_BREAK = date(2018, 1, 1)


def detect_level_break(s: pd.Series, min_jump: float = 2.0) -> pd.Timestamp | None:
    """Devuelve la primera fecha donde la serie salta de forma anómala (|Δ| >= min_jump)
    y cambia de signo. Sirve para validar que el quiebre metodológico sigue donde se espera."""
    s = s.dropna().sort_index()
    diff = s.diff()
    sign_change = (s.shift(1) < 0) & (s >= 0)
    hits = s.index[(diff.abs() >= min_jump) & sign_change]
    return hits[0] if len(hits) else None


def check_panel(df: pd.DataFrame) -> dict:
    """Chequeos básicos sobre el panel silver (bank, indicator, period, value)."""
    out = {
        "rows": int(len(df)),
        "duplicated_keys": int(df.duplicated(["code", "period"]).sum()),
        "null_values": int(df["value"].isna().sum()),
        "min_period": str(df["period"].min()),
        "max_period": str(df["period"].max()),
    }
    mor = df[(df["indicator"] == "morosidad") & (pd.to_datetime(df["period"]) >= pd.Timestamp(MOROSIDAD_BREAK))]
    out["morosidad_out_of_range"] = int(((mor["value"] < 0) | (mor["value"] > 40)).sum())
    return out
