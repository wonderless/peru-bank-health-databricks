"""Monitoreo explicable de la salud bancaria con datos públicos del BCRP.

Cuatro dimensiones (lógica tipo CAMELS):
  - Calidad de cartera: morosidad (sube = peor)
  - Eficiencia: gastos operativos / margen (sube = peor)
  - Solvencia: ratio de capital global, que el BCRP publica como "palanca global" (baja = peor)
  - Rentabilidad: utilidad de los últimos 12 meses (baja = peor)

Cada mes, cada banco recibe un semáforo por dimensión (verde / amarillo / rojo) y una lista de
motivos en lenguaje simple. Las señales son relativas al sistema bancario y se comparan contra la
historia previa del propio banco, sin mirar el mes actual ni el futuro.

Esto es un monitor, NO un pronóstico: el experimento de anticipación (scripts/experimento_alertas.py)
mostró que ni un Isolation Forest ni las reglas anticipan los deterioros mejor que la regla ingenua
de forma confiable (ver README).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SYSTEM = "Sistema bancario"
INDICATORS = ["morosidad", "eficiencia", "palanca_global", "utilidad_acumulada"]

# Señales orientadas para que "más alto = peor". Cada una es relativa al sistema, así los
# choques comunes (COVID, cambios de norma como el de capital en ene-2023) no disparan alertas.
SIGNALS = {
    "sig_morosidad_3m": "La morosidad subió en 3 meses más que la del sistema",
    "sig_morosidad_6m": "La morosidad subió en 6 meses más que la del sistema",
    "sig_eficiencia_12m": "La eficiencia empeoró en 12 meses más que la del sistema",
    "sig_capital_12m": "El ratio de capital cayó en 12 meses más que el del sistema",
    "sig_utilidad_12m": "La utilidad de 12 meses cayó más que la del sistema",
}
DIMENSIONS = {
    "calidad_cartera": ["sig_morosidad_3m", "sig_morosidad_6m"],
    "eficiencia": ["sig_eficiencia_12m"],
    "solvencia": ["sig_capital_12m"],
    "rentabilidad": ["sig_utilidad_12m"],
}
# Umbrales fijados de antemano (no ajustados mirando resultados).
Z_RED = 2.0           # desvío fuerte: más de 2 desviaciones respecto de la historia del banco
Z_YELLOW = 1.0        # desvío a vigilar
MIN_HISTORY = 24      # meses mínimos de historia para estandarizar
STREAK_RED = 4        # meses seguidos subiendo la morosidad
STREAK_YELLOW = 3


def eligible_banks(wide: pd.DataFrame) -> pd.DataFrame:
    """Excluye bancos sin cartera minorista (morosidad mediana 0: Citibank, ICBC). Mantiene el sistema."""
    med = wide.groupby("bank")["morosidad"].median()
    return wide[wide["bank"].isin(med[med > 0].index)].copy()


def build_signals(wide: pd.DataFrame) -> pd.DataFrame:
    """wide: columnas [bank, period] + INDICATORS (incluida la fila 'Sistema bancario')."""
    df = wide.sort_values(["bank", "period"]).copy()
    df["period"] = pd.to_datetime(df["period"])
    g = df.groupby("bank", group_keys=False)

    # la utilidad del BCRP es acumulada en el año: se reinicia cada enero
    prev = g["utilidad_acumulada"].shift(1)
    df["utilidad_mensual"] = np.where(df["period"].dt.month == 1, df["utilidad_acumulada"],
                                      df["utilidad_acumulada"] - prev)
    df["utilidad_12m"] = g["utilidad_mensual"].transform(lambda s: s.rolling(12, min_periods=12).sum())
    u12_prev = g["utilidad_12m"].shift(12)
    df["utilidad_12m_var"] = ((df["utilidad_12m"] - u12_prev) / u12_prev.abs().clip(lower=1.0)).clip(-3, 3)

    raw = pd.DataFrame(index=df.index)
    raw["sig_morosidad_3m"] = df["morosidad"] - g["morosidad"].shift(3)
    raw["sig_morosidad_6m"] = df["morosidad"] - g["morosidad"].shift(6)
    raw["sig_eficiencia_12m"] = df["eficiencia"] - g["eficiencia"].shift(12)
    raw["sig_capital_12m"] = -(df["palanca_global"] - g["palanca_global"].shift(12))
    raw["sig_utilidad_12m"] = -df["utilidad_12m_var"]
    raw[["bank", "period"]] = df[["bank", "period"]]

    # relativo al sistema
    sys_vals = raw[raw["bank"] == SYSTEM].set_index("period")[list(SIGNALS)]
    rel = raw.join(sys_vals, on="period", rsuffix="_sys")
    for c in SIGNALS:
        df[c] = rel[c] - rel[f"{c}_sys"]

    # racha de meses seguidos con la morosidad subiendo
    up = (g["morosidad"].diff() > 0).astype(int)
    df["morosidad_racha"] = up.groupby(df["bank"]).transform(
        lambda s: s.groupby((s == 0).cumsum()).cumsum())
    df["paso_a_perdida"] = (df["utilidad_12m"] < 0) & (g["utilidad_12m"].shift(12) >= 0)
    return df[df["bank"] != SYSTEM].reset_index(drop=True)


def standardize_expanding(df: pd.DataFrame) -> pd.DataFrame:
    """z-score de cada señal contra la historia previa del propio banco (sin usar el mes actual)."""
    out = df.copy()
    for c in SIGNALS:
        grp = out.groupby("bank")[c]
        mean = grp.transform(lambda s: s.shift(1).expanding(MIN_HISTORY).mean())
        std = grp.transform(lambda s: s.shift(1).expanding(MIN_HISTORY).std())
        out[f"z_{c}"] = ((out[c] - mean) / std.replace(0, np.nan)).clip(-6, 6)
    return out


def z_columns() -> list[str]:
    return [f"z_{c}" for c in SIGNALS]


def _dimension_status(r: pd.Series, dim: str) -> tuple[str, list[str], list[str]]:
    """Devuelve (estado, motivos rojos, motivos amarillos) de una dimensión para una fila."""
    red, yellow = [], []
    for c in DIMENSIONS[dim]:
        z = r[f"z_{c}"]
        if pd.notna(z) and z > Z_RED:
            red.append(SIGNALS[c])
        elif pd.notna(z) and z > Z_YELLOW:
            yellow.append(SIGNALS[c])
    if dim == "calidad_cartera" and pd.notna(r["morosidad_racha"]):
        n = int(r["morosidad_racha"])
        if n >= STREAK_RED:
            red.append(f"La morosidad sube {n} meses seguidos")
        elif n >= STREAK_YELLOW:
            yellow.append(f"La morosidad sube {n} meses seguidos")
    if dim == "rentabilidad" and bool(r["paso_a_perdida"]):
        red.append("Pasó a pérdidas en los últimos 12 meses")
    estado = "rojo" if red else ("amarillo" if yellow else "verde")
    return estado, red, yellow


def semaforo(df: pd.DataFrame) -> pd.DataFrame:
    """Agrega estado_<dimensión>, dimensiones_en_rojo/amarillo, motivos, desvio_max y nivel_alerta."""
    out = df.copy()
    estados = {d: [] for d in DIMENSIONS}
    motivos, vigilar = [], []
    for _, r in out.iterrows():
        red_all, yellow_all = [], []
        for d in DIMENSIONS:
            estado, red, yellow = _dimension_status(r, d)
            estados[d].append(estado)
            red_all += red
            yellow_all += yellow
        motivos.append("; ".join(red_all))
        vigilar.append("; ".join(yellow_all))
    for d in DIMENSIONS:
        out[f"estado_{d}"] = estados[d]
    est = out[[f"estado_{d}" for d in DIMENSIONS]]
    out["dimensiones_en_rojo"] = (est == "rojo").sum(axis=1)
    out["dimensiones_en_amarillo"] = (est == "amarillo").sum(axis=1)
    out["motivos"] = motivos
    out["motivos_vigilar"] = vigilar
    out["desvio_max"] = out[z_columns()].max(axis=1)
    out["nivel_alerta"] = alert_level(out)
    return out


def alert_level(df: pd.DataFrame) -> pd.Series:
    """alta: 2+ dimensiones en rojo · media: 1 en rojo · vigilar: alguna en amarillo · normal."""
    rojo, amarillo = df["dimensiones_en_rojo"], df["dimensiones_en_amarillo"]
    return pd.Series(np.select([rojo >= 2, rojo == 1, amarillo >= 1], ["alta", "media", "vigilar"], "normal"),
                     index=df.index)


def run(wide: pd.DataFrame) -> pd.DataFrame:
    """Pipeline completo: bancos elegibles -> señales -> z-scores -> semáforo."""
    sig = standardize_expanding(build_signals(eligible_banks(wide)))
    sig = sig[sig[z_columns()].notna().any(axis=1)]
    return semaforo(sig).reset_index(drop=True)


# ---------------------------------------------------------------- evaluación descriptiva

def deterioration_events(wide: pd.DataFrame, rise_pp: float = 1.0, months: int = 6) -> pd.DataFrame:
    """Marca los meses t en que la morosidad sube >= rise_pp entre t y t+months."""
    df = wide[wide["bank"] != SYSTEM].sort_values(["bank", "period"]).copy()
    fut = df.groupby("bank")["morosidad"].transform(lambda s: s.shift(-months))
    df["subida_futura"] = fut - df["morosidad"]
    df["deterioro"] = df["subida_futura"] >= rise_pp
    return df[["bank", "period", "morosidad", "subida_futura", "deterioro"]]


def backtest(alerts: pd.DataFrame, events: pd.DataFrame, start: str, end: str,
             alert_col: str = "alerta", months_after: int = 0) -> dict:
    """Precisión, cobertura y episodios marcados en [start, end].

    Un episodio (meses seguidos con deterioro de un banco) cuenta como marcado si hubo alerta entre
    3 meses antes de su inicio y `months_after` meses después. Con months_after=0 mide anticipación;
    con months_after=6 mide si el monitor lo marcó mientras ocurría.
    """
    d = alerts.merge(events, on=["bank", "period"], how="inner")
    d = d[(d["period"] >= start) & (d["period"] <= end) & d["subida_futura"].notna()]
    hit = d[alert_col] & d["deterioro"]
    precision = hit.sum() / max(d[alert_col].sum(), 1)
    recall = hit.sum() / max(d["deterioro"].sum(), 1)

    episodes = []
    for bank, x in d.sort_values("period").groupby("bank"):
        x = x.reset_index(drop=True)
        grp = (x["deterioro"] != x["deterioro"].shift()).cumsum()
        for _, e in x[x["deterioro"]].groupby(grp[x["deterioro"]]):
            s = e["period"].min()
            win = alerts[(alerts["bank"] == bank) & (alerts["period"] >= s - pd.DateOffset(months=3))
                         & (alerts["period"] <= s + pd.DateOffset(months=months_after)) & alerts[alert_col]]
            first = win["period"].min() if len(win) else None
            lead = (s.year - first.year) * 12 + (s.month - first.month) if first is not None else None
            episodes.append({"bank": bank, "inicio": s.date(), "subida_max_pp": round(e["subida_futura"].max(), 2),
                             "marcado": len(win) > 0, "meses_antes_del_inicio": lead})
    ep = pd.DataFrame(episodes)
    return {
        "meses_evaluados": int(len(d)),
        "tasa_base_deterioro": float(d["deterioro"].mean()),
        "meses_con_alerta": int(d[alert_col].sum()),
        "precision": float(precision),
        "cobertura_meses": float(recall),
        "episodios": ep,
        "episodios_marcados": int(ep["marcado"].sum()) if len(ep) else 0,
        "episodios_total": int(len(ep)),
    }
