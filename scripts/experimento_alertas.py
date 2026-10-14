"""Experimento: ¿se pueden ANTICIPAR los deterioros de morosidad con datos públicos del BCRP?

    python scripts/experimento_alertas.py

Protocolo (fijado antes de mirar el test):
  - entrenamiento: hasta 2021-06 (Isolation Forest y regresión logística)
  - validación:    2021-07 .. 2022-12   -> aquí se elige el candidato
  - prueba:        2023-01 .. último mes con 6 meses de futuro conocido (se mira UNA vez)
  - objetivo:      la morosidad del banco sube >= 1 pp en los 6 meses siguientes
  - métrica:       average precision (AP) del puntaje continuo; AUC como secundaria
  - referencia:    regla ingenua "cuánto subió la morosidad en los últimos 3 meses" (d3)

Resultado (snapshot oct-2026): en validación gana la racha de meses con morosidad subiendo
(AP 0.42). En prueba: racha AP 0.16 / AUC 0.64 vs ingenua AP 0.12 / AUC 0.59, con tasa base 6.1%
y solo 9 episodios. El Isolation Forest queda por debajo del azar (AUC 0.44).
Conclusión: no hay evidencia suficiente para vender "anticipación"; el proyecto se presenta como
monitoreo explicable (src/bank_health/early_warning.py).
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from bank_health import early_warning as ew  # noqa: E402
from backtest_early_warning import load_wide  # noqa: E402

warnings.filterwarnings("ignore")
TRAIN_END, VAL_START, VAL_END, TEST_START = "2021-06-01", "2021-07-01", "2022-12-01", "2023-01-01"


def isolation_forest_score(sig: pd.DataFrame, train_end: str) -> pd.Series:
    """Puntaje 0-100 = percentil de anomalía frente a los meses de entrenamiento (solo desvíos adversos)."""
    X = sig[ew.z_columns()].fillna(0).clip(lower=0)
    train = sig["period"] <= pd.Timestamp(train_end)
    model = IsolationForest(n_estimators=300, contamination="auto", random_state=42).fit(X[train])
    ref = np.sort(-model.score_samples(X[train]))
    return pd.Series(100 * np.searchsorted(ref, -model.score_samples(X), side="right") / len(ref), index=sig.index)


def candidates(wide: pd.DataFrame) -> pd.DataFrame:
    sig = ew.run(wide)
    sig["isolation_forest"] = isolation_forest_score(sig, TRAIN_END)
    sig["reglas_rojas"] = sig["dimensiones_en_rojo"]
    sig["racha_morosidad"] = sig["morosidad_racha"]

    w = wide.sort_values(["bank", "period"]).copy()
    g = w.groupby("bank")["morosidad"]
    w["ingenua_d3"], w["d6"] = g.diff(3), g.diff(6)
    d = sig.merge(w[["bank", "period", "ingenua_d3", "d6"]], on=["bank", "period"], how="left")

    ev = ew.deterioration_events(wide)
    d = d.merge(ev[["bank", "period", "deterioro", "subida_futura"]], on=["bank", "period"])
    d = d[d["subida_futura"].notna()].copy()
    d["y"] = d["deterioro"].astype(int)

    # regresión logística supervisada: solo etiquetas ya conocidas al cierre del entrenamiento
    feats = ew.z_columns() + ["ingenua_d3", "d6"]
    tr = d[d["period"] <= pd.Timestamp(TRAIN_END) - pd.DateOffset(months=6)]
    lr = LogisticRegression(C=0.3, max_iter=2000).fit(tr[feats].fillna(0), tr["y"])
    d["logistica"] = lr.predict_proba(d[feats].fillna(0))[:, 1]
    return d


COLS = ["ingenua_d3", "racha_morosidad", "isolation_forest", "reglas_rojas", "logistica"]


def report(d: pd.DataFrame, start: str, end: str | None, label: str) -> pd.DataFrame:
    x = d[(d["period"] >= start) & ((d["period"] <= end) if end else True)]
    print(f"\n[{label}] {x['period'].min():%Y-%m} .. {x['period'].max():%Y-%m} · banco-mes={len(x)} "
          f"· positivos={x['y'].sum()} · tasa base={x['y'].mean():.1%}")
    rows = [{"candidato": c, "AP": average_precision_score(x["y"], x[c].fillna(-9)),
             "AUC": roc_auc_score(x["y"], x[c].fillna(-9))} for c in COLS]
    t = pd.DataFrame(rows).round(3)
    print(t.to_string(index=False))
    return t


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    d = candidates(load_wide(ROOT / "data" / "snapshot"))
    val = report(d, VAL_START, VAL_END, "validación")
    chosen = val.sort_values("AP", ascending=False).iloc[0]["candidato"]
    print(f"\nElegido en validación (mayor AP): {chosen}")
    report(d, TEST_START, None, "prueba — se mira una sola vez")


if __name__ == "__main__":
    main()
