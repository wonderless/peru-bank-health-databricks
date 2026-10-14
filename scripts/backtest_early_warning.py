"""Evaluación descriptiva del monitor de salud bancaria con datos reales del BCRP (sin Databricks).

    python scripts/backtest_early_warning.py                 # usa data/snapshot/ (descarga oct-2026)
    python scripts/backtest_early_warning.py --raw data/raw  # o una descarga nueva (scripts/download_bcrp.py)

Qué mide (periodo 2023-01 .. último mes con 6 meses de futuro conocido):
  - deterioro: la morosidad del banco sube >= 1 pp en los 6 meses siguientes
  - anticipación: ¿hubo alerta entre 3 meses antes y el inicio del episodio?
  - durante:      ¿el monitor marcó al banco hasta 6 meses después del inicio?
  - comparación:  regla ingenua "la morosidad subió en los últimos 3 meses"
El monitor no se ajustó con estos datos: los umbrales están fijos en early_warning.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import warnings
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from bank_health import bcrp, early_warning as ew, quality  # noqa: E402

warnings.filterwarnings("ignore")


def load_wide(raw_dir: Path) -> pd.DataFrame:
    """Reproduce localmente gold_bank_monthly: banco × mes con los indicadores comparables."""
    rows = [r for f in sorted(glob.glob(str(raw_dir / "*.json")))
            for r in bcrp.parse_api_payload(json.loads(Path(f).read_text(encoding="utf-8")))]
    cat = pd.DataFrame([s.__dict__ for s in bcrp.load_catalog(ROOT / "config" / "series.csv")])
    df = pd.DataFrame(rows).merge(cat, on="code")
    df["period"] = pd.to_datetime(df["period"])
    df = df[df["value"].notna()]
    df = df[~((df["indicator"] == "morosidad") & (df["period"] < pd.Timestamp(quality.MOROSIDAD_BREAK)))]
    return df[df["group"] == "bank"].pivot_table(index=["bank", "period"], columns="indicator",
                                                 values="value").reset_index()


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "data" / "snapshot"))
    ap.add_argument("--test-start", default="2023-01-01")
    args = ap.parse_args()

    wide = load_wide(Path(args.raw))
    alerts = ew.run(wide)
    alerts["alerta"] = alerts["nivel_alerta"].isin(["alta", "media"])
    w = wide.sort_values(["bank", "period"]).copy()
    w["d3"] = w.groupby("bank")["morosidad"].diff(3)
    alerts = alerts.merge(w[["bank", "period", "d3"]], on=["bank", "period"], how="left")
    alerts["ingenua"] = alerts["d3"] > 0

    events = ew.deterioration_events(ew.eligible_banks(wide))
    test_end = (events.dropna(subset=["subida_futura"])["period"].max()).strftime("%Y-%m-%d")
    print(f"Periodo: {args.test_start} .. {test_end}")
    for col in ["alerta", "ingenua"]:
        r = ew.backtest(alerts, events, args.test_start, test_end, col)
        r6 = ew.backtest(alerts, events, args.test_start, test_end, col, months_after=6)
        print(f"\n[{col}] banco-mes={r['meses_evaluados']} tasa_base={r['tasa_base_deterioro']:.1%} "
              f"meses_con_alerta={r['meses_con_alerta']} precision={r['precision']:.1%} "
              f"anticipados={r['episodios_marcados']}/{r['episodios_total']} "
              f"marcados_durante={r6['episodios_marcados']}/{r6['episodios_total']}")
        if col == "alerta":
            print(r6["episodios"].to_string(index=False))

    last = alerts["period"].max()
    print(f"\nSemáforo vigente ({last:%Y-%m}):")
    cols = ["bank", "nivel_alerta"] + [f"estado_{d}" for d in ew.DIMENSIONS] + ["motivos", "motivos_vigilar"]
    cur = alerts[alerts["period"] == last].sort_values(["dimensiones_en_rojo", "dimensiones_en_amarillo",
                                                        "desvio_max"], ascending=False)
    print(cur[cols].to_string(index=False))


if __name__ == "__main__":
    main()
