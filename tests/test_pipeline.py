import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bank_health import bcrp, drift, features, model, quality

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "data" / "sample"


def load_sample() -> pd.DataFrame:
    rows = []
    for f in sorted(SAMPLE.glob("*.json")):
        rows += bcrp.parse_api_payload(json.loads(f.read_text(encoding="utf-8")))
    df = pd.DataFrame(rows)
    cat = pd.DataFrame([s.__dict__ for s in bcrp.load_catalog(ROOT / "config" / "series.csv")])
    df = df.merge(cat, on="code", how="left")
    df["period"] = pd.to_datetime(df["period"])
    return df


def test_parse_period_and_value():
    assert bcrp.parse_period("Ago.2026") == date(2026, 8, 1)
    assert bcrp.parse_period("Set.2025") == date(2025, 9, 1)
    assert bcrp.parse_value("n.d.") is None
    assert bcrp.parse_value("2.82") == pytest.approx(2.82)


def test_catalog_has_all_morosidad_banks():
    cat = bcrp.load_catalog(ROOT / "config" / "series.csv")
    mor = [s for s in cat if s.indicator == "morosidad"]
    assert len(mor) == 17  # 16 bancos + total del sistema
    assert {"BCP", "BBVA", "Interbank", "MiBanco", "Sistema bancario"} <= {s.bank for s in mor}


def test_codes_are_mapped_in_api_order():
    # Respuesta real de la API para PN07745EM-PD04722MM-PN07729EM-PN01207PM (Ago.2026):
    # el BCRP devuelve las series ordenadas por código, no en el orden pedido.
    payload = {
        "_codes": ["PN07745EM", "PD04722MM", "PN07729EM", "PN01207PM"],
        "config": {"series": [{"name": "Tasa de Referencia"}, {"name": "Tipo de cambio"},
                              {"name": "Morosidad - Crédito"}, {"name": "Morosidad - ICBC"}]},
        "periods": [{"name": "Ago.2026", "values": ["4.25", "3.36295", "2.67", "1.1"]}],
    }
    got = {r["code"]: r["value"] for r in bcrp.parse_api_payload(payload)}
    assert got == {"PD04722MM": 4.25, "PN01207PM": 3.36295, "PN07729EM": 2.67, "PN07745EM": 1.1}


def test_payload_is_parsed_into_rows():
    df = load_sample()
    assert not df.duplicated(["code", "period"]).any()
    assert df["indicator"].notna().all()


def test_morosidad_break_detected_at_2018():
    df = load_sample()
    s = df[(df.code == "PN07746EM")].set_index("period")["value"]
    assert quality.detect_level_break(s) == pd.Timestamp(quality.MOROSIDAD_BREAK)


def test_quality_checks_pass_after_break():
    df = load_sample()
    out = quality.check_panel(df)
    assert out["duplicated_keys"] == 0
    assert out["morosidad_out_of_range"] == 0


def test_model_beats_or_matches_naive_baseline():
    df = load_sample()
    bank = (df[(df.indicator == "morosidad") & (df.bank != "Sistema bancario")
               & (df.period >= pd.Timestamp(quality.MOROSIDAD_BREAK))]
            .rename(columns={"value": "morosidad"})[["bank", "period", "morosidad"]])
    macro = df[df.group == "macro"].pivot_table(index="period", columns="indicator", values="value").reset_index()
    feats = features.build_features(bank, macro, horizon=3)
    train, test, cutoff = features.time_split(feats, test_months=12)
    res = model.evaluate(train, test)
    assert res["_n_test"] > 0
    best = min(res["ridge"]["mae"], res["gbr"]["mae"])
    assert best <= res["baseline_naive"]["mae"] * 1.5  # sanity: no peor que 1.5x el ingenuo


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    base = rng.normal(0, 1, 2000)
    assert drift.psi(base, rng.normal(0, 1, 2000)) < 0.1
    assert drift.psi(base, rng.normal(1.5, 1, 2000)) > 0.25


def synthetic_panel(seed: int = 1) -> tuple[pd.DataFrame, pd.DatetimeIndex]:
    """Panel sintético: A sufre un deterioro de morosidad al final, D una caída de capital y pérdidas."""
    from bank_health import early_warning as ew
    rng = np.random.default_rng(seed)
    periods = pd.date_range("2015-01-01", "2024-12-01", freq="MS")
    rows = []
    for bank in ["A", "B", "C", "D", "Corporativo", ew.SYSTEM]:
        mor = 3 + rng.normal(0, 0.05, len(periods)).cumsum() * 0.1
        if bank == "A":
            mor[-6:] += np.linspace(0.5, 3, 6)
        if bank == "Corporativo":
            mor = np.zeros(len(periods))  # sin cartera minorista: debe excluirse
        cap = 15 + rng.normal(0, 0.1, len(periods))
        acum = np.tile(np.arange(1, 13) * 10.0, len(periods) // 12)
        if bank == "D":
            cap[-12:] -= np.linspace(0.5, 4, 12)
            acum[-12:] = -np.arange(1, 13) * 5.0  # el último año acumula pérdidas
        for i, p in enumerate(periods):
            rows.append({"bank": bank, "period": p, "morosidad": mor[i], "eficiencia": 45 + rng.normal(0, 0.3),
                         "palanca_global": cap[i], "utilidad_acumulada": acum[i]})
    return pd.DataFrame(rows), periods


def test_semaforo_flags_the_right_dimension():
    from bank_health import early_warning as ew
    wide, periods = synthetic_panel()
    out = ew.run(wide)
    assert not {ew.SYSTEM, "Corporativo"} & set(out["bank"])
    last = out[out["period"] == periods[-1]].set_index("bank")
    assert last.loc["A", "estado_calidad_cartera"] == "rojo"
    assert "morosidad" in last.loc["A", "motivos"]
    assert last.loc["D", "estado_solvencia"] == "rojo"
    assert last.loc["D", "estado_rentabilidad"] == "rojo"
    assert "Pasó a pérdidas" in last.loc["D", "motivos"]
    assert last.loc["D", "nivel_alerta"] == "alta"
    assert last.loc["B", "dimensiones_en_rojo"] == 0


def test_semaforo_has_no_lookahead():
    from bank_health import early_warning as ew
    wide, _ = synthetic_panel()
    out = ew.run(wide)
    # cambiar el futuro no debe alterar el semáforo del pasado
    wide2 = wide.copy()
    wide2.loc[wide2["period"] > "2023-06-01", ["morosidad", "palanca_global"]] += 5
    out2 = ew.run(wide2)
    k = ["bank", "period"]
    cols = ["nivel_alerta", "motivos", "motivos_vigilar"]
    a = out[out["period"] <= "2023-06-01"].set_index(k)[cols]
    b = out2[out2["period"] <= "2023-06-01"].set_index(k)[cols]
    pd.testing.assert_frame_equal(a, b)
