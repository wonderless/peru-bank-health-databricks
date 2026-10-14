"""Cliente mínimo para la API pública de series estadísticas del BCRP (BCRPData).

Documentación de la API: https://estadisticas.bcrp.gob.pe/estadisticas/series/ayuda/api
Formato de URL: /estadisticas/series/api/{codigos}/json/{inicio}/{fin}
"""
from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterable

API_BASE = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"

# El BCRP usa abreviaturas de mes en español ("Ene.2026", "Set.2025" o "Sep.2025").
MESES = {
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6,
    "jul": 7, "ago": 8, "sep": 9, "set": 9, "oct": 10, "nov": 11, "dic": 12,
}


@dataclass(frozen=True)
class Serie:
    code: str
    group: str        # "bank" o "macro"
    indicator: str    # p. ej. "morosidad"
    bank_code: str    # nombre del banco tal como lo publica el BCRP ("" en macro)
    bank: str         # nombre comercial ("BCP", "BBVA"...)
    unit: str


def load_catalog(path: str | Path) -> list[Serie]:
    """Lee config/series.csv y devuelve el catálogo de series."""
    with open(path, encoding="utf-8") as f:
        return [
            Serie(r["code"], r["group"], r["indicator"], r["bank_code"], r["bank"], r["unit"])
            for r in csv.DictReader(f)
        ]


def api_url(codes: Iterable[str], start: str = "2010-1", end: str = "2030-12") -> str:
    """URL de la API para una o varias series (máx. ~10 códigos por llamada)."""
    return f"{API_BASE}/{'-'.join(codes)}/json/{start}/{end}"


def parse_period(name: str) -> date:
    """Convierte 'Ago.2026' (mensual) en date(2026, 8, 1)."""
    mes, anio = name.strip().split(".")
    return date(int(anio), MESES[mes[:3].lower()], 1)


def parse_value(raw: str | None) -> float | None:
    """El BCRP usa 'n.d.' para datos no disponibles."""
    if raw is None:
        return None
    raw = str(raw).strip()
    if raw.lower() in {"n.d.", "nd", ""}:
        return None
    try:
        v = float(raw)
    except ValueError:
        return None
    return None if math.isnan(v) else v


def parse_api_payload(payload: dict) -> list[dict]:
    """Aplana la respuesta JSON de la API en filas (code, name, period, value).

    La respuesta tiene la forma:
      {"config": {"series": [{"name": ...}, ...]},
       "periods": [{"name": "Ene.2026", "values": ["2.82", ...]}, ...]}
    OJO: la API devuelve las series ordenadas alfabéticamente por código, no en el orden
    en que se pidieron (p. ej. pedir PN07745EM-PD04722MM devuelve primero PD04722MM).
    Por eso los códigos se ordenan antes de asignarlos a 'values'.
    """
    series = payload.get("config", {}).get("series", [])
    codes = sorted(payload.get("_codes") or [s.get("code") for s in series])
    rows = []
    for p in payload.get("periods", []):
        period = parse_period(p["name"])
        for i, raw in enumerate(p.get("values", [])):
            rows.append({
                "code": codes[i] if i < len(codes) else None,
                "series_name": series[i]["name"] if i < len(series) else None,
                "period": period,
                "value": parse_value(raw),
            })
    return rows


def fetch(codes: list[str], start: str = "2010-1", end: str = "2030-12", timeout: int = 30) -> dict:
    """Descarga una o varias series. Devuelve el JSON con la lista de códigos añadida."""
    import requests  # import tardío: no se necesita para probar el parser

    r = requests.get(api_url(codes, start, end), timeout=timeout)
    r.raise_for_status()
    payload = r.json()
    payload["_codes"] = list(codes)
    return payload


def save_payload(payload: dict, folder: str | Path, name: str) -> Path:
    path = Path(folder) / f"{name}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path
