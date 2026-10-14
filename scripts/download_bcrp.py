"""Descarga todas las series del catálogo desde la API del BCRP a data/raw/.

Úsalo si tu workspace de Databricks no tiene salida a internet: corre esto en tu PC
y sube la carpeta data/raw/ al volumen /Volumes/workspace/bank_health/raw/bcrp/.

    pip install requests
    python scripts/download_bcrp.py            # desde 2010
    python scripts/download_bcrp.py --start 2018-1
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bank_health import bcrp  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-1")
    ap.add_argument("--end", default="2030-12")
    ap.add_argument("--out", default=str(ROOT / "data" / "raw"))
    ap.add_argument("--batch", type=int, default=10)
    args = ap.parse_args()

    codes = [s.code for s in bcrp.load_catalog(ROOT / "config" / "series.csv")]
    print(f"{len(codes)} series -> {args.out}")
    for i in range(0, len(codes), args.batch):
        batch = codes[i:i + args.batch]
        payload = bcrp.fetch(batch, args.start, args.end)
        path = bcrp.save_payload(payload, args.out, f"batch_{i // args.batch:02d}")
        print(f"  {path.name}: {len(batch)} series, {len(payload.get('periods', []))} meses")
        time.sleep(0.5)  # ser amable con la API
    print("Listo.")


if __name__ == "__main__":
    main()
