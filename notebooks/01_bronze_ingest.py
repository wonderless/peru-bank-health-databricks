# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Bronze — ingesta de la API del BCRP
# MAGIC 1. Intenta descargar todas las series del catálogo (`config/series.csv`) desde la API pública del BCRP
# MAGIC    y guarda el JSON crudo en el volumen `raw` (una carpeta por fecha de ingesta).
# MAGIC 2. Si el workspace no tiene salida a internet (Free Edition restringe dominios), usa los JSON
# MAGIC    que hayas subido al volumen con `scripts/download_bcrp.py`.
# MAGIC 3. Aplana todo y escribe la tabla Delta `bronze_bcrp_series`, sin transformar los valores.

# COMMAND ----------

import json, os, sys
from datetime import datetime, timezone

sys.path.append(os.path.abspath("../src"))
from bank_health import bcrp

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "bank_health")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
raw_root = f"/Volumes/{catalog}/{schema}/raw/bcrp"

catalog_rows = bcrp.load_catalog(os.path.abspath("../config/series.csv"))
codes = [s.code for s in catalog_rows]
print(f"{len(codes)} series en el catálogo")

# COMMAND ----------

# MAGIC %md ### 1) Descarga (con plan B si no hay internet)

# COMMAND ----------

run_ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
run_dir = f"{raw_root}/{run_ts}"
downloaded, errors = 0, []
for i in range(0, len(codes), 10):  # la API acepta varios códigos por llamada
    batch = codes[i:i + 10]
    try:
        payload = bcrp.fetch(batch, start="2010-1")
        bcrp.save_payload(payload, run_dir, f"batch_{i // 10:02d}")
        downloaded += len(batch)
    except Exception as e:  # sin salida a internet o API caída
        errors.append(f"{batch[0]}..: {type(e).__name__}")
        break

if downloaded:
    print(f"API OK: {downloaded} series guardadas en {run_dir}")
else:
    print("No se pudo llegar a la API:", errors[:1])
    print("Usando los JSON subidos manualmente al volumen (scripts/download_bcrp.py).")

# COMMAND ----------

# MAGIC %md ### 2) Aplanar los JSON y escribir Bronze

# COMMAND ----------

def latest_files(root: str) -> list[str]:
    """Toma la carpeta de ingesta más reciente que tenga archivos JSON."""
    folders = sorted([f.path for f in dbutils.fs.ls(root) if f.isDir()], reverse=True)
    for folder in folders + [root]:
        files = [f.path for f in dbutils.fs.ls(folder) if f.path.endswith(".json")]
        if files:
            return files
    raise FileNotFoundError(f"No hay JSON en {root}. Sube los archivos de data/raw al volumen.")

files = latest_files(raw_root)
rows = []
for path in files:
    local = path.replace("dbfs:", "")
    with open(local, encoding="utf-8") as f:
        payload = json.load(f)
    for r in bcrp.parse_api_payload(payload):
        r["source_file"] = os.path.basename(local)
        rows.append(r)

bronze = (spark.createDataFrame(rows)
          .withColumnRenamed("period", "period_date")
          .selectExpr("code", "series_name", "period_date", "value", "source_file",
                      f"timestamp'{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}' AS ingested_at"))

(bronze.write.mode("overwrite").option("overwriteSchema", "true")
       .saveAsTable(f"{catalog}.{schema}.bronze_bcrp_series"))
print(f"bronze_bcrp_series: {bronze.count():,} filas desde {len(files)} archivos")
