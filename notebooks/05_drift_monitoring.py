# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Monitoreo de drift (PSI)
# MAGIC Compara la distribución de cada señal del monitor (relativa al sistema) en la historia vs. los últimos
# MAGIC 12 meses. Un PSI alto avisa que el régimen cambió (p. ej. una nueva norma de capital) y que conviene
# MAGIC revisar los umbrales del semáforo. Guarda el resultado en `monitoring_drift` para seguirlo en el dashboard.

# COMMAND ----------

import os, sys
from datetime import datetime, timezone
import pandas as pd

sys.path.append(os.path.abspath("../src"))
from bank_health import drift, early_warning as ew

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "bank_health")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
T = lambda name: f"{catalog}.{schema}.{name}"

wide = spark.table(T("gold_bank_monthly")).select("bank", "period", *ew.INDICATORS).toPandas()
wide["period"] = pd.to_datetime(wide["period"])
signals = ew.build_signals(ew.eligible_banks(wide))

cutoff = signals["period"].max() - pd.DateOffset(months=11)
ref, recent = signals[signals["period"] < cutoff], signals[signals["period"] >= cutoff]

# COMMAND ----------

rows = []
for col, desc in ew.SIGNALS.items():
    value = drift.psi(ref[col], recent[col])
    rows.append({"feature": col, "descripcion": desc, "psi": value, "estado": drift.psi_label(value),
                 "ventana_reciente_desde": cutoff.date(), "calculado_en": datetime.now(timezone.utc)})

report = pd.DataFrame(rows).sort_values("psi", ascending=False)
spark.createDataFrame(report).write.mode("append").option("mergeSchema", "true").saveAsTable(T("monitoring_drift"))
display(report)
