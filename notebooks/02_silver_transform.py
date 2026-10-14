# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Silver — limpieza, catálogo y calidad de datos
# MAGIC - Une cada serie con el catálogo (indicador, banco, unidad).
# MAGIC - Marca el **quiebre metodológico de la morosidad** (antes de 2018 la serie viene neta de provisiones).
# MAGIC - Aplica reglas de calidad y falla si algo está mal (para que el job no publique datos rotos).

# COMMAND ----------

import os, sys
import pyspark.sql.functions as F

sys.path.append(os.path.abspath("../src"))
from bank_health import bcrp, quality

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "bank_health")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
T = lambda name: f"{catalog}.{schema}.{name}"

cat = spark.createDataFrame([s.__dict__ for s in bcrp.load_catalog(os.path.abspath("../config/series.csv"))])
bronze = spark.table(T("bronze_bcrp_series"))

# COMMAND ----------

silver = (bronze.join(cat, "code", "left")
          .withColumnRenamed("period_date", "period")
          .dropDuplicates(["code", "period"])
          .withColumn("is_comparable",
                      ~((F.col("indicator") == "morosidad") & (F.col("period") < F.lit(str(quality.MOROSIDAD_BREAK)))))
          .select("code", "group", "indicator", "bank", "unit", "period", "value", "is_comparable", "ingested_at"))

# COMMAND ----------

# MAGIC %md ### Reglas de calidad

# COMMAND ----------

dq = {
    "series_sin_catalogo": silver.filter("indicator IS NULL").select("code").distinct().count(),
    "claves_duplicadas": silver.groupBy("code", "period").count().filter("count > 1").count(),
    "morosidad_fuera_de_rango": silver.filter("indicator = 'morosidad' AND is_comparable AND (value < 0 OR value > 40)").count(),
}
print(dq)
assert all(v == 0 for v in dq.values()), f"Fallaron reglas de calidad: {dq}"

# COMMAND ----------

bank = silver.filter(F.col("group") == "bank")
macro = (silver.filter(F.col("group") == "macro")
         .groupBy("period").pivot("indicator").agg(F.first("value"))
         .orderBy("period"))

bank.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T("silver_bank_indicators"))
macro.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T("silver_macro"))
print("silver_bank_indicators:", bank.count(), "| silver_macro:", macro.count())
