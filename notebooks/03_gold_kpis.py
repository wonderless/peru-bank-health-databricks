# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Gold — KPIs por banco y por sistema
# MAGIC Tabla ancha banco × mes con los indicadores comparables, variaciones y ranking de morosidad.

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql import Window

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "bank_health")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
T = lambda name: f"{catalog}.{schema}.{name}"

bank = spark.table(T("silver_bank_indicators")).filter("is_comparable AND value IS NOT NULL")

# COMMAND ----------

wide = (bank.groupBy("bank", "period").pivot("indicator").agg(F.first("value")))

w = Window.partitionBy("bank").orderBy("period")
gold = (wide
        .withColumn("morosidad_var_12m", F.col("morosidad") - F.lag("morosidad", 12).over(w))
        .withColumn("morosidad_ma6", F.avg("morosidad").over(w.rowsBetween(-5, 0)))
        .withColumn("utilidad_mensual", F.col("utilidad_acumulada")
                    - F.when(F.month("period") == 1, F.lit(0)).otherwise(F.lag("utilidad_acumulada", 1).over(w))))

rank_w = Window.partitionBy("period").orderBy(F.col("morosidad").desc())
gold = gold.withColumn("ranking_morosidad",
                       F.when((F.col("bank") != "Sistema bancario") & F.col("morosidad").isNotNull(),
                              F.dense_rank().over(rank_w)))

gold.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T("gold_bank_monthly"))

system = gold.filter("bank = 'Sistema bancario'").drop("ranking_morosidad")
system.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T("gold_system_monthly"))

last = gold.filter("morosidad IS NOT NULL").agg(F.max("period")).first()[0]
print("Último mes con morosidad publicada:", last)
display(gold.filter(F.col("period") == last).orderBy("ranking_morosidad"))
