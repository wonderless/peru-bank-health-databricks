# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Setup
# MAGIC Crea el esquema y el volumen del proyecto en Unity Catalog.
# MAGIC En Databricks Free Edition el catálogo por defecto se llama `workspace`.

# COMMAND ----------

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "bank_health")
catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema} COMMENT 'Monitor de salud bancaria del Perú (datos públicos BCRP)'")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.raw COMMENT 'JSON crudos de la API del BCRP'")
dbutils.fs.mkdirs(f"/Volumes/{catalog}/{schema}/raw/bcrp")
print(f"Listo: {catalog}.{schema} y /Volumes/{catalog}/{schema}/raw")
