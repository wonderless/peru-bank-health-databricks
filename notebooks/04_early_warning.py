# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Monitor de salud bancaria — semáforo CAMELS explicable
# MAGIC Para cada banco y mes calcula un semáforo (verde / amarillo / rojo) en cuatro dimensiones:
# MAGIC calidad de cartera (morosidad), eficiencia, solvencia (ratio de capital) y rentabilidad (utilidad 12m).
# MAGIC - Las señales son **relativas al sistema bancario** (así un choque común como el COVID no enciende todo)
# MAGIC   y se comparan contra la **historia previa del propio banco** (z-score sin mirar el futuro).
# MAGIC - Cada alerta trae sus **motivos** en lenguaje simple.
# MAGIC - Es un **monitor**, no un pronóstico: el experimento de anticipación no le ganó de forma confiable a la
# MAGIC   regla ingenua (ver README). La evaluación descriptiva desde 2023 queda registrada en MLflow.
# MAGIC
# MAGIC Escribe `gold_alertas_tempranas` (histórico completo banco × mes).

# COMMAND ----------

import os, sys
import mlflow
import pandas as pd

sys.path.append(os.path.abspath("../src"))
from bank_health import early_warning as ew

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "bank_health")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
T = lambda name: f"{catalog}.{schema}.{name}"

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/bank_health_monitor")

# COMMAND ----------

wide = spark.table(T("gold_bank_monthly")).select("bank", "period", *ew.INDICATORS).toPandas()
wide["period"] = pd.to_datetime(wide["period"])
assert ew.SYSTEM in set(wide["bank"]), "Falta la fila 'Sistema bancario' en gold_bank_monthly"

alerts = ew.run(wide)
last = alerts["period"].max()
print(f"{alerts['bank'].nunique()} bancos · {alerts['period'].min():%Y-%m} .. {last:%Y-%m}")

# COMMAND ----------

# MAGIC %md ### Evaluación descriptiva desde 2023 (¿el monitor marcó los episodios de deterioro?)
# MAGIC Deterioro = la morosidad del banco sube ≥ 1 pp en los 6 meses siguientes. Se compara con la regla
# MAGIC ingenua "la morosidad subió en 3 meses". Los umbrales del monitor están fijos en el código.

# COMMAND ----------

alerts["alerta"] = alerts["nivel_alerta"].isin(["alta", "media"])
w = wide.sort_values(["bank", "period"]).copy()
w["d3"] = w.groupby("bank")["morosidad"].diff(3)
eval_df = alerts.merge(w[["bank", "period", "d3"]], on=["bank", "period"], how="left")
eval_df["ingenua"] = eval_df["d3"] > 0

events = ew.deterioration_events(ew.eligible_banks(wide))
test_start = "2023-01-01"
test_end = events.dropna(subset=["subida_futura"])["period"].max().strftime("%Y-%m-%d")

with mlflow.start_run(run_name=f"monitor_{last:%Y-%m}"):
    mlflow.log_params({"z_rojo": ew.Z_RED, "z_amarillo": ew.Z_YELLOW, "racha_rojo": ew.STREAK_RED,
                       "racha_amarillo": ew.STREAK_YELLOW, "historia_minima": ew.MIN_HISTORY,
                       "evaluacion_desde": test_start, "evaluacion_hasta": test_end, "ultimo_mes": f"{last:%Y-%m}"})
    for col in ["alerta", "ingenua"]:
        r = ew.backtest(eval_df, events, test_start, test_end, col)
        r6 = ew.backtest(eval_df, events, test_start, test_end, col, months_after=6)
        mlflow.log_metrics({f"{col}_precision": r["precision"], f"{col}_meses_con_alerta": r["meses_con_alerta"],
                            f"{col}_episodios_anticipados": r["episodios_marcados"],
                            f"{col}_episodios_marcados_durante": r6["episodios_marcados"],
                            "episodios_total": r["episodios_total"], "tasa_base_deterioro": r["tasa_base_deterioro"]})
        print(f"[{col}] alertas={r['meses_con_alerta']} precisión={r['precision']:.1%} "
              f"anticipados={r['episodios_marcados']}/{r['episodios_total']} "
              f"marcados durante={r6['episodios_marcados']}/{r6['episodios_total']}")
        if col == "alerta":
            mlflow.log_text(r6["episodios"].to_csv(index=False), "episodios_2023.csv")
    mlflow.log_text(alerts[alerts["period"] == last].to_csv(index=False), "semaforo_vigente.csv")

# COMMAND ----------

# MAGIC %md ### Guardar el histórico del semáforo

# COMMAND ----------

cols = (["bank", "period", "nivel_alerta", "dimensiones_en_rojo", "dimensiones_en_amarillo"]
        + [f"estado_{d}" for d in ew.DIMENSIONS] + ["motivos", "motivos_vigilar", "desvio_max", "morosidad",
        "eficiencia", "palanca_global", "utilidad_12m", "morosidad_racha"] + ew.z_columns())
out = alerts[cols].copy()
out["period"] = out["period"].dt.date

(spark.createDataFrame(out).write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(T("gold_alertas_tempranas")))
spark.sql(f"COMMENT ON TABLE {T('gold_alertas_tempranas')} IS "
          "'Semáforo CAMELS explicable por banco y mes (monitor, no pronóstico). Fuente: BCRP.'")

display(spark.table(T("gold_alertas_tempranas")).filter(f"period = '{last:%Y-%m-%d}'")
        .orderBy("dimensiones_en_rojo", "dimensiones_en_amarillo", "desvio_max", ascending=False))
