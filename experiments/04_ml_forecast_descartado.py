# Databricks notebook source
# MAGIC %md
# MAGIC # (Descartado) Pronóstico de morosidad por banco (MLflow)
# MAGIC > **Experimento archivado.** Ningún modelo le ganó de forma estable al baseline ingenuo
# MAGIC > (ver README, sección "Lo que no funcionó"). Se conserva como referencia; no forma parte del job.
# MAGIC - Objetivo: morosidad de cada banco **3 meses adelante**.
# MAGIC - Variables: rezagos y promedios de la propia morosidad + macro del BCRP (tasa de referencia,
# MAGIC   inflación, PBI, tipo de cambio) con 3 meses de rezago.
# MAGIC - Validación temporal: los últimos 12 meses con objetivo conocido son el set de prueba.
# MAGIC - Se compara contra un **baseline ingenuo** (la morosidad no cambia). Todo queda en MLflow y el
# MAGIC   mejor modelo se registra en Unity Catalog.

# COMMAND ----------

import os, sys
import mlflow
import pandas as pd
from mlflow.models import infer_signature

sys.path.append(os.path.abspath("../src"))
from bank_health import features, model

dbutils.widgets.text("catalog", "workspace")
dbutils.widgets.text("schema", "bank_health")
dbutils.widgets.text("horizon", "3")
catalog, schema = dbutils.widgets.get("catalog"), dbutils.widgets.get("schema")
horizon = int(dbutils.widgets.get("horizon"))
T = lambda name: f"{catalog}.{schema}.{name}"

mlflow.set_registry_uri("databricks-uc")
user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/bank_health_morosidad")

# COMMAND ----------

bank = (spark.table(T("gold_bank_monthly"))
        .filter("bank != 'Sistema bancario' AND morosidad IS NOT NULL")
        .select("bank", "period", "morosidad").toPandas())
macro = spark.table(T("silver_macro")).toPandas()
bank["period"], macro["period"] = pd.to_datetime(bank["period"]), pd.to_datetime(macro["period"])

panel = features.build_features(bank, macro, horizon=horizon)
train, test, cutoff = features.time_split(panel, test_months=12)
print(f"Corte de prueba: {cutoff:%Y-%m} · train={len(train)} · test={len(test)}")

# COMMAND ----------

results = model.evaluate(train, test)
feats = results["_features"]
best_name, best_uri = None, None

for name in ["baseline_naive", "ridge", "gbr"]:
    r = results[name]
    with mlflow.start_run(run_name=f"{name}_h{horizon}"):
        mlflow.log_params({"model": name, "horizon": horizon, "test_months": 12,
                           "n_train": results["_n_train"], "n_test": results["_n_test"],
                           "cutoff": f"{cutoff:%Y-%m}"})
        mlflow.log_metrics({"mae": r["mae"], "rmse": r["rmse"]})
        if r["model"] is not None:
            X = train.dropna(subset=feats + ["target"])[feats]
            info = mlflow.sklearn.log_model(r["model"], name="model", input_example=X.head(3),
                                            signature=infer_signature(X, r["model"].predict(X)),
                                            code_paths=[os.path.abspath("../src/bank_health")])
            if best_name is None or r["mae"] < results[best_name]["mae"]:
                best_name, best_uri = name, info.model_uri

print({k: round(v["mae"], 3) for k, v in results.items() if not k.startswith("_")})
mejora = 1 - results[best_name]["mae"] / results["baseline_naive"]["mae"]
print(f"Mejor modelo: {best_name} · MAE {results[best_name]['mae']:.3f} vs baseline {results['baseline_naive']['mae']:.3f} "
      f"({mejora:+.1%} de mejora)")

# COMMAND ----------

# MAGIC %md ### Registrar el mejor modelo y guardar el pronóstico vigente

# COMMAND ----------

model_name = f"{catalog}.{schema}.morosidad_forecaster"
mv = mlflow.register_model(best_uri, model_name)
print(f"Registrado {model_name} v{mv.version}")

final = model.candidate_models()[best_name]
full = panel.dropna(subset=feats + ["target"])
final.fit(full[feats], full["target"])

latest = panel[panel["period"] == panel["period"].max()].dropna(subset=feats).copy()
assert len(latest), "El último mes no tiene todas las variables: revisa silver_macro"
latest["morosidad_pronosticada"] = final.predict(latest[feats])
out = latest[["bank", "period", "target_date", "morosidad", "morosidad_pronosticada"]].rename(
    columns={"period": "fecha_base", "target_date": "fecha_objetivo", "morosidad": "morosidad_actual"})
out["modelo"], out["version"] = best_name, int(mv.version)

spark.createDataFrame(out).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(T("gold_morosidad_forecast"))
display(out.sort_values("morosidad_pronosticada", ascending=False))
