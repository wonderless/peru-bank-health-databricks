-- Consultas para el dashboard (Databricks SQL / AI/BI Dashboard).
-- Catálogo y esquema por defecto: workspace.bank_health

-- 1. Morosidad del sistema bancario en el tiempo
SELECT period, morosidad, morosidad_ma6
FROM workspace.bank_health.gold_system_monthly
WHERE morosidad IS NOT NULL
ORDER BY period;

-- 2. Ranking de morosidad por banco (último mes disponible)
SELECT bank, morosidad, morosidad_var_12m, participacion_colocaciones, ranking_morosidad
FROM workspace.bank_health.gold_bank_monthly
WHERE period = (SELECT max(period) FROM workspace.bank_health.gold_bank_monthly WHERE morosidad IS NOT NULL)
  AND bank <> 'Sistema bancario'
ORDER BY ranking_morosidad;

-- 3. Morosidad vs. tasa de referencia del BCRP
SELECT s.period, s.morosidad, m.tasa_referencia, m.inflacion_12m
FROM workspace.bank_health.gold_system_monthly s
JOIN workspace.bank_health.silver_macro m USING (period)
WHERE s.morosidad IS NOT NULL
ORDER BY s.period;

-- 4. Semáforo vigente: qué bancos se desvían y por qué
SELECT bank, nivel_alerta,
       estado_calidad_cartera, estado_eficiencia, estado_solvencia, estado_rentabilidad,
       motivos, motivos_vigilar, round(morosidad, 2) AS morosidad, round(palanca_global, 1) AS ratio_capital
FROM workspace.bank_health.gold_alertas_tempranas
WHERE period = (SELECT max(period) FROM workspace.bank_health.gold_alertas_tempranas)
ORDER BY dimensiones_en_rojo DESC, dimensiones_en_amarillo DESC, desvio_max DESC;

-- 5. Histórico de alertas (mapa de calor banco × mes)
SELECT bank, period,
       CASE nivel_alerta WHEN 'alta' THEN 3 WHEN 'media' THEN 2 WHEN 'vigilar' THEN 1 ELSE 0 END AS nivel,
       nivel_alerta, motivos
FROM workspace.bank_health.gold_alertas_tempranas
WHERE period >= add_months((SELECT max(period) FROM workspace.bank_health.gold_alertas_tempranas), -35)
ORDER BY bank, period;

-- 6. Motivos más frecuentes en los últimos 12 meses
SELECT trim(motivo) AS motivo, count(*) AS banco_meses
FROM workspace.bank_health.gold_alertas_tempranas
LATERAL VIEW explode(split(motivos, ';')) t AS motivo
WHERE motivos <> ''
  AND period >= add_months((SELECT max(period) FROM workspace.bank_health.gold_alertas_tempranas), -11)
GROUP BY trim(motivo)
ORDER BY banco_meses DESC;

-- 7. Drift de las señales del monitor (última corrida)
SELECT feature, descripcion, round(psi, 3) AS psi, estado
FROM workspace.bank_health.monitoring_drift
WHERE calculado_en = (SELECT max(calculado_en) FROM workspace.bank_health.monitoring_drift)
ORDER BY psi DESC;
