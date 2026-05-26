# Experiments

Registro de corridas de optimización Optuna sobre estrategias de Oteador.

Cada archivo `YYYY-MM-DD-<symbol>-<scope>.md` documenta una corrida concreta:
comando exacto, ventana de datos, configuración ganadora, métricas por fold y
lectura cualitativa. El objetivo es que cualquiera (incluyendo tu yo de dentro
de tres meses) pueda **reproducir** la corrida y **comparar** resultados nuevos
contra los anteriores.

Convención de nombres:

- `YYYY-MM-DD` — fecha (UTC) de la corrida.
- `<symbol>` — símbolo principal (e.g., `btcusdt`).
- `<scope>` — etiqueta corta del experimento (e.g., `12m-r1` = 12 meses, run 1;
  `cross-validation`, `tuning`, etc.).

Los datasets, caches y storages de Optuna (`data/`, `optuna_studies/`) están
ignorados en git; el markdown es la única traza permanente.
