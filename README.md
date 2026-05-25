# oteador

Supervisor de histogramas para trading adaptativo en Binance Spot.

`oteador` asigna un proceso scout por símbolo de exchange. El scout caracteriza
la distribución empírica de `(close − MA) / MA` sobre datos históricos
(descargados en bloque desde [data.binance.vision](https://data.binance.vision)),
identifica el régimen de oscilación del símbolo y opera reversión a la media
usando los percentiles vivos de esa distribución como umbrales de entrada y
salida.

## Estado

**Pre-alpha.** Esqueleto del paquete + CI. Próximo paso: descargador de la
Binance Vision, caracterizador espectral (FFT/Welch) y estimador de
percentiles online (t-digest).

## Componentes previstos

- **Ingesta**: histórico vía bulk download de Binance Vision; en vivo vía
  WebSocket (`kline`, `depth`, `trade`); REST para `exchangeInfo` y snapshots.
- **Caracterización**: FFT/wavelet sobre el residuo `close − MA` para estimar
  el período dominante de oscilación; histograma rodante del z-score con
  t-digest.
- **Estudios**: Optuna con walk-forward sobre `(periodo_MA, percentil_entrada,
  percentil_salida, holding_max, tipo_orden)`.
- **Supervisor**: máquina de estados `FLAT → BUY_PENDING → LONG → SELL_PENDING
  → FLAT` con política limit-then-market.
- **Riesgo**: tope por posición, tope diario, kill-switch, modo solo-cerrar.

## Desarrollo

Requiere Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux/macOS

pip install -e ".[dev]"

pytest
ruff check src tests
ruff format --check src tests
mypy src

oteador version
```

## Licencia

MIT.
