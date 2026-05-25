# Oteador

[![CI](https://github.com/CuevazaArt/Oteador/actions/workflows/ci.yml/badge.svg)](https://github.com/CuevazaArt/Oteador/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Supervisor de histogramas para trading adaptativo en Binance Spot.

`oteador` asigna un proceso *scout* por símbolo de exchange. El scout caracteriza
la distribución empírica de `(close − MA) / MA` sobre datos históricos
(descargados en bloque desde [data.binance.vision](https://data.binance.vision)),
identifica el régimen de oscilación del símbolo y, en la fase de tiempo real,
operará reversión a la media usando los percentiles vivos de esa distribución
como umbrales de entrada y salida.

## Estado

**Pre-alpha.** Lo que ya funciona:

- ✅ Esqueleto del paquete (`src/oteador`) instalable con `pip install -e .`.
- ✅ Descargador bulk de Binance Vision (klines mensuales) con caché y
  verificación SHA-256.
- ✅ Media móvil simple + residuo `close − MA`.
- ✅ Caracterizador espectral (PSD vía Welch) con periodo dominante y top-k modos.
- ✅ Histograma offline del ratio: media, std, skew, curtosis y percentiles.
- ✅ CLI `oteador characterize SYMBOL`.
- ✅ CI en GitHub Actions: `ruff` + `ruff format` + `mypy --strict` + `pytest`
  sobre Python 3.11 y 3.12.

Próximas iteraciones:

- 🔜 Detector de régimen (Hurst / ADX) para distinguir oscilación vs tendencia.
- 🔜 Barrido Optuna sobre `(timeframe, ma_window)` con walk-forward.
- 🔜 Cliente WebSocket de Binance + estimador online del histograma (t-digest).
- 🔜 Máquina de estados de ejecución: `FLAT → BUY_PENDING → LONG → SELL_PENDING
  → FLAT`, política limit-then-market, capa de riesgo, paper-trading en testnet.

## Quickstart

Requiere Python 3.11+.

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
# source .venv/bin/activate      # Linux/macOS

pip install -e ".[dev]"

# Caracteriza 3 meses de BTCUSDT en 1m con MA(6) → descarga + reporta + parquet
oteador characterize BTCUSDT --interval 1m --months 3 --ma 6
```

Salida típica:

```
=== BTCUSDT @ 1m (3 meses) ===
Velas analizadas: 128,160
Ventana MA:       6

Espectro del residuo (close - MA):
  periodo dominante: 10240.0s (≈ 170.7 velas)
  fracción de potencia del modo dominante: 0.90%
  ...

Histograma del ratio (close - MA) / MA:
  n=128,155  media=-0.000000  std=0.000935
  skew=+0.445  kurtosis_excess=+16.497
  percentiles:
    p  1.0 = -0.002723
    p  5.0 = -0.001363
    ...
    p 95.0 = +0.001333
    p 99.0 = +0.002716
```

El parquet de salida (`data/characterization/<SYMBOL>-<interval>-ma<window>.parquet`)
incluye OHLCV + `sma_<window>`, `residual` y `ratio` para análisis posterior.

## Desarrollo

```bash
pytest
ruff check src tests
ruff format --check src tests
mypy src
```

Estructura:

```
src/oteador/
├── cli.py                          # Typer CLI: version, characterize
├── data/vision.py                  # Descargador Binance Vision
├── features/
│   ├── moving_average.py           # sma, residual, price_ma_ratio
│   └── histogram.py                # RatioHistogram (percentiles offline)
└── studies/
    ├── spectral.py                 # PSD Welch, periodo dominante, top-k modos
    └── characterization.py         # Orquestación + reporte
```

## Licencia

MIT.
