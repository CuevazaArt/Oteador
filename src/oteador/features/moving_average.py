from __future__ import annotations

import polars as pl


def sma(close: pl.Series, window: int) -> pl.Series:
    """Media móvil simple. Los primeros `window - 1` valores son null."""
    if window < 1:
        raise ValueError(f"window debe ser >= 1, recibido {window}")
    return close.rolling_mean(window_size=window)


def residual(close: pl.Series, ma: pl.Series) -> pl.Series:
    """Diferencia close - ma. Entrada al análisis espectral."""
    return close - ma


def price_ma_ratio(close: pl.Series, ma: pl.Series) -> pl.Series:
    """Ratio (close - ma) / ma. Entrada al histograma de percentiles."""
    return (close - ma) / ma
