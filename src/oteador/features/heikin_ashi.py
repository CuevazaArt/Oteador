"""Velas Heikin-Ashi y una media móvil sobre HA_Close como referencia tendencial.

Definición estándar:

    HA_Close[i] = (Open[i] + High[i] + Low[i] + Close[i]) / 4
    HA_Open[0]  = (Open[0] + Close[0]) / 2
    HA_Open[i]  = (HA_Open[i-1] + HA_Close[i-1]) / 2     # recursivo
    HA_High[i]  = max(High[i], HA_Open[i], HA_Close[i])
    HA_Low[i]   = min(Low[i], HA_Open[i], HA_Close[i])

La recursión de HA_Open hace que la serie sea suavizada pero también
introduce un pequeño retardo respecto al precio crudo: cambios de tendencia
se reflejan con 1-2 velas de delay.

Color: una vela HA es *verde* si HA_Close > HA_Open, *roja* si <, *doji* si =.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
import polars as pl


def heikin_ashi(df: pl.DataFrame) -> pl.DataFrame:
    """Devuelve df con columnas ha_open, ha_high, ha_low, ha_close añadidas."""
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"faltan columnas requeridas: {sorted(missing)}")

    o = df["open"].to_numpy().astype(np.float64, copy=False)
    h = df["high"].to_numpy().astype(np.float64, copy=False)
    low = df["low"].to_numpy().astype(np.float64, copy=False)
    c = df["close"].to_numpy().astype(np.float64, copy=False)
    n = o.size

    ha_close = (o + h + low + c) * 0.25
    ha_open = np.empty(n, dtype=np.float64)
    ha_open[0] = (o[0] + c[0]) * 0.5
    for i in range(1, n):
        ha_open[i] = (ha_open[i - 1] + ha_close[i - 1]) * 0.5
    ha_high = np.maximum(np.maximum(h, ha_open), ha_close)
    ha_low = np.minimum(np.minimum(low, ha_open), ha_close)

    return df.with_columns(
        [
            pl.Series("ha_open", ha_open),
            pl.Series("ha_high", ha_high),
            pl.Series("ha_low", ha_low),
            pl.Series("ha_close", ha_close),
        ]
    )


def add_ha_ma(df: pl.DataFrame, window: int) -> pl.DataFrame:
    """Añade columna sma_ha_close: SMA(HA_Close, window). Asume df ya pasó por heikin_ashi."""
    if window < 1:
        raise ValueError(f"window debe ser >= 1, recibido {window}")
    if "ha_close" not in df.columns:
        raise ValueError("falta la columna ha_close; aplica heikin_ashi primero")
    return df.with_columns(df["ha_close"].rolling_mean(window_size=window).alias("sma_ha_close"))


def add_ha_open_mas(df: pl.DataFrame, fast_window: int, slow_window: int) -> pl.DataFrame:
    """Añade sma_ha_open_fast / sma_ha_open_slow para detectar cambios de tendencia.

    El cruce de estas dos MAs sobre HA_Open suele ser más sensible que el color
    de la vela HA para identificar volteos de tendencia.
    """
    if fast_window < 1 or slow_window < 1:
        raise ValueError(f"windows deben ser >= 1; recibido fast={fast_window}, slow={slow_window}")
    if fast_window >= slow_window:
        raise ValueError(f"fast_window ({fast_window}) debe ser < slow_window ({slow_window})")
    if "ha_open" not in df.columns:
        raise ValueError("falta la columna ha_open; aplica heikin_ashi primero")
    return df.with_columns(
        [
            df["ha_open"].rolling_mean(window_size=fast_window).alias("sma_ha_open_fast"),
            df["ha_open"].rolling_mean(window_size=slow_window).alias("sma_ha_open_slow"),
        ]
    )


def _longest_streak(mask: np.ndarray) -> int:
    """Longitud del bloque True consecutivo más largo."""
    if mask.size == 0:
        return 0
    best = 0
    cur = 0
    for v in mask:
        if v:
            cur += 1
            if cur > best:
                best = cur
        else:
            cur = 0
    return int(best)


@dataclass(frozen=True)
class HeikinAshiReport:
    bars: int
    pct_green: float
    pct_red: float
    pct_doji: float
    longest_green_streak: int
    longest_red_streak: int
    pct_ha_above_ma: float
    pct_ha_below_ma: float
    ma_window: int
    mean_abs_distance_to_ma_pct: float

    @classmethod
    def from_df(cls, df: pl.DataFrame, ma_window: int) -> Self:
        """Calcula estadísticas sobre las columnas ha_* y sma_ha_close del df."""
        for col in ("ha_open", "ha_close", "sma_ha_close"):
            if col not in df.columns:
                raise ValueError(f"falta la columna {col}; aplica heikin_ashi + add_ha_ma primero")
        ha_open = df["ha_open"].to_numpy()
        ha_close = df["ha_close"].to_numpy()
        sma = df["sma_ha_close"].to_numpy()
        n = int(ha_close.size)

        green = ha_close > ha_open
        red = ha_close < ha_open
        doji = ~(green | red)

        valid_ma = ~np.isnan(sma)
        above = (ha_close > sma) & valid_ma
        below = (ha_close < sma) & valid_ma
        denom_ma = int(valid_ma.sum())
        pct_above = float(above.sum() / denom_ma) if denom_ma else 0.0
        pct_below = float(below.sum() / denom_ma) if denom_ma else 0.0

        with np.errstate(invalid="ignore", divide="ignore"):
            distance_pct = np.where(valid_ma, np.abs(ha_close - sma) / sma, np.nan)
        mean_abs_dist = float(np.nanmean(distance_pct)) if denom_ma else 0.0

        return cls(
            bars=n,
            pct_green=float(green.sum() / n) if n else 0.0,
            pct_red=float(red.sum() / n) if n else 0.0,
            pct_doji=float(doji.sum() / n) if n else 0.0,
            longest_green_streak=_longest_streak(green),
            longest_red_streak=_longest_streak(red),
            pct_ha_above_ma=pct_above,
            pct_ha_below_ma=pct_below,
            ma_window=ma_window,
            mean_abs_distance_to_ma_pct=mean_abs_dist,
        )


@dataclass(frozen=True)
class HaOpenMaCrossover:
    fast_window: int
    slow_window: int
    pct_uptrend: float
    pct_downtrend: float
    pct_flat: float
    n_crossovers: int
    avg_bars_per_trend: float
    longest_up_streak: int
    longest_down_streak: int
    current_alignment: str
    samples: int

    @classmethod
    def from_df(cls, df: pl.DataFrame, fast_window: int, slow_window: int) -> Self:
        for col in ("sma_ha_open_fast", "sma_ha_open_slow"):
            if col not in df.columns:
                raise ValueError(f"falta la columna {col}; aplica add_ha_open_mas primero")
        fast = df["sma_ha_open_fast"].to_numpy()
        slow = df["sma_ha_open_slow"].to_numpy()
        valid = ~(np.isnan(fast) | np.isnan(slow))
        if int(valid.sum()) < 2:
            raise ValueError("muestras válidas insuficientes para crossover")
        sign = np.where(fast > slow, 1, np.where(fast < slow, -1, 0)).astype(np.int8)
        sign_valid = sign[valid]
        n = int(sign_valid.size)
        # Crossovers: signo cambia de +1 a -1 o viceversa (ignorando ceros).
        s = sign_valid
        crossings = int(np.sum((s[1:] * s[:-1]) < 0))
        avg_trend = float(n / crossings) if crossings > 0 else float(n)
        last = int(sign_valid[-1])
        current = "UP" if last > 0 else ("DOWN" if last < 0 else "FLAT")
        return cls(
            fast_window=fast_window,
            slow_window=slow_window,
            pct_uptrend=float((sign_valid == 1).sum() / n),
            pct_downtrend=float((sign_valid == -1).sum() / n),
            pct_flat=float((sign_valid == 0).sum() / n),
            n_crossovers=crossings,
            avg_bars_per_trend=avg_trend,
            longest_up_streak=_longest_streak(sign_valid == 1),
            longest_down_streak=_longest_streak(sign_valid == -1),
            current_alignment=current,
            samples=n,
        )


__all__ = [
    "HaOpenMaCrossover",
    "HeikinAshiReport",
    "add_ha_ma",
    "add_ha_open_mas",
    "heikin_ashi",
]
