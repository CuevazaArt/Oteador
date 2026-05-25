"""Estrategia trend-following por cruce de dos MAs sobre HA_Open.

Política:

- LONG cuando fast_MA(HA_Open) > slow_MA(HA_Open).
- Salir cuando fast_MA < slow_MA, sujeto a un holding mínimo para evitar
  whipsaws (cruces inmediatos por ruido).
- Spot only: no se opera SHORT.
- Fees por lado en `fee_pct`.

Es la lógica simétrica a la reversión por percentiles: aquí el sistema *sigue*
la tendencia que el cruce de MMs delata, en vez de apostar contra ella.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from oteador.backtest import BacktestMetrics, compute_metrics
from oteador.features.heikin_ashi import add_ha_open_mas, heikin_ashi

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class HaCrossParams:
    fast_window: int
    slow_window: int
    min_holding_bars: int = 1
    fee_pct: float = 0.001

    def __post_init__(self) -> None:
        if self.fast_window < 1:
            raise ValueError(f"fast_window debe ser >= 1, recibido {self.fast_window}")
        if self.slow_window <= self.fast_window:
            raise ValueError(
                f"slow_window ({self.slow_window}) debe ser > fast_window ({self.fast_window})"
            )
        if self.min_holding_bars < 1:
            raise ValueError(f"min_holding_bars debe ser >= 1, recibido {self.min_holding_bars}")
        if not (0.0 <= self.fee_pct < 0.05):
            raise ValueError(f"fee_pct fuera de rango razonable: {self.fee_pct}")


def compute_signals(
    df: pl.DataFrame, params: HaCrossParams
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Devuelve (close, fast_ma_on_ha_open, slow_ma_on_ha_open)."""
    df_ha = add_ha_open_mas(heikin_ashi(df), params.fast_window, params.slow_window)
    return (
        df["close"].to_numpy().astype(np.float64, copy=False),
        df_ha["sma_ha_open_fast"].to_numpy().astype(np.float64, copy=False),
        df_ha["sma_ha_open_slow"].to_numpy().astype(np.float64, copy=False),
    )


def simulate(
    close: FloatArray,
    fast: FloatArray,
    slow: FloatArray,
    min_holding_bars: int,
    fee_pct: float,
    force_close_at_end: bool = True,
) -> tuple[FloatArray, list[dict[str, Any]]]:
    """Loop trend-following: long mientras fast > slow."""
    n = int(close.size)
    if not (fast.size == slow.size == n):
        raise ValueError("close, fast, slow deben tener la misma longitud")
    if min_holding_bars < 1:
        raise ValueError(f"min_holding_bars debe ser >= 1, recibido {min_holding_bars}")
    equity = np.ones(n, dtype=np.float64)
    eq = 1.0
    pos_open = False
    entry_price = 0.0
    entry_bar = 0
    trades: list[dict[str, Any]] = []
    fee_round_trip = (1.0 - fee_pct) ** 2

    for i in range(n):
        c = float(close[i])
        if pos_open:
            equity[i] = eq * c / entry_price * fee_round_trip
        else:
            equity[i] = eq

        f = fast[i]
        s = slow[i]
        if not (np.isfinite(f) and np.isfinite(s)):
            continue

        is_up = f > s
        if not pos_open:
            if is_up:
                pos_open = True
                entry_price = c
                entry_bar = i
            continue

        holding = i - entry_bar
        if not is_up and holding >= min_holding_bars:
            net = c / entry_price * fee_round_trip
            eq *= net
            equity[i] = eq
            trades.append(
                {
                    "entry_bar": entry_bar,
                    "exit_bar": i,
                    "holding_bars": holding,
                    "entry_price": entry_price,
                    "exit_price": c,
                    "net_return": net - 1.0,
                    "forced": False,
                }
            )
            pos_open = False

    if pos_open and force_close_at_end:
        c = float(close[-1])
        net = c / entry_price * fee_round_trip
        eq *= net
        equity[-1] = eq
        trades.append(
            {
                "entry_bar": entry_bar,
                "exit_bar": n - 1,
                "holding_bars": n - 1 - entry_bar,
                "entry_price": entry_price,
                "exit_price": c,
                "net_return": net - 1.0,
                "forced": True,
            }
        )

    return equity, trades


def backtest(
    df: pl.DataFrame, params: HaCrossParams, sampling_period_seconds: float
) -> BacktestMetrics:
    close, fast, slow = compute_signals(df, params)
    equity, trades = simulate(
        close,
        fast,
        slow,
        min_holding_bars=params.min_holding_bars,
        fee_pct=params.fee_pct,
    )
    return compute_metrics(equity, trades, sampling_period_seconds)


__all__ = ["HaCrossParams", "backtest", "compute_signals", "simulate"]
