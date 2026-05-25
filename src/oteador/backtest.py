"""Backtester de la estrategia de reversión por percentiles.

La estrategia es la siguiente, evaluada al cierre de cada vela:

- Calcular MA(ma_window) y ratio = (close - MA) / MA.
- Estimar percentiles rodantes del ratio en una ventana lookback.
- Entrar LONG cuando ratio[t] <= percentil_entrada[t] (cola izquierda).
- Salir cuando:
    a) ratio[t] >= percentil_salida[t] Y profit_neto >= min_profit_pct, o
    b) holding_bars >= holding_max (salida forzada).
- Fees por lado en `fee_pct` (e.g., 0.001 = 0.1%).

El simulador hace un loop bar-por-bar en numpy. Es lo bastante rápido
para 100k barras x decenas de trials sin necesidad de numba.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class BacktestParams:
    ma_window: int
    entry_percentile: float
    exit_percentile: float
    rolling_window_bars: int
    holding_max_bars: int
    min_profit_pct: float
    fee_pct: float = 0.001

    def __post_init__(self) -> None:
        if self.ma_window < 1:
            raise ValueError(f"ma_window debe ser >= 1, recibido {self.ma_window}")
        if not (0.0 < self.entry_percentile < self.exit_percentile < 100.0):
            raise ValueError(
                f"se requiere 0 < entry_percentile ({self.entry_percentile}) "
                f"< exit_percentile ({self.exit_percentile}) < 100"
            )
        if self.rolling_window_bars < 32:
            raise ValueError(
                f"rolling_window_bars debe ser >= 32, recibido {self.rolling_window_bars}"
            )
        if self.holding_max_bars < 1:
            raise ValueError(f"holding_max_bars debe ser >= 1, recibido {self.holding_max_bars}")
        if self.min_profit_pct < 0.0:
            raise ValueError(f"min_profit_pct debe ser >= 0, recibido {self.min_profit_pct}")
        if not (0.0 <= self.fee_pct < 0.05):
            raise ValueError(f"fee_pct fuera de rango razonable: {self.fee_pct}")


@dataclass(frozen=True)
class BacktestMetrics:
    n_trades: int
    final_equity: float
    total_return_pct: float
    sharpe_annualized: float
    max_drawdown_pct: float
    win_rate: float
    avg_trade_pct: float
    avg_holding_bars: float
    forced_exits: int
    trades: list[dict[str, Any]] = field(default_factory=list)

    def to_summary(self) -> dict[str, Any]:
        return {
            "n_trades": self.n_trades,
            "final_equity": self.final_equity,
            "total_return_pct": self.total_return_pct,
            "sharpe_annualized": self.sharpe_annualized,
            "max_drawdown_pct": self.max_drawdown_pct,
            "win_rate": self.win_rate,
            "avg_trade_pct": self.avg_trade_pct,
            "avg_holding_bars": self.avg_holding_bars,
            "forced_exits": self.forced_exits,
        }


def compute_signals(
    df: pl.DataFrame,
    params: BacktestParams,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Devuelve (ratio, p_low, p_high) alineados con df. Todo causal."""
    close = df["close"]
    ma = close.rolling_mean(window_size=params.ma_window)
    ratio = (close - ma) / ma
    min_samples = max(64, params.rolling_window_bars // 4)
    p_low = ratio.rolling_quantile(
        params.entry_percentile / 100.0,
        window_size=params.rolling_window_bars,
        min_samples=min_samples,
    )
    p_high = ratio.rolling_quantile(
        params.exit_percentile / 100.0,
        window_size=params.rolling_window_bars,
        min_samples=min_samples,
    )
    return (
        ratio.to_numpy().astype(np.float64, copy=False),
        p_low.to_numpy().astype(np.float64, copy=False),
        p_high.to_numpy().astype(np.float64, copy=False),
    )


def simulate(
    close: FloatArray,
    ratio: FloatArray,
    p_low: FloatArray,
    p_high: FloatArray,
    holding_max_bars: int,
    min_profit_pct: float,
    fee_pct: float,
    force_close_at_end: bool = True,
) -> tuple[FloatArray, list[dict[str, Any]]]:
    """Loop principal del backtest. Devuelve (equity_curve, trades)."""
    n = int(close.size)
    if not (ratio.size == p_low.size == p_high.size == n):
        raise ValueError("close, ratio, p_low, p_high deben tener la misma longitud")
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
            mark = c / entry_price * fee_round_trip
            equity[i] = eq * mark
        else:
            equity[i] = eq

        r = ratio[i]
        pl_i = p_low[i]
        ph_i = p_high[i]
        if not (np.isfinite(r) and np.isfinite(pl_i) and np.isfinite(ph_i)):
            continue

        if not pos_open:
            if r <= pl_i:
                pos_open = True
                entry_price = c
                entry_bar = i
            continue

        holding = i - entry_bar
        cur_net_return = c / entry_price * fee_round_trip - 1.0
        exit_by_percentile = bool(r >= ph_i and cur_net_return >= min_profit_pct)
        exit_by_time = bool(holding >= holding_max_bars)
        if exit_by_percentile or exit_by_time:
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
                    "forced": exit_by_time and not exit_by_percentile,
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


def compute_metrics(
    equity: FloatArray,
    trades: list[dict[str, Any]],
    sampling_period_seconds: float,
) -> BacktestMetrics:
    if equity.size == 0:
        raise ValueError("equity vacía")
    final_eq = float(equity[-1])
    total_return_pct = (final_eq - 1.0) * 100.0

    if len(trades) == 0:
        return BacktestMetrics(
            n_trades=0,
            final_equity=final_eq,
            total_return_pct=total_return_pct,
            sharpe_annualized=0.0,
            max_drawdown_pct=0.0,
            win_rate=0.0,
            avg_trade_pct=0.0,
            avg_holding_bars=0.0,
            forced_exits=0,
            trades=[],
        )

    log_eq = np.log(np.clip(equity, 1e-12, None))
    log_rets = np.diff(log_eq)
    log_rets = log_rets[np.isfinite(log_rets)]
    if log_rets.size < 2 or float(log_rets.std()) == 0.0:
        sharpe = 0.0
    else:
        bars_per_year = 365.25 * 24.0 * 3600.0 / sampling_period_seconds
        sharpe = float(log_rets.mean() / log_rets.std() * np.sqrt(bars_per_year))

    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak
    max_dd_pct = float(drawdown.min()) * 100.0

    trade_returns = np.array([t["net_return"] for t in trades], dtype=np.float64)
    holdings = np.array([t["holding_bars"] for t in trades], dtype=np.float64)
    forced = int(sum(1 for t in trades if t.get("forced")))

    return BacktestMetrics(
        n_trades=len(trades),
        final_equity=final_eq,
        total_return_pct=total_return_pct,
        sharpe_annualized=sharpe,
        max_drawdown_pct=max_dd_pct,
        win_rate=float((trade_returns > 0).mean()),
        avg_trade_pct=float(trade_returns.mean()) * 100.0,
        avg_holding_bars=float(holdings.mean()),
        forced_exits=forced,
        trades=trades,
    )


def backtest(
    df: pl.DataFrame,
    params: BacktestParams,
    sampling_period_seconds: float,
) -> BacktestMetrics:
    """Atajo: compute_signals + simulate + compute_metrics sobre el df completo."""
    close = df["close"].to_numpy().astype(np.float64, copy=False)
    ratio, p_low, p_high = compute_signals(df, params)
    equity, trades = simulate(
        close,
        ratio,
        p_low,
        p_high,
        holding_max_bars=params.holding_max_bars,
        min_profit_pct=params.min_profit_pct,
        fee_pct=params.fee_pct,
    )
    return compute_metrics(equity, trades, sampling_period_seconds)


__all__ = [
    "BacktestMetrics",
    "BacktestParams",
    "backtest",
    "compute_metrics",
    "compute_signals",
    "simulate",
]
