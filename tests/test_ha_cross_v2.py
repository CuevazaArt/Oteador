from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from oteador.strategies.ha_cross_v2 import (
    HaCrossV2Params,
    backtest,
    compute_signals,
    simulate,
)


def _ohlc_from_close(close: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame({"open": close, "high": close, "low": close, "close": close})


def test_params_defaults_disable_extra_filters() -> None:
    p = HaCrossV2Params(fast_window=2, slow_window=6)
    assert p.confirmation_bars == 1
    assert p.ha_color_confirm == 0
    assert p.min_cross_strength_pct == 0.0
    assert p.entry_max_distance_pct == float("inf")
    assert p.exit_min_distance_pct == float("inf")


def test_params_validation_rolling_percentiles_require_bounded_threshold() -> None:
    with pytest.raises(ValueError, match="percentiles rodantes"):
        HaCrossV2Params(
            fast_window=2,
            slow_window=6,
            use_rolling_percentiles=True,
            entry_max_distance_pct=5.0,  # fuera de (0,1]
            exit_min_distance_pct=0.9,
        )


def test_params_validation_static_distance_must_be_positive() -> None:
    with pytest.raises(ValueError, match="entry_max_distance_pct"):
        HaCrossV2Params(
            fast_window=2,
            slow_window=6,
            entry_max_distance_pct=0.0,
        )


def test_confirmation_bars_blocks_short_whipsaws() -> None:
    """Un cruce que dura 2 barras y se invierte no debe disparar entrada."""
    n = 300
    close = np.full(n, 100.0)
    close[200:] = 110.0  # subida real más tarde
    df = _ohlc_from_close(close)

    p = HaCrossV2Params(
        fast_window=1,
        slow_window=2,
        min_holding_bars=1,
        confirmation_bars=5,  # exige 5 barras de cruce sostenido
        ha_color_confirm=0,
    )
    sig = compute_signals(df, p)
    # Inyectamos un whipsaw artificial corto en fast/slow:
    fast = sig.fast.copy()
    slow = sig.slow.copy()
    fast[:] = 99.0
    slow[:] = 100.0
    fast[50:52] = 100.5
    slow[50:52] = 99.5  # cruce UP por 2 barras
    fast[100:200] = 100.5
    slow[100:200] = 99.5  # cruce UP sostenido (100 barras) → entra
    fast[200:] = 99.0
    slow[200:] = 100.0
    sig2 = type(sig)(
        close=sig.close,
        fast=fast,
        slow=slow,
        ha_green=sig.ha_green,
        distance=sig.distance,
        entry_threshold=sig.entry_threshold,
        exit_threshold=sig.exit_threshold,
    )
    _eq, trades = simulate(sig2, p, force_close_at_end=False)
    # Sólo el cruce largo debió haber abierto trade.
    assert len(trades) == 1
    assert trades[0]["entry_bar"] >= 100 + p.confirmation_bars - 1
    assert trades[0]["entry_bar"] < 200


def test_min_cross_strength_blocks_weak_crosses() -> None:
    """Un cruce con separación insuficiente entre fast y slow no debe disparar."""
    n = 400
    close = np.full(n, 100.0)
    df = _ohlc_from_close(close)
    p = HaCrossV2Params(
        fast_window=1,
        slow_window=2,
        min_holding_bars=1,
        min_cross_strength_pct=0.01,  # exige 1% de separación
        confirmation_bars=1,
    )
    sig = compute_signals(df, p)
    fast = np.full(n, 100.0001)
    slow = np.full(n, 100.0)  # (fast-slow)/slow ≈ 1e-6 → insuficiente
    sig2 = type(sig)(
        close=sig.close,
        fast=fast,
        slow=slow,
        ha_green=sig.ha_green,
        distance=sig.distance,
        entry_threshold=sig.entry_threshold,
        exit_threshold=sig.exit_threshold,
    )
    _eq, trades = simulate(sig2, p, force_close_at_end=False)
    assert len(trades) == 0


def test_entry_distance_filter_blocks_chase() -> None:
    """Si el precio está muy por encima del slow_MA, no debe entrar (anti-chase)."""
    n = 200
    close = np.full(n, 100.0)
    df = _ohlc_from_close(close)
    p = HaCrossV2Params(
        fast_window=1,
        slow_window=2,
        min_holding_bars=1,
        confirmation_bars=1,
        entry_max_distance_pct=0.005,  # 0.5% máximo
    )
    sig = compute_signals(df, p)
    fast = np.full(n, 110.0)
    slow = np.full(n, 100.0)
    # distance manual: close/slow - 1 = 0 (pero el filtro lee de sig.distance)
    distance = np.full(n, 0.02)  # 2% por encima → bloquea (> 0.5%)
    sig2 = type(sig)(
        close=sig.close,
        fast=fast,
        slow=slow,
        ha_green=sig.ha_green,
        distance=distance,
        entry_threshold=sig.entry_threshold,
        exit_threshold=sig.exit_threshold,
    )
    _eq, trades = simulate(sig2, p, force_close_at_end=False)
    assert len(trades) == 0


def test_exit_distance_takes_profit_above_threshold() -> None:
    """Cuando la distancia al slow_MA supera el umbral de salida y hay ganancia, debe salir."""
    n = 200
    close = np.full(n, 100.0)
    close[100:] = 120.0  # +20% para garantizar ganancia
    df = _ohlc_from_close(close)
    p = HaCrossV2Params(
        fast_window=1,
        slow_window=2,
        min_holding_bars=1,
        confirmation_bars=1,
        entry_max_distance_pct=10.0,  # entrada libre
        exit_min_distance_pct=0.05,  # sale si distance >= 5%
        fee_pct=0.0,
    )
    sig = compute_signals(df, p)
    n = sig.close.size
    # Forzamos: cruce UP sostenido todo el tiempo, distancia que sube tras bar 100.
    fast = np.full(n, 101.0)
    slow = np.full(n, 100.0)
    distance = np.zeros(n)
    distance[100:] = 0.10  # 10% por encima
    sig2 = type(sig)(
        close=sig.close,
        fast=fast,
        slow=slow,
        ha_green=sig.ha_green,
        distance=distance,
        entry_threshold=sig.entry_threshold,
        exit_threshold=sig.exit_threshold,
    )
    _eq, trades = simulate(sig2, p, force_close_at_end=False)
    assert len(trades) == 1
    assert trades[0]["exit_bar"] == 100
    assert trades[0]["exit_reason"] == "distance"
    assert trades[0]["net_return"] == pytest.approx(0.20)


def test_backtest_runs_on_synthetic_series() -> None:
    rng = np.random.default_rng(7)
    n = 2000
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5 + 0.05)
    close = np.clip(close, 1.0, None)
    df = _ohlc_from_close(close)
    p = HaCrossV2Params(
        fast_window=2,
        slow_window=10,
        min_holding_bars=2,
        fee_pct=0.0,
        confirmation_bars=2,
        ha_color_confirm=1,
        min_cross_strength_pct=0.0005,
        entry_max_distance_pct=0.05,
        exit_min_distance_pct=0.08,
    )
    m = backtest(df, p, sampling_period_seconds=3600.0)
    assert np.isfinite(m.sharpe_annualized)
    # No esperamos rentabilidad concreta; sólo que la simulación no rompa con todos los filtros.


def test_ha_color_confirm_blocks_red_entry() -> None:
    """Aunque haya cruce UP fuerte, si las velas HA son rojas no debe entrar."""
    n = 200
    close = np.full(n, 100.0)
    df = _ohlc_from_close(close)
    p = HaCrossV2Params(
        fast_window=1,
        slow_window=2,
        min_holding_bars=1,
        confirmation_bars=1,
        ha_color_confirm=3,  # exige 3 velas verdes
    )
    sig = compute_signals(df, p)
    fast = np.full(n, 101.0)
    slow = np.full(n, 100.0)
    ha_green = np.zeros(n, dtype=bool)  # todas rojas → bloquea
    sig2 = type(sig)(
        close=sig.close,
        fast=fast,
        slow=slow,
        ha_green=ha_green,
        distance=sig.distance,
        entry_threshold=sig.entry_threshold,
        exit_threshold=sig.exit_threshold,
    )
    _eq, trades = simulate(sig2, p, force_close_at_end=False)
    assert len(trades) == 0
