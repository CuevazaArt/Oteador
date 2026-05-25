from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from oteador.strategies.ha_cross import HaCrossParams, backtest, compute_signals, simulate


def _df_ohlc_from_close(close: np.ndarray) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
        }
    )


def test_params_validation() -> None:
    with pytest.raises(ValueError):
        HaCrossParams(fast_window=0, slow_window=5)
    with pytest.raises(ValueError):
        HaCrossParams(fast_window=10, slow_window=10)
    with pytest.raises(ValueError):
        HaCrossParams(fast_window=20, slow_window=10)
    with pytest.raises(ValueError):
        HaCrossParams(fast_window=5, slow_window=10, min_holding_bars=0)


def test_simulate_monotonic_uptrend_enters_once_and_force_closes() -> None:
    n = 500
    close = np.linspace(100.0, 200.0, n).astype(np.float64)
    fast = np.linspace(100.0, 200.0, n).astype(np.float64)
    slow = np.linspace(99.0, 199.0, n).astype(np.float64)  # fast siempre por encima
    _eq, trades = simulate(close, fast, slow, min_holding_bars=1, fee_pct=0.0)
    assert len(trades) == 1
    t = trades[0]
    assert t["entry_bar"] == 0
    assert t["forced"] is True
    assert t["net_return"] == pytest.approx(1.0)  # 200/100 - 1


def test_simulate_monotonic_downtrend_never_enters() -> None:
    n = 200
    close = np.linspace(100.0, 50.0, n).astype(np.float64)
    fast = np.linspace(99.0, 49.0, n).astype(np.float64)
    slow = np.linspace(100.0, 50.0, n).astype(np.float64)  # fast siempre por debajo
    eq, trades = simulate(close, fast, slow, min_holding_bars=1, fee_pct=0.0)
    assert len(trades) == 0
    assert eq[-1] == pytest.approx(1.0)


def test_simulate_single_cross_up_then_down() -> None:
    n = 200
    close = np.full(n, 100.0)
    close[100:] = 110.0
    # fast cruza arriba de slow en bar 0, baja en bar 150
    fast = np.empty(n)
    slow = np.empty(n)
    fast[:150] = 1.0
    slow[:150] = 0.5
    fast[150:] = 0.5
    slow[150:] = 1.0
    _eq, trades = simulate(
        close, fast, slow, min_holding_bars=1, fee_pct=0.0, force_close_at_end=False
    )
    assert len(trades) == 1
    assert trades[0]["entry_bar"] == 0
    assert trades[0]["exit_bar"] == 150
    assert trades[0]["forced"] is False
    assert trades[0]["net_return"] == pytest.approx(0.10)


def test_simulate_min_holding_blocks_immediate_exit() -> None:
    n = 200
    close = np.full(n, 100.0)
    close[50:] = 110.0  # subida tras whipsaw
    fast = np.empty(n)
    slow = np.empty(n)
    # Whipsaw: cruce arriba en bar 0, abajo en bar 5, arriba otra vez en bar 6
    fast[:5] = 1.0
    slow[:5] = 0.5
    fast[5:6] = 0.5
    slow[5:6] = 1.0
    fast[6:150] = 1.0
    slow[6:150] = 0.5
    fast[150:] = 0.5
    slow[150:] = 1.0
    _eq, trades = simulate(
        close, fast, slow, min_holding_bars=30, fee_pct=0.0, force_close_at_end=False
    )
    # min_holding=30 impide la salida en bar 5; mantiene hasta bar 150.
    assert len(trades) == 1
    assert trades[0]["entry_bar"] == 0
    assert trades[0]["exit_bar"] == 150
    assert trades[0]["net_return"] == pytest.approx(0.10)


def test_compute_signals_shape() -> None:
    n = 300
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.1)
    df = _df_ohlc_from_close(close)
    params = HaCrossParams(fast_window=5, slow_window=20, min_holding_bars=1)
    close_arr, fast, slow = compute_signals(df, params)
    assert close_arr.shape == (n,)
    assert fast.shape == (n,)
    assert slow.shape == (n,)
    assert np.isnan(fast[0]) or np.isfinite(fast[0])  # ok if NaN by warmup
    assert np.isfinite(fast[-1])
    assert np.isfinite(slow[-1])


def test_backtest_runs_on_synthetic_series() -> None:
    rng = np.random.default_rng(11)
    n = 2000
    # Drift positivo + ruido.
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.5 + 0.05)
    close = np.clip(close, 1.0, None)
    df = _df_ohlc_from_close(close)
    params = HaCrossParams(fast_window=5, slow_window=20, min_holding_bars=3, fee_pct=0.0)
    m = backtest(df, params, sampling_period_seconds=3600.0)
    assert m.n_trades >= 1
    assert np.isfinite(m.sharpe_annualized)


def test_ha_cross_walk_forward_penalizes_low_trades() -> None:
    from oteador.studies.ha_cross_study import walk_forward_score
    from oteador.studies.optuna_study import PENALTY_SCORE, WalkForwardConfig

    n = 2000
    close = np.full(n, 100.0)
    # Sin cruces: fast siempre debajo de slow.
    fast = np.full(n, 1.0)
    slow = np.full(n, 2.0)
    params = HaCrossParams(fast_window=3, slow_window=10, min_holding_bars=1, fee_pct=0.001)
    score, fold_metrics = walk_forward_score(
        close,
        fast,
        slow,
        params,
        sampling_period_seconds=3600.0,
        cv=WalkForwardConfig(n_folds=4, min_trades_per_fold=1),
    )
    assert score == PENALTY_SCORE
    assert all(m.n_trades == 0 for m in fold_metrics)
