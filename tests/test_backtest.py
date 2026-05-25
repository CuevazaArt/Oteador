from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from oteador.backtest import (
    BacktestParams,
    backtest,
    compute_metrics,
    compute_signals,
    simulate,
)


def _df_from_close(close: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"close": close})


def test_backtest_params_validation() -> None:
    with pytest.raises(ValueError):
        BacktestParams(
            ma_window=0,
            entry_percentile=5,
            exit_percentile=95,
            rolling_window_bars=100,
            holding_max_bars=10,
            min_profit_pct=0.0,
        )
    with pytest.raises(ValueError):
        BacktestParams(
            ma_window=6,
            entry_percentile=95,
            exit_percentile=5,
            rolling_window_bars=100,
            holding_max_bars=10,
            min_profit_pct=0.0,
        )
    with pytest.raises(ValueError):
        BacktestParams(
            ma_window=6,
            entry_percentile=5,
            exit_percentile=95,
            rolling_window_bars=10,
            holding_max_bars=10,
            min_profit_pct=0.0,
        )


def test_simulate_constant_price_makes_no_trades() -> None:
    n = 1000
    close = np.full(n, 100.0)
    ratio = np.zeros(n)
    p_low = np.full(n, -0.01)
    p_high = np.full(n, 0.01)
    eq, trades = simulate(
        close,
        ratio,
        p_low,
        p_high,
        holding_max_bars=100,
        min_profit_pct=0.0,
        fee_pct=0.001,
    )
    assert len(trades) == 0
    assert eq[-1] == pytest.approx(1.0)


def test_simulate_synthetic_round_trip_is_profitable_without_fees() -> None:
    n = 200
    # Bar 0..49: precio 100, ratio bajo (entra). Bar 50..199: precio 110, ratio
    # alto (sale y no vuelve a entrar porque ratio > p_low).
    close = np.full(n, 100.0)
    close[50:] = 110.0
    ratio = np.empty(n)
    ratio[:50] = -1.0
    ratio[50:] = +1.0
    p_low = np.zeros(n)
    p_high = np.zeros(n)
    eq, trades = simulate(
        close,
        ratio,
        p_low,
        p_high,
        holding_max_bars=10_000,
        min_profit_pct=0.05,
        fee_pct=0.0,
    )
    assert len(trades) == 1
    t = trades[0]
    assert t["entry_bar"] == 0
    assert t["entry_price"] == 100.0
    assert t["exit_bar"] == 50
    assert t["exit_price"] == 110.0
    assert t["net_return"] == pytest.approx(0.10)
    assert eq[-1] == pytest.approx(1.10)


def test_simulate_forced_exit_on_holding_max() -> None:
    n = 100
    close = np.linspace(100.0, 90.0, n)  # tendencia bajista
    ratio = np.full(n, -0.5)  # siempre por debajo de p_low
    p_low = np.zeros(n)
    p_high = np.full(n, 1.0)  # nunca dispara salida por percentil
    _eq, trades = simulate(
        close,
        ratio,
        p_low,
        p_high,
        holding_max_bars=5,
        min_profit_pct=0.0,
        fee_pct=0.0,
        force_close_at_end=False,
    )
    assert len(trades) >= 1
    assert all(t["forced"] for t in trades)
    assert trades[0]["holding_bars"] == 5


def test_simulate_fees_reduce_returns() -> None:
    n = 100
    close = np.full(n, 100.0)
    close[50:] = 110.0
    ratio = np.empty(n)
    ratio[:50] = -1.0
    ratio[50:] = +1.0
    p_low = np.zeros(n)
    p_high = np.zeros(n)
    _, trades = simulate(
        close,
        ratio,
        p_low,
        p_high,
        holding_max_bars=10_000,
        min_profit_pct=0.05,
        fee_pct=0.001,
    )
    assert len(trades) == 1
    # 110/100 * (1-0.001)^2 - 1 ≈ 0.0978
    assert trades[0]["net_return"] == pytest.approx(1.10 * (0.999**2) - 1.0, abs=1e-6)


def test_compute_signals_shape_and_causality() -> None:
    n = 500
    rng = np.random.default_rng(0)
    close = 100.0 + np.cumsum(rng.standard_normal(n) * 0.1)
    df = _df_from_close(close.tolist())
    params = BacktestParams(
        ma_window=6,
        entry_percentile=5,
        exit_percentile=95,
        rolling_window_bars=128,
        holding_max_bars=20,
        min_profit_pct=0.0,
    )
    ratio, p_low, p_high = compute_signals(df, params)
    assert ratio.shape == (n,)
    assert p_low.shape == (n,)
    assert p_high.shape == (n,)
    # Los primeros valores deben ser NaN por warmup de la MA y de las percentiles.
    assert np.isnan(ratio[0])
    assert np.isnan(p_low[0])
    # El último debe ser finito (warmup completado).
    assert np.isfinite(ratio[-1])
    assert np.isfinite(p_low[-1])
    assert np.isfinite(p_high[-1])


def test_compute_metrics_no_trades_returns_zero_sharpe() -> None:
    eq = np.ones(100)
    m = compute_metrics(eq, [], sampling_period_seconds=60.0)
    assert m.n_trades == 0
    assert m.sharpe_annualized == 0.0
    assert m.total_return_pct == 0.0


def test_backtest_end_to_end_runs() -> None:
    rng = np.random.default_rng(7)
    n = 2000
    # Serie con leve reversión: AR(1) negativo sobre returns.
    eps = rng.standard_normal(n) * 0.001
    rets = np.zeros(n)
    for i in range(1, n):
        rets[i] = -0.3 * rets[i - 1] + eps[i]
    close = 100.0 * np.exp(np.cumsum(rets))
    df = _df_from_close(close.tolist())
    params = BacktestParams(
        ma_window=8,
        entry_percentile=10.0,
        exit_percentile=90.0,
        rolling_window_bars=256,
        holding_max_bars=50,
        min_profit_pct=0.0,
        fee_pct=0.0,
    )
    m = backtest(df, params, sampling_period_seconds=60.0)
    assert m.n_trades >= 1
    assert np.isfinite(m.sharpe_annualized)


def test_walk_forward_score_penalizes_low_trades() -> None:
    from oteador.studies.optuna_study import (
        PENALTY_SCORE,
        WalkForwardConfig,
        walk_forward_score,
    )

    n = 5000
    close = np.full(n, 100.0)
    ratio = np.zeros(n)
    p_low = np.full(n, -0.01)
    p_high = np.full(n, 0.01)
    params = BacktestParams(
        ma_window=6,
        entry_percentile=5,
        exit_percentile=95,
        rolling_window_bars=64,
        holding_max_bars=10,
        min_profit_pct=0.0,
        fee_pct=0.001,
    )
    score, fold_metrics = walk_forward_score(
        close,
        ratio,
        p_low,
        p_high,
        params,
        sampling_period_seconds=60.0,
        cv=WalkForwardConfig(n_folds=4, min_trades_per_fold=1),
    )
    # Precio constante → 0 trades → penalización.
    assert score == PENALTY_SCORE
    assert all(m.n_trades == 0 for m in fold_metrics)
