from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from oteador.backtest import BacktestMetrics
from oteador.studies.ha_cross_v2_study import (
    CompositeScoreWeights,
    HaCrossV2SearchSpace,
    StabilityConfig,
    composite_score,
    run_study,
)
from oteador.studies.optuna_study import PENALTY_SCORE, WalkForwardConfig


def _synthetic_1m_df(hours: int = 240) -> pl.DataFrame:
    """Genera ~hours horas de klines 1m con drift positivo + ruido."""
    n = hours * 60
    rng = np.random.default_rng(3)
    open_times = np.arange(n, dtype=np.int64) * 60_000
    closes = 100.0 + np.cumsum(rng.standard_normal(n) * 0.05 + 0.005)
    closes = np.clip(closes, 1.0, None)
    opens = np.concatenate([[100.0], closes[:-1]])
    highs = np.maximum(opens, closes) + 0.02
    lows = np.minimum(opens, closes) - 0.02
    volumes = np.ones(n)
    return pl.DataFrame(
        {
            "open_time": open_times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "close_time": open_times + 59_999,
            "quote_volume": volumes * closes,
            "trades": np.full(n, 1, dtype=np.int64),
            "taker_buy_base_volume": volumes * 0.5,
            "taker_buy_quote_volume": volumes * closes * 0.5,
            "ignore": np.zeros(n, dtype=np.int64),
        }
    )


def _fake_metrics(
    n_trades: int = 10,
    sharpe: float = 1.0,
    dd: float = -5.0,
    wr: float = 0.55,
    forced: int = 0,
) -> BacktestMetrics:
    return BacktestMetrics(
        n_trades=n_trades,
        final_equity=1.0 + sharpe * 0.1,
        total_return_pct=sharpe * 10.0,
        sharpe_annualized=sharpe,
        max_drawdown_pct=dd,
        win_rate=wr,
        avg_trade_pct=0.5,
        avg_holding_bars=5.0,
        forced_exits=forced,
    )


def test_composite_score_combines_components() -> None:
    metrics = [_fake_metrics(sharpe=1.5, dd=-10.0, wr=0.6, forced=2) for _ in range(5)]
    cv = WalkForwardConfig(n_folds=5, min_trades_per_fold=1)
    w = CompositeScoreWeights(whipsaw=1.0, drawdown=1.0, win_rate=1.0, win_rate_cap=0.6)
    score, breakdown = composite_score(metrics, cv, w, StabilityConfig())
    # sharpe 1.5 - whipsaw(2/10=0.2) - dd(0.10) + win(0.6) = 1.8
    assert score == pytest.approx(1.8, abs=1e-6)
    assert breakdown["sharpe_mean"] == pytest.approx(1.5)
    assert breakdown["whipsaw_penalty"] == pytest.approx(0.2)


def test_composite_score_penalizes_low_trades() -> None:
    metrics = [_fake_metrics(n_trades=2, sharpe=1.0) for _ in range(4)]
    cv = WalkForwardConfig(n_folds=5, min_trades_per_fold=10)
    score, _ = composite_score(metrics, cv, CompositeScoreWeights(), StabilityConfig())
    assert score == PENALTY_SCORE


def test_composite_score_stability_min_positive_folds() -> None:
    # 2 folds positivos, 3 negativos → con min_positive_folds=4 cae a PENALTY
    metrics = [
        _fake_metrics(sharpe=1.0),
        _fake_metrics(sharpe=1.0),
        _fake_metrics(sharpe=-0.5),
        _fake_metrics(sharpe=-0.5),
        _fake_metrics(sharpe=-0.5),
    ]
    cv = WalkForwardConfig(n_folds=5, min_trades_per_fold=1)
    stab = StabilityConfig(min_positive_folds=4)
    score, _ = composite_score(metrics, cv, CompositeScoreWeights(), stab)
    assert score == PENALTY_SCORE


def test_search_space_validation() -> None:
    with pytest.raises(ValueError):
        HaCrossV2SearchSpace(timeframes=())
    with pytest.raises(ValueError):
        HaCrossV2SearchSpace(fast_max=20, slow_max=10)


def test_run_study_smoke_few_trials() -> None:
    """Ejecuta Optuna con pocos trials sobre datos sintéticos pequeños."""
    df = _synthetic_1m_df(hours=240)  # 240h ≈ 10 días
    space = HaCrossV2SearchSpace(
        timeframes=("1h",),  # 1 sólo TF para velocidad
        fast_max=4,
        slow_max=10,
        holding_max=5,
        confirmation_max=2,
        ha_color_max=1,
        use_rolling_percentiles_choices=(False,),
    )
    cv = WalkForwardConfig(n_folds=3, min_trades_per_fold=1)
    art = run_study(
        df_1m=df,
        n_trials=5,
        cv=cv,
        fee_pct=0.0,
        space=space,
        weights=CompositeScoreWeights(),
        stability=StabilityConfig(),
        seed=0,
    )
    assert len(art.study.trials) == 5
    # No exigimos un best_value concreto; sólo que la mecánica corra.
    assert art.best_value is not None
