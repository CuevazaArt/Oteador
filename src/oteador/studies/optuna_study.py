"""Estudio Optuna con walk-forward CV sobre la estrategia de reversión por percentiles.

Walk-forward: dividimos la serie en N folds contiguos. Saltamos el fold 0 (no
hay suficiente lookback para los percentiles rodantes) y evaluamos el backtest
en cada fold restante. El objetivo Optuna es la media del Sharpe out-of-sample
entre folds, con penalización dura para configuraciones que apenas tradean
(menos de `min_trades_per_fold` trades en algún fold).

Los features (ratio, percentiles rodantes) se computan una sola vez sobre la
serie completa, ya que son causales (sólo miran hacia atrás). El backtest
luego se restringe al slice del fold.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import optuna
import polars as pl
import structlog

from oteador.backtest import (
    BacktestMetrics,
    BacktestParams,
    compute_metrics,
    compute_signals,
    simulate,
)

log = structlog.get_logger(__name__)

PENALTY_SCORE: float = -10.0


@dataclass(frozen=True)
class WalkForwardConfig:
    n_folds: int = 5
    min_trades_per_fold: int = 5
    skip_first_fold: bool = True

    def __post_init__(self) -> None:
        if self.n_folds < 2:
            raise ValueError(f"n_folds debe ser >= 2, recibido {self.n_folds}")
        if self.min_trades_per_fold < 1:
            raise ValueError(
                f"min_trades_per_fold debe ser >= 1, recibido {self.min_trades_per_fold}"
            )


def walk_forward_score(
    close: np.ndarray,
    ratio: np.ndarray,
    p_low: np.ndarray,
    p_high: np.ndarray,
    params: BacktestParams,
    sampling_period_seconds: float,
    cv: WalkForwardConfig,
) -> tuple[float, list[BacktestMetrics]]:
    """Devuelve (score, métricas por fold). Score = -PENALTY si algún fold no cumple."""
    n = int(close.size)
    fold_size = n // cv.n_folds
    if fold_size < params.rolling_window_bars * 2:
        return PENALTY_SCORE, []

    fold_metrics: list[BacktestMetrics] = []
    start_fold = 1 if cv.skip_first_fold else 0
    for k in range(start_fold, cv.n_folds):
        start = k * fold_size
        end = (k + 1) * fold_size if k < cv.n_folds - 1 else n
        eq, trades = simulate(
            close[start:end].astype(np.float64, copy=False),
            ratio[start:end].astype(np.float64, copy=False),
            p_low[start:end].astype(np.float64, copy=False),
            p_high[start:end].astype(np.float64, copy=False),
            holding_max_bars=params.holding_max_bars,
            min_profit_pct=params.min_profit_pct,
            fee_pct=params.fee_pct,
        )
        fold_metrics.append(compute_metrics(eq, trades, sampling_period_seconds))

    sharpes: list[float] = []
    for m in fold_metrics:
        if m.n_trades < cv.min_trades_per_fold:
            return PENALTY_SCORE, fold_metrics
        sharpes.append(m.sharpe_annualized)

    return float(np.mean(sharpes)), fold_metrics


def make_objective(
    df: pl.DataFrame,
    sampling_period_seconds: float,
    cv: WalkForwardConfig,
    fee_pct: float,
) -> Any:
    close = df["close"].to_numpy().astype(np.float64, copy=False)

    def objective(trial: optuna.Trial) -> float:
        try:
            params = BacktestParams(
                ma_window=trial.suggest_int("ma_window", 4, 100, log=True),
                entry_percentile=trial.suggest_float("entry_percentile", 1.0, 25.0),
                exit_percentile=trial.suggest_float("exit_percentile", 75.0, 99.0),
                rolling_window_bars=trial.suggest_int("rolling_window_bars", 500, 10_000, log=True),
                holding_max_bars=trial.suggest_int("holding_max_bars", 5, 500, log=True),
                min_profit_pct=trial.suggest_float("min_profit_pct", 0.0, 0.005),
                fee_pct=fee_pct,
            )
        except ValueError:
            return PENALTY_SCORE

        ratio, p_low, p_high = compute_signals(df, params)
        score, fold_metrics = walk_forward_score(
            close, ratio, p_low, p_high, params, sampling_period_seconds, cv
        )
        trial.set_user_attr(
            "fold_metrics",
            [m.to_summary() for m in fold_metrics],
        )
        return score

    return objective


@dataclass(frozen=True)
class StudyArtifacts:
    study: optuna.Study
    best_params: dict[str, Any]
    best_value: float
    best_fold_metrics: list[dict[str, Any]]


def run_study(
    df: pl.DataFrame,
    sampling_period_seconds: float,
    n_trials: int = 100,
    cv: WalkForwardConfig | None = None,
    fee_pct: float = 0.001,
    seed: int = 42,
    study_name: str | None = None,
    storage_path: Path | None = None,
) -> StudyArtifacts:
    cv = cv or WalkForwardConfig()
    sampler = optuna.samplers.TPESampler(seed=seed)
    storage: str | None = None
    if storage_path is not None:
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage = f"sqlite:///{storage_path.as_posix()}"
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=study_name,
        storage=storage,
        load_if_exists=storage is not None,
    )
    objective = make_objective(df, sampling_period_seconds, cv, fee_pct)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    fold_metrics = study.best_trial.user_attrs.get("fold_metrics", []) if study.best_trial else []
    return StudyArtifacts(
        study=study,
        best_params=dict(study.best_params) if study.best_trial else {},
        best_value=float(study.best_value) if study.best_trial else PENALTY_SCORE,
        best_fold_metrics=fold_metrics,
    )


def format_study_report(art: StudyArtifacts, top_k: int = 5) -> str:
    lines: list[str] = []
    s = art.study
    lines.append(f"=== Optuna study: {s.study_name} ===")
    lines.append(f"Trials completos: {len(s.trials)}")
    if art.best_value <= PENALTY_SCORE:
        lines.append("No se encontró ninguna configuración válida (todos los folds penalizados).")
        return "\n".join(lines)
    lines.append(f"Mejor score (Sharpe medio OOS): {art.best_value:+.4f}")
    lines.append("")
    lines.append("Mejores parámetros:")
    for k, v in art.best_params.items():
        if isinstance(v, float):
            lines.append(f"  {k}: {v:.6f}")
        else:
            lines.append(f"  {k}: {v}")
    sorted_trials = sorted(
        (t for t in s.trials if t.value is not None and t.value > PENALTY_SCORE),
        key=lambda t: t.value or PENALTY_SCORE,
        reverse=True,
    )[:top_k]
    if sorted_trials:
        lines.append("")
        lines.append(f"Top {len(sorted_trials)} trials válidos:")
        for i, t in enumerate(sorted_trials, 1):
            p = t.params
            lines.append(
                f"  #{i:>2} score={t.value:+.4f}  ma={p['ma_window']:>3}  "
                f"entry_p={p['entry_percentile']:>5.2f}  exit_p={p['exit_percentile']:>5.2f}  "
                f"rw={p['rolling_window_bars']:>5}  hmax={p['holding_max_bars']:>4}  "
                f"min_p={p['min_profit_pct']:.4f}"
            )
    if art.best_fold_metrics:
        lines.append("")
        lines.append("Desglose del mejor trial por fold:")
        for idx, m in enumerate(art.best_fold_metrics, 1):
            lines.append(
                f"  Fold {idx}: n_trades={m['n_trades']:>4}  "
                f"sharpe={m['sharpe_annualized']:+.3f}  "
                f"return={m['total_return_pct']:+.2f}%  "
                f"win_rate={m['win_rate']:.1%}  "
                f"max_dd={m['max_drawdown_pct']:+.2f}%  "
                f"forced={m['forced_exits']}"
            )
    return "\n".join(lines)


__all__ = [
    "PENALTY_SCORE",
    "StudyArtifacts",
    "WalkForwardConfig",
    "format_study_report",
    "make_objective",
    "run_study",
    "walk_forward_score",
]
