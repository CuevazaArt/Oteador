"""Optuna walk-forward sobre la estrategia HA-cross.

Búsqueda de (fast_window, slow_window, min_holding_bars) que maximiza el
Sharpe medio out-of-sample. Por defecto explora desde fast=1, slow=2 (lo más
granular posible) hasta fast=50, slow=200; min_holding_bars 1..100.

Optimización clave: HA_Open se computa **una vez** por dataset; cada trial
sólo recalcula las dos rolling means, que es lo barato.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import optuna
import polars as pl
import structlog

from oteador.backtest import BacktestMetrics, compute_metrics
from oteador.features.heikin_ashi import heikin_ashi
from oteador.strategies.ha_cross import HaCrossParams, simulate
from oteador.studies.optuna_study import PENALTY_SCORE, WalkForwardConfig

log = structlog.get_logger(__name__)

FloatArray = npt.NDArray[np.float64]


def walk_forward_score(
    close: FloatArray,
    fast: FloatArray,
    slow: FloatArray,
    params: HaCrossParams,
    sampling_period_seconds: float,
    cv: WalkForwardConfig,
) -> tuple[float, list[BacktestMetrics]]:
    n = int(close.size)
    fold_size = n // cv.n_folds
    if fold_size < params.slow_window * 2:
        return PENALTY_SCORE, []

    fold_metrics: list[BacktestMetrics] = []
    start_fold = 1 if cv.skip_first_fold else 0
    for k in range(start_fold, cv.n_folds):
        start = k * fold_size
        end = (k + 1) * fold_size if k < cv.n_folds - 1 else n
        eq, trades = simulate(
            close[start:end].astype(np.float64, copy=False),
            fast[start:end].astype(np.float64, copy=False),
            slow[start:end].astype(np.float64, copy=False),
            min_holding_bars=params.min_holding_bars,
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
    fast_max: int = 50,
    slow_max: int = 200,
    holding_max: int = 100,
) -> Any:
    df_ha = heikin_ashi(df)
    ha_open_series = df_ha["ha_open"]
    close = df["close"].to_numpy().astype(np.float64, copy=False)

    def objective(trial: optuna.Trial) -> float:
        fast_window = trial.suggest_int("fast_window", 1, fast_max)
        slow_window = trial.suggest_int("slow_window", 2, slow_max, log=True)
        if slow_window <= fast_window:
            return PENALTY_SCORE
        min_holding = trial.suggest_int("min_holding_bars", 1, holding_max, log=True)
        try:
            params = HaCrossParams(
                fast_window=fast_window,
                slow_window=slow_window,
                min_holding_bars=min_holding,
                fee_pct=fee_pct,
            )
        except ValueError:
            return PENALTY_SCORE

        fast_arr = (
            ha_open_series.rolling_mean(window_size=fast_window)
            .to_numpy()
            .astype(np.float64, copy=False)
        )
        slow_arr = (
            ha_open_series.rolling_mean(window_size=slow_window)
            .to_numpy()
            .astype(np.float64, copy=False)
        )

        score, fold_metrics = walk_forward_score(
            close, fast_arr, slow_arr, params, sampling_period_seconds, cv
        )
        trial.set_user_attr("fold_metrics", [m.to_summary() for m in fold_metrics])
        return score

    return objective


@dataclass(frozen=True)
class HaCrossStudyArtifacts:
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
    fast_max: int = 50,
    slow_max: int = 200,
    holding_max: int = 100,
    seed: int = 42,
    study_name: str | None = None,
    storage_path: Path | None = None,
) -> HaCrossStudyArtifacts:
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
    objective = make_objective(
        df, sampling_period_seconds, cv, fee_pct, fast_max, slow_max, holding_max
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    fold_metrics = study.best_trial.user_attrs.get("fold_metrics", []) if study.best_trial else []
    return HaCrossStudyArtifacts(
        study=study,
        best_params=dict(study.best_params) if study.best_trial else {},
        best_value=float(study.best_value) if study.best_trial else PENALTY_SCORE,
        best_fold_metrics=fold_metrics,
    )


def format_study_report(art: HaCrossStudyArtifacts, top_k: int = 10) -> str:
    lines: list[str] = []
    s = art.study
    lines.append(f"=== HA-Cross Optuna study: {s.study_name} ===")
    lines.append(f"Trials completos: {len(s.trials)}")
    if art.best_value <= PENALTY_SCORE:
        lines.append("No se encontró ninguna configuración válida.")
        return "\n".join(lines)
    lines.append(f"Mejor score (Sharpe medio OOS): {art.best_value:+.4f}")
    lines.append("")
    lines.append("Mejores parámetros:")
    for k, v in art.best_params.items():
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
                f"  #{i:>2} score={t.value:+.4f}  fast={p['fast_window']:>3}  "
                f"slow={p['slow_window']:>4}  hold={p['min_holding_bars']:>3}"
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
                f"max_dd={m['max_drawdown_pct']:+.2f}%"
            )
    return "\n".join(lines)


__all__ = [
    "HaCrossStudyArtifacts",
    "format_study_report",
    "make_objective",
    "run_study",
    "walk_forward_score",
]
