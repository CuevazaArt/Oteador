"""Optuna walk-forward sobre HA-Cross v2 con barrido multi-timeframe.

El timeframe es un hiperparámetro categórico más
(``{15m, 30m, 1h, 2h, 4h}`` por defecto). Para no pagar la descarga repetida
desde Binance Vision, partimos de un df 1m y agregamos in-memory a cada TF
explorado, cacheando el resultado.

El score por trial es un agregado robusto sobre los folds out-of-sample:

    score = sharpe_oos_medio
            - λ_whipsaw · forced_exits_ratio
            - λ_dd      · |max_drawdown_pct| / 100
            + λ_win     · min(win_rate, 0.6)

con penalización dura si algún fold viola los mínimos de actividad
(``min_trades_per_fold``) o si la estabilidad cruz-fold es insuficiente
(``min_positive_folds`` con sharpe > 0).
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
from oteador.data.aggregator import aggregate_klines
from oteador.features.heikin_ashi import heikin_ashi
from oteador.strategies.ha_cross_v2 import (
    HaCrossV2Params,
    SignalArrays,
    compute_signals,
    simulate,
)
from oteador.studies.characterization import interval_to_seconds
from oteador.studies.optuna_study import PENALTY_SCORE, WalkForwardConfig

log = structlog.get_logger(__name__)

FloatArray = npt.NDArray[np.float64]

DEFAULT_TIMEFRAMES: tuple[str, ...] = ("15m", "30m", "1h", "2h", "4h")


@dataclass(frozen=True)
class CompositeScoreWeights:
    """Pesos λ del score compuesto. Valores por defecto calibrados para
    Sharpe ~ O(1)."""

    whipsaw: float = 1.0
    drawdown: float = 1.0
    win_rate: float = 1.0
    win_rate_cap: float = 0.6

    def __post_init__(self) -> None:
        for name in ("whipsaw", "drawdown", "win_rate"):
            v = getattr(self, name)
            if v < 0.0:
                raise ValueError(f"{name} debe ser >= 0, recibido {v}")
        if not (0.0 < self.win_rate_cap <= 1.0):
            raise ValueError(f"win_rate_cap ∈ (0, 1]; recibido {self.win_rate_cap}")


@dataclass(frozen=True)
class StabilityConfig:
    """Requisitos adicionales sobre los folds OOS."""

    min_positive_folds: int = 0
    max_fold_sharpe_std: float | None = None

    def __post_init__(self) -> None:
        if self.min_positive_folds < 0:
            raise ValueError(
                f"min_positive_folds debe ser >= 0, recibido {self.min_positive_folds}"
            )
        if self.max_fold_sharpe_std is not None and self.max_fold_sharpe_std <= 0.0:
            raise ValueError(
                f"max_fold_sharpe_std debe ser > 0 o None, recibido {self.max_fold_sharpe_std}"
            )


@dataclass(frozen=True)
class HaCrossV2SearchSpace:
    timeframes: tuple[str, ...] = DEFAULT_TIMEFRAMES
    fast_max: int = 10
    slow_max: int = 30
    holding_max: int = 20
    confirmation_max: int = 5
    ha_color_max: int = 3
    cross_strength_max_pct: float = 0.005
    entry_distance_max_pct: float = 0.05
    exit_distance_max_pct: float = 0.10
    use_rolling_percentiles_choices: tuple[bool, ...] = (False, True)
    lookback_min: int = 100
    lookback_max: int = 2000

    def __post_init__(self) -> None:
        if not self.timeframes:
            raise ValueError("timeframes no puede estar vacío")
        if self.fast_max < 2:
            raise ValueError(f"fast_max debe ser >= 2, recibido {self.fast_max}")
        if self.slow_max <= self.fast_max:
            raise ValueError(f"slow_max ({self.slow_max}) debe ser > fast_max ({self.fast_max})")
        if self.holding_max < 1 or self.confirmation_max < 1 or self.ha_color_max < 0:
            raise ValueError("rangos enteros inválidos")
        if self.lookback_min < 32 or self.lookback_max <= self.lookback_min:
            raise ValueError("lookback_min >= 32 y lookback_max > lookback_min")


@dataclass
class _TimeframeBundle:
    """Cache por TF: df agregado + HA precomputado + sampling_period."""

    df: pl.DataFrame
    ha_open: pl.Series
    ha_green: npt.NDArray[np.bool_]
    sampling_period_seconds: float


def _build_tf_bundle(df_1m: pl.DataFrame, target_interval: str) -> _TimeframeBundle:
    df_tf = aggregate_klines(df_1m, "1m", target_interval)
    df_ha = heikin_ashi(df_tf)
    ha_open = df_ha["ha_open"]
    ha_green = (df_ha["ha_close"].to_numpy() > df_ha["ha_open"].to_numpy()).astype(bool)
    return _TimeframeBundle(
        df=df_tf,
        ha_open=ha_open,
        ha_green=ha_green,
        sampling_period_seconds=interval_to_seconds(target_interval),
    )


def composite_score(
    fold_metrics: list[BacktestMetrics],
    cv: WalkForwardConfig,
    weights: CompositeScoreWeights,
    stability: StabilityConfig,
) -> tuple[float, dict[str, float]]:
    """Devuelve (score, breakdown). PENALTY si no se cumplen los mínimos."""
    if not fold_metrics:
        return PENALTY_SCORE, {}

    sharpes: list[float] = []
    forced_ratios: list[float] = []
    dd_pcts: list[float] = []
    win_rates: list[float] = []
    for m in fold_metrics:
        if m.n_trades < cv.min_trades_per_fold:
            return PENALTY_SCORE, {"reason": float("nan")}
        sharpes.append(m.sharpe_annualized)
        forced_ratios.append(m.forced_exits / m.n_trades if m.n_trades else 0.0)
        dd_pcts.append(abs(m.max_drawdown_pct) / 100.0)
        win_rates.append(m.win_rate)

    positive_folds = int(sum(1 for s in sharpes if s > 0.0))
    if positive_folds < stability.min_positive_folds:
        return PENALTY_SCORE, {
            "sharpe_mean": float(np.mean(sharpes)),
            "positive_folds": float(positive_folds),
        }
    sharpe_std = float(np.std(sharpes))
    if stability.max_fold_sharpe_std is not None and sharpe_std > stability.max_fold_sharpe_std:
        return PENALTY_SCORE, {
            "sharpe_mean": float(np.mean(sharpes)),
            "sharpe_std": sharpe_std,
        }

    sharpe_mean = float(np.mean(sharpes))
    whipsaw_pen = weights.whipsaw * float(np.mean(forced_ratios))
    dd_pen = weights.drawdown * float(np.mean(dd_pcts))
    win_bonus = weights.win_rate * float(np.mean([min(w, weights.win_rate_cap) for w in win_rates]))
    score = sharpe_mean - whipsaw_pen - dd_pen + win_bonus
    breakdown = {
        "sharpe_mean": sharpe_mean,
        "sharpe_std": sharpe_std,
        "positive_folds": float(positive_folds),
        "whipsaw_penalty": whipsaw_pen,
        "drawdown_penalty": dd_pen,
        "win_rate_bonus": win_bonus,
        "score": score,
    }
    return score, breakdown


def walk_forward_eval(
    sig: SignalArrays,
    params: HaCrossV2Params,
    sampling_period_seconds: float,
    cv: WalkForwardConfig,
) -> list[BacktestMetrics]:
    n = int(sig.close.size)
    fold_size = n // cv.n_folds
    if fold_size < max(params.slow_window * 2, params.lookback_bars):
        return []

    fold_metrics: list[BacktestMetrics] = []
    start_fold = 1 if cv.skip_first_fold else 0
    for k in range(start_fold, cv.n_folds):
        start = k * fold_size
        end = (k + 1) * fold_size if k < cv.n_folds - 1 else n
        sl = slice(start, end)
        fold_sig = SignalArrays(
            close=sig.close[sl],
            fast=sig.fast[sl],
            slow=sig.slow[sl],
            ha_green=sig.ha_green[sl],
            distance=sig.distance[sl],
            entry_threshold=sig.entry_threshold[sl],
            exit_threshold=sig.exit_threshold[sl],
        )
        eq, trades = simulate(fold_sig, params)
        fold_metrics.append(compute_metrics(eq, trades, sampling_period_seconds))
    return fold_metrics


def make_objective(
    df_1m: pl.DataFrame,
    cv: WalkForwardConfig,
    fee_pct: float,
    space: HaCrossV2SearchSpace,
    weights: CompositeScoreWeights,
    stability: StabilityConfig,
) -> Any:
    bundles: dict[str, _TimeframeBundle] = {}

    def _bundle(tf: str) -> _TimeframeBundle:
        if tf not in bundles:
            log.info("aggregating_timeframe", target=tf)
            bundles[tf] = _build_tf_bundle(df_1m, tf)
        return bundles[tf]

    def objective(trial: optuna.Trial) -> float:
        tf = trial.suggest_categorical("timeframe", list(space.timeframes))
        bundle = _bundle(tf)
        if bundle.df.height < 200:
            return PENALTY_SCORE

        fast = trial.suggest_int("fast_window", 1, space.fast_max, log=True)
        slow = trial.suggest_int("slow_window", 2, space.slow_max, log=True)
        if slow <= fast:
            return PENALTY_SCORE
        min_holding = trial.suggest_int("min_holding_bars", 1, space.holding_max, log=True)
        cross_strength = trial.suggest_float(
            "min_cross_strength_pct", 0.0, space.cross_strength_max_pct
        )
        confirmation = trial.suggest_int("confirmation_bars", 1, space.confirmation_max)
        ha_color = trial.suggest_int("ha_color_confirm", 0, space.ha_color_max)
        use_rolling = trial.suggest_categorical(
            "use_rolling_percentiles", list(space.use_rolling_percentiles_choices)
        )
        # Nombres distintos por modo para que Optuna no se queje de configuraciones
        # `log` incompatibles entre trials (el "rolling" usa rango lineal en (0,1)
        # como percentil; el "static" usa rango log sobre proporciones pequeñas).
        if use_rolling:
            entry_dist = trial.suggest_float("entry_distance_quantile", 0.50, 0.95)
            exit_dist = trial.suggest_float("exit_distance_quantile", 0.70, 0.99)
            lookback = trial.suggest_int(
                "lookback_bars", space.lookback_min, space.lookback_max, log=True
            )
        else:
            entry_dist = trial.suggest_float(
                "entry_distance_static_pct", 0.001, space.entry_distance_max_pct, log=True
            )
            exit_dist = trial.suggest_float(
                "exit_distance_static_pct", 0.001, space.exit_distance_max_pct, log=True
            )
            lookback = 500  # ignorado cuando use_rolling=False

        try:
            params = HaCrossV2Params(
                fast_window=fast,
                slow_window=slow,
                min_holding_bars=min_holding,
                min_cross_strength_pct=cross_strength,
                confirmation_bars=confirmation,
                ha_color_confirm=ha_color,
                entry_max_distance_pct=entry_dist,
                exit_min_distance_pct=exit_dist,
                use_rolling_percentiles=bool(use_rolling),
                lookback_bars=lookback,
                fee_pct=fee_pct,
            )
        except ValueError:
            return PENALTY_SCORE

        sig = compute_signals(bundle.df, params, ha_open=bundle.ha_open, ha_green=bundle.ha_green)
        fold_metrics = walk_forward_eval(sig, params, bundle.sampling_period_seconds, cv)
        score, breakdown = composite_score(fold_metrics, cv, weights, stability)
        trial.set_user_attr("fold_metrics", [m.to_summary() for m in fold_metrics])
        trial.set_user_attr("score_breakdown", breakdown)
        trial.set_user_attr("timeframe_bars", bundle.df.height)
        return score

    return objective


@dataclass(frozen=True)
class HaCrossV2StudyArtifacts:
    study: optuna.Study
    best_params: dict[str, Any]
    best_value: float
    best_fold_metrics: list[dict[str, Any]]
    best_breakdown: dict[str, float]


def run_study(
    df_1m: pl.DataFrame,
    n_trials: int = 200,
    cv: WalkForwardConfig | None = None,
    fee_pct: float = 0.001,
    space: HaCrossV2SearchSpace | None = None,
    weights: CompositeScoreWeights | None = None,
    stability: StabilityConfig | None = None,
    seed: int = 42,
    study_name: str | None = None,
    storage_path: Path | None = None,
) -> HaCrossV2StudyArtifacts:
    cv = cv or WalkForwardConfig(n_folds=6, min_trades_per_fold=10)
    space = space or HaCrossV2SearchSpace()
    weights = weights or CompositeScoreWeights()
    stability = stability or StabilityConfig()

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
    objective = make_objective(df_1m, cv, fee_pct, space, weights, stability)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    if study.best_trial is None:
        return HaCrossV2StudyArtifacts(
            study=study,
            best_params={},
            best_value=PENALTY_SCORE,
            best_fold_metrics=[],
            best_breakdown={},
        )
    fold_metrics = study.best_trial.user_attrs.get("fold_metrics", [])
    breakdown = study.best_trial.user_attrs.get("score_breakdown", {})
    return HaCrossV2StudyArtifacts(
        study=study,
        best_params=dict(study.best_params),
        best_value=float(study.best_value),
        best_fold_metrics=fold_metrics,
        best_breakdown=breakdown,
    )


def format_study_report(art: HaCrossV2StudyArtifacts, top_k: int = 10) -> str:
    lines: list[str] = []
    s = art.study
    lines.append(f"=== HA-Cross v2 Optuna study: {s.study_name} ===")
    lines.append(f"Trials completos: {len(s.trials)}")
    if art.best_value <= PENALTY_SCORE:
        lines.append("No se encontró ninguna configuración válida.")
        return "\n".join(lines)
    lines.append(f"Mejor score compuesto: {art.best_value:+.4f}")
    if art.best_breakdown:
        b = art.best_breakdown
        lines.append(
            f"  desglose: sharpe={b.get('sharpe_mean', 0):+.3f}±{b.get('sharpe_std', 0):.3f}  "
            f"whipsaw_pen={b.get('whipsaw_penalty', 0):.3f}  "
            f"dd_pen={b.get('drawdown_penalty', 0):.3f}  "
            f"win_bonus={b.get('win_rate_bonus', 0):+.3f}"
        )
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
                f"  #{i:>2} score={t.value:+.4f}  tf={p.get('timeframe', '?'):>3}  "
                f"fast={p['fast_window']:>2}  slow={p['slow_window']:>3}  "
                f"hold={p['min_holding_bars']:>2}  conf={p['confirmation_bars']}  "
                f"green={p['ha_color_confirm']}  "
                f"x_str={p['min_cross_strength_pct']:.4f}  "
                f"rolling={p.get('use_rolling_percentiles', False)}"
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
    "DEFAULT_TIMEFRAMES",
    "CompositeScoreWeights",
    "HaCrossV2SearchSpace",
    "HaCrossV2StudyArtifacts",
    "StabilityConfig",
    "composite_score",
    "format_study_report",
    "make_objective",
    "run_study",
    "walk_forward_eval",
]
