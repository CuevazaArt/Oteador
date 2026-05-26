"""HA-Cross v2 — Trend-following con filtros anti-whipsaw y confirmación por distancia a la MM.

Mejoras frente a :mod:`oteador.strategies.ha_cross`:

1. **Fuerza mínima del cruce** (``min_cross_strength_pct``): rechaza cruces cuya
   separación relativa ``(fast - slow) / slow`` no supere un umbral. Los cruces
   débiles son los que más frecuentemente revierten.
2. **Barras de confirmación** (``confirmation_bars``): exige N barras
   consecutivas con el cruce en el mismo sentido antes de entrar. Filtra
   whipsaws de 1-2 barras.
3. **Confirmación por color HA** (``ha_color_confirm``): exige N velas HA
   verdes consecutivas antes de entrar LONG.
4. **Distancia máxima al MM lento en entrada** (``entry_max_distance_pct``):
   no entrar si el precio ya está demasiado lejos por encima del slow_MA
   (anti-chase).
5. **Distancia mínima al MM lento en salida** (``exit_min_distance_pct``):
   forzar salida con beneficio cuando el precio se extiende mucho por encima
   del slow_MA, incluso si el cruce sigue alcista.
6. **Percentiles rodantes opcionales** (``use_rolling_percentiles``): si está
   activo, los umbrales de distancia se interpretan como percentiles (en
   tanto por uno) sobre una ventana rodante ``lookback_bars`` del ratio
   ``(close - slow_MA) / slow_MA``. Eso adapta los filtros a la volatilidad
   local.

Spot only: no se opera SHORT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import polars as pl

from oteador.backtest import BacktestMetrics, compute_metrics
from oteador.features.heikin_ashi import heikin_ashi

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True)
class HaCrossV2Params:
    """Parámetros completos de la estrategia HA-cross v2.

    Los valores por defecto reproducen el comportamiento clásico de
    :mod:`oteador.strategies.ha_cross` (filtros desactivados), lo que permite
    medir el impacto incremental de cada filtro.
    """

    fast_window: int
    slow_window: int
    min_holding_bars: int = 1
    # Anti-whipsaw
    min_cross_strength_pct: float = 0.0
    confirmation_bars: int = 1
    ha_color_confirm: int = 0
    # Filtros de distancia al slow_MA
    entry_max_distance_pct: float = float("inf")
    exit_min_distance_pct: float = float("inf")
    use_rolling_percentiles: bool = False
    lookback_bars: int = 500
    # Económicos
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
        if self.min_cross_strength_pct < 0.0:
            raise ValueError(
                f"min_cross_strength_pct debe ser >= 0, recibido {self.min_cross_strength_pct}"
            )
        if self.confirmation_bars < 1:
            raise ValueError(f"confirmation_bars debe ser >= 1, recibido {self.confirmation_bars}")
        if self.ha_color_confirm < 0:
            raise ValueError(f"ha_color_confirm debe ser >= 0, recibido {self.ha_color_confirm}")
        if self.use_rolling_percentiles:
            if not (0.0 < self.entry_max_distance_pct <= 1.0):
                raise ValueError(
                    "con percentiles rodantes, entry_max_distance_pct ∈ (0, 1]; "
                    f"recibido {self.entry_max_distance_pct}"
                )
            if not (0.0 < self.exit_min_distance_pct <= 1.0):
                raise ValueError(
                    "con percentiles rodantes, exit_min_distance_pct ∈ (0, 1]; "
                    f"recibido {self.exit_min_distance_pct}"
                )
            if self.lookback_bars < 32:
                raise ValueError(f"lookback_bars debe ser >= 32, recibido {self.lookback_bars}")
        else:
            if self.entry_max_distance_pct <= 0.0:
                raise ValueError(
                    f"entry_max_distance_pct debe ser > 0, recibido {self.entry_max_distance_pct}"
                )
            if self.exit_min_distance_pct <= 0.0:
                raise ValueError(
                    f"exit_min_distance_pct debe ser > 0, recibido {self.exit_min_distance_pct}"
                )
        if not (0.0 <= self.fee_pct < 0.05):
            raise ValueError(f"fee_pct fuera de rango razonable: {self.fee_pct}")


@dataclass(frozen=True)
class SignalArrays:
    """Bundle de arrays alineados con el df, listos para ``simulate``."""

    close: FloatArray
    fast: FloatArray
    slow: FloatArray
    ha_green: npt.NDArray[np.bool_]
    distance: FloatArray
    entry_threshold: FloatArray
    exit_threshold: FloatArray


def _rolling_quantile(arr: FloatArray, q: float, window: int) -> FloatArray:
    """Quantile rodante usando polars (más rápido y consistente con el resto)."""
    s = pl.Series("v", arr)
    min_samples = max(32, window // 4)
    return (
        s.rolling_quantile(q, window_size=window, min_samples=min_samples)
        .to_numpy()
        .astype(np.float64, copy=False)
    )


def compute_signals(
    df: pl.DataFrame,
    params: HaCrossV2Params,
    ha_open: pl.Series | None = None,
    ha_green: npt.NDArray[np.bool_] | None = None,
) -> SignalArrays:
    """Calcula todos los arrays necesarios para la simulación.

    Si se pasan ``ha_open`` y ``ha_green`` precomputados (e.g., calculados una
    sola vez fuera del bucle Optuna), se evita recomputar Heikin-Ashi por trial.
    """
    if ha_open is None or ha_green is None:
        df_ha = heikin_ashi(df)
        ha_open = df_ha["ha_open"]
        ha_green = (df_ha["ha_close"].to_numpy() > df_ha["ha_open"].to_numpy()).astype(bool)

    close_arr = df["close"].to_numpy().astype(np.float64, copy=False)
    fast_arr = (
        ha_open.rolling_mean(window_size=params.fast_window)
        .to_numpy()
        .astype(np.float64, copy=False)
    )
    slow_arr = (
        ha_open.rolling_mean(window_size=params.slow_window)
        .to_numpy()
        .astype(np.float64, copy=False)
    )

    with np.errstate(invalid="ignore", divide="ignore"):
        distance = np.where(
            np.isfinite(slow_arr) & (slow_arr > 0.0), (close_arr - slow_arr) / slow_arr, np.nan
        )

    n = close_arr.size
    if params.use_rolling_percentiles:
        entry_threshold = _rolling_quantile(
            distance, params.entry_max_distance_pct, params.lookback_bars
        )
        exit_threshold = _rolling_quantile(
            distance, params.exit_min_distance_pct, params.lookback_bars
        )
    else:
        entry_threshold = np.full(n, params.entry_max_distance_pct, dtype=np.float64)
        exit_threshold = np.full(n, params.exit_min_distance_pct, dtype=np.float64)

    return SignalArrays(
        close=close_arr,
        fast=fast_arr,
        slow=slow_arr,
        ha_green=ha_green,
        distance=distance.astype(np.float64, copy=False),
        entry_threshold=entry_threshold,
        exit_threshold=exit_threshold,
    )


def simulate(
    sig: SignalArrays,
    params: HaCrossV2Params,
    force_close_at_end: bool = True,
) -> tuple[FloatArray, list[dict[str, Any]]]:
    """Loop bar-por-bar con todos los filtros aplicados."""
    close = sig.close
    fast = sig.fast
    slow = sig.slow
    ha_green = sig.ha_green
    distance = sig.distance
    entry_threshold = sig.entry_threshold
    exit_threshold = sig.exit_threshold

    n = int(close.size)
    if not (fast.size == slow.size == ha_green.size == distance.size == n):
        raise ValueError("todos los arrays deben tener la misma longitud")

    equity = np.ones(n, dtype=np.float64)
    eq = 1.0
    pos_open = False
    entry_price = 0.0
    entry_bar = 0
    trades: list[dict[str, Any]] = []
    fee_round_trip = (1.0 - params.fee_pct) ** 2

    up_streak = 0
    green_streak = 0

    for i in range(n):
        c = float(close[i])
        if pos_open:
            equity[i] = eq * c / entry_price * fee_round_trip
        else:
            equity[i] = eq

        f = fast[i]
        s = slow[i]
        if not (np.isfinite(f) and np.isfinite(s) and s > 0.0):
            up_streak = 0
            green_streak = 0
            continue

        cross_strength = (f - s) / s
        is_up = cross_strength > 0.0
        strong_up = is_up and cross_strength >= params.min_cross_strength_pct

        if strong_up:
            up_streak += 1
        else:
            up_streak = 0

        if ha_green[i]:
            green_streak += 1
        else:
            green_streak = 0

        if not pos_open:
            # Condiciones de entrada
            if not strong_up:
                continue
            if up_streak < params.confirmation_bars:
                continue
            if params.ha_color_confirm > 0 and green_streak < params.ha_color_confirm:
                continue
            d = distance[i]
            d_thr = entry_threshold[i]
            if np.isfinite(d) and np.isfinite(d_thr) and d > d_thr:
                # Demasiado lejos por encima del slow_MA: anti-chase
                continue
            pos_open = True
            entry_price = c
            entry_bar = i
            continue

        # Posición abierta: lógica de salida
        holding = i - entry_bar
        if holding < params.min_holding_bars:
            continue

        cross_exit = not is_up
        d = distance[i]
        d_thr_exit = exit_threshold[i]
        distance_exit = (
            np.isfinite(d)
            and np.isfinite(d_thr_exit)
            and d >= d_thr_exit
            and c > entry_price  # sólo tomar profit si va a ganancia bruta
        )
        if not (cross_exit or distance_exit):
            continue

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
                "exit_reason": "distance" if distance_exit and not cross_exit else "cross",
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
                "exit_reason": "forced_end",
            }
        )

    return equity, trades


def backtest(
    df: pl.DataFrame,
    params: HaCrossV2Params,
    sampling_period_seconds: float,
) -> BacktestMetrics:
    """Atajo: compute_signals + simulate + compute_metrics."""
    sig = compute_signals(df, params)
    equity, trades = simulate(sig, params)
    return compute_metrics(equity, trades, sampling_period_seconds)


__all__ = [
    "HaCrossV2Params",
    "SignalArrays",
    "backtest",
    "compute_signals",
    "simulate",
]
