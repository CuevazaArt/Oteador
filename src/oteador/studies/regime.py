"""Detección de régimen vía exponente de Hurst sobre log-returns.

Interpretación del Hurst sobre returns (no sobre el precio acumulado):

- H ≈ 0.5  → returns independientes; precio se comporta como random walk.
- H > 0.5  → returns con persistencia positiva; precio tiende a tendencia.
- H < 0.5  → returns con persistencia negativa (anti-persistencia); precio
             tiende a reversión a la media. Régimen donde la estrategia de
             percentiles del histograma tiene ventaja.

Hurst R/S es ruidoso en series cortas. Para 1m de BTCUSDT con ~128k muestras
la varianza de la estimación es razonable; para ventanas más pequeñas hay que
suavizar o promediar entre ejecuciones.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Regime = str
MEAN_REVERTING: Regime = "MEAN_REVERTING"
NEUTRAL: Regime = "NEUTRAL"
TRENDING: Regime = "TRENDING"


@dataclass(frozen=True)
class RegimeReport:
    hurst: float
    classification: Regime
    samples: int
    mean_reverting_threshold: float
    trending_threshold: float


def log_returns(close: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """log(close_t / close_{t-1}). Longitud = close.size - 1."""
    if close.size < 2:
        raise ValueError("se necesitan >= 2 closes")
    close64 = close.astype(np.float64, copy=False)
    if (close64 <= 0).any():
        raise ValueError("hay closes <= 0; log no está definido")
    return np.log(close64[1:] / close64[:-1])


def hurst_rs(
    series: npt.NDArray[np.floating],
    min_window: int = 16,
    max_window: int | None = None,
    num_windows: int = 24,
) -> float:
    """Exponente de Hurst por rescaled-range (Hurst-Mandelbrot).

    Aplicar a una serie estacionaria (log-returns), no al precio.
    Devuelve la pendiente de log(R/S) vs log(window).
    """
    if min_window < 8:
        raise ValueError(f"min_window debe ser >= 8, recibido {min_window}")
    clean = series[~np.isnan(series)].astype(np.float64, copy=False)
    n = clean.size
    if n < 100:
        raise ValueError(f"se necesitan >= 100 muestras, hay {n}")
    if max_window is None:
        max_window = n // 4
    if max_window <= min_window:
        raise ValueError(
            f"max_window ({max_window}) debe ser > min_window ({min_window}); "
            f"serie demasiado corta ({n} muestras)"
        )

    windows = np.unique(
        np.logspace(
            np.log10(min_window),
            np.log10(max_window),
            num=num_windows,
        ).astype(np.int64)
    )

    rs_means: list[float] = []
    for w in windows:
        w_int = int(w)
        k = n // w_int
        if k < 1:
            rs_means.append(np.nan)
            continue
        rs_chunk: list[float] = []
        for i in range(k):
            chunk = clean[i * w_int : (i + 1) * w_int]
            mean = chunk.mean()
            std = chunk.std(ddof=1)
            if std == 0.0:
                continue
            cum = np.cumsum(chunk - mean)
            rng = float(cum.max() - cum.min())
            rs_chunk.append(rng / float(std))
        rs_means.append(float(np.mean(rs_chunk)) if rs_chunk else np.nan)

    rs_arr = np.asarray(rs_means, dtype=np.float64)
    valid = np.isfinite(rs_arr) & (rs_arr > 0.0)
    if int(valid.sum()) < 4:
        raise ValueError("ventanas válidas insuficientes para la regresión de Hurst")

    log_w = np.log(windows[valid].astype(np.float64))
    log_rs = np.log(rs_arr[valid])
    slope = float(np.polyfit(log_w, log_rs, 1)[0])
    return slope


def classify_regime(
    returns: npt.NDArray[np.floating],
    mean_reverting_below: float = 0.45,
    trending_above: float = 0.55,
) -> RegimeReport:
    """Clasifica el régimen del símbolo a partir de log-returns."""
    if not (0.0 < mean_reverting_below < trending_above < 1.0):
        raise ValueError("los umbrales deben cumplir 0 < mean_reverting_below < trending_above < 1")
    clean = returns[~np.isnan(returns)]
    h = hurst_rs(clean)
    if h < mean_reverting_below:
        cls: Regime = MEAN_REVERTING
    elif h > trending_above:
        cls = TRENDING
    else:
        cls = NEUTRAL
    return RegimeReport(
        hurst=h,
        classification=cls,
        samples=int(clean.size),
        mean_reverting_threshold=mean_reverting_below,
        trending_threshold=trending_above,
    )
