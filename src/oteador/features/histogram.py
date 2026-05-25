"""Histograma empírico del ratio (close - MA) / MA.

Versión offline para la etapa de caracterización: percentiles, momentos y
asimetría de la distribución completa. El estimador online (t-digest sobre
ventana deslizante) llega cuando arranquemos el supervisor en tiempo real.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Self

import numpy as np
import numpy.typing as npt

DEFAULT_PERCENTILES: tuple[float, ...] = (1.0, 5.0, 10.0, 25.0, 50.0, 75.0, 90.0, 95.0, 99.0)


@dataclass(frozen=True)
class RatioHistogram:
    count: int
    mean: float
    std: float
    skew: float
    kurtosis_excess: float
    min: float
    max: float
    percentiles: dict[float, float]

    @classmethod
    def from_array(
        cls,
        values: npt.NDArray[np.floating],
        percentiles: Sequence[float] = DEFAULT_PERCENTILES,
    ) -> Self:
        clean = values[~np.isnan(values)].astype(np.float64, copy=False)
        if clean.size < 2:
            raise ValueError(f"se necesitan >=2 muestras no-NaN, hay {clean.size}")
        mean = float(clean.mean())
        std = float(clean.std(ddof=1))
        if std == 0.0:
            skew = 0.0
            kurt_excess = 0.0
        else:
            z = (clean - mean) / std
            skew = float((z**3).mean())
            kurt_excess = float((z**4).mean()) - 3.0
        pct_array = np.percentile(clean, percentiles)
        return cls(
            count=int(clean.size),
            mean=mean,
            std=std,
            skew=skew,
            kurtosis_excess=kurt_excess,
            min=float(clean.min()),
            max=float(clean.max()),
            percentiles={float(p): float(v) for p, v in zip(percentiles, pct_array, strict=True)},
        )
