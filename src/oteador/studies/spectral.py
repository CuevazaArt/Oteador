"""Caracterización espectral del residuo close - MA via Welch.

La estimación PSD por Welch (segmentos + ventana + promedio) tiene mucha menos
varianza que un único periodograma, lo cual importa para identificar el modo
oscilatorio dominante de un símbolo a partir de una serie ruidosa.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import signal


@dataclass(frozen=True)
class SpectralReport:
    sampling_period_seconds: float
    dominant_period_seconds: float
    dominant_period_samples: float
    dominant_power_fraction: float
    top_modes: list[tuple[float, float]]


def welch_psd(
    series: npt.NDArray[np.floating],
    fs: float,
    nperseg: int | None = None,
) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """PSD via Welch. fs en Hz (=1/sampling_period_seconds)."""
    clean = series[~np.isnan(series)].astype(np.float64, copy=False)
    if clean.size < 16:
        raise ValueError(f"se necesitan >=16 muestras no-NaN, hay {clean.size}")
    if nperseg is None:
        nperseg = int(min(clean.size, 1024))
    freqs, psd = signal.welch(clean, fs=fs, nperseg=nperseg, detrend="constant")
    return freqs, psd


def characterize(
    series: npt.NDArray[np.floating],
    sampling_period_seconds: float,
    top_k: int = 5,
    nperseg: int | None = None,
) -> SpectralReport:
    """Reporta el periodo de oscilación dominante y los top-k modos."""
    if sampling_period_seconds <= 0:
        raise ValueError(
            f"sampling_period_seconds debe ser > 0, recibido {sampling_period_seconds}"
        )
    if top_k < 1:
        raise ValueError(f"top_k debe ser >= 1, recibido {top_k}")
    fs = 1.0 / sampling_period_seconds
    freqs, psd = welch_psd(series, fs, nperseg=nperseg)
    mask = freqs > 0
    freqs_pos = freqs[mask]
    psd_pos = psd[mask]
    if psd_pos.size == 0:
        raise ValueError("no hay componentes en frecuencia positiva tras enmascarar DC")
    total_power = float(psd_pos.sum())
    order = np.argsort(psd_pos)[::-1][: min(top_k, psd_pos.size)]
    top_modes: list[tuple[float, float]] = [
        (float(1.0 / freqs_pos[i]), float(psd_pos[i] / total_power)) for i in order
    ]
    dominant_period_s = float(1.0 / freqs_pos[order[0]])
    return SpectralReport(
        sampling_period_seconds=sampling_period_seconds,
        dominant_period_seconds=dominant_period_s,
        dominant_period_samples=dominant_period_s / sampling_period_seconds,
        dominant_power_fraction=float(psd_pos[order[0]] / total_power),
        top_modes=top_modes,
    )
