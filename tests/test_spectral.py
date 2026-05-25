from __future__ import annotations

import numpy as np
import pytest

from oteador.studies.spectral import characterize, welch_psd


def test_characterize_recovers_known_period() -> None:
    # Señal sintética: seno de periodo 120s muestreado cada 1s.
    period_s = 120.0
    fs = 1.0
    t = np.arange(4096) / fs
    rng = np.random.default_rng(0)
    signal_series = np.sin(2 * np.pi * t / period_s) + 0.05 * rng.standard_normal(t.size)

    report = characterize(signal_series, sampling_period_seconds=1.0, top_k=3, nperseg=512)
    # Debe estar dentro del bin de frecuencia de Welch (resolución ≈ fs/nperseg).
    assert report.dominant_period_seconds == pytest.approx(period_s, rel=0.1)
    assert 0.0 < report.dominant_power_fraction <= 1.0
    assert len(report.top_modes) == 3


def test_welch_psd_rejects_too_short() -> None:
    with pytest.raises(ValueError, match=">=16 muestras"):
        welch_psd(np.zeros(4), fs=1.0)


def test_characterize_rejects_bad_inputs() -> None:
    s = np.random.default_rng(0).standard_normal(256)
    with pytest.raises(ValueError):
        characterize(s, sampling_period_seconds=0.0)
    with pytest.raises(ValueError):
        characterize(s, sampling_period_seconds=1.0, top_k=0)
