from __future__ import annotations

import numpy as np
import pytest

from oteador.studies.regime import (
    MEAN_REVERTING,
    NEUTRAL,
    TRENDING,
    classify_regime,
    hurst_rs,
    log_returns,
)


def test_log_returns_basic() -> None:
    closes = np.array([100.0, 110.0, 99.0, 99.0])
    r = log_returns(closes)
    assert r.shape == (3,)
    assert r[0] == pytest.approx(np.log(110 / 100))
    assert r[2] == pytest.approx(0.0)


def test_log_returns_rejects_short_or_nonpositive() -> None:
    with pytest.raises(ValueError):
        log_returns(np.array([100.0]))
    with pytest.raises(ValueError):
        log_returns(np.array([100.0, 0.0]))


def test_hurst_of_white_noise_is_near_half() -> None:
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(8192)
    h = hurst_rs(noise)
    assert h == pytest.approx(0.5, abs=0.07)


def test_hurst_of_trending_returns_is_above_half() -> None:
    # AR(1) con phi > 0 → persistencia positiva → H > 0.5
    rng = np.random.default_rng(7)
    n = 8192
    phi = 0.7
    eps = rng.standard_normal(n) * 0.01
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    h = hurst_rs(x)
    assert h > 0.55


def test_hurst_of_antipersistent_returns_is_below_half() -> None:
    # AR(1) con phi < 0 → anti-persistencia → H < 0.5
    rng = np.random.default_rng(11)
    n = 8192
    phi = -0.7
    eps = rng.standard_normal(n) * 0.01
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + eps[i]
    h = hurst_rs(x)
    assert h < 0.45


def test_hurst_rejects_too_short() -> None:
    with pytest.raises(ValueError, match=">= 100 muestras"):
        hurst_rs(np.zeros(50))


def test_classify_regime_thresholds() -> None:
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(4096)
    report = classify_regime(noise)
    assert report.classification in {MEAN_REVERTING, NEUTRAL, TRENDING}
    assert report.samples == 4096
    assert 0.0 < report.mean_reverting_threshold < report.trending_threshold < 1.0


def test_classify_regime_neutral_for_white_noise() -> None:
    rng = np.random.default_rng(123)
    noise = rng.standard_normal(8192)
    report = classify_regime(noise)
    # White noise debe caer en NEUTRAL con los umbrales por defecto (0.45 / 0.55).
    assert report.classification == NEUTRAL


def test_classify_regime_rejects_bad_thresholds() -> None:
    rng = np.random.default_rng(0)
    noise = rng.standard_normal(1024)
    with pytest.raises(ValueError):
        classify_regime(noise, mean_reverting_below=0.6, trending_above=0.55)
