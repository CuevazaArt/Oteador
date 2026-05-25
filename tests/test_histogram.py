from __future__ import annotations

import numpy as np
import pytest

from oteador.features.histogram import DEFAULT_PERCENTILES, RatioHistogram


def test_from_array_matches_normal_distribution_statistics() -> None:
    rng = np.random.default_rng(42)
    x = rng.standard_normal(50_000)
    h = RatioHistogram.from_array(x)
    assert h.count == 50_000
    assert h.mean == pytest.approx(0.0, abs=0.02)
    assert h.std == pytest.approx(1.0, abs=0.02)
    assert h.skew == pytest.approx(0.0, abs=0.05)
    assert h.kurtosis_excess == pytest.approx(0.0, abs=0.1)
    assert h.percentiles[50.0] == pytest.approx(0.0, abs=0.02)
    assert h.percentiles[5.0] == pytest.approx(-1.645, abs=0.05)
    assert h.percentiles[95.0] == pytest.approx(1.645, abs=0.05)


def test_default_percentiles_are_present() -> None:
    x = np.linspace(-1, 1, 1000)
    h = RatioHistogram.from_array(x)
    assert set(h.percentiles) == set(DEFAULT_PERCENTILES)


def test_ignores_nans() -> None:
    x = np.array([1.0, np.nan, 2.0, np.nan, 3.0])
    h = RatioHistogram.from_array(x)
    assert h.count == 3
    assert h.min == 1.0
    assert h.max == 3.0


def test_rejects_too_few_samples() -> None:
    with pytest.raises(ValueError, match=">=2 muestras"):
        RatioHistogram.from_array(np.array([np.nan, 1.0]))
