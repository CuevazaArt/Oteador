from __future__ import annotations

import polars as pl
import pytest

from oteador.features.moving_average import price_ma_ratio, residual, sma


def test_sma_window_3() -> None:
    s = pl.Series("close", [1.0, 2.0, 3.0, 4.0, 5.0])
    out = sma(s, 3)
    # rolling_mean en polars deja null para los primeros window-1
    assert out.to_list()[:2] == [None, None]
    assert out.to_list()[2:] == [2.0, 3.0, 4.0]


def test_sma_invalid_window() -> None:
    with pytest.raises(ValueError):
        sma(pl.Series("x", [1.0, 2.0]), 0)


def test_residual_and_ratio() -> None:
    close = pl.Series("c", [110.0, 100.0, 90.0])
    ma = pl.Series("ma", [100.0, 100.0, 100.0])
    assert residual(close, ma).to_list() == [10.0, 0.0, -10.0]
    assert price_ma_ratio(close, ma).to_list() == [0.1, 0.0, -0.1]
