from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from oteador.features.heikin_ashi import HeikinAshiReport, add_ha_ma, heikin_ashi


def _df_ohlc(o: list[float], h: list[float], low: list[float], c: list[float]) -> pl.DataFrame:
    return pl.DataFrame({"open": o, "high": h, "low": low, "close": c})


def test_heikin_ashi_manual_first_bars() -> None:
    df = _df_ohlc(
        o=[10.0, 12.0, 11.0],
        h=[14.0, 14.0, 13.0],
        low=[9.0, 11.0, 10.0],
        c=[12.0, 11.0, 12.0],
    )
    out = heikin_ashi(df)
    # HA_Close = (O+H+L+C)/4
    expected_close = [(10 + 14 + 9 + 12) / 4, (12 + 14 + 11 + 11) / 4, (11 + 13 + 10 + 12) / 4]
    assert out["ha_close"].to_list() == pytest.approx(expected_close)
    # HA_Open[0] = (O[0]+C[0])/2 = 11; HA_Open[i] = (prev_HA_Open + prev_HA_Close)/2
    assert out["ha_open"][0] == pytest.approx(11.0)
    assert out["ha_open"][1] == pytest.approx((11.0 + expected_close[0]) / 2)
    assert out["ha_open"][2] == pytest.approx((out["ha_open"][1] + expected_close[1]) / 2)
    # HA_High y HA_Low contienen a HA_Open y HA_Close
    for i in range(3):
        assert out["ha_high"][i] >= max(out["ha_open"][i], out["ha_close"][i])
        assert out["ha_low"][i] <= min(out["ha_open"][i], out["ha_close"][i])


def test_heikin_ashi_requires_ohlc_columns() -> None:
    df = pl.DataFrame({"close": [1.0, 2.0]})
    with pytest.raises(ValueError, match="faltan columnas"):
        heikin_ashi(df)


def test_add_ha_ma_requires_ha_close() -> None:
    df = pl.DataFrame({"open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0]})
    with pytest.raises(ValueError, match="ha_close"):
        add_ha_ma(df, window=3)


def test_add_ha_ma_basic() -> None:
    df = _df_ohlc(
        o=[10.0] * 5,
        h=[10.0] * 5,
        low=[10.0] * 5,
        c=[10.0] * 5,
    )
    out = add_ha_ma(heikin_ashi(df), window=3)
    # Con OHLC constante a 10, HA_Close = 10 ∀ → SMA también 10 después del warmup.
    assert out["sma_ha_close"].to_list()[2:] == pytest.approx([10.0, 10.0, 10.0])


def test_ha_report_on_trending_series() -> None:
    n = 200
    # Tendencia alcista pura: cada vela verde.
    o = np.linspace(100.0, 119.9, n)
    c = o + 0.5
    h = c + 0.1
    low = o - 0.1
    df = _df_ohlc(o.tolist(), h.tolist(), low.tolist(), c.tolist())
    df_ha = add_ha_ma(heikin_ashi(df), window=10)
    rep = HeikinAshiReport.from_df(df_ha, ma_window=10)
    assert rep.pct_green > 0.9
    assert rep.pct_red < 0.05
    assert rep.longest_green_streak > 100
    assert rep.pct_ha_above_ma > 0.5


def test_ha_report_on_constant_series() -> None:
    df = _df_ohlc([10.0] * 100, [10.0] * 100, [10.0] * 100, [10.0] * 100)
    df_ha = add_ha_ma(heikin_ashi(df), window=5)
    rep = HeikinAshiReport.from_df(df_ha, ma_window=5)
    # Todo doji (HA_Close == HA_Open == 10).
    assert rep.pct_doji == pytest.approx(1.0, abs=0.05)
    assert rep.mean_abs_distance_to_ma_pct == pytest.approx(0.0)
