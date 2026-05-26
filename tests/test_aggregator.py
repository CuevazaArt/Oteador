from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from oteador.data.aggregator import INTERVAL_MS, aggregate_klines, interval_to_ms


def _make_1m_df(n_bars: int, base_price: float = 100.0) -> pl.DataFrame:
    """Crea un df con n_bars klines 1m alineadas al epoch (open_time en ms)."""
    rng = np.random.default_rng(0)
    open_times = np.arange(n_bars, dtype=np.int64) * 60_000
    closes = base_price + np.cumsum(rng.standard_normal(n_bars) * 0.1)
    opens = np.concatenate([[base_price], closes[:-1]])
    highs = np.maximum(opens, closes) + 0.05
    lows = np.minimum(opens, closes) - 0.05
    volumes = rng.uniform(1.0, 10.0, n_bars)
    return pl.DataFrame(
        {
            "open_time": open_times,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
            "close_time": open_times + 59_999,
            "quote_volume": volumes * closes,
            "trades": np.full(n_bars, 5, dtype=np.int64),
            "taker_buy_base_volume": volumes * 0.5,
            "taker_buy_quote_volume": volumes * closes * 0.5,
            "ignore": np.zeros(n_bars, dtype=np.int64),
        }
    )


def test_interval_to_ms_known() -> None:
    assert interval_to_ms("1m") == 60_000
    assert interval_to_ms("1h") == 3_600_000
    assert interval_to_ms("4h") == 14_400_000


def test_interval_to_ms_unknown_raises() -> None:
    with pytest.raises(ValueError):
        interval_to_ms("7m")


def test_aggregate_1m_to_1h_preserves_ohlcv_semantics() -> None:
    df = _make_1m_df(180)  # 3h exactas
    out = aggregate_klines(df, "1m", "1h")
    assert out.height == 3
    # Open de cada bucket = open del primer minuto
    first_opens = df.filter(pl.col("open_time") % 3_600_000 == 0)["open"].to_numpy()
    assert np.allclose(out["open"].to_numpy(), first_opens)
    # Volume suma
    expected_volume_h0 = float(df["volume"].head(60).sum())
    assert out["volume"][0] == pytest.approx(expected_volume_h0)
    # High = max de las 60 barras
    expected_high_h0 = float(df["high"].head(60).max())
    assert out["high"][0] == pytest.approx(expected_high_h0)
    # Low = min de las 60 barras
    expected_low_h0 = float(df["low"].head(60).min())
    assert out["low"][0] == pytest.approx(expected_low_h0)
    # Close = close de la última barra del bucket
    assert out["close"][0] == pytest.approx(float(df["close"][59]))


def test_aggregate_same_tf_is_noop_modulo_sort() -> None:
    df = _make_1m_df(50)
    out = aggregate_klines(df, "1m", "1m")
    assert out.height == df.height
    assert np.array_equal(out["open_time"].to_numpy(), df["open_time"].to_numpy())


def test_aggregate_drops_incomplete_last_bucket() -> None:
    df = _make_1m_df(125)  # 2h completas + 5min sueltos
    out = aggregate_klines(df, "1m", "1h", drop_incomplete_buckets=True)
    assert out.height == 2  # los 5min residuales se descartan


def test_aggregate_keeps_incomplete_when_requested() -> None:
    df = _make_1m_df(125)
    out = aggregate_klines(df, "1m", "1h", drop_incomplete_buckets=False)
    assert out.height == 3


def test_aggregate_target_smaller_than_source_raises() -> None:
    df = _make_1m_df(60)
    with pytest.raises(ValueError, match=">= source_interval"):
        aggregate_klines(df, "1h", "1m")


def test_aggregate_non_multiple_raises() -> None:
    # 3m no divide 5m: 300_000 % 180_000 != 0
    df3m = aggregate_klines(_make_1m_df(60), "1m", "3m")
    with pytest.raises(ValueError, match="múltiplo"):
        aggregate_klines(df3m, "3m", "5m")


def test_aggregate_missing_columns_raises() -> None:
    df = pl.DataFrame({"open_time": [0, 60_000], "close": [1.0, 2.0]})
    with pytest.raises(ValueError, match="faltan columnas"):
        aggregate_klines(df, "1m", "1h")


def test_all_known_intervals_are_strictly_increasing() -> None:
    from itertools import pairwise

    sizes = sorted(INTERVAL_MS.values())
    assert sizes == sorted(set(sizes))  # unicidad
    assert all(b > a for a, b in pairwise(sizes))
