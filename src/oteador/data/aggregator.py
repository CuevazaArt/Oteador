"""Agregación de klines 1m a timeframes superiores (15m, 30m, 1h, 2h, 4h, ...).

Permite descargar el histórico una sola vez en 1m y construir múltiples TFs
in-memory para barridos multi-timeframe. Más barato que volver a descargar
una serie distinta de Binance Vision por cada TF.

Reglas de agregación estándar OHLCV:

    open   = first      high  = max     low   = min     close = last
    volume = sum        trades = sum    quote_volume = sum
    taker_buy_base_volume = sum         taker_buy_quote_volume = sum

El bucket se obtiene como ``(open_time_ms // period_ms) * period_ms`` —
alineado al epoch. Esto coincide con la malla de Binance para los TFs
estándar (15m, 30m, 1h, 2h, 4h, ...).
"""

from __future__ import annotations

import polars as pl

INTERVAL_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}


def interval_to_ms(interval: str) -> int:
    """Devuelve el período del kline en milisegundos."""
    try:
        return INTERVAL_MS[interval]
    except KeyError as exc:
        raise ValueError(
            f"intervalo desconocido {interval!r}; usa uno de {sorted(INTERVAL_MS)}"
        ) from exc


def detect_timestamp_unit_factor(df: pl.DataFrame, source_interval: str) -> int:
    """Detecta el multiplicador para pasar de ms a la unidad real de ``open_time``.

    Binance Vision cambió en 2025 la resolución del campo ``open_time`` de
    milisegundos a microsegundos en algunos endpoints. Esta función infiere
    la unidad a partir de la diferencia entre dos klines consecutivas y
    devuelve:

    - ``1``    si ``open_time`` está en milisegundos (formato histórico).
    - ``1000`` si ``open_time`` está en microsegundos (formato actual).

    Lanza ``ValueError`` si la delta observada no coincide con ninguno
    de los dos casos.
    """
    if df.height < 2:
        return 1
    src_ms = interval_to_ms(source_interval)
    ot = df["open_time"].sort()
    delta = int(ot[1] - ot[0])
    if delta == src_ms:
        return 1
    if delta == src_ms * 1000:
        return 1000
    raise ValueError(
        f"delta entre klines = {delta}; no coincide ni con {src_ms}ms ni con "
        f"{src_ms * 1000}µs para source_interval={source_interval!r}"
    )


def aggregate_klines(
    df: pl.DataFrame,
    source_interval: str,
    target_interval: str,
    drop_incomplete_buckets: bool = True,
) -> pl.DataFrame:
    """Agrega klines de ``source_interval`` a ``target_interval``.

    Args:
        df: DataFrame con el esquema estándar de Binance Vision klines.
            Debe contener al menos ``open_time``, ``open``, ``high``, ``low``,
            ``close``, ``volume``.
        source_interval: TF del df de entrada (e.g., ``"1m"``).
        target_interval: TF destino (e.g., ``"1h"``). Debe ser múltiplo entero
            del source.
        drop_incomplete_buckets: si True, descarta el último bucket si tiene
            menos barras que las esperadas. Evita un cierre incompleto al final.

    Returns:
        DataFrame agregado, ordenado por ``open_time`` ascendente.
    """
    src_ms = interval_to_ms(source_interval)
    tgt_ms = interval_to_ms(target_interval)
    if tgt_ms < src_ms:
        raise ValueError(
            f"target_interval ({target_interval}) debe ser >= source_interval ({source_interval})"
        )
    if tgt_ms % src_ms != 0:
        raise ValueError(
            f"target_interval ({target_interval}={tgt_ms}ms) no es múltiplo de "
            f"source_interval ({source_interval}={src_ms}ms)"
        )
    if tgt_ms == src_ms:
        return df.sort("open_time")

    required = {"open_time", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"faltan columnas requeridas: {sorted(missing)}")

    unit_factor = detect_timestamp_unit_factor(df, source_interval)
    tgt_unit = tgt_ms * unit_factor
    bars_per_bucket = tgt_ms // src_ms

    sorted_df = df.sort("open_time")
    bucketed = sorted_df.with_columns(
        ((pl.col("open_time") // tgt_unit) * tgt_unit).alias("_bucket")
    )

    agg_exprs: list[pl.Expr] = [
        pl.col("open").first().alias("open"),
        pl.col("high").max().alias("high"),
        pl.col("low").min().alias("low"),
        pl.col("close").last().alias("close"),
        pl.col("volume").sum().alias("volume"),
        pl.len().alias("_n_bars"),
    ]
    optional_sum_cols = (
        "quote_volume",
        "trades",
        "taker_buy_base_volume",
        "taker_buy_quote_volume",
    )
    for col in optional_sum_cols:
        if col in df.columns:
            agg_exprs.append(pl.col(col).sum().alias(col))
    if "close_time" in df.columns:
        agg_exprs.append(pl.col("close_time").last().alias("close_time"))
    if "ignore" in df.columns:
        agg_exprs.append(pl.col("ignore").first().alias("ignore"))

    grouped = (
        bucketed.group_by("_bucket", maintain_order=True)
        .agg(agg_exprs)
        .rename({"_bucket": "open_time"})
        .sort("open_time")
    )

    if drop_incomplete_buckets:
        grouped = grouped.filter(pl.col("_n_bars") == bars_per_bucket)

    return grouped.drop("_n_bars")


__all__ = [
    "INTERVAL_MS",
    "aggregate_klines",
    "detect_timestamp_unit_factor",
    "interval_to_ms",
]
