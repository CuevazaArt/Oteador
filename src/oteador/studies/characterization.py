"""Orquesta la caracterización end-to-end: descarga → MA → espectro + histograma."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from oteador.data.vision import VisionDownloader, months_back
from oteador.features.histogram import RatioHistogram
from oteador.features.moving_average import price_ma_ratio, residual, sma
from oteador.studies.spectral import SpectralReport
from oteador.studies.spectral import characterize as spectral_characterize

INTERVAL_SECONDS: dict[str, float] = {
    "1s": 1,
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "2h": 7200,
    "4h": 14400,
    "6h": 21600,
    "8h": 28800,
    "12h": 43200,
    "1d": 86400,
}


def interval_to_seconds(interval: str) -> float:
    try:
        return INTERVAL_SECONDS[interval]
    except KeyError as exc:
        raise ValueError(
            f"intervalo desconocido {interval!r}; usa uno de {sorted(INTERVAL_SECONDS)}"
        ) from exc


@dataclass(frozen=True)
class CharacterizationReport:
    symbol: str
    interval: str
    months: int
    ma_window: int
    bars: int
    spectral: SpectralReport
    histogram: RatioHistogram


def run_characterization(
    symbol: str,
    interval: str,
    months: int,
    ma_window: int,
    cache_dir: Path | str = "data/vision",
    output_dir: Path | str | None = "data/characterization",
) -> CharacterizationReport:
    """Descarga histórico, calcula MA y residuo, y reporta espectro + histograma.

    Si output_dir no es None, guarda un parquet con close, ma y ratio.
    """
    sampling_period_s = interval_to_seconds(interval)
    with VisionDownloader(cache_dir=cache_dir) as dl:
        df = dl.load_months(symbol, interval, months_back(months))

    close = df["close"]
    ma = sma(close, ma_window)
    res = residual(close, ma)
    ratio = price_ma_ratio(close, ma)

    spec = spectral_characterize(res.to_numpy(), sampling_period_s)
    hist = RatioHistogram.from_array(ratio.to_numpy())

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        out_path = out / f"{symbol.upper()}-{interval}-ma{ma_window}.parquet"
        df.with_columns(
            [
                ma.alias(f"sma_{ma_window}"),
                res.alias("residual"),
                ratio.alias("ratio"),
            ]
        ).write_parquet(out_path)

    return CharacterizationReport(
        symbol=symbol.upper(),
        interval=interval,
        months=months,
        ma_window=ma_window,
        bars=int(df.height),
        spectral=spec,
        histogram=hist,
    )


def format_report(report: CharacterizationReport) -> str:
    """Devuelve el reporte formateado como texto plano para la CLI."""
    lines: list[str] = []
    lines.append(f"=== {report.symbol} @ {report.interval} ({report.months} meses) ===")
    lines.append(f"Velas analizadas: {report.bars:,}")
    lines.append(f"Ventana MA:       {report.ma_window}")
    lines.append("")
    lines.append("Espectro del residuo (close - MA):")
    spec = report.spectral
    lines.append(
        f"  periodo dominante: {spec.dominant_period_seconds:.1f}s "
        f"(≈ {spec.dominant_period_samples:.1f} velas)"
    )
    lines.append(f"  fracción de potencia del modo dominante: {spec.dominant_power_fraction:.2%}")
    lines.append("  top modos (periodo s, fracción potencia):")
    for period_s, frac in spec.top_modes:
        lines.append(f"    {period_s:>10.1f}s  {frac:>7.2%}")
    lines.append("")
    lines.append("Histograma del ratio (close - MA) / MA:")
    h = report.histogram
    lines.append(f"  n={h.count:,}  media={h.mean:+.6f}  std={h.std:.6f}")
    lines.append(f"  skew={h.skew:+.3f}  kurtosis_excess={h.kurtosis_excess:+.3f}")
    lines.append(f"  min={h.min:+.6f}  max={h.max:+.6f}")
    lines.append("  percentiles:")
    for p in sorted(h.percentiles):
        lines.append(f"    p{p:>5.1f} = {h.percentiles[p]:+.6f}")
    return "\n".join(lines)


__all__ = [
    "CharacterizationReport",
    "format_report",
    "interval_to_seconds",
    "run_characterization",
]
