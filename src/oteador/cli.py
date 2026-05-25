from __future__ import annotations

import sys
from pathlib import Path

import typer

for _stream in (sys.stdout, sys.stderr):
    reconfigure = getattr(_stream, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8", errors="replace")

app = typer.Typer(
    help="Oteador — supervisor de histogramas para trading adaptativo en Binance Spot.",
    no_args_is_help=True,
)


@app.callback()
def _main() -> None:
    """Punto de entrada raíz; fuerza a Typer a tratar los comandos como subcomandos."""


@app.command()
def version() -> None:
    """Muestra la versión instalada de Oteador."""
    from oteador import __version__

    typer.echo(__version__)


@app.command()
def characterize(
    symbol: str = typer.Argument(..., help="Símbolo Binance Spot, p.ej. BTCUSDT"),
    interval: str = typer.Option("1m", "--interval", "-i", help="Intervalo de kline"),
    months: int = typer.Option(3, "--months", "-m", min=1, help="Meses históricos a descargar"),
    ma_window: int = typer.Option(6, "--ma", min=1, help="Periodo de la SMA"),
    cache_dir: Path = typer.Option(Path("data/vision"), "--cache-dir"),
    output_dir: Path = typer.Option(Path("data/characterization"), "--out"),
    no_save: bool = typer.Option(False, "--no-save", help="No escribir parquet de salida"),
) -> None:
    """Descarga histórico de Binance Vision y reporta espectro + histograma del ratio."""
    from oteador.studies.characterization import format_report, run_characterization

    report = run_characterization(
        symbol=symbol,
        interval=interval,
        months=months,
        ma_window=ma_window,
        cache_dir=cache_dir,
        output_dir=None if no_save else output_dir,
    )
    typer.echo(format_report(report))
    if not no_save:
        out_path = output_dir / f"{symbol.upper()}-{interval}-ma{ma_window}.parquet"
        typer.echo(f"\nParquet: {out_path}")


@app.command()
def study(
    symbol: str = typer.Argument(..., help="Símbolo Binance Spot, p.ej. BTCUSDT"),
    interval: str = typer.Option("1m", "--interval", "-i"),
    months: int = typer.Option(6, "--months", "-m", min=1),
    n_trials: int = typer.Option(50, "--n-trials", min=1),
    n_folds: int = typer.Option(5, "--n-folds", min=2),
    min_trades_per_fold: int = typer.Option(5, "--min-trades", min=1),
    fee_pct: float = typer.Option(0.001, "--fee"),
    seed: int = typer.Option(42, "--seed"),
    cache_dir: Path = typer.Option(Path("data/vision"), "--cache-dir"),
    storage_dir: Path = typer.Option(Path("optuna_studies"), "--storage-dir"),
    fresh: bool = typer.Option(
        False, "--fresh", help="Ignorar storage previo y crear estudio nuevo"
    ),
) -> None:
    """Optuna walk-forward sobre parámetros de la estrategia de reversión por percentiles."""
    from oteador.data.vision import VisionDownloader, months_back
    from oteador.studies.characterization import interval_to_seconds
    from oteador.studies.optuna_study import (
        WalkForwardConfig,
        format_study_report,
        run_study,
    )

    sampling_period_s = interval_to_seconds(interval)

    typer.echo(f"[1/3] Descargando {months} meses de {symbol} @ {interval} ...")
    with VisionDownloader(cache_dir=cache_dir) as dl:
        df = dl.load_months(symbol, interval, months_back(months))
    typer.echo(f"      {df.height:,} velas")

    storage_path: Path | None
    if fresh:
        storage_path = None
        study_name = None
    else:
        storage_path = storage_dir / f"{symbol.upper()}-{interval}.db"
        study_name = f"{symbol.upper()}-{interval}-{months}m"

    typer.echo(
        f"[2/3] {n_trials} trials, {n_folds} folds walk-forward "
        f"(min {min_trades_per_fold} trades/fold) ..."
    )
    art = run_study(
        df=df,
        sampling_period_seconds=sampling_period_s,
        n_trials=n_trials,
        cv=WalkForwardConfig(n_folds=n_folds, min_trades_per_fold=min_trades_per_fold),
        fee_pct=fee_pct,
        seed=seed,
        study_name=study_name,
        storage_path=storage_path,
    )

    typer.echo("[3/3] Resultados:\n")
    typer.echo(format_study_report(art))


if __name__ == "__main__":
    app()
