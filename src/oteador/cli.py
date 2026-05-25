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


if __name__ == "__main__":
    app()
