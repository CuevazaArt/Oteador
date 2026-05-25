from __future__ import annotations

import typer

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


if __name__ == "__main__":
    app()
