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
    ma_window: int = typer.Option(6, "--ma", min=1, help="Periodo de la SMA principal"),
    ha_fast: int = typer.Option(6, "--ha-fast", min=1, help="MA rápida sobre HA_Open"),
    ha_slow: int = typer.Option(18, "--ha-slow", min=2, help="MA lenta sobre HA_Open"),
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
        ha_fast_window=ha_fast,
        ha_slow_window=ha_slow,
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


@app.command(name="ha-study")
def ha_study(
    symbol: str = typer.Argument(..., help="Símbolo Binance Spot, p.ej. BTCUSDT"),
    interval: str = typer.Option("1h", "--interval", "-i", help="Intervalo (1h+ recomendado)"),
    months: int = typer.Option(12, "--months", "-m", min=1),
    n_trials: int = typer.Option(100, "--n-trials", min=1),
    n_folds: int = typer.Option(5, "--n-folds", min=2),
    min_trades_per_fold: int = typer.Option(3, "--min-trades", min=1),
    fee_pct: float = typer.Option(0.001, "--fee"),
    fast_max: int = typer.Option(50, "--fast-max", min=2, help="Máximo fast_window"),
    slow_max: int = typer.Option(200, "--slow-max", min=3, help="Máximo slow_window"),
    holding_max: int = typer.Option(100, "--holding-max", min=1, help="Máximo min_holding_bars"),
    seed: int = typer.Option(42, "--seed"),
    cache_dir: Path = typer.Option(Path("data/vision"), "--cache-dir"),
    storage_dir: Path = typer.Option(Path("optuna_studies"), "--storage-dir"),
    fresh: bool = typer.Option(False, "--fresh", help="Ignorar storage previo"),
) -> None:
    """Optuna walk-forward sobre HA-cross (trend-following por cruce de MMs en HA_Open)."""
    from oteador.data.vision import VisionDownloader, months_back
    from oteador.studies.characterization import interval_to_seconds
    from oteador.studies.ha_cross_study import format_study_report, run_study
    from oteador.studies.optuna_study import WalkForwardConfig

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
        storage_path = storage_dir / f"ha-{symbol.upper()}-{interval}.db"
        study_name = f"ha-{symbol.upper()}-{interval}-{months}m"

    typer.echo(
        f"[2/3] {n_trials} trials, {n_folds} folds walk-forward "
        f"(min {min_trades_per_fold} trades/fold); "
        f"fast 1-{fast_max} x slow 2-{slow_max} x hold 1-{holding_max} ..."
    )
    art = run_study(
        df=df,
        sampling_period_seconds=sampling_period_s,
        n_trials=n_trials,
        cv=WalkForwardConfig(n_folds=n_folds, min_trades_per_fold=min_trades_per_fold),
        fee_pct=fee_pct,
        fast_max=fast_max,
        slow_max=slow_max,
        holding_max=holding_max,
        seed=seed,
        study_name=study_name,
        storage_path=storage_path,
    )

    typer.echo("[3/3] Resultados:\n")
    typer.echo(format_study_report(art))


@app.command(name="ha-study-v2")
def ha_study_v2(
    symbol: str = typer.Argument(..., help="Símbolo Binance Spot, p.ej. BTCUSDT"),
    months: int = typer.Option(12, "--months", "-m", min=1, help="Meses de 1m a descargar"),
    n_trials: int = typer.Option(200, "--n-trials", min=1),
    n_folds: int = typer.Option(6, "--n-folds", min=2),
    min_trades_per_fold: int = typer.Option(10, "--min-trades", min=1),
    min_positive_folds: int = typer.Option(
        0, "--min-positive-folds", min=0, help="Folds OOS con sharpe>0 mínimos"
    ),
    fee_pct: float = typer.Option(0.001, "--fee"),
    timeframes: str = typer.Option(
        "15m,30m,1h,2h,4h",
        "--timeframes",
        help="Lista CSV de TFs candidatos para que Optuna escoja",
    ),
    fast_max: int = typer.Option(10, "--fast-max", min=2),
    slow_max: int = typer.Option(30, "--slow-max", min=3),
    holding_max: int = typer.Option(20, "--holding-max", min=1),
    confirmation_max: int = typer.Option(5, "--confirmation-max", min=1),
    whipsaw_weight: float = typer.Option(1.0, "--w-whipsaw", min=0.0),
    drawdown_weight: float = typer.Option(1.0, "--w-dd", min=0.0),
    win_rate_weight: float = typer.Option(1.0, "--w-win", min=0.0),
    seed: int = typer.Option(42, "--seed"),
    cache_dir: Path = typer.Option(Path("data/vision"), "--cache-dir"),
    storage_dir: Path = typer.Option(Path("optuna_studies"), "--storage-dir"),
    fresh: bool = typer.Option(False, "--fresh", help="Ignorar storage previo"),
) -> None:
    """Optuna walk-forward HA-Cross v2: multi-TF + filtros anti-whipsaw + distancia a la MM."""
    from oteador.data.vision import VisionDownloader, months_back
    from oteador.studies.ha_cross_v2_study import (
        CompositeScoreWeights,
        HaCrossV2SearchSpace,
        StabilityConfig,
        format_study_report,
        run_study,
    )
    from oteador.studies.optuna_study import WalkForwardConfig

    tfs = tuple(t.strip() for t in timeframes.split(",") if t.strip())
    if not tfs:
        raise typer.BadParameter("--timeframes vacío")

    typer.echo(f"[1/3] Descargando {months} meses de {symbol} @ 1m ...")
    with VisionDownloader(cache_dir=cache_dir) as dl:
        df_1m = dl.load_months(symbol, "1m", months_back(months))
    typer.echo(f"      {df_1m.height:,} velas 1m base")

    space = HaCrossV2SearchSpace(
        timeframes=tfs,
        fast_max=fast_max,
        slow_max=slow_max,
        holding_max=holding_max,
        confirmation_max=confirmation_max,
    )
    weights = CompositeScoreWeights(
        whipsaw=whipsaw_weight,
        drawdown=drawdown_weight,
        win_rate=win_rate_weight,
    )
    stability = StabilityConfig(min_positive_folds=min_positive_folds)
    cv = WalkForwardConfig(n_folds=n_folds, min_trades_per_fold=min_trades_per_fold)

    storage_path: Path | None
    if fresh:
        storage_path = None
        study_name = None
    else:
        storage_path = storage_dir / f"ha-v2-{symbol.upper()}-{months}m.db"
        study_name = f"ha-v2-{symbol.upper()}-{months}m"

    typer.echo(
        f"[2/3] {n_trials} trials, {n_folds} folds walk-forward (min {min_trades_per_fold} "
        f"trades/fold); TFs candidatos: {','.join(tfs)} ..."
    )
    art = run_study(
        df_1m=df_1m,
        n_trials=n_trials,
        cv=cv,
        fee_pct=fee_pct,
        space=space,
        weights=weights,
        stability=stability,
        seed=seed,
        study_name=study_name,
        storage_path=storage_path,
    )

    typer.echo("[3/3] Resultados:\n")
    typer.echo(format_study_report(art))


if __name__ == "__main__":
    app()
