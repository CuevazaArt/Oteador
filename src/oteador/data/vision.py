"""Descarga de klines mensuales desde https://data.binance.vision.

Cada zip mensual contiene un único CSV con el esquema documentado en
https://github.com/binance/binance-public-data. Los CSV recientes traen
cabecera; los antiguos no. Detectamos ambos casos.
"""

from __future__ import annotations

import hashlib
import io
import zipfile
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from types import TracebackType
from typing import Self

import httpx
import polars as pl
import structlog

log = structlog.get_logger(__name__)

VISION_BASE = "https://data.binance.vision/data/spot/monthly/klines"

KLINE_COLUMNS: list[str] = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base_volume",
    "taker_buy_quote_volume",
    "ignore",
]

KLINE_SCHEMA: dict[str, pl.DataType] = {
    "open_time": pl.Int64(),
    "open": pl.Float64(),
    "high": pl.Float64(),
    "low": pl.Float64(),
    "close": pl.Float64(),
    "volume": pl.Float64(),
    "close_time": pl.Int64(),
    "quote_volume": pl.Float64(),
    "trades": pl.Int64(),
    "taker_buy_base_volume": pl.Float64(),
    "taker_buy_quote_volume": pl.Float64(),
    "ignore": pl.Int64(),
}


@dataclass(frozen=True)
class KlineFile:
    """Identifica un zip mensual concreto en Binance Vision."""

    symbol: str
    interval: str
    year: int
    month: int

    @property
    def stem(self) -> str:
        return f"{self.symbol.upper()}-{self.interval}-{self.year:04d}-{self.month:02d}"

    @property
    def url(self) -> str:
        return f"{VISION_BASE}/{self.symbol.upper()}/{self.interval}/{self.stem}.zip"

    @property
    def checksum_url(self) -> str:
        return f"{self.url}.CHECKSUM"


def months_back(n: int, today: date | None = None) -> list[tuple[int, int]]:
    """Devuelve los N meses completos más recientes como (año, mes), en orden cronológico.

    El mes actual se excluye porque Binance Vision sólo publica meses cerrados.
    """
    if n < 1:
        raise ValueError(f"n debe ser >= 1, recibido {n}")
    if today is None:
        today = date.today()
    year, month = today.year, today.month
    out: list[tuple[int, int]] = []
    for _ in range(n):
        month -= 1
        if month == 0:
            month = 12
            year -= 1
        out.append((year, month))
    return list(reversed(out))


def verify_checksum(zip_bytes: bytes, checksum_text: str) -> None:
    """Valida SHA256. El texto de .CHECKSUM tiene formato '<sha256>  <filename>'."""
    expected = checksum_text.split()[0].strip().lower()
    actual = hashlib.sha256(zip_bytes).hexdigest()
    if actual != expected:
        raise ValueError(f"checksum mismatch: got {actual}, expected {expected}")


def parse_kline_zip(zip_bytes: bytes) -> pl.DataFrame:
    """Extrae el CSV interno y lo carga a polars con esquema tipado."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        names = zf.namelist()
        if len(names) != 1:
            raise ValueError(f"se esperaba 1 archivo en el zip, hay {len(names)}: {names}")
        with zf.open(names[0]) as fp:
            data = fp.read()
    first_line = data[:200].split(b"\n", 1)[0].decode("utf-8", errors="replace")
    has_header = first_line.startswith("open_time")
    if has_header:
        df = pl.read_csv(io.BytesIO(data), schema_overrides=KLINE_SCHEMA)
    else:
        df = pl.read_csv(
            io.BytesIO(data),
            has_header=False,
            new_columns=KLINE_COLUMNS,
            schema_overrides=KLINE_SCHEMA,
        )
    return df.drop("ignore")


class VisionDownloader:
    """Descargador sincrónico con caché en disco para klines de Binance Vision."""

    def __init__(
        self,
        cache_dir: Path | str = "data/vision",
        client: httpx.Client | None = None,
        verify_checksums: bool = True,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = client or httpx.Client(timeout=60.0, follow_redirects=True)
        self._owns_client = client is None
        self._verify = verify_checksums

    def local_zip_path(self, kf: KlineFile) -> Path:
        return self.cache_dir / kf.symbol.upper() / kf.interval / f"{kf.stem}.zip"

    def fetch(self, kf: KlineFile) -> Path:
        """Devuelve la ruta local del zip, descargándolo si no está en caché."""
        local = self.local_zip_path(kf)
        if local.exists() and local.stat().st_size > 0:
            log.debug("vision.cache_hit", file=str(local))
            return local
        local.parent.mkdir(parents=True, exist_ok=True)
        log.info("vision.fetch", url=kf.url)
        resp = self._client.get(kf.url)
        resp.raise_for_status()
        zip_bytes = resp.content
        if self._verify:
            try:
                cs = self._client.get(kf.checksum_url)
                cs.raise_for_status()
                verify_checksum(zip_bytes, cs.text)
            except httpx.HTTPStatusError as exc:
                log.warning("vision.no_checksum", status=exc.response.status_code, file=kf.stem)
        local.write_bytes(zip_bytes)
        return local

    def load(self, kf: KlineFile) -> pl.DataFrame:
        path = self.fetch(kf)
        return parse_kline_zip(path.read_bytes())

    def load_months(
        self,
        symbol: str,
        interval: str,
        months: Iterable[tuple[int, int]],
    ) -> pl.DataFrame:
        frames = [
            self.load(KlineFile(symbol=symbol, interval=interval, year=y, month=m))
            for y, m in months
        ]
        if not frames:
            raise ValueError("la iterable de meses está vacía")
        return pl.concat(frames).sort("open_time").unique(subset=["open_time"], keep="first")

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
