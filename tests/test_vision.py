from __future__ import annotations

import hashlib
import io
import zipfile
from datetime import date
from pathlib import Path

import httpx
import polars as pl
import pytest

from oteador.data.vision import (
    KLINE_COLUMNS,
    KlineFile,
    VisionDownloader,
    months_back,
    parse_kline_zip,
    verify_checksum,
)


def _make_kline_zip(*, with_header: bool, n_rows: int = 10) -> bytes:
    rows: list[list[str]] = []
    for i in range(n_rows):
        rows.append(
            [
                str(1_700_000_000_000 + i * 60_000),
                "100.0",
                "101.5",
                "99.5",
                f"{100.0 + i * 0.1:.4f}",
                "12.34",
                str(1_700_000_000_000 + i * 60_000 + 59_999),
                "1234.5",
                "42",
                "5.0",
                "500.0",
                "0",
            ]
        )
    csv_lines: list[str] = []
    if with_header:
        csv_lines.append(",".join(KLINE_COLUMNS))
    csv_lines.extend(",".join(r) for r in rows)
    csv_bytes = ("\n".join(csv_lines) + "\n").encode("utf-8")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("BTCUSDT-1m-2024-01.csv", csv_bytes)
    return buf.getvalue()


def test_kline_file_url_and_paths() -> None:
    kf = KlineFile(symbol="btcusdt", interval="1m", year=2024, month=3)
    assert kf.stem == "BTCUSDT-1m-2024-03"
    assert kf.url.endswith("/BTCUSDT/1m/BTCUSDT-1m-2024-03.zip")
    assert kf.checksum_url.endswith(".zip.CHECKSUM")


def test_months_back_excludes_current_month_and_is_chronological() -> None:
    out = months_back(3, today=date(2025, 3, 15))
    assert out == [(2024, 12), (2025, 1), (2025, 2)]


def test_months_back_rejects_zero() -> None:
    with pytest.raises(ValueError):
        months_back(0)


def test_verify_checksum_ok() -> None:
    payload = b"hello"
    digest = hashlib.sha256(payload).hexdigest()
    verify_checksum(payload, f"{digest}  some-file.zip\n")


def test_verify_checksum_mismatch() -> None:
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_checksum(b"hello", "deadbeef  some-file.zip\n")


def test_parse_kline_zip_with_header() -> None:
    df = parse_kline_zip(_make_kline_zip(with_header=True, n_rows=5))
    assert df.height == 5
    assert "ignore" not in df.columns
    assert df["close"].dtype == pl.Float64
    assert df["open_time"].dtype == pl.Int64


def test_parse_kline_zip_without_header() -> None:
    df = parse_kline_zip(_make_kline_zip(with_header=False, n_rows=5))
    assert df.height == 5
    assert df.columns == [c for c in KLINE_COLUMNS if c != "ignore"]


def test_downloader_caches_and_round_trips(tmp_path: Path) -> None:
    zip_bytes = _make_kline_zip(with_header=True, n_rows=20)
    digest = hashlib.sha256(zip_bytes).hexdigest()

    call_count = {"zip": 0, "checksum": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if url.endswith(".CHECKSUM"):
            call_count["checksum"] += 1
            return httpx.Response(200, text=f"{digest}  whatever.zip\n")
        call_count["zip"] += 1
        return httpx.Response(200, content=zip_bytes)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    dl = VisionDownloader(cache_dir=tmp_path, client=client, verify_checksums=True)
    kf = KlineFile(symbol="BTCUSDT", interval="1m", year=2024, month=1)

    df1 = dl.load(kf)
    assert df1.height == 20
    assert call_count["zip"] == 1

    # Segunda llamada: cache hit, no debe haber descarga.
    df2 = dl.load(kf)
    assert df2.height == 20
    assert call_count["zip"] == 1

    client.close()
