from __future__ import annotations

from typer.testing import CliRunner

from oteador import __version__
from oteador.cli import app


def test_version_is_non_empty_string() -> None:
    assert isinstance(__version__, str)
    assert __version__


def test_cli_version_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.stdout
