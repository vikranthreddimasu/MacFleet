"""Tests for `macfleet pair` (v2.2 PR 16a)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from macfleet.cli.main import cli


class TestPairFromStdin:
    def test_valid_url_writes_token(self, tmp_path: Path, monkeypatch):
        # Redirect TOKEN_FILE to a temp path so we don't clobber real config
        token_path = tmp_path / "token"
        monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(token_path))

        url = "macfleet://pair?token=secret-pair-token&fleet=my-fleet"
        runner = CliRunner()
        result = runner.invoke(cli, ["pair", "--stdin"], input=url)

        assert result.exit_code == 0, result.output
        assert "Paired" in result.output
        assert "my-fleet" in result.output
        assert token_path.read_text().strip() == "secret-pair-token"

    def test_url_without_fleet_id_works(self, tmp_path: Path, monkeypatch):
        token_path = tmp_path / "token"
        monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(token_path))

        runner = CliRunner()
        result = runner.invoke(
            cli, ["pair", "--stdin"], input="macfleet://pair?token=lonely-token",
        )
        assert result.exit_code == 0
        assert "Paired" in result.output
        assert token_path.read_text().strip() == "lonely-token"

    def test_empty_stdin_errors(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["pair", "--stdin"], input="")
        assert result.exit_code == 1
        # Rich may wrap output; normalize
        clean = " ".join(result.output.split())
        assert "no URL" in clean

    def test_malformed_url_errors(self):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["pair", "--stdin"], input="http://not-macfleet/pair?token=x",
        )
        assert result.exit_code == 1
        assert "Error" in result.output or "error" in result.output

    def test_missing_token_errors(self):
        runner = CliRunner()
        result = runner.invoke(
            cli, ["pair", "--stdin"], input="macfleet://pair?fleet=x",
        )
        assert result.exit_code == 1

    def test_whitespace_stripped(self, tmp_path: Path, monkeypatch):
        """Pasteboard often picks up trailing newlines; we must strip."""
        token_path = tmp_path / "token"
        monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(token_path))

        runner = CliRunner()
        result = runner.invoke(
            cli, ["pair", "--stdin"],
            input="   macfleet://pair?token=spaced-token   \n\n",
        )
        assert result.exit_code == 0
        assert token_path.read_text().strip() == "spaced-token"


class TestPairFromPasteboard:
    def test_reads_from_pasteboard(self, tmp_path: Path, monkeypatch):
        token_path = tmp_path / "token"
        monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(token_path))

        url = "macfleet://pair?token=pasteboard-token&fleet=cluster-a"
        with patch(
            "macfleet.security.bootstrap.read_from_pasteboard",
            return_value=url,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["pair"])

        assert result.exit_code == 0
        assert "Paired" in result.output
        assert token_path.read_text().strip() == "pasteboard-token"

    def test_empty_pasteboard_errors(self):
        with patch(
            "macfleet.security.bootstrap.read_from_pasteboard",
            return_value=None,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["pair"])
        assert result.exit_code == 1
        clean = " ".join(result.output.split())
        assert "pasteboard" in clean

    def test_pasteboard_suggests_stdin_fallback(self):
        with patch(
            "macfleet.security.bootstrap.read_from_pasteboard",
            return_value=None,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["pair"])
        # Error message should hint at the --stdin workaround
        clean = " ".join(result.output.split())
        assert "--stdin" in clean


class TestBootstrapFlag:
    def test_open_plus_bootstrap_rejected(self):
        """--open (no token) + --bootstrap makes no sense — reject at CLI."""
        runner = CliRunner()
        result = runner.invoke(cli, ["join", "--open", "--bootstrap"])
        assert result.exit_code == 1
        clean = " ".join(result.output.split())
        assert "open" in clean.lower() and "bootstrap" in clean.lower()


class TestCliHelp:
    def test_pair_in_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "pair" in result.output

    def test_bootstrap_flag_in_join_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["join", "--help"])
        assert "--bootstrap" in result.output
