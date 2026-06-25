"""Tests for `macfleet pair` (v2.2 PR 16a)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from click.testing import CliRunner

from macfleet.cli.main import cli
from macfleet.security.enrollment import EnrollmentResult


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

    def test_existing_token_from_stdin_requires_yes(self, tmp_path: Path, monkeypatch):
        token_path = tmp_path / "token"
        token_path.write_text("existing-token")
        monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(token_path))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["pair", "--stdin"],
            input="macfleet://pair?token=new-token",
        )

        assert result.exit_code == 1
        assert "--yes" in result.output
        assert token_path.read_text() == "existing-token"

    def test_existing_token_can_be_replaced_with_yes(self, tmp_path: Path, monkeypatch):
        token_path = tmp_path / "token"
        token_path.write_text("existing-token")
        monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(token_path))

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["pair", "--stdin", "--yes"],
            input="macfleet://pair?token=new-token",
        )

        assert result.exit_code == 0, result.output
        assert token_path.read_text().strip() == "new-token"


class TestPairFromPasteboard:
    def test_default_requires_explicit_input_source(self, tmp_path: Path, monkeypatch):
        token_path = tmp_path / "token"
        monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(token_path))

        url = "macfleet://pair?token=pasteboard-token&fleet=cluster-a"
        with patch(
            "macfleet.security.bootstrap.read_from_pasteboard",
            return_value=url,
        ) as read_from_pasteboard:
            runner = CliRunner()
            result = runner.invoke(cli, ["pair"])

        assert result.exit_code == 1
        assert "explicit input source" in result.output
        assert not token_path.exists()
        read_from_pasteboard.assert_not_called()

    def test_reads_from_pasteboard_when_explicit(self, tmp_path: Path, monkeypatch):
        token_path = tmp_path / "token"
        monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(token_path))

        url = "macfleet://pair?token=pasteboard-token&fleet=cluster-a"
        with patch(
            "macfleet.security.bootstrap.read_from_pasteboard",
            return_value=url,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["pair", "--pasteboard"])

        assert result.exit_code == 0
        assert "Paired" in result.output
        assert token_path.read_text().strip() == "pasteboard-token"

    def test_existing_token_from_pasteboard_can_be_cancelled(self, tmp_path: Path, monkeypatch):
        token_path = tmp_path / "token"
        token_path.write_text("existing-token")
        monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(token_path))

        url = "macfleet://pair?token=pasteboard-token&fleet=cluster-a"
        with patch(
            "macfleet.security.bootstrap.read_from_pasteboard",
            return_value=url,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["pair", "--pasteboard"], input="n\n")

        assert result.exit_code == 1
        assert "left unchanged" in result.output
        assert token_path.read_text() == "existing-token"

    def test_empty_pasteboard_errors(self):
        with patch(
            "macfleet.security.bootstrap.read_from_pasteboard",
            return_value=None,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["pair", "--pasteboard"])
        assert result.exit_code == 1
        clean = " ".join(result.output.split())
        assert "pasteboard" in clean

    def test_pasteboard_suggests_stdin_fallback(self):
        with patch(
            "macfleet.security.bootstrap.read_from_pasteboard",
            return_value=None,
        ):
            runner = CliRunner()
            result = runner.invoke(cli, ["pair", "--pasteboard"])
        # Error message should hint at the --stdin workaround
        clean = " ".join(result.output.split())
        assert "--stdin" in clean

    def test_rejects_multiple_legacy_sources(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["pair", "--stdin", "--pasteboard"],
            input="macfleet://pair?token=secret-token",
        )
        assert result.exit_code == 1
        assert "choose only one" in result.output


class TestPairFromEnrollment:
    def test_host_code_writes_token(self, tmp_path: Path, monkeypatch):
        token_path = tmp_path / "token"
        monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(token_path))

        async def fake_request(host, port, code):
            assert host == "192.168.1.10"
            assert port == 4242
            assert code == "ABCD-EFGH-IJKL-MNOP"
            return EnrollmentResult(
                token="enrolled-token-long-enough",
                fleet_id="lab",
                server_node="studio",
            )

        monkeypatch.setattr("macfleet.security.enrollment.request_enrollment", fake_request)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "pair",
                "--host",
                "192.168.1.10:4242",
                "--code",
                "ABCD-EFGH-IJKL-MNOP",
            ],
        )

        assert result.exit_code == 0, result.output
        assert "Paired" in result.output
        assert "studio" in result.output
        assert token_path.read_text().strip() == "enrolled-token-long-enough"

    def test_host_code_confirmation_happens_before_enrollment_request(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        token_path = tmp_path / "token"
        token_path.write_text("existing-token")
        monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(token_path))

        async def fail_if_called(host, port, code):
            raise AssertionError("enrollment request should not run after cancel")

        monkeypatch.setattr("macfleet.security.enrollment.request_enrollment", fail_if_called)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "pair",
                "--host",
                "192.168.1.10:4242",
                "--code",
                "ABCD-EFGH-IJKL-MNOP",
            ],
            input="n\n",
        )

        assert result.exit_code == 1
        assert "left unchanged" in result.output
        assert token_path.read_text() == "existing-token"

    def test_host_code_yes_replaces_existing_token(self, tmp_path: Path, monkeypatch):
        token_path = tmp_path / "token"
        token_path.write_text("existing-token")
        monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(token_path))

        async def fake_request(host, port, code):
            return EnrollmentResult(
                token="new-enrolled-token",
                fleet_id="lab",
                server_node="studio",
            )

        monkeypatch.setattr("macfleet.security.enrollment.request_enrollment", fake_request)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "pair",
                "--host",
                "192.168.1.10:4242",
                "--code",
                "ABCD-EFGH-IJKL-MNOP",
                "--yes",
            ],
        )

        assert result.exit_code == 0, result.output
        assert token_path.read_text().strip() == "new-enrolled-token"

    def test_host_requires_code(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["pair", "--host", "127.0.0.1:1234"])
        assert result.exit_code == 1
        assert "--host and --code" in result.output

    def test_host_code_rejects_pasteboard(self):
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "pair",
                "--host",
                "127.0.0.1:1234",
                "--code",
                "ABCD-EFGH-IJKL-MNOP",
                "--pasteboard",
            ],
        )
        assert result.exit_code == 1
        assert "cannot be combined" in result.output


class TestBootstrapFlag:
    def test_open_plus_bootstrap_rejected(self):
        """--open (no token) + --bootstrap makes no sense — reject at CLI."""
        runner = CliRunner()
        result = runner.invoke(cli, ["join", "--open", "--bootstrap"])
        assert result.exit_code == 1
        clean = " ".join(result.output.split())
        assert "open" in clean.lower() and "bootstrap" in clean.lower()

    def test_open_requires_explicit_insecure_ack(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["join", "--open"])
        assert result.exit_code == 1
        clean = " ".join(result.output.split())
        assert "--allow-insecure-open" in clean


class TestJoinTokenSetup:
    def test_join_reports_token_write_failure_without_traceback(
        self,
        tmp_path: Path,
        monkeypatch,
    ):
        target = tmp_path / "target-token"
        target.write_text("existing-token")
        token_path = tmp_path / "fleet-token"
        token_path.symlink_to(target)
        monkeypatch.setattr("macfleet.security.auth.TOKEN_FILE", str(token_path))
        monkeypatch.setattr("macfleet.security.auth.TOKEN_DIR", str(tmp_path))

        runner = CliRunner()
        result = runner.invoke(cli, ["join"])

        assert result.exit_code == 1
        assert "couldn't configure fleet token" in result.output
        assert "Traceback" not in result.output
        assert target.read_text() == "existing-token"


class TestCliHelp:
    def test_pair_in_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert "pair" in result.output

    def test_bootstrap_flag_in_join_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["join", "--help"])
        assert "--bootstrap" in result.output
        assert "--allow-insecure-open" in result.output

    def test_pasteboard_flag_in_pair_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["pair", "--help"])
        assert "--pasteboard" in result.output

    def test_yes_flag_in_pair_help(self):
        runner = CliRunner()
        result = runner.invoke(cli, ["pair", "--help"])
        assert "--yes" in result.output
