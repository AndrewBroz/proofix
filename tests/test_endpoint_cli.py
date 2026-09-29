import io
import os

import pytest

from proofix import cli


def test_isolation_hides_real_config(isolated_llm_config):
    assert os.environ["XDG_CONFIG_HOME"] == str(isolated_llm_config)
    assert not any(os.environ.get(f"{app}_LLM_{s}") for app in ("ADJUDICATE", "STYLEFIX", "PROOFIX")
                   for s in ("URL", "MODEL", "KEY"))


def test_dgx_preset_is_gone():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["x.md", "--llm", "dgx"])


def test_timeout_flag_defaults_to_none():
    assert cli.build_parser().parse_args(["x.md"]).timeout is None


def test_bad_config_exits_2(isolated_llm_config, tmp_path, capsys):
    cfg = isolated_llm_config / "proofix" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("[llm\n")
    src = tmp_path / "d.md"
    src.write_text("Some text.\n")
    assert cli.main([str(src), "--no-typos", "--no-cache", "--dry-run"]) == 2
    err = capsys.readouterr().err
    assert err.startswith("proofix: ") and str(cfg) in err


def test_no_endpoint_suggests_setup(tmp_path, capsys):
    src = tmp_path / "d.md"
    src.write_text("Some text.\n")
    assert cli.main([str(src), "--no-lt", "--no-typos", "--no-cache", "--dry-run"]) == 0
    assert "proofix --setup" in capsys.readouterr().err


def test_check_endpoint_needs_no_input_file(capsys):
    assert cli.main(["--check-endpoint"]) == 1
    assert "proofix --setup" in capsys.readouterr().out


def test_setup_needs_a_terminal(monkeypatch, capsys):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert cli.main(["--setup"]) == 2


def test_input_still_required_for_a_run(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main([])
    assert e.value.code == 2 and "input" in capsys.readouterr().err


def test_stylefix_env_no_longer_configures_proofix(monkeypatch):
    monkeypatch.setenv("STYLEFIX_LLM_URL", "http://stylefix-only/v1")
    monkeypatch.setenv("STYLEFIX_LLM_MODEL", "m")
    from adjudicate import resolve_endpoint
    assert resolve_endpoint("proofix") is None
