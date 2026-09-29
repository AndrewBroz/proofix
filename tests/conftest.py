"""No test may see the developer's real endpoint config or reach a model."""

import pytest

LLM_ENV = [f"{app}_LLM_{s}" for app in ("ADJUDICATE", "STYLEFIX", "PROOFIX")
           for s in ("URL", "MODEL", "KEY")]


@pytest.fixture(autouse=True)
def isolated_llm_config(tmp_path_factory, monkeypatch):
    xdg = tmp_path_factory.mktemp("xdg")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(xdg))
    for name in LLM_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "http://127.0.0.1:9")
    return xdg
