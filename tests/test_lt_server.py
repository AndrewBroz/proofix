"""LanguageToolServer: a local languagetool-server for callers that check many files."""

import os
import shutil
import time

import pytest

from adjudicate import Cache
from proofix import api, lt
from proofix.commas import CommaCache
from proofix.lt import LanguageToolServer, LTError, run_languagetool

if shutil.which("languagetool-server") is None:
    pytest.skip("languagetool-server not installed", allow_module_level=True)


def _alive(pid):
    for _ in range(50):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        time.sleep(0.1)
    return True


@pytest.fixture(scope="module")
def srv():
    with LanguageToolServer() as s:
        yield s


def test_server_matches_command_line(srv):
    text = "We recieved the the new color scheme.\n"
    via_server = run_languagetool(text, "us", server=srv)
    via_cli = run_languagetool(text, "us")
    assert [(m.start, m.end, m.rule) for m in via_server] == [(m.start, m.end, m.rule) for m in via_cli]
    assert via_server


def test_server_honours_disabled_rules_and_utf16_offsets(srv):
    text = "😀😀 Finalise the  plan.\n"
    on = run_languagetool(text, "uk", server=srv)
    assert any(m.rule == "OXFORD_SPELLING_Z_NOT_S" and text[m.start:m.end] == "Finalise" for m in on)
    # the server drops overlapping matches (WHITESPACE_RULE loses to CONSECUTIVE_SPACES),
    # so fix() disables the duplicate: api.lt_disabled
    off = run_languagetool(text, "uk", disabled=api.lt_disabled(api.Options(target="uk")), server=srv)
    assert not any(m.rule == "OXFORD_SPELLING_Z_NOT_S" for m in off)
    assert any(m.rule == "WHITESPACE_RULE" and text[m.start:m.end] == "  " for m in off)


def test_fix_through_the_server_matches_the_command_line(srv):
    text = "We recieved the the new color scheme , and it was fine.\n"
    opts = api.Options(skip=frozenset({"commas", "typos"}))
    a = api.fix(text, opts, None, Cache(None), CommaCache(None), lt_server=srv)
    b = api.fix(text, opts, None, Cache(None), CommaCache(None))
    assert a.text == b.text
    assert [(i.line, i.check, i.message) for i in a.review] == [(i.line, i.check, i.message) for i in b.review]


def test_server_listens_on_localhost_only(srv):
    assert srv.url.startswith("http://127.0.0.1:")
    import subprocess
    port = srv.url.rsplit(":", 1)[1]
    out = subprocess.run(["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN"], capture_output=True,
                         text=True).stdout if shutil.which("lsof") else ""
    if out:
        assert f"127.0.0.1:{port}" in out and "*:" + port not in out


def test_server_stops_on_exit_and_on_error():
    with LanguageToolServer() as s:
        pid = s.process.pid
    assert not _alive(pid)
    with pytest.raises(RuntimeError):
        with LanguageToolServer() as s:
            pid = s.process.pid
            raise RuntimeError("boom")
    assert not _alive(pid)


def test_server_start_failure_raises_lterror(monkeypatch):
    monkeypatch.setattr(lt, "SERVER_START_TIMEOUT", 0.01)
    with pytest.raises(LTError):
        with LanguageToolServer():
            pass


def test_server_request_failure_raises_lterror():
    class Gone:
        url = "http://127.0.0.1:9"
    with pytest.raises(LTError):
        run_languagetool("Some text.\n", "us", server=Gone())


def test_interrupt_during_start_stops_the_server(monkeypatch):
    started = []
    real_popen = lt.subprocess.Popen

    def popen(*a, **k):
        p = real_popen(*a, **k)
        started.append(p)
        return p
    monkeypatch.setattr(lt.subprocess, "Popen", popen)

    def interrupted(*a, **k):
        raise KeyboardInterrupt
    monkeypatch.setattr(lt.urllib.request, "urlopen", interrupted)
    with pytest.raises(KeyboardInterrupt):
        with LanguageToolServer():
            pass
    assert started and not _alive(started[0].pid)


def test_http_exception_during_start_is_not_ready_yet(monkeypatch):
    import http.client
    real = lt.urllib.request.urlopen
    calls = []

    def flaky(*a, **k):
        calls.append(1)
        if len(calls) == 1:
            raise http.client.BadStatusLine("x")
        return real(*a, **k)
    monkeypatch.setattr(lt.urllib.request, "urlopen", flaky)
    with LanguageToolServer() as s:
        pid = s.process.pid
    assert len(calls) >= 2 and not _alive(pid)


def test_early_exit_is_reported_with_its_code(monkeypatch, tmp_path):
    script = tmp_path / "languagetool-server"
    script.write_text("#!/bin/sh\nexit 3\n")
    script.chmod(0o755)
    monkeypatch.setattr(lt.shutil, "which", lambda name: str(script))
    with pytest.raises(LTError, match="exited.*3"):
        with LanguageToolServer():
            pass




def _parity(text, srv, ep_factory):
    out = []
    for server in (srv, None):
        ep = ep_factory()
        r = api.fix(text, api.Options(skip=frozenset({"commas", "typos", "compounds"})), ep,
                    Cache(None), CommaCache(None), lt_server=server)
        out.append((r.text, sorted((e.original, e.replacement, e.source) for e in r.applied),
                    sorted((i.line, i.kind, i.check) for i in r.review),
                    sorted((d.item.word, d.item.options) for d in r.decisions)))
    return out


@pytest.mark.parametrize("name", ["however", "typos.md"])
@pytest.mark.parametrize("model", [False, True])
def test_server_parity_with_command_line(srv, name, model):
    import json as _json
    from pathlib import Path
    from adjudicate.llm import Endpoint

    class First(Endpoint):
        def __init__(self):
            super().__init__(url="http://x/v1", model="m")

        def chat(self, system, user, timeout=None, json_mode=True):
            ids = [int(line.split()[1]) for line in user.splitlines() if line.startswith("Item ")]
            return _json.dumps({"decisions": [{"id": i, "choice": 1, "confidence": 0.95, "reason": "t"}
                                              for i in ids]})
    text = ("It was clear.However the chair disagreed.\n" if name == "however" else
            (Path(__file__).parent.parent / "examples" / "typos.md").read_text())
    via_server, via_cli = _parity(text, srv, (First if model else (lambda: None)))
    assert via_server == via_cli
