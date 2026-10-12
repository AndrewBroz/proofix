from pathlib import Path

from adjudicate import Cache, apply_changes
from proofix import api
from proofix.commas import CommaCache

TYPOS = (Path(__file__).parent.parent / "examples" / "typos.md").read_text()


def run(text, **kw):
    return api.fix(text, api.Options(**kw), None, Cache(None), CommaCache(None))


def test_changes_reproduce_the_output(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")             # no LanguageTool, no typos
    r = run("Bad​space and&#32;entity.\r\n")
    assert apply_changes("Bad​space and&#32;entity.\r\n", r.changes) == r.text
    assert r.lt_missing
    assert {s.why for s in r.skipped} >= {"LanguageTool not found"}


def test_skip_hygiene():
    text = "Bad​space.\n"
    assert run(text, skip=frozenset({"hygiene", "mechanical", "adjudicated", "typos"})).text == text


def test_no_file_io(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PATH", "/nonexistent")
    run(TYPOS)
    assert list(tmp_path.iterdir()) == []


def test_cached_comma_decision_is_labelled_cache(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    text = "The committee, decided to postpone the vote.\n"
    cc = CommaCache(None)
    cc.data[CommaCache.key("us|leave", "The committee, decided to postpone the vote.")] = {
        "text": "The committee decided to postpone the vote.", "confidence": 0.95, "reason": "stub"}
    r = api.fix(text, api.Options(skip=frozenset({"typos"})), None, Cache(None), cc)
    assert r.text == "The committee decided to postpone the vote.\n"
    commas = [c for c in r.changes if c.check == "commas"]
    assert len(commas) == 1 and commas[0].decided_by == "cache"
    assert apply_changes(text, r.changes) == r.text
