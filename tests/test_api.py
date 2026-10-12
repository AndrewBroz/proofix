import json
from pathlib import Path

from adjudicate import Cache, apply_changes
from adjudicate.llm import Endpoint
from proofix import api
from proofix.commas import CommaCache
from proofix.lt import parse_matches

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


class Fake(Endpoint):
    def chat(self, system, user, timeout=None, json_mode=True):
        if "comma placement" in system:
            return json.dumps({"changed": [], "notes": []})
        ds = []
        for line in user.splitlines():
            if line.startswith("Item "):
                options = [o.split("] ", 1)[-1] for o in line.split("options: ")[1].split(" | ")]
                ds.append({"id": int(line.split()[1]), "choice": options[0],
                           "confidence": 0.9, "reason": "model says so"})
        return json.dumps({"decisions": ds})


def test_model_decided_compound_is_labelled_model(monkeypatch):
    fixture = (Path(__file__).parent / "fixtures" / "typos.lt.json").read_text()
    monkeypatch.setattr(api, "find_languagetool", lambda: "/replay/languagetool")
    monkeypatch.setattr(api, "run_languagetool", lambda t, target, level="default": parse_matches(fixture, t))
    r = api.fix(TYPOS, api.Options(skip=frozenset({"typos", "commas"})),
                Fake(url="http://x/v1", model="m"), Cache(None), CommaCache(None))
    assert apply_changes(TYPOS, r.changes) == r.text
    third = [c for c in r.changes if c.after == "-" and c.decided_by == "model"]
    assert third and third[0].check == "adjudicated" and third[0].reason == "model says so"


def test_no_endpoint_is_not_a_review_item_per_sentence(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    r = run(TYPOS)
    assert not [i for i in r.review if "no model configured" in i.message]
