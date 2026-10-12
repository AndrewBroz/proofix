"""The basics tier (no LanguageTool) and doubled-word routing, for basics and
for LanguageTool's ENGLISH_WORD_REPEAT_RULE alike."""

import json

import pytest

from adjudicate import Cache
from adjudicate.llm import Endpoint
from proofix import api, basics
from proofix.commas import CommaCache
from proofix.lt import parse_matches


class FakeEndpoint(Endpoint):
    """Answers every item in the prompt with the same choice."""

    def __init__(self, choice):
        super().__init__(url="http://x/v1", model="m")
        self.choice = choice

    def chat(self, system, user, timeout=None, json_mode=True):
        if "comma placement" in system:
            return json.dumps({"changed": [], "notes": []})
        ids = [int(line.split()[1]) for line in user.splitlines() if line.startswith("Item ")]
        return json.dumps({"decisions": [{"id": i, "choice": self.choice, "confidence": 0.95,
                                          "reason": "t"} for i in ids]})


def run(text, ep=None, **kw):
    return api.fix(text, api.Options(**kw), ep, Cache(None), CommaCache(None))


def _lt_json(text, word):
    """A hand-written LanguageTool repeat match on the first '<word> <word>' (UTF-16 offsets)."""
    doubled = f"{word} {word}"
    off = len(text[:text.index(doubled)].encode("utf-16-le")) // 2
    return json.dumps({"matches": [{
        "message": "Possible typo: you repeated a word.", "shortMessage": "Word repetition",
        "replacements": [{"value": word}], "offset": off, "length": len(doubled),
        "context": {"text": text, "offset": off, "length": len(doubled)},
        "rule": {"id": "ENGLISH_WORD_REPEAT_RULE", "category": {"id": "MISC"}}}]})


def _replay(monkeypatch, fixture):
    monkeypatch.setenv("PATH", "/nonexistent")                     # no typos
    monkeypatch.setattr(api, "find_languagetool", lambda: "/replay/languagetool")
    monkeypatch.setattr(api, "run_languagetool",
                        lambda text, target, level="default": parse_matches(fixture, text))


@pytest.fixture
def replay_lt_repeat(monkeypatch):
    _replay(monkeypatch, _lt_json("How do you know that that is wrong?\n", "that"))


@pytest.fixture
def replay_lt_repeat_the(monkeypatch):
    _replay(monkeypatch, _lt_json("We got the the scheme.\n", "the"))


def test_closed_list_doubles_fixed_without_model(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    assert run("We got the the scheme, most of of it.\n").text == "We got the scheme, most of it.\n"


@pytest.mark.parametrize("text", ["How do you know that that is wrong?\n", "What it is is unclear.\n",
                                  "I had had enough.\n", "It was very very good.\n"])
def test_other_doubles_reviewed_without_model(monkeypatch, text):
    monkeypatch.setenv("PATH", "/nonexistent")
    r = run(text)
    assert r.text == text
    assert any(i.kind == "undecided" or "repeated word" in i.message for i in r.review)


def test_other_doubles_adjudicated_with_model(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    keep = FakeEndpoint(choice="KEEP")
    assert run("How do you know that that is wrong?\n", ep=keep).text == "How do you know that that is wrong?\n"
    drop = FakeEndpoint(choice=1)          # option 1 = one copy
    assert run("It was very very good.\n", ep=drop).text == "It was very good.\n"


@pytest.mark.parametrize("text", ["We flew to Bora Bora.\n", "New York, New York is a song.\n",
                                  "- the\n- the\n", "# The\n\nthe end\n"])
def test_doubles_left_alone(monkeypatch, text):
    monkeypatch.setenv("PATH", "/nonexistent")
    assert run(text).text == text


def test_basics_spacing(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    assert run("The programme , which covers analysis,design and testing.\n").text == \
        "The programme, which covers analysis, design and testing.\n"


@pytest.mark.parametrize("text", ["It cost 1,000 dollars at 10:30.\n", "See https://a.b/c,d now.\n",
                                  "A ratio of 3:2 holds, e.g., here.\n", "Wait ... then.\n", "`a ,b`\n"])
def test_basics_leaves_numbers_times_urls(monkeypatch, text):
    monkeypatch.setenv("PATH", "/nonexistent")
    assert run(text).text == text


def test_basics_does_not_run_with_languagetool(monkeypatch):
    monkeypatch.setattr(api, "find_languagetool", lambda: "/replay/languagetool")
    monkeypatch.setattr(api, "run_languagetool", lambda *a, **k: [])     # LanguageTool "runs", finds nothing
    r = run("The the end , ok.\n")
    assert not [e for e in r.applied if e.source == "basics"]


def test_languagetool_repeat_outside_closed_list_is_reviewed(replay_lt_repeat):
    # replay a LanguageTool ENGLISH_WORD_REPEAT_RULE match on "that that", no model
    r = run("How do you know that that is wrong?\n")
    assert r.text == "How do you know that that is wrong?\n"
    assert any("repeated word 'that that'" in i.message for i in r.review)


def test_languagetool_repeat_in_closed_list_still_fixed(replay_lt_repeat_the):
    assert run("We got the the scheme.\n").text == "We got the scheme.\n"


# Beyond the brief: the routes' edges.

def test_languagetool_repeat_adjudicated_with_model(replay_lt_repeat):
    r = run("How do you know that that is wrong?\n", ep=FakeEndpoint(choice=1))
    assert r.text == "How do you know that is wrong?\n"


def test_languagetool_repeat_into_a_list_line_left_alone(monkeypatch):
    text = "Buy the\n- the milk\n"
    off = text.index("the")
    _replay(monkeypatch, json.dumps({"matches": [{
        "message": "Possible typo: you repeated a word.", "replacements": [{"value": "the"}],
        "offset": off, "length": 9, "context": {"text": text, "offset": off, "length": 9},
        "rule": {"id": "ENGLISH_WORD_REPEAT_RULE", "category": {"id": "MISC"}}}]}))
    r = run(text)
    assert r.text == text
    assert not [i for i in r.review if "repeated word" in i.message]


def test_double_across_a_line_break_in_a_paragraph(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    assert run("We sent the\nthe report.\n").text == "We sent the report.\n"   # separator goes with the copy


def test_capitalised_first_copy_keeps_its_case(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    assert run("The the end.\n").text == "The end.\n"
    assert run("Very very good.\n", ep=FakeEndpoint(choice=1)).text == "Very good.\n"


def test_accept_listed_double_untouched(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    r = run("It was very very good.\n", accept=frozenset({"very very"}))
    assert not [i for i in r.review if "repeated word" in i.message]


def test_basics_can_be_skipped(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    text = "The the end , ok.\n"
    assert run(text, skip=frozenset({"basics"})).text == text
    assert "basics" in api.TIERS and api.TIERS.index("basics") == api.TIERS.index("mechanical") + 1


def test_doubled_words_reports_second_copy_span():
    text = "most of of it"
    [d] = basics.doubled_words(text, [])
    assert (text[d.start:d.end], d.word, d.closed) == (" of", "of", True)


def test_space_before_comma_and_none_after(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    assert run("the budget ,the timeline\n").text == "the budget, the timeline\n"


def test_basics_review_reaches_the_cli_list(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    r = run("I had had enough.\n")
    assert [(m.rule, m.message) for m in r.lt_review] == [("basics", "repeated word 'had had'")]
    assert [(i.kind, i.check) for i in r.review if "repeated word" in i.message] == [("alert", "basics")]
