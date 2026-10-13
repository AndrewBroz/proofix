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
                        lambda text, target, level="default", **kw: parse_matches(fixture, text))


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
    # Fix round 2 ruling: doubled words run alongside LanguageTool; spacing does not.
    # Final review ruling R1: a closed-list double LanguageTool did not flag is routed.
    assert not [e for e in r.applied if e.source == "basics"]
    assert r.text == "The the end , ok.\n"
    assert "repeated word 'The the'" in [i.message for i in r.review]


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


# Fix round 1: spacing applies only provably-right cases; the rest are reviewed or left alone.

@pytest.mark.parametrize("text", ["Rename it to .bashrc first\n", "Built on .NET today\n",
                                  "Great :) see you\n", "ok ;)\n", "He said :-( and left\n",
                                  "Smile :D now\n", "Use :P often\n", "hello :wave: there\n",
                                  "Dots . . . trail\n", "Press : to enter command mode\n",
                                  "Type ! to run a shell\n", "a ratio of a :b here\n",
                                  "Use e.g. , this\n", "Item:Value\n", "Wow !!\n",
                                  "key item:wave: pairs\n", "say ok ,\"quoted\"\n"])
def test_spacing_leaves_non_prose_alone(monkeypatch, text):
    monkeypatch.setenv("PATH", "/nonexistent")
    assert run(text).text == text


@pytest.mark.parametrize("text,fixed", [
    ("Is it ready ?\n", "Is it ready?\n"), ("one word ,\nnext line\n", "one word,\nnext line\n"),
    ("items x , y and z\n", "items x, y and z\n"), ("Made in the U.S.A .\n", "Made in the U.S.A.\n"),
    ("The programme , which\n", "The programme, which\n"), ("analysis,design\n", "analysis, design\n"),
    ("end of list :\n- a\n", "end of list:\n- a\n"), ("Stop !\n", "Stop!\n"),
    ("(see this ,) and\n", "(see this,) and\n"), ("key item:value pairs\n", "key item: value pairs\n"),
    ("Wait ,really?\n", "Wait, really?\n")])
def test_spacing_fixes_provable_cases(monkeypatch, text, fixed):
    monkeypatch.setenv("PATH", "/nonexistent")
    assert run(text).text == fixed


@pytest.mark.parametrize("text,mark", [("Press : to enter\n", ":"), ("Note : the thing\n", ":"),
                                       ("Type ! to run\n", "!"), ("Press ? for help\n", "?"),
                                       ("Press ; to repeat\n", ";")])
def test_space_before_mark_mid_line_is_reviewed(monkeypatch, text, mark):
    monkeypatch.setenv("PATH", "/nonexistent")
    r = run(text)
    assert r.text == text
    assert f"space before '{mark}'" in [m.message for m in r.lt_review]


def test_unicode_and_apostrophe_doubles(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    for text, msg in [("The café café was shut.\n", "repeated word 'café café'"),
                      ("It's it's raining.\n", "repeated word 'It's it's'"),
                      ("The über über plan.\n", "repeated word 'über über'")]:
        r = run(text)
        assert r.text == text and msg in [i.message for i in r.review]
    assert run("The café café was shut.\n", ep=FakeEndpoint(choice=1)).text == "The café was shut.\n"


def test_languagetool_repeat_basics_cannot_match_is_still_routed(monkeypatch):
    text = "We saw 3rd 3rd place.\n"            # _DOUBLE takes letters only
    off = text.index("3rd")
    _replay(monkeypatch, json.dumps({"matches": [{
        "message": "Possible typo: you repeated a word.", "replacements": [{"value": "3rd"}],
        "offset": off, "length": 7, "context": {"text": text, "offset": off, "length": 7},
        "rule": {"id": "ENGLISH_WORD_REPEAT_RULE", "category": {"id": "MISC"}}}]}))
    r = run(text)
    assert r.text == text
    assert "repeated word '3rd 3rd'" in [i.message for i in r.review]
    assert run(text, ep=FakeEndpoint(choice=1)).text == "We saw 3rd place.\n"


def test_double_with_two_spaces(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    assert run("We saw the  the dog.\n").text == "We saw the dog.\n"


def test_languagetool_repeat_with_two_spaces(monkeypatch):
    text = "We saw the  the dog.\n"
    off = text.index("the")
    _replay(monkeypatch, json.dumps({"matches": [{
        "message": "Possible typo: you repeated a word.", "replacements": [{"value": "the"}],
        "offset": off, "length": 8, "context": {"text": text, "offset": off, "length": 8},
        "rule": {"id": "ENGLISH_WORD_REPEAT_RULE", "category": {"id": "MISC"}}}]}))
    assert run(text).text == "We saw the dog.\n"


def test_capitalised_closed_pair_is_reviewed(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    for ep in (None, FakeEndpoint(choice=1)):
        r = run("The The report is late.\n", ep=ep)
        assert r.text == "The The report is late.\n"
        assert "repeated word 'The The'" in [i.message for i in r.review]
    assert run("We flew to Bora Bora.\n").review == []


def test_repeat_review_kept_when_adjudicated_skipped(replay_lt_repeat):
    r = run("How do you know that that is wrong?\n", skip=frozenset({"adjudicated"}))
    assert r.text == "How do you know that that is wrong?\n"
    assert "repeated word 'that that'" in [i.message for i in r.review]


# Fix round 2: doubled_words runs alongside real LanguageTool, which misses "that that".

needs_lt = pytest.mark.skipif(api.find_languagetool() is None, reason="languagetool not installed")


def _repeat_reviews(r):
    return [i for i in r.review if "repeated word" in i.message]


@needs_lt
def test_real_lt_that_that_is_reviewed():
    r = run("How do you know that that is wrong?\n", skip=frozenset({"typos"}))
    assert r.text == "How do you know that that is wrong?\n"
    assert [i.message for i in _repeat_reviews(r)] == ["repeated word 'that that'"]


@needs_lt
def test_real_lt_closed_double_one_edit():
    r = run("We got the the scheme.\n", skip=frozenset({"typos"}))
    assert r.text == "We got the scheme.\n"
    assert len([e for e in r.applied if "the the" in e.original]) == 1
    assert _repeat_reviews(r) == []


@needs_lt
def test_real_lt_very_very_one_review():
    r = run("It was very very good.\n", skip=frozenset({"typos"}))
    assert r.text == "It was very very good.\n"
    assert len(_repeat_reviews(r)) == 1


@needs_lt
def test_real_lt_that_that_adjudicated_once():
    ep = FakeEndpoint(choice=1)
    r = run("How do you know that that is wrong?\n", ep=ep, skip=frozenset({"typos"}))
    assert r.text == "How do you know that is wrong?\n"
    assert len([d for d in r.decisions if d.item.source == "repeat"]) == 1


def test_doubles_dedup_against_replayed_lt(replay_lt_repeat):
    r = run("How do you know that that is wrong?\n")
    assert len(_repeat_reviews(r)) == 1


def test_skip_basics_with_lt_no_double_routing(monkeypatch):
    monkeypatch.setattr(api, "find_languagetool", lambda: "/replay/languagetool")
    monkeypatch.setattr(api, "run_languagetool", lambda *a, **k: [])
    r = run("How do you know that that is wrong?\n", skip=frozenset({"basics", "typos"}))
    assert _repeat_reviews(r) == []
    r = run("How do you know that that is wrong?\n", skip=frozenset({"typos"}))
    assert len(_repeat_reviews(r)) == 1
    assert run("The end , ok.\n", skip=frozenset({"typos"})).text == "The end , ok.\n"   # no spacing with LT


# Fix round 3: pass 1's repeat edit loses to an overlapping whitespace edit and there is no pass 2.

def _two_space_replay(monkeypatch):
    text = "We saw the  the dog.\n"
    off = text.index("the")
    ws = text.index("  ")
    _replay(monkeypatch, json.dumps({"matches": [
        {"message": "Possible typo: you repeated a word.", "replacements": [{"value": "the"}],
         "offset": off, "length": 8, "context": {"text": text, "offset": off, "length": 8},
         "rule": {"id": "ENGLISH_WORD_REPEAT_RULE", "category": {"id": "MISC"}}},
        {"message": "Possible typo: you repeated a whitespace", "replacements": [{"value": " "}],
         "offset": ws, "length": 2, "context": {"text": text, "offset": ws, "length": 2},
         "rule": {"id": "WHITESPACE_RULE", "category": {"id": "TYPOGRAPHY"}}}]}))
    return text


def _check_two_spaces(text, skip):
    r = run(text, skip=frozenset(skip))
    if "basics" in skip:
        assert r.text in ("We saw the the dog.\n", "We saw the dog.\n")
        if r.text != "We saw the dog.\n":
            assert "repeated word 'the the'" in [i.message for i in r.review]
    else:
        assert r.text == "We saw the dog.\n"
        assert _repeat_reviews(r) == []


@pytest.mark.parametrize("skip", [{"typos", "adjudicated"}, {"typos", "adjudicated", "basics"}])
def test_rejected_repeat_without_pass_2_replayed(monkeypatch, skip):
    _check_two_spaces(_two_space_replay(monkeypatch), skip)


@needs_lt
@pytest.mark.parametrize("skip", [{"typos", "adjudicated"}, {"typos", "adjudicated", "basics"}])
def test_rejected_repeat_without_pass_2_real_lt(skip):
    _check_two_spaces("We saw the  the dog.\n", skip)


# Final review, ruling R1: the closed list holds only words whose doubling is never
# grammatical; a hyphen touching either copy is no double; LanguageTool's silence vetoes.

PROBES = ["Add it to to-do list.\n", "Check the in in-house tools.\n", "A for for-profit firm.\n",
          "Please log in in the top-right corner.\n", "He came in in a hurry.\n",
          "Turn it on on Monday.\n"]


def test_closed_list_drops_in_on_by_about():
    assert not {"in", "on", "by", "about"} & basics.CLOSED_DOUBLES
    assert basics.CLOSED_DOUBLES == frozenset(
        "a an the of to at for with from into onto and or but nor it we they my your our their".split())


@pytest.mark.parametrize("text", ["Add it to to-do list.\n", "for for-profit\n", "in in-house\n",
                                  "a pre-the the\n", "the-the the\n"])
def test_a_hyphen_touching_either_copy_is_no_double(text):
    assert basics.doubled_words(text, []) == []


@pytest.mark.parametrize("text", PROBES)
def test_probes_never_applied_without_tools(monkeypatch, text):
    monkeypatch.setenv("PATH", "/nonexistent")
    assert run(text).text == text
    assert run(text, ep=FakeEndpoint(choice="KEEP")).text == text


@pytest.mark.parametrize("text", PROBES)
def test_probes_never_applied_with_silent_languagetool(monkeypatch, text):
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setattr(api, "find_languagetool", lambda: "/replay/languagetool")
    monkeypatch.setattr(api, "run_languagetool", lambda *a, **k: [])
    assert run(text).text == text
    assert run(text, skip=frozenset({"adjudicated"})).text == text


def test_closed_double_languagetool_does_not_flag_is_routed_not_applied(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    monkeypatch.setattr(api, "find_languagetool", lambda: "/replay/languagetool")
    monkeypatch.setattr(api, "run_languagetool", lambda *a, **k: [])     # runs, flags nothing
    for skip in (frozenset(), frozenset({"adjudicated"})):
        r = run("We got the the scheme.\n", skip=skip)
        assert r.text == "We got the the scheme.\n"
        assert "repeated word 'the the'" in [i.message for i in r.review]
    r = run("We got the the scheme.\n", ep=FakeEndpoint(choice="KEEP"))
    assert r.text == "We got the the scheme.\n"
    assert [d.item.word for d in r.decisions] == ["the the"]
    assert run("We got the the scheme.\n", ep=FakeEndpoint(choice=1)).text == "We got the scheme.\n"


@needs_lt
@pytest.mark.parametrize("text", PROBES)
def test_probes_never_applied_with_real_languagetool(text):
    assert run(text, skip=frozenset({"typos"})).text == text
    assert run(text, skip=frozenset({"typos"}), ep=FakeEndpoint(choice="KEEP")).text == text


@needs_lt
def test_real_lt_still_fixes_the_the_and_of_of():
    assert run("We got the the scheme, most of of it.\n", skip=frozenset({"typos"})).text == \
        "We got the scheme, most of it.\n"


# Final review, issue 4: an indented code block is code.

CODE = "Some prose here.\n\n    print(a,b)\n    # of of\n    x = recieve(a)  # the the\n\nMore prose.\n"


def test_indented_code_untouched_without_tools(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    r = run(CODE)
    assert r.text == CODE and r.review == []


@needs_lt
def test_indented_code_untouched_with_tools():
    r = run(CODE)
    assert r.text == CODE
