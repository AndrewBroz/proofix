"""The audit of edits applied outright (spec §7): every source that edits without
the model is right in every prose context, or its matches go to the model (KEEP
offered) or to review. LanguageTool is replayed from real output recorded for
each risky input (fixtures/audit.lt.json); the last tests run the real tools."""

import json
import shutil
from pathlib import Path

import pytest

from adjudicate import Cache
from adjudicate.llm import Endpoint
from proofix import api, hygiene, tiers
from proofix.commas import CommaCache
from proofix.lt import parse_matches

FIX = json.loads((Path(__file__).parent / "fixtures" / "audit.lt.json").read_text(encoding="utf-8"))


class FakeEndpoint(Endpoint):
    """Answers every item with the same choice; remembers the prompts."""

    def __init__(self, choice):
        super().__init__(url="http://x/v1", model="m")
        self.choice, self.prompts = choice, []

    def chat(self, system, user, timeout=None, json_mode=True):
        if "comma placement" in system:
            return json.dumps({"changed": [], "notes": []})
        self.prompts.append(user)
        ids = [int(line.split()[1]) for line in user.splitlines() if line.startswith("Item ")]
        return json.dumps({"decisions": [{"id": i, "choice": self.choice, "confidence": 0.95,
                                          "reason": "t"} for i in ids]})


def replay(monkeypatch, text, lang="en-US", fixture=None):
    """Replay LanguageTool's recorded output for `text` on every call (pass 2 sees only
    the matches whose context survives, as parse_matches's search does), honouring
    `disabled` as LanguageTool does. Returns the list of `disabled` values passed."""
    fixture = fixture or FIX[f"{lang}|{text}"]
    calls = []
    monkeypatch.setenv("PATH", "/nonexistent")                     # no typos
    monkeypatch.setattr(api, "find_languagetool", lambda: "/replay/languagetool")

    def run(t, target, level="default", disabled=None, timeout=600, server=None, ensure=None):
        calls.append(disabled)
        return [m for m in parse_matches(fixture, t) if m.rule not in (disabled or ())]
    monkeypatch.setattr(api, "run_languagetool", run)
    return calls


def fix(text, ep=None, **kw):
    kw.setdefault("skip", frozenset({"commas"}))
    return api.fix(text, api.Options(**kw), ep, Cache(None), CommaCache(None))


def reviewed(r, check):
    return [i for i in r.review if i.check == check]


# --- kept: applied outright ---------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("Ok,, then.\n", "Ok, then.\n"),                                   # DOUBLE_PUNCTUATION, one answer
    ("We saw a  b in the text.\n", "We saw a b in the text.\n"),       # WHITESPACE_RULE mid-line
    ("He said , ok.\n", "He said, ok.\n"),                             # COMMA_PARENTHESIS_WHITESPACE
    ("Hello ,world.\n", "Hello, world.\n"),
    ("It was big ( as noted ) here.\n", "It was big (as noted) here.\n"),
    ("We split the issues;the budget.\n", "We split the issues; the budget.\n"),   # MISSING_SPACE
])
def test_kept_rules_apply(monkeypatch, text, expected):
    replay(monkeypatch, text)
    r = fix(text)
    assert r.text == expected
    assert not r.review


# --- not faults: left alone, nothing to review ---------------------------------------

@pytest.mark.parametrize("text", [
    "Really?! That is odd.\n", "Wow!! That is odd.\n", "Wait... then it rained.\n",
    "See ../notes for more.\n", "...and then it rained.\n", "(as noted) the plan held.\n",
    "— a dash opening the line.\n",
    "This line ends with a break  \nand continues here.\n",           # Markdown hard line break
    "| Name    | Value |\n| ------- | ----- |\n| alpha   | 1     |\n",  # aligned table
    "Intro.\n\n    x  =  1\n",                                          # indented code
    "A ratio a:b here.\n", "Smile :D now.\n", "Say hello :wave: there.\n",   # MISSING_SPACE's old misfires
])
def test_not_faults_left_alone(monkeypatch, text):
    replay(monkeypatch, text)
    r = fix(text)
    assert r.text == text
    assert not [i for i in r.review if i.check in tiers.DETERMINISTIC | {"MISSING_SPACE"}]


# --- moved: a model decides (KEEP offered), else review --------------------------------

MOVED = [
    ("Run cd .. now.\n", "DOUBLE_PUNCTUATION", ".."),              # a path, not a typo
    ("The teams..\n", "DOUBLE_PUNCTUATION", ".."),                 # '.' or '…': two answers
    (", and so on.\n", "UNLIKELY_OPENING_PUNCTUATION", ","),       # may continue a quote or list
    ("Use Node.Js here.\n", "SENTENCE_WHITESPACE", "Js"),          # a name, not two sentences
    ("The result was clear.However it rained.\n", "SENTENCE_WHITESPACE", "However"),
    ("Read the .bashrc file.\n", "COMMA_PARENTHESIS_WHITESPACE", " ."),
    ("Smile ( ͡° ͜ʖ ͡°) ok.\n", "COMMA_PARENTHESIS_WHITESPACE", "("),
    ("Ellipsis . . . spaced.\n", "COMMA_PARENTHESIS_WHITESPACE", " ."),
    ("The fee was 1 ,000 dollars.\n", "COMMA_PARENTHESIS_WHITESPACE", ","),
]


@pytest.mark.parametrize("text, rule, part", MOVED)
def test_moved_rules_reviewed_without_model(monkeypatch, text, rule, part):
    replay(monkeypatch, text)
    r = fix(text)
    assert r.text == text
    assert reviewed(r, rule)


@pytest.mark.parametrize("text, rule, part", MOVED)
def test_moved_rules_adjudicated_with_model(monkeypatch, text, rule, part):
    replay(monkeypatch, text)
    ep = FakeEndpoint("KEEP")
    r = fix(text, ep)
    assert r.text == text
    items = [d.item for d in r.decisions if rule in d.item.rule]
    assert items and all(i.options[-1] == "KEEP" for i in items)
    assert any(part in i.word or part in i.context for i in items)


@pytest.mark.parametrize("text, expected", [
    ("The teams..\n", "The teams.\n"),
    (", and so on.\n", "and so on.\n"),
    ("Use Node.Js here.\n", "Use Node. Js here.\n"),          # the fake model always picks option 1
])
def test_moved_rules_applied_when_the_model_chooses(monkeypatch, text, expected):
    replay(monkeypatch, text)
    assert fix(text, FakeEndpoint(1)).text == expected


def test_moved_rule_options_are_readable(monkeypatch):
    """No option is blank or a bare space: the item spans the neighbouring characters."""
    for text in (", and so on.\n", "The result was clear.However it rained.\n"):
        replay(monkeypatch, text)
        r = fix(text, FakeEndpoint("KEEP"))
        for d in r.decisions:
            assert all(o.strip() == o and o for o in d.item.options), d.item


def test_moved_rules_reviewed_when_pass_2_is_skipped(monkeypatch):
    replay(monkeypatch, "The teams..\n")
    r = fix("The teams..\n", skip=frozenset({"commas", "adjudicated"}))
    assert r.text == "The teams..\n"
    assert reviewed(r, "DOUBLE_PUNCTUATION")


# --- an outright edit that loses to an overlapping one is reviewed, not dropped ---------

def _lt(text, *ms):
    return json.dumps({"matches": [{
        "message": msg, "shortMessage": "", "replacements": [{"value": rep}],
        "offset": off, "length": len(orig), "context": {"text": text, "offset": off, "length": len(orig)},
        "rule": {"id": rule, "category": {"id": "PUNCTUATION"}}} for rule, off, orig, rep, msg in ms]})


def test_rejected_outright_edit_is_reviewed(monkeypatch):
    text = "Ok,,, then.\n"
    replay(monkeypatch, text, fixture=_lt(text, ("DOUBLE_PUNCTUATION", 2, ",,", ",", "Two commas"),
                                          ("DOUBLE_PUNCTUATION", 3, ",,", ",", "Two commas")))
    r = fix(text, skip=frozenset({"commas", "adjudicated"}))         # no pass 2 to fix it
    assert r.text == "Ok,, then.\n"
    msgs = [i.message for i in reviewed(r, "DOUBLE_PUNCTUATION")]
    assert len(msgs) == 1 and "',,'" in msgs[0] and "overlap" in msgs[0]
    r = fix(text)                                                      # pass 2 fixes it: no review
    assert r.text == "Ok, then.\n" and not r.review


# --- the variety filter -------------------------------------------------------------

def test_uk_ise_disables_oxford_spelling(monkeypatch):
    text = "Finalise and organise the programme.\n"
    calls = replay(monkeypatch, text, lang="en-GB")
    r = fix(text, target="uk")
    assert r.text == text and not r.review
    assert calls and all(set(c or ()) >= {"OXFORD_SPELLING_Z_NOT_S"} for c in calls)


def test_uk_oxford_ize_keeps_oxford_spelling(monkeypatch):
    text = "Finalise and organise the programme.\n"
    calls = replay(monkeypatch, text, lang="en-GB")
    r = fix(text, target="uk", oxford_ize=True)
    assert reviewed(r, "OXFORD_SPELLING_Z_NOT_S")
    assert all("OXFORD_SPELLING_Z_NOT_S" not in (c or ()) for c in calls)


def test_us_british_spellings_are_not_flagged(monkeypatch):
    # en-US reports colour/organise only as MORFOLOGIK_RULE_EN_US "... is British English"
    text = "The colour of the organise plan.\n"
    calls = replay(monkeypatch, text)
    r = fix(text, target="us")
    assert r.text == text and not r.review
    assert calls and all(set(c or ()) - api.LT_DISABLED == set(tiers.VARIETY_DISABLED["us"]) for c in calls)


def test_variety_disabled_lists():
    assert "OXFORD_SPELLING_Z_NOT_S" in tiers.VARIETY_DISABLED["uk_ise"]
    assert set(tiers.VARIETY_DISABLED) == {"uk_ise", "us"}


# --- typos ---------------------------------------------------------------------------

def _typo(line, offset, typo, *corrections):
    return json.dumps({"type": "typo", "path": "-", "line_num": line, "byte_offset": offset,
                       "typo": typo, "corrections": list(corrections)})


@pytest.mark.parametrize("text, typo, offset, corrections, outright", [
    ("teh plan\n", "teh", 0, ["the"], True),
    ("A teh-based plan\n", "teh", 2, ["the"], True),
    ("Teh plan\n", "Teh", 0, ["The"], False),               # capitalised: may be a name
    ("TEH plan\n", "TEH", 0, ["THE"], False),
    ("Use foo_bar_teh here\n", "teh", 12, ["the"], False),  # an identifier
    ("Use teh2 here\n", "teh", 4, ["the"], False),
    ("Wir sind alle hier\n", "alle", 9, ["all", "alley"], False),   # two answers: not provably one
])
def test_typos_edits_only_when_provably_right(text, typo, offset, corrections, outright):
    out = _typo(1, offset, typo, *corrections)
    edits = tiers.typos_edits(text, out)
    assert bool(edits) == outright
    held = tiers.typos_matches(text, out)
    assert bool(held) != outright
    if held:
        assert held[0].rule == "typos" and held[0].replacements == corrections


def test_typos_accept_list():
    out = _typo(1, 0, "teh", "the")
    assert tiers.typos_edits("teh plan\n", out, accept={"teh"}) == []
    assert tiers.typos_matches("teh plan\n", out, accept={"teh"}) == []


needs_typos = pytest.mark.skipif(shutil.which("typos") is None, reason="typos not installed")


def _no_lt(monkeypatch):
    monkeypatch.setattr(api, "find_languagetool", lambda: None)


@needs_typos
def test_typos_lower_case_fixed(monkeypatch):
    _no_lt(monkeypatch)
    assert fix("We saw teh plan.\n").text == "We saw the plan.\n"


@needs_typos
@pytest.mark.parametrize("text", ["Teh plan was late.\n", "Use foo_bar_teh here.\n",
                                  "Use teh2 here.\n", "Wir sind alle hier.\n"])
def test_typos_held_back_reviewed_without_model(monkeypatch, text):
    _no_lt(monkeypatch)
    r = fix(text)
    assert r.text == text
    assert reviewed(r, "typos")


@needs_typos
def test_typos_held_back_adjudicated_with_model(monkeypatch):
    _no_lt(monkeypatch)
    keep = FakeEndpoint("KEEP")
    r = fix("Teh plan was late.\n", keep, skip=frozenset({"commas", "compounds"}))
    assert r.text == "Teh plan was late.\n"
    item = next(d.item for d in r.decisions if d.item.word == "Teh")
    assert item.rule == tiers.SPELLING_RULE and item.options == ("The", "KEEP")
    pick = FakeEndpoint(1)
    assert fix("Teh plan was late.\n", pick, skip=frozenset({"commas", "compounds"})).text == \
        "The plan was late.\n"


# --- hygiene -------------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "2020&#46; A good year.\n",                 # would become an ordered list
    "&#45; not a list\n",                       # a bullet
    "&#43; not a list\n",
    "&#32;&#32;&#32;&#32;not code\n",           # an indented code block
    "Heading\n&#61;&#61;&#61;\n",               # a setext underline
    "Hard&#32;&#32;\nbreak\n",                  # a hard line break
])
def test_hygiene_leaves_entities_that_would_become_markdown(text):
    edits, notes = hygiene.plan(text)
    assert edits == [] and notes


def test_hygiene_still_decodes_mid_line_entities():
    edits, _ = hygiene.plan("Monitor&#32;**x** &mdash; see.\n")
    assert [e.replacement for e in edits] == [" ", "—"]


def test_hygiene_keeps_zwnj_in_scripts_that_use_it():
    text = "Persian می‌خواهم here; Hindi क्‌ष.\n"
    edits, _ = hygiene.plan(text)
    assert edits == []
    edits, _ = hygiene.plan("A wo‌rd here.\n")
    assert len(edits) == 1


# --- real tools ----------------------------------------------------------------------

needs_lt = pytest.mark.skipif(shutil.which("languagetool") is None, reason="LanguageTool not installed")


@needs_lt
def test_audit_with_real_languagetool(monkeypatch):
    text = ("Read the .bashrc file. Run cd .. now. Use Node.Js here. He said , ok.\n"
            "This line ends with a break  \nand we saw a  b today.\n")
    r = fix(text, skip=frozenset({"commas", "typos"}))
    assert r.text == ("Read the .bashrc file. Run cd .. now. Use Node.Js here. He said, ok.\n"
                      "This line ends with a break  \nand we saw a b today.\n")
    checks = {i.check for i in r.review}
    assert {"COMMA_PARENTHESIS_WHITESPACE", "DOUBLE_PUNCTUATION", "SENTENCE_WHITESPACE"} <= checks


@needs_lt
def test_real_languagetool_honours_the_variety_filter():
    text = "Finalise and organise the programme.\n"
    skip = frozenset({"commas", "typos"})
    assert not fix(text, target="uk", skip=skip).review
    assert reviewed(fix(text, target="uk", oxford_ize=True, skip=skip), "OXFORD_SPELLING_Z_NOT_S")


# --- fix round 1 ----------------------------------------------------------------------

@pytest.mark.parametrize("typo, correction", [
    ("youre", "your"), ("whos", "whose"), ("hed", "head"), ("ein", "in"), ("ser", "set"),
    ("som", "some"), ("nam", "name"), ("seh", "she"), ("nto", "not"), ("dont", "do"),
])
def test_typos_short_or_contraction_shaped_held_back(typo, correction):
    text = f"x {typo} y\n"
    out = _typo(1, 2, typo, correction)
    assert tiers.typos_edits(text, out) == []
    assert [m.rule for m in tiers.typos_matches(text, out)] == ["typos"]


@pytest.mark.parametrize("typo, correction", [
    ("teh", "the"), ("hte", "the"), ("adn", "and"), ("wih", "with"), ("nd", "and"),
    ("recieve", "receive"), ("seperate", "separate"), ("mesage", "message"), ("shoud", "should"),
    ("thier", "their"), ("reccomend", "recommend"), ("occured", "occurred"), ("anual", "annual"),
    ("ment", "meant"),          # four letters, not contraction-shaped: rule (a) applies it
])
def test_typos_safe_list_and_long_typos_applied(typo, correction):
    text = f"x {typo} y\n"
    assert [e.replacement for e in tiers.typos_edits(text, _typo(1, 2, typo, correction))] == [correction]


def test_safe_short_typos_list():
    assert {"teh", "hte", "adn", "wih", "nd"} <= tiers.SAFE_SHORT_TYPOS
    assert not {"ein", "ser", "som", "nam", "seh", "nto", "hed"} & tiers.SAFE_SHORT_TYPOS
    assert all(len(w) <= 3 and w.isalpha() and w.islower() for w in tiers.SAFE_SHORT_TYPOS)


@needs_typos
@pytest.mark.parametrize("text", ["Thanks, youre welcome.\n", "Wir sind ein Team.\n",
                                  "He hit his hed on it.\n"])
def test_real_typos_wrong_short_fixes_held_back(monkeypatch, text):
    _no_lt(monkeypatch)
    r = fix(text)
    assert r.text == text and reviewed(r, "typos")


@pytest.mark.parametrize("text, expected", [
    ("The report ,,and more.\n", "The report, and more.\n"),
    ("He said ,, that it was.\n", "He said, that it was.\n"),
])
def test_pass_2_applies_audited_edits_pass_1_lost(monkeypatch, text, expected):
    """Pass 1's overlapping edits lose to each other; pass 2's outright edits are kept,
    and the overlap review they resolve is dropped."""
    fixture = {"The report ,,and more.\n": _lt(text, ("COMMA_PARENTHESIS_WHITESPACE", 10, " ,", ", ", "sp"),
                                                 ("DOUBLE_PUNCTUATION", 11, ",,", ",", "two")),
               "He said ,, that it was.\n": _lt(text, ("COMMA_PARENTHESIS_WHITESPACE", 7, " ,", ",", "sp"),
                                                 ("DOUBLE_PUNCTUATION", 8, ",,", ",", "two"))}[text]
    replay(monkeypatch, text, fixture=fixture)
    r = fix(text)
    assert r.text == expected
    assert not [i for i in r.review if "overlap" in i.message]


@pytest.mark.parametrize("text", [
    "​# not a heading\n", "2020⁠. A good year\n", "⁠- not a bullet\n",
    "​> not a quote\n", "Use ​#tag literally\n", "Mail ​@someone here\n",
    "a​*b* c\n", "**bold**​text\n",
])
def test_hygiene_keeps_invisible_characters_that_hold_off_markdown(text):
    edits, notes = hygiene.plan(text)
    assert edits == [] and notes


def test_hygiene_still_removes_stray_invisible_characters():
    for text in ("A wo​rd here.\n", "Soft­hyphen here.\n", "ten⁠km away\n"):
        edits, _ = hygiene.plan(text)
        assert len(edits) == 1, text


# --- fix round 2 ----------------------------------------------------------------------

@pytest.mark.parametrize("typo, correction", [
    ("wheres", "whereas"), ("hows", "how"), ("whys", "why"), ("whens", "when"),
    ("couldve", "could"), ("shouldve", "should"), ("wouldve", "would"), ("mustve", "must"),
    ("mightve", "might"),
])
def test_more_contraction_shaped_typos_held_back(typo, correction):
    text = f"x {typo} y\n"
    out = _typo(1, 2, typo, correction)
    assert tiers.typos_edits(text, out) == []
    assert tiers.typos_matches(text, out)


def test_plausible_short_words_not_on_the_safe_list():
    assert not {"vey", "wth", "wya", "ehr", "hsa", "suh"} & tiers.SAFE_SHORT_TYPOS


@needs_typos
@pytest.mark.parametrize("text", ["So wheres the file?\n", "Oy vey, what a day.\n"])
def test_real_typos_wheres_and_oy_vey_not_applied(monkeypatch, text):
    _no_lt(monkeypatch)
    assert fix(text).text == text


# --- final review: review items name the flagged text; lines are the input's ---------

def test_review_message_names_the_flagged_text(monkeypatch):
    text = "The colur is nice.Then we went.\n"
    replay(monkeypatch, text, fixture=_lt(
        text, ("MORFOLOGIK_RULE_EN_US", 4, "colur", "color", "Possible spelling mistake found."),
        ("SENTENCE_WHITESPACE", 18, "Then", " Then", "Add a space between sentences.")))
    r = fix(text)
    assert sorted(i.message for i in r.review) == [
        "'Then': Add a space between sentences.", "'colur': Possible spelling mistake found."]
    assert sorted(m.message for m in r.lt_review)[1].startswith("'colur': ")   # proofix's own CLI too


def test_review_message_that_quotes_the_text_is_unchanged(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")
    r = fix("How do you know that that is wrong?\n")
    assert [i.message for i in r.review] == ["repeated word 'that that'"]


@needs_typos
def test_typos_message_preferred_over_languagetool_at_the_same_word(monkeypatch):
    text = "Teh plan was late.\n"
    typos_dir = str(Path(shutil.which("typos")).parent)
    replay(monkeypatch, text, fixture=_lt(
        text, ("MORFOLOGIK_RULE_EN_US", 0, "Teh", "The", "Possible spelling mistake found.")))
    monkeypatch.setenv("PATH", typos_dir)
    r = fix(text)
    assert [i.message for i in r.review] == ["'Teh' may be a misspelling of 'The'"]


def test_review_lines_are_the_input_lines_after_a_cross_line_double(monkeypatch):
    text = "We saw the\nthe dog.\n\nThe colur is nice.\n"
    off = text.index("colur")
    replay(monkeypatch, text, fixture=_lt(
        text, ("ENGLISH_WORD_REPEAT_RULE", 7, "the\nthe", "the", "Possible typo: you repeated a word."),
        ("MORFOLOGIK_RULE_EN_US", off, "colur", "color", "Possible spelling mistake found.")))
    r = fix(text)
    assert r.text == "We saw the dog.\n\nThe colur is nice.\n"
    assert [(i.line, i.message) for i in r.review] == [(4, "'colur': Possible spelling mistake found.")]


def test_undecided_lines_are_the_input_lines(monkeypatch):
    monkeypatch.setenv("PATH", "/nonexistent")

    class Unsure(FakeEndpoint):
        def chat(self, system, user, timeout=None, json_mode=True):
            if "comma placement" in system:
                return json.dumps({"changed": [], "notes": []})
            ids = [int(l.split()[1]) for l in user.splitlines() if l.startswith("Item ")]
            return json.dumps({"decisions": [{"id": i, "choice": 1, "confidence": 0.1,
                                              "reason": "t"} for i in ids]})
    r = fix("We saw the\nthe dog.\n\nIt was very very good.\n", Unsure(1),
            skip=frozenset({"commas", "compounds"}))
    assert [i.line for i in r.review if i.kind == "undecided"] == [4]


def test_overlap_review_names_the_underlying_rule(monkeypatch):
    text = "We recieve it now.\n"
    replay(monkeypatch, text, fixture=_lt(
        text, ("GRAMMAR_A", 3, "recieve it", "receive it", "Spelling."),
        ("GRAMMAR_B", 11, "it now", "it then", "Tense.")))
    r = fix(text, FakeEndpoint(1), skip=frozenset({"commas", "compounds"}))
    assert r.text == "We recieve it then.\n"
    [lost] = [i for i in r.review if "overlaps another edit" in i.message]
    assert "GRAMMAR_A" in lost.message


@needs_typos
@pytest.mark.parametrize("text, expected, labels", [
    ("We saw teh the the plan.\n", "We saw the plan.\n", ["typos", "basics"]),
    ("We saw teh the plan.\n", "We saw the plan.\n", ["typos", "basics"]),
])
def test_typo_that_makes_a_double_keeps_both_labels(monkeypatch, text, expected, labels):
    # pass 1 fixes the typo, which makes (another) double that pass 2 removes: the
    # changes are labelled from both passes, not left as one unlabelled deletion
    _no_lt(monkeypatch)
    r = fix(text)
    assert r.text == expected
    assert [c.check for c in r.changes] == labels
    from adjudicate import apply_changes
    assert apply_changes(text, r.changes) == expected
