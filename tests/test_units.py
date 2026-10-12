import json
from pathlib import Path

from adjudicate import apply_edits

from proofix.commas import comma_edits, only_commas, sentences
from proofix.lt import parse_matches
from proofix.tiers import partition

FIX = Path(__file__).parent / "fixtures" / "typos.lt.json"
EXAMPLE = Path(__file__).parent.parent / "examples" / "typos.md"


def test_parse_matches_offsets_line_up():
    text = EXAMPLE.read_text()
    ms = parse_matches(FIX.read_text(), text)
    assert len(ms) >= 15
    by_rule = {}
    for m in ms:
        by_rule.setdefault(m.rule, []).append(text[m.start:m.end])
    assert "Teh" in by_rule["MORFOLOGIK_RULE_EN_US"] and "recieved" in by_rule["MORFOLOGIK_RULE_EN_US"]
    assert "the the" in by_rule["ENGLISH_WORD_REPEAT_RULE"]
    assert ".." in by_rule["DOUBLE_PUNCTUATION"]


def test_partition_tiers_and_protection():
    text = EXAMPLE.read_text()
    ms = parse_matches(FIX.read_text(), text)
    part = partition(text, ms, accept={"programme"})
    det = {(e.original, e.replacement) for e in part.edits}
    assert (" ,", ",") in det or (" ,", ", ") in det
    assert ("the the", "the") in det
    words = {it.word for it in part.items}
    assert ".." in words                  # two answers ('.' or '…'): the audit routes it to the model
    assert {"Teh", "recieved", "definately", "seperate"} <= words
    assert "programme" not in words and "analysed" not in words   # accept list; variety is stylefix's job
    assert "teh" in words and "`recieved`" not in words   # link text is prose; code is protected
    # a grammar suggestion is adjudicated, not applied
    assert any(it.source == "grammar" for it in part.items)
    new, skipped = apply_edits(text, part.edits)
    assert not skipped and "the the" not in new and "report ," not in new and "said ." not in new
    assert "issues; the" in new and "http://x.y/teh" in new


def test_sentences_skip_headings_tables_code():
    text = "# Head line here\n\nOne two three four five. Six seven eight nine ten!\n\n| a | b |\n|---|---|\n\n```\ncode one two three four\n```\n\nLast one here now.\n"
    ss = sentences(text, gate=False)
    assert [s.text for s in ss] == ["One two three four five.", "Six seven eight nine ten!", "Last one here now."]
    assert text[ss[1].start:ss[1].end] == ss[1].text


def test_only_commas_guard():
    assert only_commas("The committee, decided.", "The committee decided.")
    assert only_commas("After the meeting we went.", "After the meeting, we went.")
    assert not only_commas("It was late, we went.", "It was late; we went.")
    assert not only_commas("The committee decided.", "The committee decided")


def test_comma_edits_apply():
    text = "Intro.\n\nThe committee, decided to go. The report which came, was ignored.\n"
    ss = sentences(text, gate=False)
    edits = comma_edits(text, ss[0], "The committee decided to go.") + \
        comma_edits(text, ss[1], "The report, which came, was ignored.")
    new, skipped = apply_edits(text, edits)
    assert not skipped
    assert new == "Intro.\n\nThe committee decided to go. The report, which came, was ignored.\n"


def test_gate():
    from proofix.commas import needs_check
    assert needs_check("The committee, decided to go.")
    assert needs_check("After the meeting we went home.")
    assert needs_check("The report which came was ignored.")
    assert not needs_check("The committee approved the budget.")
    assert not needs_check("We went home early and slept.")
    assert needs_check("We went home early and slept soundly because the day had been very long indeed.")


def test_typos_edits_respect_protection():
    from proofix.tiers import typos_edits, typos_matches
    text = "# T\n\nTeh `recieved` report was recieved.\n"
    js = ('{"type":"typo","path":"-","line_num":3,"byte_offset":0,"typo":"Teh","corrections":["The"]}\n'
          '{"type":"typo","path":"-","line_num":3,"byte_offset":5,"typo":"recieved","corrections":["received"]}\n'
          '{"type":"typo","path":"-","line_num":3,"byte_offset":26,"typo":"recieved","corrections":["received"]}\n')
    edits = typos_edits(text, js)
    # "Teh" is capitalised, so not provably right (a name?): held back for the model or review
    assert [(e.start, e.original) for e in edits] == [(27, "recieved")]
    assert [m.start for m in typos_matches(text, js)] == [5]
    new, skipped = apply_edits(text, edits)
    assert new == "# T\n\nTeh `recieved` report was received.\n" and not skipped


def test_sentences_span_links_and_map_back():
    from proofix.commas import PLACEHOLDER
    text = "Intro.\n\nThe report, [which came](http://x.y/z) was ignored by `code` and all.\n"
    ss = sentences(text, gate=False)
    assert len(ss) == 1
    s = ss[0]
    assert s.text == f"The report, [which came{PLACEHOLDER} was ignored by {PLACEHOLDER} and all."
    edits = comma_edits(text, s, f"The report, [which came{PLACEHOLDER}, was ignored by {PLACEHOLDER} and all.")
    new, skipped = apply_edits(text, edits)
    assert not skipped
    assert new == "Intro.\n\nThe report, [which came](http://x.y/z), was ignored by `code` and all.\n"
    # an edit inside a placeholder is refused
    assert comma_edits(text, s, s.text.replace(PLACEHOLDER, "⟨,…⟩", 1)) == []


def test_consistency_families_and_items():
    from proofix.consistency import items
    text = ("Third-party auditors and third-party monitors are needed. Every third party risk "
            "assessor must be third-party certified. A `third party` in code. Frontier AI labs, "
            "frontier AI labs, frontier AI labs and Frontier AI labs again. America's edge.\n")
    its, groups = items(text)
    words = {i.word: i.options for i in its}
    assert words["third party"] == ("third-party", "KEEP")           # the minority shape
    assert "third-party" not in words and "Third-party" not in words  # the majority is not asked
    assert words["Frontier AI"][0] == "frontier AI"
    assert "s edge" not in words


def test_compound_candidates_from_dictionary():
    from proofix.consistency import items, split_closed
    text = ("A cyber attack hit the non-profit. Another cyber attack. The longterm plan is a "
            "long-term plan in the long term. It may be that we can not e-mail them. The geopolitical "
            "and geo-political views. Highly regarded people.\n")
    its, groups = items(text, "us")
    words = {i.word: i.options for i in its}
    assert words["cyber attack"] == ("cyberattack", "KEEP")
    assert len(groups[next(i.id for i in its if i.word == "cyber attack")]) == 2   # one decision, two places
    assert words["non-profit"] == ("nonprofit", "KEEP")
    assert words["longterm"] == ("long-term", "long term", "KEEP")
    assert words["geo-political"] == ("geopolitical", "KEEP") and "geopolitical" not in words
    assert "may be" not in words and "can not" not in words
    assert words["e-mail"] == ("email", "KEEP")
    assert split_closed("thirdparty", "us") == ("third", "party")
    assert split_closed("cyberattack", "us") is None


def test_hygiene_entities_and_invisibles():
    from proofix.hygiene import normalize_text, plan
    text = ("correct&#32;**bold** &quot;q&quot; an em&mdash;dash, keep &lt;tag&gt; and &amp;co and "
            "&nbsp;here, zero\u200bwidth, soft\u00adhyphen, code `&#32;` and [x](http://a/b&#32;c).\r\n"
            "caf\u0065\u0301\r\n")
    text, notes = normalize_text(text)
    assert "\r" not in text and "café" in text and len(notes) == 2
    edits, plan_notes = plan(text)
    new, skipped = apply_edits(text, edits)
    assert not skipped
    assert new.startswith("correct **bold** \"q\" an em—dash, keep &lt;tag&gt; and &amp;co and &nbsp;here, "
                          "zerowidth, softhyphen, code `&#32;` and [x](http://a/b&#32;c).\n")
    msgs = " ".join(n.message for n in plan_notes)
    assert "&lt;" in msgs and "&amp;" in msgs and "&nbsp;" in msgs and "zero-width space" in msgs


def _lt_json(text, start16, len16, ctx_text, ctx16, replacement=" ", rule="CONSECUTIVE_SPACES"):
    """A LanguageTool match as LanguageTool writes it: offsets in UTF-16 code units."""
    return json.dumps({"matches": [{
        "message": "m", "shortMessage": "", "replacements": [{"value": replacement}],
        "offset": start16, "length": len16,
        "context": {"text": ctx_text, "offset": ctx16, "length": len16},
        "rule": {"id": rule, "category": {"id": "TYPOGRAPHY"}}}]})


def _u16(s):
    return len(s.encode("utf-16-le")) // 2


def test_parse_matches_converts_utf16_offsets_from_languagetool():
    # Captured from LanguageTool 6.8: offset 17 for the double space at code point 15.
    text = "😀😀 It was nice,  ok.\n"
    ms = parse_matches(_lt_json(text, 17, 2, "😀😀 It was nice,  ok. ", 17), text)
    assert [(m.start, m.end) for m in ms] == [(15, 17)]
    assert text[ms[0].start:ms[0].end] == "  "


def test_parse_matches_utf16_zwj_sequence_in_document_and_context():
    head = "Line one 👩\u200d💻 and 🏳\ufe0f\u200d🌈 here.\n" * 3
    text = head + "Then teh word.\n"
    pos = text.index("teh")
    ctx = "🏳\ufe0f\u200d🌈 here.\nThen teh word."
    ms = parse_matches(_lt_json(text, _u16(text[:pos]), 3, ctx, _u16(ctx[:ctx.index("teh")]),
                                "the", "MORFOLOGIK_RULE_EN_US"), text)
    assert [(m.start, m.end) for m in ms] == [(pos, pos + 3)]


def test_proofix_fix_lands_on_the_right_characters_after_emoji(monkeypatch):
    from adjudicate import Cache
    from proofix import api
    from proofix.commas import CommaCache
    text = "😀😀 It was nice,  ok.\n"
    fixture = _lt_json(text, 17, 2, "😀😀 It was nice,  ok. ", 17, rule="WHITESPACE_RULE")
    monkeypatch.setattr(api, "find_languagetool", lambda: "/replay/languagetool")
    monkeypatch.setattr(api, "run_languagetool", lambda t, target, level="default", **kw: parse_matches(fixture, t))
    r = api.fix(text, api.Options(skip=frozenset({"commas"})), None, Cache(None), CommaCache(None))
    assert r.text == "😀😀 It was nice, ok.\n"


def test_hygiene_keeps_zwj_inside_emoji_sequences():
    from proofix.hygiene import plan
    for seq in ("👩\u200d💻", "👩\u200d👩\u200d👧", "🏳\ufe0f\u200d🌈", "👋🏽\u200d🙂", "❤\ufe0f\u200d🔥"):
        text = f"Look {seq} here.\n"
        edits, notes = plan(text)
        new, _ = apply_edits(text, edits)
        assert new == text, seq
    text = "zero\u200dwidth joiner and \u200d alone\n"
    new, _ = apply_edits(text, plan(text)[0])
    assert new == "zerowidth joiner and  alone\n"
