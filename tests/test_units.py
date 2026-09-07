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
    assert ("the the", "the") in det and ("..", ".") in det
    words = {it.word for it in part.items}
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
    from proofix.tiers import typos_edits
    text = "# T\n\nTeh `recieved` report was recieved.\n"
    js = ('{"type":"typo","path":"-","line_num":3,"byte_offset":0,"typo":"Teh","corrections":["The"]}\n'
          '{"type":"typo","path":"-","line_num":3,"byte_offset":5,"typo":"recieved","corrections":["received"]}\n'
          '{"type":"typo","path":"-","line_num":3,"byte_offset":26,"typo":"recieved","corrections":["received"]}\n')
    edits = typos_edits(text, js)
    assert [(e.start, e.original) for e in edits] == [(1, "Teh"), (27, "recieved")]
    new, skipped = apply_edits(text, edits)
    assert new == "# T\n\nThe `recieved` report was received.\n" and not skipped
