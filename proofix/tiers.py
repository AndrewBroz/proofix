"""Sort LanguageTool matches into deterministic edits, adjudication items
and review, with Markdown structure protected."""

from __future__ import annotations

import re
from dataclasses import dataclass

from adjudicate import Edit, Item, protected_spans, sentence_context

from .lt import Match

# Rules whose single replacement is mechanical and always right in prose.
DETERMINISTIC = {
    "COMMA_PARENTHESIS_WHITESPACE", "SENTENCE_WHITESPACE", "DOUBLE_PUNCTUATION",
    "ENGLISH_WORD_REPEAT_RULE", "UNLIKELY_OPENING_PUNCTUATION", "WHITESPACE_RULE",
}
# Rules stylefix owns, or that misfire on Markdown.
IGNORED = {"EN_QUOTES", "DASH_RULE", "PUNCTUATION_PARAGRAPH_END", "UPPERCASE_SENTENCE_START",
           "WORD_CONTAINS_UNDERSCORE", "ARROWS", "MULTIPLICATION_SIGN", "EN_UNPAIRED_BRACKETS"}
SPELLING = {"MORFOLOGIK_RULE_EN_US", "MORFOLOGIK_RULE_EN_GB", "MORFOLOGIK_RULE_EN_AU",
            "MORFOLOGIK_RULE_EN_CA", "MORFOLOGIK_RULE_EN_NZ", "MORFOLOGIK_RULE_EN_ZA"}

SPELLING_RULE = ("Choose the correctly spelled word only if the marked text is an ordinary word "
                 "misspelled. KEEP if it is a name, an acronym, a product, jargon, a code "
                 "identifier, a foreign word, or a deliberate spelling.")


@dataclass
class Partition:
    edits: list[Edit]
    items: list[Item]
    review: list[Match]


def _line_col(text: str, pos: int) -> tuple[int, int]:
    line = text.count("\n", 0, pos) + 1
    col = pos - (text.rfind("\n", 0, pos) + 1) + 1
    return line, col


def to_edit(text: str, m: Match, replacement: str, source: str) -> Edit:
    line, col = _line_col(text, m.start)
    return Edit(line, col, col + m.length - 1, text[m.start:m.end], replacement, source)


def _overlaps(m: Match, spans: list[tuple[int, int]]) -> bool:
    return any(s < m.end and m.start < e for s, e in spans)


def _line_start(text: str, pos: int) -> bool:
    return pos == 0 or text[pos - 1] == "\n" or text[text.rfind("\n", 0, pos) + 1:pos].strip() == ""


_MISSING_SPACE = re.compile(r"[;:](?=[A-Za-z])")   # "issues;the", "note:the"; digits (10:30) excluded


def mechanical_edits(text: str, spans: list[tuple[int, int]]) -> list[Edit]:
    """Faults LanguageTool has no rule for."""
    out = []
    for m in _MISSING_SPACE.finditer(text):
        if any(s <= m.start() < e for s, e in spans):
            continue
        line, col = _line_col(text, m.start())
        out.append(Edit(line, col, col, m.group(0), m.group(0) + " ", "MISSING_SPACE"))
    return out


def partition(text: str, matches: list[Match], accept: set[str] | None = None,
              adjudicate_available: bool = True) -> Partition:
    accept = {w.lower() for w in (accept or ())}
    spans = protected_spans(text)
    edits: list[Edit] = mechanical_edits(text, spans)
    items: list[Item] = []
    review: list[Match] = []
    next_id = 1
    for m in matches:
        if m.rule in IGNORED or _overlaps(m, spans):
            continue
        original = text[m.start:m.end]
        if m.rule in SPELLING and original.lower().strip("'’") in accept:
            continue
        if m.rule in SPELLING and ("British English" in m.message or "American English" in m.message):
            continue          # a variety difference, not a typo: stylefix's job
        if m.rule in DETERMINISTIC and m.replacements:
            if m.rule == "WHITESPACE_RULE" and _line_start(text, m.start):
                continue          # list indentation, code-ish alignment
            edits.append(to_edit(text, m, m.replacements[0], m.rule))
            continue
        if m.replacements and adjudicate_available:
            options = tuple(dict.fromkeys(m.replacements[:4])) + ("KEEP",)
            rule = SPELLING_RULE if m.rule in SPELLING else (
                f"LanguageTool ({m.rule}): {m.message} Choose the option that makes the sentence "
                "correct; KEEP if the original is acceptable or the suggestion changes the meaning.")
            items.append(Item(next_id, original, options, rule,
                              sentence_context(text, m.start, m.end),
                              *_line_col(text, m.start), _line_col(text, m.start)[1] + m.length - 1,
                              source="spelling" if m.rule in SPELLING else "grammar"))
            next_id += 1
            continue
        review.append(m)
    return Partition(edits, items, review)


def typos_edits(text: str, json_lines: str, spans: list[tuple[int, int]] | None = None) -> list[Edit]:
    """Edits from `typos --format json` output, kept out of protected spans."""
    import json
    spans = protected_spans(text) if spans is None else spans
    lines = text.split("\n")
    starts = [0]
    for ln in lines[:-1]:
        starts.append(starts[-1] + len(ln) + 1)
    out = []
    for raw in json_lines.splitlines():
        try:
            d = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if d.get("type") != "typo" or not d.get("corrections"):
            continue
        ln = int(d["line_num"]) - 1
        if ln >= len(lines):
            continue
        line = lines[ln]
        col = len(line.encode("utf-8")[:int(d["byte_offset"])].decode("utf-8", "ignore"))
        word = d["typo"]
        if line[col:col + len(word)] != word:
            continue
        pos = starts[ln] + col
        if any(s <= pos < e for s, e in spans):
            continue
        out.append(Edit(ln + 1, col + 1, col + len(word), word, d["corrections"][0], "typos"))
    return out
