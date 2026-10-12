"""Sort LanguageTool matches into deterministic edits, adjudication items
and review, with Markdown structure protected."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace

from adjudicate import Edit, Item, protected_spans, sentence_context

from .basics import CLOSED_DOUBLES, DOUBLE_RULE, Double, double_edit, doubled_words
from .lt import Match

# Rules whose single replacement is mechanical and always right in prose.
# ENGLISH_WORD_REPEAT_RULE is not here: "that that" and "had had" are often
# grammatical, so its matches are routed in partition() as basics routes doubles.
DETERMINISTIC = {
    "COMMA_PARENTHESIS_WHITESPACE", "SENTENCE_WHITESPACE", "DOUBLE_PUNCTUATION",
    "UNLIKELY_OPENING_PUNCTUATION", "WHITESPACE_RULE",
}
REPEAT = "ENGLISH_WORD_REPEAT_RULE"
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


def double_item(text: str, d: Double, item_id: int) -> Item:
    """An adjudication item for a doubled word: one copy (as the first is cased) or KEEP."""
    s = d.first_start
    word = text[s:d.end]
    line, col = _line_col(text, s)
    return Item(item_id, word, (text[s:d.start], "KEEP"), DOUBLE_RULE,
                sentence_context(text, s, d.end), line, col, col + len(word) - 1, source="repeat")


def double_message(text: str, d: Double) -> str:
    return f"repeated word '{d.text}'"


def accepted_double(text: str, d: Double, accept: set[str]) -> bool:
    return d.text.lower() in accept


def double_route(d: Double, adjudicate_available: bool) -> str:
    """"edit" (closed list, provably a typo), "item" (a model decides, KEEP offered)
    or "review". A capitalised closed-list pair ("The The") may be a name: reviewed."""
    if d.capitalised:
        return "review"
    if d.closed:
        return "edit"
    return "item" if adjudicate_available else "review"


def _repeat_double(text: str, m: Match) -> Double | None:
    """A LanguageTool repeat basics' pattern did not match ("3rd 3rd"): routed as a
    double that is not on the closed list, unless it spans a line (the structural
    cases: a blank line, a list, quote, heading or table line) or is a capitalised
    pair that is not on the closed list (a name)."""
    original = text[m.start:m.end]
    parts = original.split()
    if "\n" in original or len(parts) != 2:
        return None
    first, second = parts
    capitalised = first[:1].isupper() and second[:1].isupper()
    if capitalised and first.lower() not in CLOSED_DOUBLES:
        return None
    return Double(m.start + len(first), m.end, second, False, first, capitalised)


def partition(text: str, matches: list[Match], accept: set[str] | None = None,
              adjudicate_available: bool = True) -> Partition:
    accept = {w.lower() for w in (accept or ())}
    spans = protected_spans(text)
    edits: list[Edit] = mechanical_edits(text, spans)
    items: list[Item] = []
    review: list[Match] = []
    next_id = 1
    doubles = {(d.first_start, d.end): d for d in doubled_words(text, spans)}
    for m in matches:
        if m.rule in IGNORED or _overlaps(m, spans):
            continue
        original = text[m.start:m.end]
        if m.rule in SPELLING and original.lower().strip("'’") in accept:
            continue
        if m.rule in SPELLING and ("British English" in m.message or "American English" in m.message):
            continue          # a variety difference, not a typo: stylefix's job
        if m.rule == REPEAT:
            # Routed exactly as basics routes its doubles; see _repeat_double for
            # matches basics' pattern does not cover.
            d = doubles.get((m.start, m.end)) or _repeat_double(text, m)
            if d is None or accepted_double(text, d, accept):
                continue
            route = double_route(d, adjudicate_available)
            if route == "edit":
                edits.append(replace(double_edit(text, d), source=m.rule))
            elif route == "item":
                items.append(double_item(text, d, next_id))
                next_id += 1
            else:
                review.append(replace(m, message=double_message(text, d)))
            continue
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
