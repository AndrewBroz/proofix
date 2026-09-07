"""Within-document consistency: the same term written two ways.

Families are found deterministically (hyphenated / open / closed compounds;
capitalisation of multi-word terms). The minority occurrences become
adjudication items: the model chooses the majority form or KEEP, using the
rule that distinguishes the cases where both are right ("third-party
assessors" but "a third party")."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from adjudicate import Item, at_sentence_start, protected_spans, sentence_context

_WORD = r"[A-Za-z][A-Za-z']*"
_COMPOUND = re.compile(rf"\b(?=({_WORD})([- ])({_WORD})\b)")   # lookahead: overlapping pairs
_CLOSED = re.compile(r"\b([A-Za-z]{5,})\b")
_TERM2 = re.compile(rf"\b(?=({_WORD} {_WORD})\b)")

COMPOUND_RULE = ("This document writes the same compound both ways: {counts}. A compound modifier "
                 "before a noun is hyphenated ('third-party assessors', 'long-term plan'); as a "
                 "noun phrase it is open ('a third party', 'in the long term'). Choose the form "
                 "that is correct at this position, or KEEP.")
CASE_RULE = ("This document capitalises the same term inconsistently: {counts}. Choose the "
             "document's dominant form unless this occurrence is a heading, a sentence start, "
             "or a different sense (a proper name versus a common noun).")


@dataclass
class Occurrence:
    start: int
    end: int
    form: str


def _in_span(pos: int, spans: list[tuple[int, int]]) -> bool:
    return any(s <= pos < e for s, e in spans)


def compound_families(text: str, spans: list[tuple[int, int]]) -> dict[str, list[Occurrence]]:
    """key (lowercase letters only) -> occurrences of 'a b', 'a-b' forms."""
    fam: dict[str, list[Occurrence]] = defaultdict(list)
    for m in _COMPOUND.finditer(text):
        if _in_span(m.start(), spans):
            continue
        form = m.group(1) + m.group(2) + m.group(3)
        key = (m.group(1) + m.group(3)).lower()
        fam[key].append(Occurrence(m.start(), m.start() + len(form), form))
    # closed forms ("longterm") join a family only if that family already has
    # a hyphenated or open member; a single long word is not evidence.
    open_or_hyphen = {k for k, occ in fam.items() if occ}
    for m in _CLOSED.finditer(text):
        w = m.group(0)
        if _in_span(m.start(), spans) or w.lower() not in open_or_hyphen:
            continue
        fam[w.lower()].append(Occurrence(m.start(), m.end(), w))
    out = {}
    for key, occ in fam.items():
        shapes = {re.sub(r"[A-Za-z']", "", o.form) or "closed" for o in occ}
        if len(shapes) >= 2:
            out[key] = occ
    return out


def case_families(text: str, spans: list[tuple[int, int]]) -> dict[str, list[Occurrence]]:
    """Two-word terms whose capitalisation varies, ignoring sentence starts."""
    fam: dict[str, list[Occurrence]] = defaultdict(list)
    for m in _TERM2.finditer(text):
        if _in_span(m.start(), spans) or at_sentence_start(text, m.start()):
            continue
        fam[m.group(1).lower()].append(Occurrence(m.start(), m.start() + len(m.group(1)), m.group(1)))
    out = {}
    for key, occ in fam.items():
        forms = {o.form for o in occ}
        if len(forms) >= 2 and len(occ) >= 3 and any(f[0].isupper() for f in forms):
            out[key] = occ
    return out


def _line_col(text: str, pos: int) -> tuple[int, int]:
    return text.count("\n", 0, pos) + 1, pos - (text.rfind("\n", 0, pos) + 1) + 1


def items(text: str, next_id: int = 1) -> list[Item]:
    spans = protected_spans(text)
    out: list[Item] = []
    for kind, families in (("compound", compound_families(text, spans)),
                           ("case", case_families(text, spans))):
        for key, occ in families.items():
            counts = Counter(o.form if kind == "case" else o.form.lower() for o in occ)
            ranked = [f for f, _ in counts.most_common()]
            majority = counts[ranked[0]]
            desc = ", ".join(f"'{f}' {n}×" for f, n in counts.most_common())
            rule = (COMPOUND_RULE if kind == "compound" else CASE_RULE).format(counts=desc)
            for o in occ:
                form_key = o.form if kind == "case" else o.form.lower()
                minority = counts[form_key] * 2 <= majority
                if not minority and counts[form_key] == majority and kind == "compound" and len(ranked) > 1 \
                        and counts[ranked[1]] * 2 > majority:
                    minority = True          # counts are close: ask about every occurrence
                if not minority:
                    continue
                if kind == "case":
                    options = tuple(f for f in ranked if f != o.form) + ("KEEP",)
                else:
                    options = tuple(_recase(o.form, f) for f in ranked if f != form_key) + ("KEEP",)
                line, col = _line_col(text, o.start)
                out.append(Item(next_id, o.form, options, rule, sentence_context(text, o.start, o.end),
                                line, col, col + (o.end - o.start) - 1, source="consistency"))
                next_id += 1
    return out


def _recase(original: str, target_lower: str) -> str:
    """Give the target form the original's capitalisation pattern."""
    if original.isupper():
        return target_lower.upper()
    if original[:1].isupper():
        return target_lower[:1].upper() + target_lower[1:]
    return target_lower
