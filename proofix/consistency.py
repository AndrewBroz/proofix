"""Compounds and consistency: how a term is written, decided per occurrence.

Detection is deterministic and dictionary-driven. A candidate is:

- a hyphenated or open pair whose closed form is a word ("cyber attack",
  "e-mail"), or
- a closed word that is not itself a word but splits into two words
  ("longterm", "thirdparty"), or
- any term the document itself writes two ways ("third-party" and "third
  party"; "Frontier AI" and "frontier AI").

The decision is bounded: closed, hyphenated, open, or KEEP. The rule of
thumb names the target variety's dictionary (Merriam-Webster for US, Oxford
for UK) and the grammatical rule that a compound modifier before a noun is
hyphenated while the same words as a noun phrase are open. Closing is
position-independent; hyphen versus open is not, which is why the model
sees the sentence."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from adjudicate import Item, at_sentence_start, known, known_anywhere, protected_spans, sentence_context

_WORD = r"[A-Za-z][A-Za-z']*"
_PAIR = re.compile(rf"(?<!['’\w])(?=({_WORD})([- ])({_WORD})\b)")   # lookahead: overlapping pairs
_TERM2 = re.compile(rf"(?<!['’\w])(?=({_WORD} {_WORD})\b)")
_LONGWORD = re.compile(r"\b([A-Za-z]{6,})\b")

# Function words that form false compounds ("may be" is not "maybe" here).
STOP = {"a", "an", "the", "in", "on", "to", "of", "be", "may", "can", "not", "any", "some",
        "every", "no", "one", "all", "at", "by", "for", "so", "up", "out", "over", "under",
        "into", "onto", "with", "as", "is", "it", "its", "we", "us", "he", "she", "they", "them",
        "who", "what", "which", "that", "this", "these", "those", "there", "here", "where",
        "when", "then", "than", "and", "or", "but", "if", "also", "more", "most", "less", "such",
        "very", "much", "many", "each", "other", "own", "same", "new", "old", "first", "last",
        "next", "well", "will", "would", "could", "should", "might", "must", "shall", "do",
        "does", "did", "has", "have", "had", "was", "were", "been", "being", "are", "am",
        "per", "via"}
# Prefixes that Merriam-Webster and Oxford both close as a matter of rule;
# the model may close them even when the dictionary lacks the exact word.
PREFIXES = ("non", "pre", "post", "re", "co", "anti", "multi", "inter", "intra", "sub", "super",
            "counter", "over", "under", "cyber", "geo", "micro", "macro", "semi", "un")

DICTIONARY = {"us": "Merriam-Webster (the dictionary Chicago, GPO and APA follow)",
              "uk": "the Oxford dictionaries (which New Hart's Rules follows)"}

RULE = ("{dictionary} decides how this compound is written, and outranks the document's own "
        "usage. {closing} When the dictionary has the closed form, use it everywhere: closing does "
        "not depend on position. Hyphen versus open does: a compound modifier before a noun is "
        "hyphenated ('third-party assessor', 'long-term plan', 'well-known author'); the same "
        "words as a noun phrase are open ('a third party', 'in the long term', 'is well known'). "
        "Adverbs in -ly are never hyphenated ('highly regarded'). {style}{counts}"
        "Choose the form that is correct here, or KEEP if none is.")
CLOSING = {"us": "American usage closes established compounds: cyberattack, cybersecurity, "
                 "nonprofit, healthcare, policymaker, website, email, online, geopolitical.",
           "uk": "British usage hyphenates more compounds than American: cyber-attack, "
                 "non-profit, co-operate; but email, online, website and healthcare are closed."}
STYLE_NOTE = {"ap": "AP style hyphenates less than Chicago: omit the hyphen in a compound modifier "
                    "when the meaning is clear without it ('first quarter earnings'). ",
              "chicago": "", "gpo": "", "apa": "", "oxford": "", "guardian": "", "none": ""}
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


def _line_col(text: str, pos: int) -> tuple[int, int]:
    return text.count("\n", 0, pos) + 1, pos - (text.rfind("\n", 0, pos) + 1) + 1


def _recase(original: str, target_lower: str) -> str:
    if original.isupper() and len(original) > 1:
        return target_lower.upper()
    if original[:1].isupper():
        return target_lower[:1].upper() + target_lower[1:]
    return target_lower


def split_closed(word: str, variety: str) -> tuple[str, str] | None:
    """('long', 'term') for 'longterm' when both halves are words; longest
    first half wins. None if the word itself is a word."""
    lw = word.lower()
    if known(lw, variety):
        return None
    for i in range(len(lw) - 3, 2, -1):
        a, b = lw[:i], lw[i:]
        if a in STOP or b in STOP:
            continue
        if known(a, variety) and known(b, variety):
            return a, b
    return None


def compound_candidates(text: str, variety: str, spans: list[tuple[int, int]]) -> dict[str, list[Occurrence]]:
    """key (letters only, lowercase) -> occurrences of every surface form."""
    fam: dict[str, list[Occurrence]] = defaultdict(list)
    closed_ok: dict[str, bool] = {}
    shapes: dict[str, set[str]] = defaultdict(set)
    for m in _PAIR.finditer(text):
        if _in_span(m.start(), spans):
            continue
        a, sep, b = m.group(1), m.group(2), m.group(3)
        al, bl = a.lower(), b.lower()
        if "'" in al or "'" in bl or al in STOP or bl in STOP:
            continue
        key = al + bl
        if key not in closed_ok:
            # The closed form counts if either dictionary has it (en_US is thin:
            # it lacks "cyberattack"), or a closing prefix is hyphenated on.
            closed_ok[key] = known_anywhere(key) or (sep == "-" and al in PREFIXES and known(bl, variety))
        form = a + sep + b
        fam[key].append(Occurrence(m.start(), m.start() + len(form), form))
        shapes[key].add(sep)
    for m in _LONGWORD.finditer(text):
        if _in_span(m.start(), spans):
            continue
        w = m.group(0)
        key = w.lower()
        if key in fam or split_closed(w, variety):
            fam[key].append(Occurrence(m.start(), m.end(), w))
    out = {}
    for key, occ in fam.items():
        kinds = {re.sub(r"[A-Za-z']", "", o.form) or "closed" for o in occ}
        # A question exists when the closed form is a word, or the document
        # itself writes the term more than one way. An open pair whose
        # closed form is not a word ("risk assessor") is an ordinary phrase.
        if not closed_ok.get(key) and len(kinds) < 2:
            continue
        out[key] = occ
    return out


def case_families(text: str, spans: list[tuple[int, int]]) -> dict[str, list[Occurrence]]:
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


def _closed_is_word(closed: str, parts: tuple[str, str] | None) -> bool:
    return known_anywhere(closed) or bool(parts and parts[0] in PREFIXES and known_anywhere(parts[1]))


def _parts(occ: list[Occurrence], variety: str) -> tuple[str, str] | None:
    for o in occ:
        f = o.form.lower()
        if "-" in f:
            return tuple(f.split("-", 1))
        if " " in f:
            return tuple(f.split(" ", 1))
    return split_closed(occ[0].form, variety)


def items(text: str, variety: str = "us", style: str = "none", next_id: int = 1
          ) -> tuple[list[Item], dict[int, list[Occurrence]]]:
    """Adjudication items plus, for each item, every occurrence its decision
    applies to. Closing is position-independent, so a closed-form family gets
    one item per surface form; hyphen versus open is decided per occurrence,
    and only for the minority shape."""
    spans = protected_spans(text)
    out: list[Item] = []
    groups: dict[int, list[Occurrence]] = {}

    def add(o: Occurrence, options: tuple[str, ...], rule: str, source: str, group: list[Occurrence]):
        nonlocal next_id
        line, col = _line_col(text, o.start)
        out.append(Item(next_id, o.form, options + ("KEEP",), rule,
                        sentence_context(text, o.start, o.end), line, col,
                        col + (o.end - o.start) - 1, source=source))
        groups[next_id] = group
        next_id += 1

    for key, occ in compound_candidates(text, variety, spans).items():
        parts = _parts(occ, variety)
        closed = key
        counts = Counter(o.form.lower() for o in occ)
        desc = ""
        if len(counts) > 1:
            desc = "This document writes it " + ", ".join(f"'{f}' {n}×" for f, n in counts.most_common()) + ". "
        rule = RULE.format(dictionary=DICTIONARY[variety], closing=CLOSING[variety],
                           style=STYLE_NOTE.get(style, ""), counts=desc)
        if _closed_is_word(closed, parts):
            # One decision per non-closed surface form, applied to all of them.
            by_form: dict[str, list[Occurrence]] = defaultdict(list)
            for o in occ:
                if "-" in o.form or " " in o.form:
                    by_form[o.form].append(o)
            for form, group in by_form.items():
                add(group[0], (_recase(form, closed),), rule, "compound", group)
            continue
        if not parts:
            continue
        hyph, open_ = "-".join(parts), " ".join(parts)
        shape = Counter(("-" if "-" in o.form else " " if " " in o.form else "closed") for o in occ)
        majority = max(shape.values())
        for o in occ:
            sh = "-" if "-" in o.form else " " if " " in o.form else "closed"
            if sh == "closed":                       # a non-word like "longterm"
                add(o, (_recase(o.form, hyph), _recase(o.form, open_)), rule, "compound", [o])
                continue
            if shape[sh] * 2 > majority and len(shape) > 1:
                continue                             # the majority shape is not questioned
            other = open_ if sh == "-" else hyph
            add(o, (_recase(o.form, other),), rule, "compound", [o])
    for key, occ in case_families(text, spans).items():
        counts = Counter(o.form for o in occ)
        ranked = [f for f, _ in counts.most_common()]
        majority = counts[ranked[0]]
        desc = ", ".join(f"'{f}' {n}×" for f, n in counts.most_common())
        for o in occ:
            if counts[o.form] * 2 > majority:
                continue
            add(o, tuple(f for f in ranked if f != o.form), CASE_RULE.format(counts=desc), "consistency", [o])
    return out, groups
