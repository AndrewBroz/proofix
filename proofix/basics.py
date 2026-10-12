"""Pure-Python mechanical checks that run when LanguageTool does not."""

import re
from dataclasses import dataclass

from adjudicate import Edit

# Words whose doubling is never grammatical in prose: removing one copy is
# always right. Any other double ("that that", "had had", "very very") is a
# judgement call: adjudicated with a model, listed for review without one.
CLOSED_DOUBLES = frozenset("a an the of to in on at for with from by into onto about "
                           "and or but nor it we they my your our their".split())
# The same word twice, separated by one space or by a newline within the
# paragraph that does not open a list, quote, heading or table line.
_DOUBLE = re.compile(r"\b([A-Za-z]+)( |\n(?![ \t]*(?:[-*+>#|]|\d+[.)])))(\1)\b", re.I)
_SPACE_BEFORE = re.compile(r"(?<=\w)[ \t]+(?=[,;:!?]|\.(?!\.))")
_NO_SPACE_AFTER = re.compile(r"(?<=[A-Za-z])([,;:])(?=[A-Za-z])")

DOUBLE_RULE = ("A word is repeated. Choose the single word if the repeat is a typo. KEEP if it is "
               "grammatical or deliberate — repeats like 'that that', 'had had', 'is is' and "
               "emphatic doubles like 'very very' are often correct.")


@dataclass(frozen=True)
class Double:
    start: int          # start of the separator before the second copy
    end: int            # end of the second copy
    word: str
    closed: bool

    @property
    def first_start(self) -> int:
        """Start of the first copy (both copies are the same length)."""
        return self.start - len(self.word)


def _inside(pos, spans):
    return any(s <= pos < e for s, e in spans)


def doubled_words(text, spans):
    out = []
    for m in _DOUBLE.finditer(text):
        first, second = m.group(1), m.group(3)
        if _inside(m.start(), spans) or _inside(m.end() - 1, spans):
            continue
        if first[:1].isupper() and second[:1].isupper():
            continue                                   # Bora Bora: a name
        if "\n\n" in m.group(2):
            continue
        out.append(Double(m.start(2), m.end(3), second, first.lower() in CLOSED_DOUBLES))
    return out


def _edit(text, start, original, replacement):
    line = text.count("\n", 0, start) + 1
    col = start - (text.rfind("\n", 0, start) + 1) + 1
    return Edit(line, col, col + len(original) - 1, original, replacement, "basics")


def double_edit(text, d: Double) -> Edit:
    """Remove the second copy and its separator (closed-list doubles only), written
    as both copies -> the first, so the change reads 'the the' -> 'the'."""
    return _edit(text, d.first_start, text[d.first_start:d.end], text[d.first_start:d.start])


def spacing_edits(text, spans):
    """Space before , . ; : ! ? removed; a missing space after , ; : between letters added.
    Protected spans (code, URLs, links, Obsidian syntax) and anything next to a digit are skipped."""
    edits = []
    for m in _SPACE_BEFORE.finditer(text):
        if _inside(m.start(), spans) or re.search(r"\d", text[max(0, m.start() - 2):m.end() + 2]):
            continue
        punct = text[m.end()]
        if punct in ",;:" and text[m.end() + 1:m.end() + 2].isalpha():
            # "budget ,the": the space moves after the mark ("budget, the")
            edits.append(_edit(text, m.start(), m.group(0) + punct, punct + " "))
            continue
        edits.append(_edit(text, m.start(), m.group(0), ""))
    for m in _NO_SPACE_AFTER.finditer(text):
        if _inside(m.start(), spans) or re.search(r"\d", text[max(0, m.start() - 2):m.end() + 2]):
            continue
        edits.append(_edit(text, m.start(), m.group(1), m.group(1) + " "))
    return edits
