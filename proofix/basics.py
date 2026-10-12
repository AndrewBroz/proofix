"""Pure-Python mechanical checks that run when LanguageTool does not."""

import re
from dataclasses import dataclass

from adjudicate import Edit

# Words whose doubling is never grammatical in prose: removing one copy is
# always right. Any other double ("that that", "had had", "very very") is a
# judgement call: adjudicated with a model, listed for review without one.
CLOSED_DOUBLES = frozenset("a an the of to in on at for with from by into onto about "
                           "and or but nor it we they my your our their".split())
# The same word (any letters, inner apostrophes: "café", "it's") twice, separated
# by spaces or tabs, or by a newline within the paragraph that does not open a
# list, quote, heading or table line.
_WORD = r"[^\W\d_]+(?:['\u2019][^\W\d_]+)*"
_DOUBLE = re.compile(rf"(?<![\w'\u2019])({_WORD})([ \t]+|\n(?![ \t]*(?:[-*+>#|]|\d+[.)])))(\1)(?![\w'\u2019])",
                     re.I)

DOUBLE_RULE = ("A word is repeated. Choose the single word if the repeat is a typo. KEEP if it is "
               "grammatical or deliberate — repeats like 'that that', 'had had', 'is is' and "
               "emphatic doubles like 'very very' are often correct.")


@dataclass(frozen=True)
class Double:
    start: int          # start of the separator before the second copy
    end: int            # end of the second copy
    word: str           # the second copy as written
    closed: bool        # in CLOSED_DOUBLES: one copy is removed outright
    first: str = ""     # the first copy as written (defaults to the second's length)
    capitalised: bool = False   # both copies capitalised: a closed-list pair is reviewed, not fixed

    @property
    def first_start(self) -> int:
        return self.start - len(self.first or self.word)

    @property
    def text(self) -> str:
        """'<first> <second>', the separator shown as one space."""
        return f"{self.first or self.word} {self.word}"


def _inside(pos, spans):
    return any(s <= pos < e for s, e in spans)


def doubled_words(text, spans):
    out = []
    for m in _DOUBLE.finditer(text):
        first, second = m.group(1), m.group(3)
        if _inside(m.start(), spans) or _inside(m.end() - 1, spans):
            continue
        closed = first.lower() in CLOSED_DOUBLES
        capitalised = first[:1].isupper() and second[:1].isupper()
        if capitalised and not closed:
            continue                                   # Bora Bora: a name
        out.append(Double(m.start(2), m.end(3), second, closed, first, capitalised))
    return out


def _edit(text, start, original, replacement):
    line = text.count("\n", 0, start) + 1
    col = start - (text.rfind("\n", 0, start) + 1) + 1
    return Edit(line, col, col + len(original) - 1, original, replacement, "basics")


def double_edit(text, d: Double) -> Edit:
    """Remove the second copy and its separator (closed-list doubles only), written
    as both copies -> the first, so the change reads 'the the' -> 'the'."""
    return _edit(text, d.first_start, text[d.first_start:d.end], text[d.first_start:d.start])


# Spaces before a mark, where the space follows a word: "programme ,", "ready ?".
_SPACE_BEFORE = re.compile(r"(?<=[^\W\d_])[ \t]+([,.;:!?])")
_NO_SPACE_AFTER = re.compile(r"(?<=[A-Za-z])([,;])(?=[A-Za-z])|(?<=[A-Za-z]{2})(:)(?=[a-z]{2})")
_EMOTICON = re.compile(r"[:;]-?[()DPpOo3/\\|\[\]](?![^\W\d_])")
_SHORTCODE = re.compile(r":[a-z0-9_+-]+:")
_CLOSING = ")]}\u201d\u2019\u00bb"
_EOL = re.compile(r"[ \t]*(?:\n|$)")


def _near_digit(text, start, end):
    return re.search(r"\d", text[max(0, start - 2):end + 2]) is not None


def _spacing(text, spans):
    """(edits, notes): edits only where the fix is provably right; notes are
    (offset, mark) for a space before : ; ! ? with text after it on the line,
    which may be a key or a command ("Press : to enter"), so a person decides."""
    edits, notes = [], []
    for m in _SPACE_BEFORE.finditer(text):
        sp, mark, p = m.group(0)[:-1], m.group(1), m.start(1)
        if _inside(m.start(), spans) or _near_digit(text, m.start(), m.end()):
            continue
        nxt = text[p + 1:p + 2]
        at_eol = _EOL.match(text, p + 1) is not None
        if mark in ":;" and (_EMOTICON.match(text, p) or _SHORTCODE.match(text, p)):
            continue                                      # ":)", ";-(", ":D", ":wave:"
        if mark == ".":
            # not ".bashrc", ".NET", "..." or a spaced ellipsis ". . ."
            if nxt.isalnum() or nxt == "." or re.match(r"[ \t]+\.", text[p + 1:]):
                continue
            edits.append(_edit(text, m.start(), sp, ""))
        elif mark == ",":
            if at_eol or nxt.isspace() or (nxt and nxt in _CLOSING):
                edits.append(_edit(text, m.start(), sp, ""))
            elif nxt.isalpha():                           # "budget ,the" -> "budget, the"
                edits.append(_edit(text, m.start(), sp + mark, mark + " "))
        elif at_eol or (nxt and nxt in _CLOSING):
            edits.append(_edit(text, m.start(), sp, ""))
        elif nxt.isspace():
            notes.append((p, mark))
        elif mark == ";" and nxt.islower():
            edits.append(_edit(text, m.start(), sp + mark, mark + " "))
    for m in _NO_SPACE_AFTER.finditer(text):
        p = m.start()
        if _inside(p, spans) or _near_digit(text, p, m.end()) or _SHORTCODE.match(text, p):
            continue
        mark = text[p]
        if mark == ";" and _EMOTICON.match(text, p):
            continue
        edits.append(_edit(text, p, mark, mark + " "))
    return edits, notes


def spacing_edits(text, spans):
    """Space before , . ; : ! ? removed, and a missing space after , ; : added, only
    where that is provably right: never in protected spans, next to a digit, in
    emoticons and :shortcodes:, before ".bashrc" or a spaced ellipsis, or before a
    mark with more text after it on the line that could be a key (see spacing_notes)."""
    return _spacing(text, spans)[0]


def spacing_notes(text, spans):
    """[(offset of the mark, mark)] for review: "Note : the", "Press : to enter"."""
    return _spacing(text, spans)[1]


def missing_space_edits(text, spans, marks=",;:"):
    """Only the missing-space-after-a-mark edits of spacing_edits, for those marks."""
    return [e for e in _spacing(text, spans)[0] if e.original in marks and e.original
            and e.replacement == e.original + " "]
