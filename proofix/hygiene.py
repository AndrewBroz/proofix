"""Text that renders fine but is not what it looks like.

HTML entities that Markdown can carry as plain characters are decoded
(&#32; is a space, &quot; a quote, &mdash; a dash); the ones that change
Markdown's parsing (&lt;, &gt;, &amp; before a name) or carry meaning
(&nbsp;) are left alone and reported. Invisible characters (zero-width
spaces, byte-order marks, soft hyphens) are removed; a zero-width joiner
is kept where it joins an emoji sequence. Windows line endings
and decomposed accents are normalised before anything else runs. Code,
links and front matter are never touched."""

from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass

from adjudicate import Edit, protected_spans

_ENTITY = re.compile(r"&(?:#\d+|#x[0-9A-Fa-f]+|[A-Za-z][A-Za-z0-9]*);")
_INVISIBLE = re.compile("[​‌‍⁠﻿­]")
_ZWJ = "\u200d"
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# Decoded characters that would change how Markdown parses the line.
_UNSAFE_DECODED = set("<>*_`#[]\\|")
_KEEP_NAMED = {"nbsp", "ensp", "emsp", "thinsp", "zwj", "zwnj", "shy", "lrm", "rlm"}

NAMES = {"​": "zero-width space", "‌": "zero-width non-joiner",
         "‍": "zero-width joiner", "⁠": "word joiner", "﻿": "byte-order mark",
         "­": "soft hyphen"}


@dataclass
class Note:
    line: int
    what: str
    message: str


def normalize_text(text: str) -> tuple[str, list[str]]:
    """Whole-text normalisations that shift offsets: do these first."""
    notes = []
    if "\r" in text:
        n = text.count("\r\n") + text.replace("\r\n", "").count("\r")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        notes.append(f"normalised {n} Windows/old-Mac line ending(s) to LF")
    nfc = unicodedata.normalize("NFC", text)
    if nfc != text:
        changed = sum(1 for a, b in zip(text, nfc) if a != b) + abs(len(nfc) - len(text))
        text = nfc
        notes.append(f"normalised {changed} decomposed character(s) to composed form (NFC)")
    return text, notes


def _pictographic(ch: str) -> bool:
    """Emoji-like characters a zero-width joiner may join (woman + ZWJ + laptop,
    flag + VS-16 + ZWJ + rainbow): the emoji blocks, skin-tone modifiers and the
    emoji presentation selector."""
    o = ord(ch)
    return 0x1F000 <= o <= 0x1FAFF or 0x2600 <= o <= 0x27BF or o == 0xFE0F


def _joins_emoji(text: str, pos: int) -> bool:
    return (0 < pos < len(text) - 1 and _pictographic(text[pos - 1])
            and _pictographic(text[pos + 1]) and text[pos + 1] != "\ufe0f")


def _line(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _col(text: str, pos: int) -> int:
    return pos - (text.rfind("\n", 0, pos) + 1) + 1


def plan(text: str) -> tuple[list[Edit], list[Note]]:
    """Span edits for entities and invisible characters, plus notes."""
    spans = protected_spans(text, entities=False)
    edits: list[Edit] = []
    notes: list[Note] = []

    def protected(pos: int) -> bool:
        return any(s <= pos < e for s, e in spans)

    for m in _ENTITY.finditer(text):
        if protected(m.start()):
            continue
        ent = m.group(0)
        name = ent[1:-1]
        if name.lower() in _KEEP_NAMED:
            notes.append(Note(_line(text, m.start()), ent, f"'{ent}' left as written (it carries meaning)."))
            continue
        decoded = html.unescape(ent)
        if decoded == ent:
            notes.append(Note(_line(text, m.start()), ent, f"'{ent}' is not a recognised HTML entity; left as written."))
            continue
        if any(ch in _UNSAFE_DECODED for ch in decoded) or decoded == "&":
            # Decoding would change Markdown parsing ("<", "*") or could
            # form a new entity ("&" before "amp;"). Leave it, say so.
            notes.append(Note(_line(text, m.start()), ent,
                              f"'{ent}' left as written: the plain character would change how Markdown reads the line."))
            continue
        if decoded.isspace() and decoded != " ":
            decoded = " "
        col = _col(text, m.start())
        edits.append(Edit(_line(text, m.start()), col, col + len(ent) - 1, ent, decoded, "hygiene"))
    for m in _INVISIBLE.finditer(text):
        if protected(m.start()):
            continue
        if m.group(0) == _ZWJ and _joins_emoji(text, m.start()):
            continue                     # part of an emoji sequence, not stray
        col = _col(text, m.start())
        edits.append(Edit(_line(text, m.start()), col, col, m.group(0), "", "hygiene"))
        notes.append(Note(_line(text, m.start()), NAMES.get(m.group(0), "invisible character"),
                          f"removed a {NAMES.get(m.group(0), 'invisible character')}."))
    for m in _CONTROL.finditer(text):
        if protected(m.start()):
            continue
        col = _col(text, m.start())
        edits.append(Edit(_line(text, m.start()), col, col, m.group(0), "", "hygiene"))
        notes.append(Note(_line(text, m.start()), repr(m.group(0)), f"removed control character {m.group(0)!r}."))
    return edits, notes
