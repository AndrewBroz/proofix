"""Sort LanguageTool matches into deterministic edits, adjudication items
and review, with Markdown structure protected."""

from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

from adjudicate import Edit, Item, protected_spans, sentence_context
from adjudicate.apply import to_offsets

from . import basics

from .basics import CLOSED_DOUBLES, DOUBLE_RULE, Double, double_edit, doubled_words
from .lt import Match

# LanguageTool rules whose replacement may be applied outright, each only where the
# audit (spec §7) found it right in every prose context; elsewhere a match is routed
# like any other rule (a model decides, KEEP offered; review without one). See _outright.
DETERMINISTIC = {
    # Always right only where basics' audited spacing check makes the same edit ("report ,"),
    # or for a space just inside a bracket beside a letter or digit ("( as noted )").
    # Elsewhere routed: ".bashrc", ". . .", "1 ,000", "( ͡°".
    "COMMA_PARENTHESIS_WHITESPACE",
    # Always right when LanguageTool offers one answer (",," -> ","). ".." offers "." and
    # "…" and may be a path ("cd .."): routed.
    "DOUBLE_PUNCTUATION",
    # Always right mid-line in prose: Markdown renders the run as one space. Not a fault,
    # so skipped: line-start indentation, trailing spaces (a hard line break), table
    # rows (alignment) and indented code lines.
    "WHITESPACE_RULE",
}
# Moved out of DETERMINISTIC by the audit: "Node.Js", "file.Name" are not two sentences;
# a paragraph may open with ", and so on" after a quotation or list.
MOVED = {"SENTENCE_WHITESPACE", "UNLIKELY_OPENING_PUNCTUATION"}
# ENGLISH_WORD_REPEAT_RULE is not in DETERMINISTIC: "that that" and "had had" are often
# grammatical, so its matches are routed in partition() as basics routes doubles.
REPEAT = "ENGLISH_WORD_REPEAT_RULE"
# Rules stylefix owns, or that misfire on Markdown.
IGNORED = {"EN_QUOTES", "DASH_RULE", "PUNCTUATION_PARAGRAPH_END", "UPPERCASE_SENTENCE_START",
           "WORD_CONTAINS_UNDERSCORE", "ARROWS", "MULTIPLICATION_SIGN", "EN_UNPAIRED_BRACKETS"}
SPELLING = {"MORFOLOGIK_RULE_EN_US", "MORFOLOGIK_RULE_EN_GB", "MORFOLOGIK_RULE_EN_AU",
            "MORFOLOGIK_RULE_EN_CA", "MORFOLOGIK_RULE_EN_NZ", "MORFOLOGIK_RULE_EN_ZA"}

# The rules of the mechanical tier (pass 1): their review matches are kept when there is
# no pass 2 to route them again.
MECHANICAL = DETERMINISTIC | MOVED | {"ENGLISH_WORD_REPEAT_RULE"}

# LanguageTool rules that contradict the target variety, disabled when it runs:
# uk_ise (UK, the guide does not use -ize): the Oxford -ize suggestions. Of the
# BRE_STYLE_OXFORD_SPELLING category, only OXFORD_SPELLING_Z_NOT_S (and the
# gramme -> gram rule, which is not about -ize) exists in LanguageTool 6.8; the
# NOUNS/ISE_VERBS/ADJECTIVES/ADVERBS/IZE ids appear only in comments.
# us: en-US has no rule suggesting British spellings. It reports "colour" and
# "organise" only as MORFOLOGIK_RULE_EN_US "... is British English", which cannot be
# disabled without disabling spelling, and which partition() already drops as a
# variety difference (stylefix's job).
VARIETY_DISABLED: dict[str, frozenset[str]] = {
    "uk_ise": frozenset({"OXFORD_SPELLING_Z_NOT_S"}),
    "us": frozenset(),
}

SPELLING_RULE = ("Choose the correctly spelled word only if the marked text is an ordinary word "
                 "misspelled. KEEP if it is a name, an acronym, a product, jargon, a code "
                 "identifier, a foreign word, or a deliberate spelling.")


@dataclass
class Partition:
    edits: list[Edit]
    items: list[Item]
    review: list[Match]
    rules: dict[int, str] = field(default_factory=dict)     # item id -> the LanguageTool rule behind it


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


def mechanical_edits(text: str, spans: list[tuple[int, int]]) -> list[Edit]:
    """Faults LanguageTool has no rule for: a missing space after ; or : ("issues;the").
    basics' audited rule decides (not "a:b", ":D", ":wave:", "10:30"); one implementation."""
    return [replace(e, source="MISSING_SPACE") for e in basics.missing_space_edits(text, spans, ";:")]


def _minimal(start: int, original: str, replacement: str) -> tuple[int, str, str]:
    """(start, original, replacement) with the common prefix and suffix trimmed."""
    k = 0
    while k < min(len(original), len(replacement)) and original[k] == replacement[k]:
        k += 1
    j = 0
    while (j < min(len(original), len(replacement)) - k
           and original[len(original) - 1 - j] == replacement[len(replacement) - 1 - j]):
        j += 1
    return start + k, original[k:len(original) - j], replacement[k:len(replacement) - j]


def _spacing_fixes(text: str, spans: list[tuple[int, int]]) -> set[tuple[int, str, str]]:
    out = set()
    for e in basics.spacing_edits(text, spans):
        s, _ = to_offsets(text, e)
        out.add(_minimal(s, e.original, e.replacement))
    return out


def _outright(text: str, m: Match, spacing: set[tuple[int, str, str]]) -> bool | None:
    """True: apply this DETERMINISTIC match outright. False: route it (model or review).
    None: not a fault (alignment, a hard line break, code): leave it alone."""
    original = text[m.start:m.end]
    if m.rule == "WHITESPACE_RULE":
        ls = text.rfind("\n", 0, m.start) + 1
        le = text.find("\n", m.end)
        line = text[ls:le if le >= 0 else len(text)]
        if (_line_start(text, m.start) or not text[m.end:le if le >= 0 else len(text)].strip()
                or "|" in line or line.startswith(("    ", "\t"))):
            return None
        return True
    if m.rule == "DOUBLE_PUNCTUATION":
        return len(set(m.replacements)) == 1
    if m.rule == "COMMA_PARENTHESIS_WHITESPACE":
        rep = m.replacements[0]
        if _minimal(m.start, original, rep) in spacing:
            return True
        after, before = text[m.end:m.end + 1], text[m.start - 1:m.start] if m.start else ""
        if original.rstrip() in ("(", "[") and original != original.rstrip() and rep == original.rstrip():
            return after.isalnum()
        if original.lstrip() in (")", "]") and original != original.lstrip() and rep == original.lstrip():
            return before.isalnum()
        return False
    return True


def _readable(text: str, start: int, end: int, reps: list[str]) -> tuple[int, int, list[str]]:
    """Widen a span so no option is blank or edged with a space: a deletion takes the next
    word (", and" -> "and"), a space added at an edge takes the neighbouring character
    (".However" -> ". However")."""
    if any(r == "" for r in reps):
        mm = re.compile(r"[ \t]*[^\s]+").match(text, end)
        if mm:
            tail = text[end:mm.end()]
            reps = [r + tail if r else tail.lstrip() for r in reps]
            end = mm.end()
    if any(r[:1].isspace() for r in reps) and start > 0 and text[start - 1] not in "\n":
        reps = [text[start - 1] + r for r in reps]
        start -= 1
    if any(r[-1:].isspace() for r in reps) and end < len(text) and text[end] != "\n":
        reps = [r + text[end] for r in reps]
        end += 1
    return start, end, reps


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
    cases: a blank line, a list, quote, heading or table line), touches a hyphen, or
    is a capitalised pair that is not on the closed list (a name)."""
    original = text[m.start:m.end]
    parts = original.split()
    if "\n" in original or len(parts) != 2:
        return None
    if text[m.start - 1:m.start] == "-" or text[m.end:m.end + 1] == "-":
        return None                       # part of a compound: "to to-do", "for for-profit"
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
    rules: dict[int, str] = {}
    next_id = 1
    doubles = {(d.first_start, d.end): d for d in doubled_words(text, spans)}
    spacing: set[tuple[int, str, str]] | None = None
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
                rules[next_id] = m.rule
                next_id += 1
            else:
                review.append(replace(m, message=double_message(text, d)))
            continue
        if m.rule in DETERMINISTIC and m.replacements:
            if spacing is None:
                spacing = _spacing_fixes(text, spans)
            verdict = _outright(text, m, spacing)
            if verdict is None:
                continue          # list indentation, alignment, a hard line break, code
            if verdict:
                edits.append(to_edit(text, m, m.replacements[0], m.rule))
                continue
        if m.replacements and adjudicate_available:
            start, end, reps = m.start, m.end, list(m.replacements[:4])
            if m.rule in DETERMINISTIC | MOVED:
                start, end, reps = _readable(text, start, end, reps)
            word = text[start:end]
            options = tuple(dict.fromkeys(r for r in reps if r != word)) + ("KEEP",)
            rule = SPELLING_RULE if m.rule in SPELLING else (
                f"LanguageTool ({m.rule}): {m.message} Choose the option that makes the sentence "
                "correct; KEEP if the original is acceptable or the suggestion changes the meaning.")
            line, col = _line_col(text, start)
            items.append(Item(next_id, word, options, rule, sentence_context(text, start, end),
                              line, col, col + (end - start) - 1,
                              source="spelling" if m.rule in SPELLING else "grammar"))
            rules[next_id] = m.rule
            next_id += 1
            continue
        review.append(m)
    return Partition(edits, items, review, rules)


# Short typos (three letters or fewer) typos may fix outright. Chosen from typos' own data
# (all 2- and 3-letter strings fed to typos 1.51: 247 known typos): one correction only;
# rejected by LanguageTool's spell checkers for en-US, en-GB, de-DE, fr, es, it, pt-PT,
# nl, sv, da and pl; an adjacent transposition or one-letter omission of an ordinary
# word; no other common English word one transposition or insertion away, except where
# that word is no plausible intent (teh/tech, hte/hate, adn/Dan, wih/wish). Left out:
# jargon corrections (cmo, fpt, gti, jkd, slq, sxl, unx), dialect or loaded (wha, gya),
# unclear (abl, bse, daa, ede, edn), and plausible in a note (vey as in "oy vey"; wth,
# wya, suh: shorthand or slang; ehr, hsa: lower-case acronyms). "nd" is accepted by the Polish checker only, as the
# abbreviation "nd."; kept by ruling.
SAFE_SHORT_TYPOS = frozenset(
    "teh hte adn wih nd eyt fwe hda hpe hsi hve ihs kow mch muc nly nwe nwo onl oru "
    "oyu tey thm tke viw whn wht woh wrk ype yuo".split())
# A typo that one apostrophe turns into a contraction ("youre", "whos", "dont") may be
# the contraction, not typos' correction ("your", "whose"): never applied outright.
CONTRACTIONS = frozenset(
    "i'm i've i'd i'll you're you've you'd you'll he's he'd he'll she's she'd she'll it's "
    "it'd we're we've we'd we'll they're they've they'd they'll who's who'd what's that's "
    "there's here's let's don't doesn't didn't can't won't wouldn't shouldn't couldn't "
    "isn't aren't wasn't weren't hasn't haven't hadn't ain't where's how's why's when's "
    "could've should've would've must've might've".split())
_CONTRACTION_SHAPED = frozenset(c.replace("'", "") for c in CONTRACTIONS)


def _typos(text: str, json_lines: str, spans: list[tuple[int, int]] | None,
           accept: set[str] | frozenset[str] | None) -> list[tuple[int, str, list[str], bool]]:
    """(offset, typo, corrections, outright) from `typos --format json` output, kept out of
    protected spans and accept lists. Outright only when provably right: the typo is all
    lower case (not a name, an acronym or a sentence-initial word whose case is a
    guess), not within a word containing a digit or underscore (an identifier), and
    typos offers exactly one correction (two, as for "alle" -> all/alley, mean the word
    may be another language's or another word's), and the typo is in SAFE_SHORT_TYPOS or
    has four letters or more and is not contraction-shaped: short ones are often words
    elsewhere ("ein", "ser", "som") or another word ("hed": head or he'd)."""
    import json
    spans = protected_spans(text) if spans is None else spans
    accept = {w.lower() for w in (accept or ())}
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
        if line[col:col + len(word)] != word or word.lower() in accept:
            continue
        pos = starts[ln] + col
        if any(s <= pos < e for s, e in spans):
            continue
        a, b = col, col + len(word)
        while a > 0 and (line[a - 1].isalnum() or line[a - 1] == "_"):
            a -= 1
        while b < len(line) and (line[b].isalnum() or line[b] == "_"):
            b += 1
        token = line[a:b]
        corrections = list(d["corrections"])
        outright = (word.islower() and word.isalpha() and len(corrections) == 1
                    and not any(ch.isdigit() or ch == "_" for ch in token)
                    and (word in SAFE_SHORT_TYPOS
                         or (len(word) >= 4 and word not in _CONTRACTION_SHAPED)))
        out.append((pos, word, corrections, outright))
    return out


def typos_edits(text: str, json_lines: str, spans: list[tuple[int, int]] | None = None,
                accept: set[str] | frozenset[str] | None = None) -> list[Edit]:
    """Edits from `typos --format json` output: only the provably right ones (see _typos)."""
    out = []
    for pos, word, corrections, outright in _typos(text, json_lines, spans, accept):
        if outright:
            line, col = _line_col(text, pos)
            out.append(Edit(line, col, col + len(word) - 1, word, corrections[0], "typos"))
    return out


def typos_matches(text: str, json_lines: str, spans: list[tuple[int, int]] | None = None,
                  accept: set[str] | frozenset[str] | None = None) -> list[Match]:
    """The rest of typos' findings, as matches for the spelling route: a model decides
    (SPELLING_RULE, KEEP offered), or they are listed for review."""
    return [Match(pos, pos + len(word), "typos", "TYPOS",
                  f"'{word}' may be a misspelling of " + " or ".join(f"'{c}'" for c in corrections),
                  "", corrections)
            for pos, word, corrections, outright in _typos(text, json_lines, spans, accept)
            if not outright]


def typos_item(text: str, m: Match, item_id: int) -> Item:
    line, col = _line_col(text, m.start)
    word = text[m.start:m.end]
    options = tuple(dict.fromkeys(c for c in m.replacements[:4] if c != word)) + ("KEEP",)
    return Item(item_id, word, options, SPELLING_RULE, sentence_context(text, m.start, m.end),
                line, col, col + len(word) - 1, source="spelling")
