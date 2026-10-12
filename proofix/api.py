"""Library API: proofread a Markdown string; the CLI is a wrapper around fix()."""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, replace

from adjudicate import Annotation, Change, ReviewItem, Skipped, derive_changes
from adjudicate import Cache, Edit, adjudicate, apply_edits, protected_spans, system_prompt
from adjudicate.apply import to_offsets
from adjudicate.core import Decision
from adjudicate.llm import Endpoint

from . import basics, consistency, hygiene
from .commas import CommaCache, CommaResult, comma_edits, correct_commas
from .lt import LanguageToolServer, Match, find_languagetool, run_languagetool
from .tiers import (IGNORED, MECHANICAL, REPEAT, VARIETY_DISABLED, Partition, _line_col,
                    accepted_double, double_item, double_message, double_route, partition,
                    typos_edits, typos_item, typos_matches)

TASK = "You are a meticulous copy editor fixing typos and punctuation in a document written in {variety}."
VARIETY = {"us": "American English", "uk": "British English"}
TIERS = ("hygiene", "mechanical", "basics", "typos", "adjudicated", "compounds", "commas")


@dataclass
class Options:
    target: str = "us"
    accept: frozenset[str] = frozenset()
    lt_level: str = "default"
    style: str = "none"
    commas: str = "gated"
    optional_commas: str = "leave"
    max_sentences: int | None = None
    min_confidence: float = 0.8
    parallel: int = 4
    verbose: bool = False
    skip: frozenset[str] = frozenset()
    replay_cache: bool = False
    oxford_ize: bool = False      # the style guide uses -ize: LanguageTool's Oxford rules stay on


@dataclass
class Result:
    text: str
    changes: list[Change]
    review: list[ReviewItem]
    skipped: list[Skipped]
    applied: list[Edit]
    decisions: list[Decision]
    comma_results: list[CommaResult]
    lt_review: list[Match]
    typos_fixed: int
    hygiene_notes: list[str]
    lt_missing: bool


def run_typos(text: str, accept: set[str] | frozenset[str] = frozenset()) -> tuple[list[Edit], list[Match]]:
    """Known misspellings from the `typos` tool, if installed: (edits applied outright,
    matches a model or a person decides). See tiers.typos_edits."""
    exe = shutil.which("typos")
    if not exe:
        return [], []
    proc = subprocess.run([exe, "--format", "json", "-"], input=text, capture_output=True, text=True)
    return typos_edits(text, proc.stdout, accept=accept), typos_matches(text, proc.stdout, accept=accept)


# Rules partition() ignores, and CONSECUTIVE_SPACES (WHITESPACE_RULE's duplicate), are
# disabled in LanguageTool itself: a server drops overlapping matches, so a rule that is
# ignored anyway must not knock out one that is not.
LT_DISABLED = IGNORED | {"CONSECUTIVE_SPACES"}


def lt_disabled(o: Options) -> list[str]:
    """The rules LanguageTool runs without: LT_DISABLED, and those that contradict the variety."""
    key = "us" if o.target == "us" else ("uk_ise" if not o.oxford_ize else None)
    return sorted(LT_DISABLED | (VARIETY_DISABLED[key] if key else frozenset()))


def _overlap_reviews(before: str, applied: list[Edit], rejected: list[Edit], after: str,
                     routed: set[tuple[int, int]], at_before: bool = False) -> list[Match]:
    """Review matches for edits apply_edits rejected because another edit overlapped them,
    where the original is still in the text: nothing is dropped silently. Positions are in
    `after`, or in `before` with at_before (when the review list is read against it)."""
    out = []
    for e in rejected:
        s, _ = to_offsets(before, e)
        if before[s:s + len(e.original)] != e.original:
            continue                      # stale: the text no longer matches the edit
        pos = _shifted(before, applied, s)
        found = after.find(e.original, max(0, pos - len(e.original)), pos + 2 * len(e.original))
        if found < 0:
            continue                      # the overlapping edit changed this text too
        where = s if at_before else found
        key = _line_col(before if at_before else after, where)
        if key in routed:
            continue
        routed.add(key)
        out.append(Match(where, where + len(e.original), e.source, "",
                         f"'{e.original}' -> '{e.replacement}' not applied: it overlaps another edit", ""))
    return out


def _decided_by(d: Decision, model_used: bool) -> str:
    return "cache" if d.cached else ("model" if model_used else "rule")


def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _shifted(text: str, applied: list[Edit], pos: int) -> int:
    """Where offset `pos` of `text` lands once `applied` (edits on `text`) are made."""
    delta = 0
    for e in applied:
        s, t = to_offsets(text, e)
        if t <= pos:
            delta += len(e.replacement) - (t - s)
    return pos + delta


def fix(text: str, options: Options, endpoint: Endpoint | None, cache: Cache,
        comma_cache: CommaCache, lt_server: LanguageToolServer | None = None) -> Result:
    """Proofread text; no I/O on the caller's files. Raises LTError if LanguageTool fails.
    With `lt_server` (a running LanguageToolServer), LanguageTool is asked through it."""
    o, ep = options, endpoint
    skip = o.skip
    asking = ep is not None or o.replay_cache
    accept = {w.lower() for w in o.accept}
    original = text
    skipped: list[Skipped] = []
    hygiene_notes: list[str] = []
    edits: list[Edit] = []
    typos_held: list[Match] = []
    if "hygiene" not in skip:
        text, hygiene_notes = hygiene.normalize_text(text)
        h_edits, h_notes = hygiene.plan(text)
        edits += h_edits
        hygiene_notes += [f"L{n.line} {n.message}" for n in h_notes]
    if "typos" not in skip:
        if shutil.which("typos") is None:
            skipped.append(Skipped("known-misspelling fixes", "typos not found", "brew install typos-cli"))
        else:
            t_edits, typos_held = run_typos(text, accept)
            edits += t_edits
    typos_fixed = sum(1 for e in edits if e.source == "typos")
    decisions: list[Decision] = []
    review: list[Match] = []
    review_items: list[ReviewItem] = []
    lt_wanted = not {"mechanical", "adjudicated"} <= skip
    disabled = lt_disabled(o)
    # what the server's overlap filtering must not hide: every rule tier 1 applies or routes
    ensure = sorted(MECHANICAL) if lt_server is not None else None
    lt = find_languagetool() if lt_wanted else None
    lt_missing = lt_wanted and not lt
    if lt_missing:
        skipped.append(Skipped("grammar and spelling fixes (tiers 1 and 2)", "LanguageTool not found",
                               "brew install languagetool"))
    # basics' doubled words always run (LanguageTool misses "that that"), deduplicated
    # against its repeat rule; basics' spacing stands in for LanguageTool's mechanical
    # pass only when that pass will not run.
    use_doubles = "basics" not in skip
    use_spacing = use_doubles and not (lt and "mechanical" not in skip)
    # Pass 1: mechanical faults (typos, whitespace, doubled punctuation), applied.
    pass1_review: list[Match] = []
    lt_repeats: set[tuple[int, int]] = set()
    if lt and "mechanical" not in skip:
        matches = run_languagetool(text, o.target, o.lt_level, disabled=disabled, server=lt_server,
                                   ensure=ensure)
        p1 = partition(text, matches, accept, adjudicate_available=False)
        edits += p1.edits
        lt_repeats = {(e.line, e.start) for e in p1.edits if e.source == REPEAT}
        if "adjudicated" in skip:      # no pass 2 to route them: keep pass 1's mechanical reviews
            pass1_review = [m for m in p1.review if m.rule in MECHANICAL]
    if use_doubles:
        spans = protected_spans(text)
        if use_spacing:
            edits += basics.spacing_edits(text, spans)
        for d in basics.doubled_words(text, spans):
            if double_route(d, asking) != "edit" or accepted_double(text, d, accept):
                continue
            e = basics.double_edit(text, d)
            if (e.line, e.start) not in lt_repeats:        # LanguageTool's edit is the one kept
                edits.append(e)
    pre_text = text
    text, rejected_p1 = apply_edits(text, edits)
    applied = [e for e in edits if e not in rejected_p1]
    p1_rejected_repeats = [e for e in rejected_p1 if e.source == REPEAT]
    lt_pass2 = bool(lt) and "adjudicated" not in skip
    # Pass 2: judgement calls on the cleaned text, so spans no longer collide.
    review_text = text
    if lt_pass2:
        matches = run_languagetool(text, o.target, o.lt_level, disabled=disabled, server=lt_server,
                                   ensure=ensure)
        part = partition(text, matches, accept, adjudicate_available=asking)
        # Pass 2's outright edits are audited like pass 1's: anything pass 1 exposed, a repeat
        # pass 1 did not catch ("the  the"), or an edit that lost to an overlapping one there.
        edits = list(part.edits)
        review = part.review
    else:
        part = Partition([], [], [])
        edits = []
        review = [replace(m, start=_shifted(pre_text, applied, m.start),
                          end=_shifted(pre_text, applied, m.end)) for m in pass1_review]
    if use_doubles:
        # Doubles outside the closed list: adjudicated with a model, else reviewed.
        # One LanguageTool's repeat rule already routed is not raised twice.
        seen = ({(i.line, i.start) for i in part.items if i.source == "repeat"}
                | {_line_col(text, m.start) for m in review if m.rule in (REPEAT, "basics")})
        next_id = max((i.id for i in part.items), default=0) + 1
        spans = protected_spans(text)
        seen |= {(e.line, e.start) for e in edits if e.source == REPEAT}
        for d in basics.doubled_words(text, spans):
            route = double_route(d, asking)
            if accepted_double(text, d, accept) or _line_col(text, d.first_start) in seen:
                continue
            if route == "edit":       # one pass 1 could not apply ("the  the" beside a whitespace fix)
                edits.append(basics.double_edit(text, d))
                seen.add(_line_col(text, d.first_start))
            elif route == "item":
                part.items.append(double_item(text, d, next_id))
                next_id += 1
            else:      # listed with LanguageTool's review matches, so the CLI shows it too
                review.append(Match(d.first_start, d.end, "basics", "", double_message(text, d), ""))
        # A space before : ; ! ? with more text after it may be a key ("Press : to"): a person decides.
        if use_spacing:
            review += [Match(p, p + 1, "basics", "", f"space before '{mark}'", "")
                       for p, mark in basics.spacing_notes(text, spans)]
    if not lt_pass2 and p1_rejected_repeats:
        # A pass-1 repeat edit that lost to an overlapping edit, with no pass 2 to catch it
        # and basics not routing it either: a person decides, nothing is dropped silently.
        routed = ({(e.line, e.start) for e in edits} | {(i.line, i.start) for i in part.items}
                  | {_line_col(text, m.start) for m in review})
        for e in p1_rejected_repeats:
            pos = _shifted(pre_text, applied, to_offsets(pre_text, e)[0])
            m = re.match(r"([^\W\d_]\S*)[ \t]+(\S+)", text[pos:])
            if not m or m.group(1).lower() != m.group(2).lower() or _line_col(text, pos) in routed:
                continue
            review.append(Match(pos, m.end() + pos, REPEAT, "",
                                f"repeated word '{m.group(1)} {m.group(2)}'", ""))
    routed = ({(e.line, e.start) for e in edits} | {(i.line, i.start) for i in part.items}
              | {_line_col(text, m.start) for m in review})
    if typos_held:
        # typos' findings that are not provably right: a model decides (KEEP offered), else
        # a person. One LanguageTool already routed at the same place is not raised twice.
        next_id = max((i.id for i in part.items), default=0) + 1
        for m in typos_held:
            s = _shifted(pre_text, applied, m.start)
            word = pre_text[m.start:m.end]
            if text[s:s + len(word)] != word:
                found = text.find(word, max(0, s - len(word)), s + 2 * len(word))
                if found < 0:
                    continue              # an overlapping fix changed the word
                s = found
            key = _line_col(text, s)
            if key in routed:
                continue
            routed.add(key)
            held = replace(m, start=s, end=s + len(word))
            if asking:
                part.items.append(typos_item(text, held, next_id))
                next_id += 1
            else:
                review.append(held)
    # A pass-1 edit that lost to an overlapping one (other than the repeats above).
    # One a pass-2 edit covers is resolved by it: not listed.
    covered = [to_offsets(text, e) for e in edits]
    review += [m for m in _overlap_reviews(pre_text, applied,
                                           [e for e in rejected_p1 if e.source != REPEAT], text, routed)
               if not any(s < m.end and m.start < t for s, t in covered)]
    system = system_prompt(TASK.format(variety=VARIETY[o.target]))
    if part.items:
        decisions = adjudicate(ep, f"proofix|{o.target}", part.items, cache,
                               parallel=o.parallel, verbose=o.verbose, system=system,
                               log_name="proofix")
        for d in decisions:
            if d.choice is not None and d.confidence < o.min_confidence:
                d.reason, d.choice = f"low confidence ({d.confidence:.2f}): {d.reason}", None
            if d.replacement is not None:
                edits.append(Edit(d.item.line, d.item.start, d.item.end, d.item.word,
                                  d.replacement, "adjudicated"))
    new_text, rejected = apply_edits(text, edits)
    pass2_applied = [e for e in edits if e not in rejected]
    # an edit (a decision, too) that lost to an overlapping one: listed unless already listed
    review += _overlap_reviews(text, pass2_applied, rejected, new_text,
                               {_line_col(text, m.start) for m in review}, at_before=True)
    applied += pass2_applied
    text = new_text
    # Pass 3: compounds and consistency, on the text as it now stands.
    if "compounds" not in skip and asking:
        c_items, groups = consistency.items(text, o.target, o.style)
        if c_items:
            c_decisions = adjudicate(ep, f"proofix|{o.target}", c_items, cache,
                                     parallel=o.parallel, verbose=o.verbose, system=system,
                                     log_name="proofix")
            c_edits = []
            for d in c_decisions:
                if d.choice is not None and d.confidence < o.min_confidence:
                    d.reason, d.choice = f"low confidence ({d.confidence:.2f}): {d.reason}", None
                if not d.replacement:
                    continue
                for g in groups[d.item.id]:
                    line = text.count("\n", 0, g.start) + 1
                    col = g.start - (text.rfind("\n", 0, g.start) + 1) + 1
                    rep = consistency._recase(g.form, d.replacement.lower())
                    c_edits.append(Edit(line, col, col + (g.end - g.start) - 1, g.form, rep,
                                        "adjudicated"))
            decisions += c_decisions
            new_text, rejected = apply_edits(text, c_edits)
            applied += [e for e in c_edits if e not in rejected]
            text = new_text
    new_text = text

    before_commas = new_text
    comma_results: list[CommaResult] = []
    if "commas" not in skip and o.commas != "none":
        cedits, comma_results = correct_commas(new_text, o.target, ep, comma_cache,
                                               o.optional_commas, o.min_confidence,
                                               parallel=o.parallel, verbose=o.verbose,
                                               gate=o.commas == "gated",
                                               max_sentences=o.max_sentences)
        final, cskipped = apply_edits(new_text, cedits)
        applied += [e for e in cedits if e not in cskipped]
        new_text = final

    # Most specific label last-wins-first: commas, then adjudicated, then plain edits.
    ann: list[Annotation] = []
    for r in comma_results:
        if r.status == "edit" and r.new_text:
            by = "cache" if r.reason.endswith("(cached)") else ("model" if ep is not None else "rule")
            for e in comma_edits(before_commas, r.sentence, r.new_text):
                ann.append(Annotation(e.original, e.replacement, "commas", r.reason, by, r.confidence))
    for d in decisions:
        if d.replacement is not None:
            ann.append(Annotation(d.item.word, d.replacement, "adjudicated", d.reason,
                                  _decided_by(d, ep is not None), d.confidence))
    ann += [Annotation(e.original, e.replacement, e.source) for e in applied]
    changes = derive_changes(original, new_text, ann, "proofix")
    for c in changes:
        c.pass_ = "proofix"

    review_items += [ReviewItem(_line_of(review_text, m.start), "alert", m.message, m.rule, "proofix")
                     for m in review]
    review_items += [ReviewItem(d.item.line, "undecided",
                                (f"repeated word '{' '.join(d.item.word.split())}': {d.reason}"
                                 if d.item.source == "repeat" else f"'{d.item.word}': {d.reason}"),
                                d.item.source, "proofix") for d in decisions if d.choice is None]
    for r in comma_results:
        if r.status in ("rejected", "note"):
            review_items.append(ReviewItem(_line_of(before_commas, r.sentence.start),
                                           r.status, r.reason, "commas", "proofix"))
        elif r.status == "error" and r.reason != "no model configured":   # reported once, as a skipped capability
            review_items.append(ReviewItem(_line_of(before_commas, r.sentence.start),
                                           "note", r.reason, "commas", "proofix"))
    return Result(new_text, changes, review_items, skipped, applied, decisions, comma_results,
                  review, typos_fixed, hygiene_notes, lt_missing)
