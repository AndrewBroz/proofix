"""Library API: proofread a Markdown string; the CLI is a wrapper around fix()."""

from __future__ import annotations

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
from .lt import Match, find_languagetool, run_languagetool
from .tiers import (REPEAT, Partition, _line_col, accepted_double, double_item, double_message,
                    double_route, partition, typos_edits)

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


def run_typos(text: str) -> list[Edit]:
    """Known-misspelling fixes from the `typos` tool, as edits, if installed."""
    exe = shutil.which("typos")
    if not exe:
        return []
    proc = subprocess.run([exe, "--format", "json", "-"], input=text, capture_output=True, text=True)
    return typos_edits(text, proc.stdout)


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
        comma_cache: CommaCache) -> Result:
    """Proofread text; no I/O on the caller's files. Raises LTError if LanguageTool fails."""
    o, ep = options, endpoint
    skip = o.skip
    asking = ep is not None or o.replay_cache
    accept = {w.lower() for w in o.accept}
    original = text
    skipped: list[Skipped] = []
    hygiene_notes: list[str] = []
    edits: list[Edit] = []
    if "hygiene" not in skip:
        text, hygiene_notes = hygiene.normalize_text(text)
        h_edits, h_notes = hygiene.plan(text)
        edits += h_edits
        hygiene_notes += [f"L{n.line} {n.message}" for n in h_notes]
    if "typos" not in skip:
        if shutil.which("typos") is None:
            skipped.append(Skipped("known-misspelling fixes", "typos not found", "brew install typos-cli"))
        else:
            edits += run_typos(text)
    typos_fixed = sum(1 for e in edits if e.source == "typos")
    decisions: list[Decision] = []
    review: list[Match] = []
    review_items: list[ReviewItem] = []
    lt_wanted = not {"mechanical", "adjudicated"} <= skip
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
    pass1_repeats: list[Match] = []
    lt_repeats: set[tuple[int, int]] = set()
    if lt and "mechanical" not in skip:
        matches = run_languagetool(text, o.target, o.lt_level)
        p1 = partition(text, matches, accept, adjudicate_available=False)
        edits += p1.edits
        lt_repeats = {(e.line, e.start) for e in p1.edits if e.source == REPEAT}
        if "adjudicated" in skip:      # no pass 2 to route them: keep pass 1's repeat reviews
            pass1_repeats = [m for m in p1.review if m.rule == REPEAT]
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
    text, rejected = apply_edits(text, edits)
    applied = [e for e in edits if e not in rejected]
    # Pass 2: judgement calls on the cleaned text, so spans no longer collide.
    review_text = text
    if lt and "adjudicated" not in skip:
        matches = run_languagetool(text, o.target, o.lt_level)
        part = partition(text, matches, accept, adjudicate_available=asking)
        # anything pass 1 exposed, or a repeat pass 1 did not catch ("the  the")
        edits = [e for e in part.edits if e.source in ("MISSING_SPACE", REPEAT)]
        review = part.review
    else:
        part = Partition([], [], [])
        edits = []
        review = [replace(m, start=_shifted(pre_text, applied, m.start),
                          end=_shifted(pre_text, applied, m.end)) for m in pass1_repeats]
    if use_doubles:
        # Doubles outside the closed list: adjudicated with a model, else reviewed.
        # One LanguageTool's repeat rule already routed is not raised twice.
        seen = ({(i.line, i.start) for i in part.items if i.source == "repeat"}
                | {_line_col(text, m.start) for m in review if m.rule in (REPEAT, "basics")})
        next_id = max((i.id for i in part.items), default=0) + 1
        spans = protected_spans(text)
        for d in basics.doubled_words(text, spans):
            route = double_route(d, asking)
            if route == "edit" or accepted_double(text, d, accept) or _line_col(text, d.first_start) in seen:
                continue
            if route == "item":
                part.items.append(double_item(text, d, next_id))
                next_id += 1
            else:      # listed with LanguageTool's review matches, so the CLI shows it too
                review.append(Match(d.first_start, d.end, "basics", "", double_message(text, d), ""))
        # A space before : ; ! ? with more text after it may be a key ("Press : to"): a person decides.
        if use_spacing:
            review += [Match(p, p + 1, "basics", "", f"space before '{mark}'", "")
                       for p, mark in basics.spacing_notes(text, spans)]
    system = system_prompt(TASK.format(variety=VARIETY[o.target]))
    if part.items:
        decisions = adjudicate(ep, f"proofix|{o.target}", part.items, cache,
                               parallel=o.parallel, verbose=o.verbose, system=system,
                               log_name="proofix")
        for d in decisions:
            if d.choice is not None and d.confidence < o.min_confidence:
                d.reason, d.choice = f"low confidence ({d.confidence:.2f}): {d.reason}", None
            if d.replacement:
                edits.append(Edit(d.item.line, d.item.start, d.item.end, d.item.word,
                                  d.replacement, "adjudicated"))
    new_text, rejected = apply_edits(text, edits)
    applied += [e for e in edits if e not in rejected]
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
        if d.replacement:
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
