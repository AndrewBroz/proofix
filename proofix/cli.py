"""Command-line entry point."""

from __future__ import annotations

import argparse
import difflib
import os
import shutil
import subprocess
import sys
from pathlib import Path

from adjudicate import Cache, Edit, adjudicate, apply_edits, resolve_endpoint, system_prompt

from . import __version__, report
from .commas import CommaCache, correct_commas
from .lt import LTError, find_languagetool, run_languagetool
from . import consistency
from .tiers import partition, typos_edits

DESCRIPTION = """\
Fix typos and punctuation in a Markdown file, under audit.

Three tiers, each bounded so every change is one logged, reasoned edit:

1. Mechanical faults with one right answer are fixed outright: a space before a
   comma or period, a missing space after one, doubled punctuation, doubled
   words. Detected by LanguageTool (and, if installed, the `typos` tool for
   known misspellings).
2. Faults with candidate fixes are adjudicated: LanguageTool proposes
   spellings or grammar fixes, a local model picks one or KEEP from the
   sentence and the rule's message.
3. Comma placement, which no rule engine detects, is corrected sentence by
   sentence by the model under a diff guard: only insertions and deletions of
   commas are accepted; anything else the model returns is rejected and
   reported.

Code spans, fenced code, link targets, URLs, HTML and front matter are never
touched. Run it after stylefix so LanguageTool sees the target variety.
"""

EPILOG = """\
examples:
  proofix draft.md                       # -> draft.proofed.md, US English
  proofix draft.md --target uk -o out.md
  proofix draft.md --changes             # list every change as it is made
  proofix draft.md --no-commas --llm none
      LanguageTool's mechanical fixes only; nothing needs a model

Requires languagetool on PATH for tiers 1 and 2 (brew install languagetool);
tier 3 needs only a model. Accept lists: --accept, ~/.config/proofix/accept.txt,
~/.config/stylefix/accept.txt and ./.stylefix/accept.txt are all honoured.
"""

TASK = "You are a meticulous copy editor fixing typos and punctuation in a document written in {variety}."
VARIETY = {"us": "American English", "uk": "British English"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="proofix", description=DESCRIPTION, epilog=EPILOG,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, help="Markdown file to proofread")
    p.add_argument("-o", "--output", type=Path, metavar="FILE",
                   help="default: <input>.proofed.md")
    p.add_argument("-i", "--in-place", action="store_true", help="overwrite the input file")
    p.add_argument("--target", choices=["us", "uk"], default="us",
                   help="English variety the document is written in (default: %(default)s)")
    p.add_argument("--accept", type=Path, metavar="FILE",
                   help="extra words to never flag as misspelled, one per line")
    p.add_argument("--no-lt", action="store_true", help="skip LanguageTool (tiers 1 and 2)")
    p.add_argument("--lt-level", choices=["default", "picky"], default="default",
                   help="LanguageTool rule level (default: %(default)s)")
    p.add_argument("--no-typos", action="store_true",
                   help="skip the `typos` pre-pass even if installed")
    p.add_argument("--no-consistency", action="store_true",
                   help="skip the consistency tier (the same term hyphenated, open or capitalised "
                        "two ways within the document)")
    p.add_argument("--commas", choices=["gated", "all", "none"], default="gated",
                   help="comma tier: gated sends only sentences with a comma, an introductory "
                        "opener, a clause trigger or a conjunction joining clauses; all sends every "
                        "sentence; none skips it (default: %(default)s)")
    p.add_argument("--no-commas", action="store_true", help="same as --commas none")
    p.add_argument("--max-sentences", type=int, metavar="N",
                   help="cap on sentences sent to the comma tier (the rest are reported as unchecked)")
    p.add_argument("--optional-commas", choices=["leave", "add"], default="leave",
                   help="what the comma tier does with commas that are optional by convention "
                        "(default: %(default)s)")
    p.add_argument("--llm", choices=["auto", "dgx", "ollama", "none"], default="auto")
    p.add_argument("--url", metavar="URL", help="OpenAI-compatible base URL ending in /v1")
    p.add_argument("--model", metavar="NAME")
    p.add_argument("--min-confidence", type=float, default=0.8, metavar="X")
    p.add_argument("--parallel", type=int, default=4, metavar="N")
    p.add_argument("--timeout", type=float, default=120.0, metavar="SEC")
    p.add_argument("--cache", type=Path, default=Path(".proofix/cache.json"), metavar="FILE")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--report", type=Path, metavar="FILE", help="default: <output>.report.md")
    p.add_argument("--no-report", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="print a diff, write nothing")
    p.add_argument("--changes", action="store_true", help="print every change to stderr")
    p.add_argument("-v", "--verbose", action="store_true")
    p.add_argument("--version", action="version", version=f"proofix {__version__}")
    return p


def load_accept(explicit: Path | None) -> set[str]:
    cfg = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    paths = [cfg / "proofix" / "accept.txt", cfg / "stylefix" / "accept.txt",
             Path(".stylefix") / "accept.txt", Path(".proofix") / "accept.txt"]
    if explicit:
        paths.append(explicit)
    words: set[str] = set()
    for path in paths:
        if path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    words.add(line.lower())
    return words


def run_typos(text: str) -> list[Edit]:
    """Known-misspelling fixes from the `typos` tool, as edits, if installed."""
    exe = shutil.which("typos")
    if not exe:
        return []
    proc = subprocess.run([exe, "--format", "json", "-"], input=text, capture_output=True, text=True)
    return typos_edits(text, proc.stdout)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.input.exists():
        print(f"proofix: no such file: {args.input}", file=sys.stderr)
        return 2
    output = args.input if args.in_place else (
        args.output or args.input.with_name(args.input.stem + ".proofed" + (args.input.suffix or ".md")))
    report_path = args.report or output.with_suffix(output.suffix + ".report.md")
    text = args.input.read_text(encoding="utf-8")
    accept = load_accept(args.accept)
    ep = resolve_endpoint(args.llm, args.model, args.url, timeout=args.timeout)

    original = text
    edits: list[Edit] = [] if args.no_typos else run_typos(text)
    typos_fixed = len(edits)
    decisions = []
    review = []
    lt = find_languagetool() if not args.no_lt else None
    if not args.no_lt and not lt:
        print("proofix: languagetool not found; skipping tiers 1 and 2 (brew install languagetool)",
              file=sys.stderr)
    # Pass 1: mechanical faults (typos, whitespace, doubled punctuation), applied.
    if lt:
        try:
            matches = run_languagetool(text, args.target, args.lt_level)
        except LTError as e:
            print(f"proofix: {e}", file=sys.stderr)
            return 2
        edits += partition(text, matches, accept, adjudicate_available=False).edits
    text, skipped = apply_edits(text, edits)
    applied = [e for e in edits if e not in skipped]
    # Pass 2: judgement calls on the cleaned text, so spans no longer collide.
    if lt:
        try:
            matches = run_languagetool(text, args.target, args.lt_level)
        except LTError as e:
            print(f"proofix: {e}", file=sys.stderr)
            return 2
        part = partition(text, matches, accept, adjudicate_available=ep is not None)
        edits = [e for e in part.edits if e.source == "MISSING_SPACE"]   # anything pass 1 exposed
        review = part.review
    else:
        from .tiers import Partition
        part = Partition([], [], [])
        edits = []
    if not args.no_consistency and ep is not None:
        part.items += consistency.items(text, next_id=len(part.items) + 1)
    if True:
        if True:
            if part.items:
                cache = Cache(None if args.no_cache else args.cache)
                decisions = adjudicate(ep, f"proofix|{args.target}", part.items, cache,
                                       parallel=args.parallel, verbose=args.verbose,
                                       system=system_prompt(TASK.format(variety=VARIETY[args.target])),
                                       log_name="proofix")
                for d in decisions:
                    if d.choice is not None and d.confidence < args.min_confidence:
                        d.reason, d.choice = f"low confidence ({d.confidence:.2f}): {d.reason}", None
                    if d.replacement:
                        edits.append(Edit(d.item.line, d.item.start, d.item.end, d.item.word,
                                          d.replacement, "adjudicated"))
        new_text, skipped = apply_edits(text, edits)
        applied += [e for e in edits if e not in skipped]
        text = new_text
    new_text = text

    comma_results = []
    if not args.no_commas and args.commas != "none":
        ccache = CommaCache(None if args.no_cache else args.cache.with_name("commas.json"))
        cedits, comma_results = correct_commas(new_text, args.target, ep, ccache,
                                               args.optional_commas, args.min_confidence,
                                               parallel=args.parallel, verbose=args.verbose,
                                               gate=args.commas == "gated",
                                               max_sentences=args.max_sentences)
        final, cskipped = apply_edits(new_text, cedits)
        applied += [e for e in cedits if e not in cskipped]
        new_text = final

    if args.dry_run:
        sys.stdout.writelines(difflib.unified_diff(
            original.splitlines(keepends=True), new_text.splitlines(keepends=True),
            str(args.input), str(output)))
    else:
        output.write_text(new_text, encoding="utf-8")
    log = report.changelog(original, applied, decisions, comma_results, review, typos_fixed)
    if args.changes or args.verbose:
        sys.stderr.write(log)
    if not args.no_report and not args.dry_run:
        report_path.write_text("# proofix report\n\n```\n" + log + "```\n", encoding="utf-8")
    n_commas = sum(1 for r in comma_results if r.status == "edit")
    print(f"proofix: {len(applied)} edit(s) ({n_commas} comma sentence(s)), "
          f"{sum(1 for d in decisions if d.choice is None)} undecided, {len(review)} for review"
          + ("" if args.dry_run else f" -> {output}"), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
