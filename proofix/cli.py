"""Command-line entry point."""

from __future__ import annotations

import argparse
import difflib
import math
import os
import sys
from pathlib import Path

from adjudicate import Cache, ConfigError, check_endpoint, resolve_endpoint, run_setup

from . import __version__, api, report
from .commas import CommaCache
from .lt import LTError

DESCRIPTION = """\
Fix typos and punctuation in a Markdown file, under audit.

Five tiers, each bounded so every change is one logged, reasoned edit:

0. Hygiene: HTML entities Markdown can carry plainly, zero-width spaces,
   byte-order marks, soft hyphens, control characters, Windows line endings
   and decomposed accents are fixed deterministically.
1. Mechanical faults with one right answer are fixed outright: a space before a
   comma or period, a missing space after one, doubled punctuation, a doubled
   "the", "of" or other word whose repeat is never grammatical. Detected by
   LanguageTool (and, if installed, the `typos` tool for known misspellings);
   without LanguageTool, a small built-in check (tier "basics") fixes the
   spacing and doubled words. Other doubles ("that that", "had had") are
   adjudicated by the model, or listed for review without one.
2. Adjudicated: LanguageTool proposes spellings or grammar fixes, and a
   language model picks one or KEEP from the sentence and the rule's message.
3. Compounds and consistency: the model chooses closed, hyphenated or open
   for compounds the dictionaries question ("cyber attack", "non-profit"),
   and one form for terms the document capitalises two ways.
4. Commas: comma placement, which no rule engine detects, is corrected
   sentence by sentence by the model under a diff guard: only insertions and
   deletions of commas are accepted; anything else the model returns is
   rejected and reported.

Without a model, tiers 0 and 1 still apply; tier 2's fixes are listed for
review, tier 3 is skipped and sentences are not checked for commas.

Code spans, fenced code, link targets, URLs, HTML and front matter are never
touched. Run it after stylefix so LanguageTool sees the target variety.
"""

EPILOG = """\
examples:
  proofix --setup                        # choose, test and save a model endpoint
  proofix draft.md                       # -> draft.proofed.md, US English
  proofix draft.md --target uk -o out.md
  proofix draft.md --changes             # list every change as it is made
  proofix draft.md --no-commas --llm none
      LanguageTool's mechanical fixes only; nothing needs a model

Requires languagetool on PATH for tier 1's LanguageTool fixes and for tier 2
(brew install languagetool); tiers 3 and 4 need only a model. Accept lists:
--accept, ~/.config/proofix/accept.txt, ~/.config/stylefix/accept.txt,
./.stylefix/accept.txt and ./.proofix/accept.txt are all honoured.

environment:
  PROOFIX_LLM_URL, PROOFIX_LLM_MODEL, PROOFIX_LLM_KEY
      the model endpoint for proofix (same as --url, --model, and the API key)
  ADJUDICATE_LLM_URL, ADJUDICATE_LLM_MODEL, ADJUDICATE_LLM_KEY
      the same, shared with stylefix; below the proofix variables and config
  OLLAMA_HOST   Ollama base URL (default http://localhost:11434)

files:
  ~/.config/proofix/config.toml      [llm] url, model, api_key or api_key_env,
                                     timeout, thinking_switch (proofix only)
  ~/.config/adjudicate/config.toml   the same table, shared with stylefix

Full documentation: https://github.com/AndrewBroz/proofix (man page:
man/proofix.1 in the source)
"""

def positive_seconds(text: str) -> float:
    """argparse type for --timeout: a positive, finite number of seconds."""
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not (math.isfinite(value) and value > 0):
        raise argparse.ArgumentTypeError(f"must be a positive number of seconds, not {text}")
    return value


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="proofix", description=DESCRIPTION, epilog=EPILOG,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("input", type=Path, nargs="?", help="Markdown file to proofread")
    p.add_argument("-o", "--output", type=Path, metavar="FILE",
                   help="default: <input>.proofed.md")
    p.add_argument("-i", "--in-place", action="store_true", help="overwrite the input file")
    p.add_argument("--target", choices=["us", "uk"], default="us",
                   help="English variety the document is written in (default: %(default)s)")
    p.add_argument("--accept", type=Path, metavar="FILE",
                   help="extra words to never flag as misspelled, one per line")
    p.add_argument("--no-lt", action="store_true",
                   help="skip LanguageTool: its tier 1 fixes (typos still runs) and all of tier 2")
    p.add_argument("--lt-level", choices=["default", "picky"], default="default",
                   help="LanguageTool rule level (default: %(default)s)")
    p.add_argument("--no-typos", action="store_true",
                   help="skip the `typos` pre-pass even if installed")
    p.add_argument("--no-hygiene", action="store_true",
                   help="skip text hygiene (HTML entities, invisible characters, line endings, "
                        "decomposed accents)")
    p.add_argument("--no-consistency", action="store_true",
                   help="skip the compounds and consistency tier (closed, hyphenated or open "
                        "compounds; terms capitalised two ways)")
    p.add_argument("--style", default="none", metavar="GUIDE",
                   help="style guide, only for its hyphenation habit (ap hyphenates less); "
                        "chicago, gpo, ap, apa, oxford, guardian or none")
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
    p.add_argument("--llm", choices=["auto", "ollama", "none"], default="auto", help="auto: the endpoint from --url, PROOFIX_LLM_*, ~/.config/proofix/config.toml, ADJUDICATE_LLM_* or ~/.config/adjudicate/config.toml, else local Ollama; ollama: local Ollama only; none: no model, so tier 2's fixes are listed for review, tier 3 is skipped and commas are not checked (default: %(default)s)")
    p.add_argument("--url", metavar="URL",
                   help="OpenAI-compatible base URL ending in /v1; overrides every configured "
                        "url. A --url or PROOFIX_LLM_URL drops keys set in lower layers; set "
                        "PROOFIX_LLM_KEY (or api_key_env in the same config file) alongside it")
    p.add_argument("--model", metavar="NAME")
    p.add_argument("--min-confidence", type=float, default=0.8, metavar="X")
    p.add_argument("--parallel", type=int, default=4, metavar="N")
    p.add_argument("--timeout", type=positive_seconds, default=None, metavar="SEC", help="seconds per model request (default: 120, or timeout in config.toml)")
    c = p.add_argument_group("configuration", "set up or check the model endpoint, then exit")
    c.add_argument("--setup", action="store_true",
                   help="choose a model endpoint interactively, test it, and save it to "
                        "~/.config/adjudicate/config.toml or ~/.config/proofix/config.toml")
    c.add_argument("--check-endpoint", action="store_true",
                   help="show the endpoint proofix would use and where each setting comes "
                        "from, then test it; exit 0 if it answers, 1 if not, 2 on a config error")
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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.setup:
        return run_setup("proofix")
    if args.check_endpoint:
        return check_endpoint("proofix", preset=args.llm, url=args.url, model=args.model,
                              timeout=args.timeout)
    if args.input is None:
        parser.error("the following arguments are required: input")
    if not args.input.exists():
        print(f"proofix: no such file: {args.input}", file=sys.stderr)
        return 2
    output = args.input if args.in_place else (
        args.output or args.input.with_name(args.input.stem + ".proofed" + (args.input.suffix or ".md")))
    report_path = args.report or output.with_suffix(output.suffix + ".report.md")
    text = args.input.read_text(encoding="utf-8")
    accept = load_accept(args.accept)
    try:
        ep = resolve_endpoint("proofix", args.llm, url=args.url, model=args.model,
                              timeout=args.timeout)
    except ConfigError as e:
        print(f"proofix: {e}", file=sys.stderr)
        return 2
    if ep is None and args.llm == "auto":
        print("proofix: no model endpoint found; adjudicated fixes (tier 2) will be listed for "
              "review, not applied, compounds (tier 3) skipped and commas (tier 4) not checked. "
              "Run `proofix --setup` to configure one.", file=sys.stderr)

    skip = {t for t, flag in (("hygiene", args.no_hygiene), ("typos", args.no_typos),
                              ("compounds", args.no_consistency)) if flag}
    skip |= {"mechanical", "adjudicated"} if args.no_lt else set()
    skip |= {"commas"} if args.no_commas else set()
    options = api.Options(target=args.target, accept=frozenset(accept), lt_level=args.lt_level,
                          style=args.style, commas=args.commas, optional_commas=args.optional_commas,
                          max_sentences=args.max_sentences, min_confidence=args.min_confidence,
                          parallel=args.parallel, verbose=args.verbose, skip=frozenset(skip))
    cache = Cache(None if args.no_cache else args.cache)
    comma_cache = CommaCache(None if args.no_cache else args.cache.with_name("commas.json"))
    try:
        r = api.fix(text, options, ep, cache, comma_cache)
    except LTError as e:
        print(f"proofix: {e}", file=sys.stderr)
        return 2
    if r.lt_missing:
        print("proofix: languagetool not found; skipping its tier 1 fixes and tier 2 "
              "(brew install languagetool)", file=sys.stderr)
    original, new_text = text, r.text
    applied, decisions, comma_results = r.applied, r.decisions, r.comma_results
    review, typos_fixed, hygiene_notes = r.lt_review, r.typos_fixed, r.hygiene_notes

    if args.dry_run:
        sys.stdout.writelines(difflib.unified_diff(
            original.splitlines(keepends=True), new_text.splitlines(keepends=True),
            str(args.input), str(output)))
    else:
        output.write_text(new_text, encoding="utf-8")
    log = report.changelog(original, applied, decisions, comma_results, review, typos_fixed,
                           hygiene_notes)
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
