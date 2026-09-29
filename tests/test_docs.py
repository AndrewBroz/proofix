"""The man page and --help must keep up with the parser."""

import re
from pathlib import Path

from proofix import __version__
from proofix.cli import build_parser

MAN = (Path(__file__).parent.parent / "man" / "proofix.1").read_text(encoding="utf-8")
PLAIN = MAN.replace("\\-", "-")          # roff writes --llm as \-\-llm


def test_every_long_option_is_in_the_man_page():
    missing = [o for a in build_parser()._actions for o in a.option_strings
               if o.startswith("--") and o not in PLAIN]
    assert missing == []


def test_llm_choices_match():
    choices = next(a.choices for a in build_parser()._actions if "--llm" in a.option_strings)
    assert "{" + "|".join(choices) + "}" in PLAIN


def test_man_version_matches():
    m = re.search(r'^\.TH \S+ 1 "[^"]*" "proofix ([^"]+)"', MAN, re.M)
    assert m and m.group(1) == __version__


def test_no_dgx_in_help_or_man():
    assert "dgx" not in build_parser().format_help().lower()
    assert "dgx" not in MAN.lower()
