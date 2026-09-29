"""The man page and --help must keep up with the parser."""

import re
from pathlib import Path

import pytest

from proofix import __version__
from proofix.cli import build_parser

ROOT = Path(__file__).parent.parent
MAN = (ROOT / "man" / "proofix.1").read_text(encoding="utf-8")
README = (ROOT / "README.md").read_text(encoding="utf-8")
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


def flat(text):
    return " ".join(text.split())


TIERS = ["0 hygiene", "1 mechanical", "2 adjudicated", "3 compounds and consistency", "4 commas"]


def test_five_tiers_everywhere():
    import proofix.cli as cli
    desc = flat(cli.DESCRIPTION).lower()
    assert "five tiers" in desc and "three tiers" not in desc
    for n, name in enumerate(["hygiene", "mechanical", "adjudicated", "compounds and consistency",
                              "commas"]):
        assert f"{n}. {name}" in desc, name
    for tier in TIERS:
        assert f"| {tier} |" in README, tier
        assert f".B {tier}" in MAN, tier
    assert "tier 3, gated" not in README          # the comma tier is 4
    assert "tiers two and three" not in MAN and "first two tiers" not in MAN


def test_no_model_wording_matches_the_code():
    llm = next(a for a in build_parser()._actions if "--llm" in a.option_strings)
    for text in (llm.help, flat(PLAIN), flat(README)):
        assert "listed for review" in text and "tier 3 is skipped" in text
    assert "only hygiene and commas run" not in flat(README)


def test_help_points_at_documentation_that_ships():
    text = build_parser().format_help()
    assert "https://github.com/AndrewBroz/proofix" in text and "man/proofix.1" in text
    assert "https://github.com/AndrewBroz/proofix" in MAN


def test_accept_lists_name_the_project_file():
    assert "./.proofix/accept.txt" in build_parser().format_help()
    assert "./.proofix/accept.txt" in README


def test_no_local_model_wording():
    for text in (flat(README), flat(PLAIN), flat(build_parser().format_help())):
        assert "local model picks" not in text and "a local model" not in text


def test_man_llm_entry_stands_alone():
    assert "as in stylefix" not in flat(PLAIN)
    start = PLAIN.index(".B --check-endpoint")
    entry = PLAIN[start:PLAIN.index(".TP", start)]
    assert "Honours" in entry
    assert all(o in entry for o in ("--llm", "--url", "--model", "--timeout"))


def test_url_help_states_the_key_reset():
    url = next(a for a in build_parser()._actions if "--url" in a.option_strings)
    assert "overrides every configured url" in url.help
    assert "PROOFIX_LLM_KEY" in url.help and "api_key_env" in url.help


@pytest.mark.parametrize("value", ["0", "-5", "nan", "inf", "soon"])
def test_timeout_must_be_positive(value, capsys):
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(["--timeout", value])
    assert e.value.code == 2 and "--timeout" in capsys.readouterr().err


def test_timeout_accepts_a_positive_number():
    assert build_parser().parse_args(["--timeout", "0.5"]).timeout == 0.5
