"""Run LanguageTool and return its matches with character (code-point) offsets."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field

LANG = {"us": "en-US", "uk": "en-GB"}


class LTError(RuntimeError):
    pass


@dataclass
class Match:
    start: int                 # absolute character offset
    end: int
    rule: str                  # e.g. MORFOLOGIK_RULE_EN_US
    category: str              # e.g. TYPOS, PUNCTUATION
    message: str
    short: str
    replacements: list[str] = field(default_factory=list)

    @property
    def length(self) -> int:
        return self.end - self.start


def find_languagetool() -> str | None:
    return shutil.which("languagetool") or shutil.which("languagetool-commandline")


def run_languagetool(text: str, target: str, level: str = "default",
                     disabled: list[str] | None = None, timeout: float = 600) -> list[Match]:
    exe = find_languagetool()
    if not exe:
        raise LTError("languagetool not found on PATH (brew install languagetool).")
    cmd = [exe, "-l", LANG[target], "--json"]
    if level == "picky":
        cmd += ["--level", "PICKY"]
    if disabled:
        cmd += ["-d", ",".join(disabled)]
    proc = subprocess.run(cmd, input=text, capture_output=True, text=True, timeout=timeout)
    if proc.returncode != 0 and not proc.stdout.strip():
        raise LTError(f"languagetool failed: {proc.stderr.strip()[-500:]}")
    return parse_matches(proc.stdout, text)


def utf16_to_codepoints(s: str) -> list[int]:
    """Map each UTF-16 code-unit offset in ``s`` (0..len16 inclusive) to a
    code-point offset. LanguageTool is a Java program: its offsets and
    lengths count UTF-16 code units, so every astral character (an emoji,
    most of a ZWJ sequence) before a match shifts it by one per character.
    An offset that falls inside a surrogate pair maps to that character."""
    out: list[int] = []
    for i, ch in enumerate(s):
        out.append(i)
        if ord(ch) > 0xFFFF:
            out.append(i)
    out.append(len(s))
    return out


def _cp(table: list[int], u16: int) -> int:
    return table[min(max(u16, 0), len(table) - 1)]


def parse_matches(json_text: str, text: str) -> list[Match]:
    data = json.loads(json_text[json_text.index("{"):]) if "{" in json_text else {"matches": []}
    out: list[Match] = []
    table = utf16_to_codepoints(text)
    for m in data.get("matches", []):
        start = _cp(table, m["offset"])
        end = _cp(table, m["offset"] + m["length"])
        length = end - start
        # Verify against the context (whose offsets are UTF-16 too) and fall
        # back to a search near the offset when the text no longer lines up.
        ctx = m.get("context", {})
        ctext = ctx.get("text", "")
        ctable = utf16_to_codepoints(ctext)
        coff = ctx.get("offset", 0)
        expected = ctext[_cp(ctable, coff):_cp(ctable, coff + ctx.get("length", 0))]
        if expected and text[start:end] != expected:
            near = text.find(expected, max(0, start - 8), end + 8)
            if near < 0:
                continue
            start = near
            length = len(expected)
        out.append(Match(start, start + length, m["rule"]["id"],
                         m["rule"].get("category", {}).get("id", ""), m.get("message", ""),
                         m.get("shortMessage", ""),
                         [r["value"] for r in m.get("replacements", [])]))
    out.sort(key=lambda x: x.start)
    return out
