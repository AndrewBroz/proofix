"""Run LanguageTool and return its matches with character offsets."""

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


def parse_matches(json_text: str, text: str) -> list[Match]:
    data = json.loads(json_text[json_text.index("{"):]) if "{" in json_text else {"matches": []}
    out: list[Match] = []
    for m in data.get("matches", []):
        start, length = m["offset"], m["length"]
        # LanguageTool's offsets are UTF-16 code units in some builds; verify
        # against the context and fall back to a search near the offset.
        ctx = m.get("context", {})
        expected = ctx.get("text", "")[ctx.get("offset", 0):ctx.get("offset", 0) + ctx.get("length", 0)]
        if expected and text[start:start + length] != expected:
            near = text.find(expected, max(0, start - 8), start + length + 8)
            if near < 0:
                continue
            start = near
        out.append(Match(start, start + length, m["rule"]["id"],
                         m["rule"].get("category", {}).get("id", ""), m.get("message", ""),
                         m.get("shortMessage", ""),
                         [r["value"] for r in m.get("replacements", [])]))
    out.sort(key=lambda x: x.start)
    return out
