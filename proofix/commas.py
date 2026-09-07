"""Constrained rewrite tier: the model corrects comma placement sentence by
sentence, and a diff guard admits only insertions and deletions of commas.
Anything else the model returns is rejected and reported."""

from __future__ import annotations

import difflib
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from adjudicate import Edit, Endpoint, LLMError, prose_regions, strip_thinking

VARIETY = {"us": "American English", "uk": "British English"}

SYSTEM = """You are a copy editor checking comma placement only, in {variety}.
For each numbered sentence, decide whether a comma is missing or misplaced. You may ONLY insert or delete commas.
Never change, add, or remove any other character, including other punctuation and spacing.
{optional}
Never add or remove a serial (Oxford) comma before the last item of a list; that is a style choice made elsewhere.
Do not "fix" a comma splice by changing punctuation; instead list it under "notes".
Return ONLY the sentences you changed. Respond with JSON only:
{{"changed": [{{"id": 1, "text": "<corrected sentence>", "confidence": 0.0, "reason": "<under 12 words>"}}],
 "notes": [{{"id": 2, "reason": "<e.g. comma splice; needs a semicolon or period>"}}]}}"""

OPTIONAL = {
    "leave": "Where a comma is optional by convention (after a short introductory phrase, before "
             "a short final clause), leave the sentence unchanged.",
    "add": "Where a comma is optional by convention, add it: this style guide prefers commas "
           "after introductory phrases and before coordinating conjunctions joining clauses.",
}

_SENT_END = re.compile(r"(?<=[.!?])[\"”’)]*\s+(?=[A-Z“\"(])")
_MD_LINE = re.compile(r"^(\s*(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|>\s?|\|))")

# The gate: a sentence goes to the model only if a comma question can arise.
_OPENER = re.compile(r"^(?:After|Although|As|Because|Before|By|Despite|During|For|From|If|In|On|Once|"
                     r"Since|Though|Through|Unless|Until|When|Whenever|Where|Whereas|While|With|Without|"
                     r"Given|Having|Following|Under|Over|To|At|Like|Unlike|Beyond|Within|Throughout|"
                     r"According|Instead|Rather|First|Second|Third|Finally|Meanwhile|Today|Yesterday|"
                     r"Tomorrow|Now|Then|Thus|Hence|So|Yet|But|And|Or|Nor|However|Therefore|Moreover|"
                     r"Furthermore|Nevertheless|Nonetheless|Consequently|Otherwise|Indeed|Still|Also|"
                     r"Additionally|Importantly|Notably|Ultimately|Overall)\b")
_TRIGGER = re.compile(r"\b(?:which|however|therefore|moreover|for example|for instance|such as|including|"
                      r"e\.g\.|i\.e\.|said|says|asked|replied|added|noted|argued|wrote)\b|[\"“”]", re.I)
_CLAUSE_JOIN = re.compile(r"\b(?:and|but|so|yet|or|nor)\b", re.I)


def needs_check(sentence: str) -> bool:
    """True if the sentence contains a comma, an introductory opener, a
    clause-level trigger, or two clauses joined by a conjunction."""
    if "," in sentence:
        return True
    if _OPENER.match(sentence):
        return True
    if _TRIGGER.search(sentence):
        return True
    words = len(sentence.split())
    return words >= 12 and bool(_CLAUSE_JOIN.search(sentence))


@dataclass
class Sentence:
    start: int
    end: int
    text: str


@dataclass
class CommaResult:
    sentence: Sentence
    new_text: str | None        # None = unchanged / no answer
    confidence: float
    reason: str
    status: str                 # unchanged | edit | rejected | low | note | error


def sentences(text: str, min_words: int = 4, gate: bool = True) -> list[Sentence]:
    """Sentences in prose regions, excluding table rows and headings, and
    (with gate) only those where a comma question can arise."""
    out: list[Sentence] = []
    for rs, re_ in prose_regions(text):
        region = text[rs:re_]
        for para_m in re.finditer(r"[^\n]+(?:\n[^\n]+)*", region):
            para = para_m.group(0)
            pstart = rs + para_m.start()
            first = para.split("\n", 1)[0]
            if _MD_LINE.match(first) and (first.lstrip().startswith(("#", "|"))):
                continue
            cursor = 0
            pieces = []
            for m in _SENT_END.finditer(para):
                pieces.append((cursor, m.start() + len(m.group(0).rstrip())))
                cursor = m.end()
            pieces.append((cursor, len(para)))
            for a, b in pieces:
                s = para[a:b]
                lead = len(s) - len(s.lstrip())
                s2 = s.strip()
                if len(s2.split()) < min_words or (gate and not needs_check(s2)):
                    continue
                out.append(Sentence(pstart + a + lead, pstart + a + lead + len(s2), s2))
    return out


def only_commas(a: str, b: str) -> bool:
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        if set(a[i1:i2] + b[j1:j2]) - {","}:
            return False
    return True


def comma_edits(text: str, s: Sentence, new: str) -> list[Edit]:
    """Edits (line, col based) turning s.text into new; only comma changes."""
    from adjudicate.apply import line_offsets  # noqa: F401
    edits = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, s.text, new, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        pos = s.start + i1
        line = text.count("\n", 0, pos) + 1
        col = pos - (text.rfind("\n", 0, pos) + 1) + 1
        original, repl = s.text[i1:i2], new[j1:j2]
        edits.append(Edit(line, col, col + len(original) - 1, original, repl, "commas"))
    return edits


class CommaCache:
    def __init__(self, path: Path | None):
        self.path, self.data = path, {}
        if path and path.exists():
            try:
                self.data = json.loads(path.read_text())
            except json.JSONDecodeError:
                self.data = {}

    @staticmethod
    def key(scope: str, sentence: str) -> str:
        return hashlib.sha256(f"commas|{scope}|{sentence.strip()}".encode()).hexdigest()[:24]

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, indent=1, ensure_ascii=False))


def _ask(ep: Endpoint, system: str, batch: list[Sentence]) -> dict[int, dict] | None:
    """{id: {"text", "confidence", "reason"} or {"note": reason}}; None on a bad reply."""
    user = "\n".join(f"{i + 1}. {s.text}" for i, s in enumerate(batch))
    reply = strip_thinking(ep.chat(system, user))
    m = re.search(r"\{.*\}", reply, re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    out: dict[int, dict] = {}
    for d in data.get("changed") or data.get("sentences") or []:
        try:
            out[int(d["id"])] = d
        except (KeyError, TypeError, ValueError):
            continue
    for d in data.get("notes") or []:
        try:
            out.setdefault(int(d["id"]), {"note": str(d.get("reason", ""))})
        except (KeyError, TypeError, ValueError):
            continue
    return out


def correct_commas(text: str, target: str, ep: Endpoint | None, cache: CommaCache,
                   optional: str = "leave", min_confidence: float = 0.8, batch_size: int = 15,
                   parallel: int = 4, verbose: bool = False, gate: bool = True,
                   max_sentences: int | None = None) -> tuple[list[Edit], list[CommaResult]]:
    sents = sentences(text, gate=gate)
    if max_sentences is not None:
        sents = sents[:max_sentences]
    scope = f"{target}|{optional}"
    results: dict[int, CommaResult] = {}
    todo: list[tuple[int, Sentence]] = []
    for i, s in enumerate(sents):
        hit = cache.data.get(CommaCache.key(scope, s.text))
        if hit is not None:
            new = hit.get("text")
            results[i] = CommaResult(s, new if new and new != s.text else None,
                                     float(hit.get("confidence", 1.0)), hit.get("reason", "") + " (cached)",
                                     "edit" if new and new != s.text else "unchanged")
        else:
            todo.append((i, s))
    if todo and ep is not None:
        system = SYSTEM.format(variety=VARIETY[target], optional=OPTIONAL[optional])
        batches = [todo[k:k + batch_size] for k in range(0, len(todo), batch_size)]
        if verbose:
            print(f"proofix: checking commas in {len(todo)} sentence(s), {len(batches)} batch(es)",
                  file=sys.stderr)

        def run(batch):
            try:
                return batch, _ask(ep, system, [s for _, s in batch])
            except LLMError as e:
                return batch, {"error": str(e)}

        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            for batch, answers in pool.map(run, batches):
                for n, (i, s) in enumerate(batch, 1):
                    if answers is None or "error" in answers:
                        why = answers["error"] if answers else "unparseable reply"
                        results[i] = CommaResult(s, None, 0.0, f"model error: {why}", "error")
                        continue
                    a = answers.get(n)
                    if a is None:                      # not listed: unchanged
                        results[i] = CommaResult(s, None, 1.0, "", "unchanged")
                        cache.data[CommaCache.key(scope, s.text)] = {"text": s.text, "confidence": 1.0, "reason": ""}
                        continue
                    if "note" in a:
                        results[i] = CommaResult(s, None, 0.0, a["note"], "note")
                        continue
                    if not isinstance(a.get("text"), str):
                        results[i] = CommaResult(s, None, 0.0, "malformed answer", "error")
                        continue
                    new = a["text"].strip()
                    try:
                        conf = float(a.get("confidence", 0))
                    except (TypeError, ValueError):
                        conf = 0.0
                    reason = str(a.get("reason", ""))[:160]
                    if new == s.text:
                        results[i] = CommaResult(s, None, conf, reason, "unchanged")
                        cache.data[CommaCache.key(scope, s.text)] = {"text": s.text, "confidence": conf, "reason": reason}
                    elif not only_commas(s.text, new):
                        results[i] = CommaResult(s, new, conf, reason, "rejected")
                    elif conf < min_confidence:
                        results[i] = CommaResult(s, new, conf, reason, "low")
                    else:
                        results[i] = CommaResult(s, new, conf, reason, "edit")
                        cache.data[CommaCache.key(scope, s.text)] = {"text": new, "confidence": conf, "reason": reason}
        cache.save()
    elif todo:
        for i, s in todo:
            results[i] = CommaResult(s, None, 0.0, "no model configured", "error")
    ordered = [results[i] for i in range(len(sents)) if i in results]
    edits = []
    for r in ordered:
        if r.status == "edit" and r.new_text:
            edits.extend(comma_edits(text, r.sentence, r.new_text))
    return edits, ordered
