"""Change log (stderr) and Markdown report."""

from __future__ import annotations

from adjudicate import Decision, Edit

from .commas import CommaResult
from .lt import Match


def _ctx(text: str, e: Edit, width: int = 28) -> str:
    lines = text.split("\n")
    line = lines[e.line - 1] if e.line - 1 < len(lines) else ""
    s, en = e.start - 1, e.end
    return f"{'…' if s > width else ''}{line[max(0, s - width):s]}[{line[s:en]}]{line[en:en + width]}{'…' if len(line) > en + width else ''}"


def changelog(text: str, applied: list[Edit], decisions: list[Decision],
              commas: list[CommaResult], review: list[Match], typos_fixed: int) -> str:
    by_pos = {(d.item.line, d.item.start): d for d in decisions}
    out = [f"proofix changes: {len(applied)} edit(s)" + (f", {typos_fixed} by typos" if typos_fixed else "")]
    for e in sorted(applied, key=lambda e: (e.line, e.start)):
        d = by_pos.get((e.line, e.start))
        how = e.source
        if d:
            how = f"{d.item.source}, model {d.confidence:.2f}: {d.reason}"
        elif e.source == "commas":
            how = "commas (see below)"
        out.append(f"  L{e.line:<4} {e.original!r} -> {e.replacement!r:16} {how}")
        out.append(f"        {_ctx(text, e)}")
    kept = [d for d in decisions if d.choice == "KEEP"]
    if kept:
        out.append(f"kept after adjudication: {len(kept)}")
        for d in kept:
            out.append(f"  L{d.item.line:<4} {d.item.word:<20} {d.confidence:.2f}: {d.reason}")
    undecided = [d for d in decisions if d.choice is None]
    if undecided:
        out.append(f"undecided (left as is): {len(undecided)}")
        for d in undecided:
            out.append(f"  L{d.item.line:<4} {d.item.word:<20} {d.reason}")
    changed = [r for r in commas if r.status == "edit"]
    if changed:
        out.append(f"commas corrected: {len(changed)}")
        for r in changed:
            out.append(f"  {r.confidence:.2f} {r.reason}")
            out.append(f"     {r.new_text}")
    held = [r for r in commas if r.status in ("rejected", "low")]
    if held:
        out.append(f"comma suggestions not applied: {len(held)}")
        for r in held:
            why = "non-comma change" if r.status == "rejected" else f"confidence {r.confidence:.2f}"
            out.append(f"  {why}: {r.reason}")
            out.append(f"     was: {r.sentence.text}")
            out.append(f"     got: {r.new_text}")
    notes = [r for r in commas if r.status == "note"]
    if notes:
        out.append(f"comma notes (not changed): {len(notes)}")
        for r in notes:
            out.append(f"  {r.reason}")
            out.append(f"     {r.sentence.text}")
    errors = [r for r in commas if r.status == "error"]
    if errors:
        out.append(f"comma check skipped: {len(errors)} sentence(s): {errors[0].reason}")
    if review:
        out.append(f"for review (not changed): {len(review)}")
        for m in review:
            out.append(f"  {m.rule}: {m.message[:90]}")
    return "\n".join(out) + "\n"
