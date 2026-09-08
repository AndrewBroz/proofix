"""End to end with LanguageTool replayed from a fixture and a stub model."""

import json
import shutil
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from proofix import cli, lt
from proofix.lt import parse_matches

FIX = Path(__file__).parent / "fixtures" / "typos.lt.json"
EXAMPLE = Path(__file__).parent.parent / "examples" / "typos.md"


class StubModel(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        user = body["messages"][1]["content"]
        system = body["messages"][0]["content"]
        if "comma placement" in system:
            changed, notes = [], []
            for line in user.splitlines():
                n, _, s = line.partition(". ")
                fixed = s.replace("The committee, decided", "The committee decided") \
                         .replace("published in May was", "published in May, was")
                if "It was late, we" in s:
                    fixed = s.replace(", we", "; we")          # must be rejected by the guard
                if "she said" in s:
                    notes.append({"id": int(n), "reason": "stub note"})
                elif fixed != s:
                    changed.append({"id": int(n), "text": fixed, "confidence": 0.95, "reason": "stub"})
            content = json.dumps({"changed": changed, "notes": notes})
        else:
            decisions = []
            for line in user.splitlines():
                if line.startswith("Item "):
                    num = int(line.split()[1])
                    options = [o.split("] ", 1)[-1] for o in line.split("options: ")[1].split(" | ")]
                    word = line.split('word: "')[1].split('"')[0]
                    choice = options[0] if word[0].islower() or word == "Teh" else "KEEP"
                    if word in ("third-party", "long-term"):
                        choice = "KEEP"                            # already right as modifiers
                    if "However" in line:
                        choice = "KEEP"
                    decisions.append({"id": num, "choice": choice, "confidence": 0.9, "reason": "stub"})
            content = json.dumps({"decisions": decisions})
        resp = json.dumps({"choices": [{"message": {"content": content}}]}).encode()
        self.send_response(200); self.send_header("Content-Type", "application/json")
        self.end_headers(); self.wfile.write(resp)

    def do_GET(self):
        self.send_response(200); self.end_headers(); self.wfile.write(b"{}")

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def stub_url():
    srv = HTTPServer(("127.0.0.1", 0), StubModel)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}/v1"
    srv.shutdown()


@pytest.fixture
def replay_lt(monkeypatch):
    """First call replays the fixture; the second (on cleaned text) replays only
    the matches whose context still exists, as parse_matches's search does."""
    fixture = FIX.read_text()
    monkeypatch.setattr(cli, "find_languagetool", lambda: "/replay/languagetool")
    monkeypatch.setattr(cli, "run_languagetool", lambda text, target, level="default": parse_matches(fixture, text))


def test_end_to_end(tmp_path, replay_lt, stub_url, capsys):
    src = tmp_path / "doc.md"
    shutil.copy(EXAMPLE, src)
    out = tmp_path / "out.md"
    rc = cli.main([str(src), "-o", str(out), "--url", stub_url, "--model", "stub",
                   "--no-typos", "--no-cache", "--changes", "--target", "us"])
    assert rc == 0
    text = out.read_text()
    err = capsys.readouterr().err
    # tier 1: mechanical
    assert "the the" not in text and "report ," not in text and "said ." not in text and "teams.." not in text
    # tier 2: spelling via the stub
    assert "The committee received" in text and "definitely clear" in text and "three separate teams" in text
    # tier 3: commas under the guard
    assert "The committee decided to postpone" in text
    assert "The report, which was published in May, was ignored" in text
    assert "It was late, we went home anyway" in text          # semicolon rewrite rejected
    assert "non-comma change" in err and "stub note" in err
    # consistency tier
    assert "Every third-party risk assessor" in text
    # hygiene
    assert "correct **unsanctioned actions** — see &lt;config&gt;." in text
    assert "text hygiene" in err and "&lt;" in err
    # protection
    assert "`recieved`" in text and "[the link](http://x.y/teh)" in text   # text fixed, target kept
    assert "issues; the budget" in text
    assert (tmp_path / "out.md.report.md").exists()


def test_no_model_lists_items(tmp_path, replay_lt, capsys):
    src = tmp_path / "doc.md"
    shutil.copy(EXAMPLE, src)
    out = tmp_path / "out.md"
    rc = cli.main([str(src), "-o", str(out), "--llm", "none", "--no-typos", "--no-cache",
                   "--changes", "--no-report"])
    assert rc == 0
    text = out.read_text()
    err = capsys.readouterr().err
    assert "the the" not in text and "recieved" in text
    assert "for review" in err and "MORFOLOGIK_RULE_EN_US" in err
