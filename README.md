# proofix

Fix typos and punctuation in a Markdown file, under audit. Every change is
one bounded, logged, reasoned edit; the model never rewrites prose.

```
proofix draft.md                      # -> draft.proofed.md (US English)
proofix draft.md --target uk --changes
```

Run it after [stylefix](../stylefix), so the grammar checker sees the
target variety, and before [mdmemo](../mdmemo) turns the file into a PDF.

## Four tiers

| tier | what | how | who decides |
|---|---|---|---|
| 1 mechanical | space before a comma or period, missing space after one, doubled punctuation, doubled words, known misspellings | LanguageTool's single-answer rules, the `typos` tool, a small missing-space rule | applied outright |
| 2 adjudicated | unknown words with candidate spellings, grammar rules with candidate fixes ("a comma may be missing after However") | LanguageTool proposes, a local model picks one or KEEP from the sentence and the rule's message | model |
| 3 consistency | the same term written two ways in one document: "third-party" eight times and "third party" once, "Frontier AI" against "frontier AI" | families found deterministically (hyphenated, open and closed compounds; capitalisation of two-word terms, sentence starts excluded); the minority occurrences are put to the model with the compound-modifier rule | model |
| 4 commas | a comma between subject and verb, a missing comma after an introductory phrase or around a non-restrictive clause, a stray comma before "and" joining objects: nothing a rule engine detects | the model returns the sentence with commas corrected; a diff guard admits only insertions and deletions of commas, rejects anything else, and reports it | model, under the guard |

LanguageTool runs twice: once for the mechanical fixes, which are applied,
and again on the cleaned text for the judgement calls, so overlapping
faults on one span ("clear.However" needs a space *and* a comma) are both
fixed. Code spans, fenced code, link targets, URLs, HTML and front matter
are never touched; link text is prose and is checked.

Variety spellings ("programme", "analysed") are left to stylefix. Serial
commas are left to stylefix's style guide. Comma splices are not "fixed"
by changing punctuation; they are reported as notes.

## What goes to the model, and how much

Only the residue after the rule engines:

- Tier 2: one item per LanguageTool suggestion, with the sentence and one
  neighbour as context. Typically 1 to 4 per page.
- Tier 3: only the minority occurrences of an inconsistent term, typically
  a handful per document.
- Tier 4: sentences pass a gate first. A sentence is sent only if it
  contains a comma, opens with an introductory word (After, Although,
  However, In, When ...), contains a clause trigger (which, for example,
  such as, said ...), or is long and joins clauses with a conjunction.
  About two thirds of sentences pass. The model returns only the sentences
  it changed, so replies are short. `--commas all` sends every sentence;
  `--max-sentences N` caps the tier.

Measured on a 20-page document (8,000 words, 390 sentences) against a
DGX Spark serving Qwen3.8-27B, 8 requests in parallel:

| stage | sent to the model | time |
|---|---|---|
| stylefix | 119 words | 43 s |
| LanguageTool, one pass | nothing (local Java) | 4.5 s |
| proofix tier 2 | 77 items | included below |
| proofix tier 3, gated | 260 sentences | 19 s |
| proofix tier 3, all | 390 sentences | 24 s |
| proofix, everything | | 57 s |

Costs scale linearly with length: expect a 100-page document to take
about four minutes in stylefix and five in proofix. Decisions are cached
by sentence, so re-runs after edits pay only for what changed.

## Install

```
brew install languagetool typos-cli      # Java for LanguageTool; typos optional
uv tool install ~/Code/adjudicate
uv tool install .
```

Accept lists: `--accept FILE`, `~/.config/proofix/accept.txt`, and
stylefix's `~/.config/stylefix/accept.txt` and `./.stylefix/accept.txt`
are all honoured for spelling.

## Output

`<name>.proofed.md`, a report next to it, and with `--changes` a log on
stderr: every edit with its line, before and after, context and how it was
decided; the comma corrections with the model's reasons; comma suggestions
the guard rejected, with what the model wanted; notes on faults the model
noticed but is not allowed to fix (comma splices, agreement errors, a
missing closing quote); and LanguageTool matches with no fix.

## Development

```
uv sync
uv run pytest   # LanguageTool is replayed from a fixture; the model is a stub
```
