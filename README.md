# proofix

Fix typos and punctuation in a Markdown file, under audit. Every change is
one bounded, logged, reasoned edit; the model never rewrites prose.

```
proofix draft.md                      # -> draft.proofed.md (US English)
proofix draft.md --target uk --changes
```

Run it after [stylefix](https://github.com/AndrewBroz/stylefix), so the grammar checker sees the target variety.

## Five tiers

| tier | what | how | who decides |
|---|---|---|---|
| 0 hygiene | text that renders fine but is not what it looks like: HTML entities Markdown can carry plainly (`&#32;`, `&quot;`, `&mdash;`), zero-width spaces, byte-order marks, soft hyphens, control characters, Windows line endings, decomposed accents | fixed deterministically and logged; entities that would change Markdown's parsing (`&lt;`, `&gt;`, `&amp;`) or carry meaning (`&nbsp;`) are left alone and reported | applied outright, or noted |
| 1 mechanical | space before a comma or period, missing space after one, doubled punctuation, doubled words whose repeat is never grammatical (`the the`, `of of`), known misspellings | LanguageTool's single-answer rules, the `typos` tool, a small missing-space rule; the built-in `basics` check (doubled words always; spacing without LanguageTool) | applied outright only where right in every context (audited rule by rule): a typos fix only when the word is all lower case, has one correction and is not part of a word with a digit or underscore; `..` (`.` or `…`, or `cd ..`), `Node.Js`, `.bashrc`, a paragraph opening `, and` go to the model with KEEP, or are listed for review without one; indentation, table alignment and Markdown hard line breaks are left alone. Other doubles (`that that`, `had had`, `very very`) go to the model with KEEP, or are listed for review |
| 2 adjudicated | unknown words with candidate spellings, grammar rules with candidate fixes ("a comma may be missing after However") | LanguageTool proposes, a language model picks one or KEEP from the sentence and the rule's message | model |
| 3 compounds and consistency | how a compound is written: "cyber attack" or "cyberattack", "non-profit" or "nonprofit", "long term" or "long-term"; and terms the document capitalises two ways | candidates found from the dictionaries (an open or hyphenated pair whose closed form is a word; a non-word that splits into two words; anything the document itself writes two ways). The model chooses closed, hyphenated, open or KEEP, told which dictionary rules (Merriam-Webster for US, Oxford for UK) and the rule that a compound modifier before a noun is hyphenated while a noun phrase is open | model |
| 4 commas | a comma between subject and verb, a missing comma after an introductory phrase or around a non-restrictive clause, a stray comma before "and" joining objects: nothing a rule engine detects | the model returns the sentence with commas corrected; a diff guard admits only insertions and deletions of commas, rejects anything else, and reports it | model, under the guard |

LanguageTool runs twice: once for the mechanical fixes, which are applied,
and again on the cleaned text for the judgement calls, so overlapping
faults on one span ("clear.However" needs a space *and* a comma) are both
fixed. Code spans, fenced code, link targets, URLs, HTML and front matter
are never touched; link text is prose and is checked.

Variety spellings ("programme", "analysed") are left to stylefix. Serial
commas are left to stylefix's style guide. Compound spelling follows the
target variety's dictionary, because that is what every style guide
defers to; the guide only adds AP's lighter hyphenation. Comma splices are not "fixed"
by changing punctuation; they are reported as notes.

## What goes to the model, and how much

Only the residue after the rule engines:

- Tier 2: one item per LanguageTool suggestion, with the sentence and one
  neighbour as context. Typically 1 to 4 per page.
- Tier 3: one item per occurrence of a compound that has a question: a
  closed form exists, or the document writes it two ways. A document that
  is consistent and closes what the dictionary closes sends nothing.
- Tier 4: sentences pass a gate first. A sentence is sent only if it
  contains a comma, opens with an introductory word (After, Although,
  However, In, When ...), contains a clause trigger (which, for example,
  such as, said ...), or is long and joins clauses with a conjunction.
  About two thirds of sentences pass. The model returns only the sentences
  it changed, so replies are short. `--commas all` sends every sentence;
  `--max-sentences N` caps the tier.

Measured on a 20-page document (8,000 words, 390 sentences). One measurement, on a DGX Spark serving Qwen3.8-27B, 8 requests in parallel:

| stage | sent to the model | time |
|---|---|---|
| stylefix | 119 words | 43 s |
| LanguageTool, one pass | nothing (local Java) | 4.5 s |
| proofix tier 2 | 77 items | included below |
| proofix tier 4, gated | 260 sentences | 19 s |
| proofix tier 4, all | 390 sentences | 24 s |
| proofix, everything | | 57 s |

Costs scale linearly with length: expect a 100-page document to take
about four minutes in stylefix and five in proofix. Decisions are cached
by sentence, so re-runs after edits pay only for what changed.

## Install

```
uv tool install proofix       # or: pipx install proofix
```

Optional, and worth having:

- [LanguageTool](https://languagetool.org) on your PATH as `languagetool`
  (`brew install languagetool`; it needs Java). Without it, LanguageTool's
  tier 1 fixes and all of tier 2 are skipped with a warning; hygiene, the
  `typos` fixes, compounds and commas still run.
- [typos](https://github.com/crate-ci/typos) (`brew install typos-cli` or
  `cargo install typos-cli`) for known-misspelling fixes in tier 1.

Run `proofix --setup` to choose and test a model endpoint, and
`proofix --check-endpoint` to see which one proofix will use. The endpoint
is configured as for stylefix: `--url`/`--model`,
`PROOFIX_LLM_URL`/`_MODEL`/`_KEY`, `~/.config/proofix/config.toml`, then
the shared `ADJUDICATE_LLM_*` and `~/.config/adjudicate/config.toml`; with
nothing configured, a local Ollama if one is running. A `--url` or
`PROOFIX_LLM_URL` drops keys set in lower layers; set `PROOFIX_LLM_KEY` (or
`api_key_env` in the same config file) alongside it. Without a model, tiers
0 and 1 still apply; tier 2's fixes are listed for review, tier 3 is skipped
and sentences are not checked for commas. See
[adjudicate](https://github.com/AndrewBroz/adjudicate#choosing-the-model-endpoint)
for the config table. `STYLEFIX_LLM_*` no longer configures proofix; use the
shared file or `ADJUDICATE_LLM_*` to set both tools at once.

Accept lists: `--accept FILE`, `~/.config/proofix/accept.txt`,
`./.proofix/accept.txt`, and stylefix's `~/.config/stylefix/accept.txt` and
`./.stylefix/accept.txt` are all honoured for spelling.

## Output

`<name>.proofed.md`, a report next to it, and with `--changes` a log on
stderr: every edit with its line, before and after, context and how it was
decided; the comma corrections with the model's reasons; comma suggestions
the guard rejected, with what the model wanted; notes on faults the model
noticed but is not allowed to fix (comma splices, agreement errors, a
missing closing quote); and LanguageTool matches with no fix.

## License

MIT (see `LICENSE`).

## Development

```
uv sync
uv run pytest   # LanguageTool is replayed from a fixture; the model is a stub
```
