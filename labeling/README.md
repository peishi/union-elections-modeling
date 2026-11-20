# NAICS labeling

Tools for assigning 2022 NAICS codes to employers in `analytical_elections`,
for training a model that predicts an employer's industry. Labels live in the
`llm_naics_labels` table in the project DuckDB.

Two entry points:

- **`llm_pipeline.py`** — batch-classifies every unique employer with the LLM
  and writes 2-digit and 3-digit codes (the primary path).
- **`app.py`** — a small local web app for reviewing/correcting labels.

Both require `analytical_elections` to exist (run
`transformations/build_analytical_view.py` first).

## Batch pipeline

```bash
uv run python labeling/llm_pipeline.py --limit 20     # small test run
uv run python labeling/llm_pipeline.py                # every unique employer
uv run python labeling/llm_pipeline.py --overwrite     # re-query existing rows
```

Options:

```bash
--db PATH          # database (default data/union_elections.duckdb)
--limit N          # process at most N employers
--overwrite        # re-query and replace already-labeled employers
--one-step         # single call with both catalogs (instead of two-step)
--workers N        # concurrent LLM requests (default 4)
--model NAME       # primary model (default gemini-4-argon)
--fallback-model   # used when the primary is exhausted (default claude-haiku-5-5)
--base-url URL     # default https://opencode.ai/zen/go/v1
--max-tokens N     # default 1024
-v, --verbose
```

For each unique employer it sends the employer name, filing year, city, state,
and bargaining-unit description to the LLM and asks for the **two most likely
2-digit sectors** and the **two most likely 3-digit subsectors**, each with a
probability. This is done in **two steps** by default (2-digit first, then only
the subsectors under the chosen sectors), which cuts token use roughly in half
versus the `--one-step` single call. The static catalogs come first in each
prompt so their prefix can be cached (this benefits the `claude-haiku-5-5`
fallback; `gemini-4-argon` does not cache). Results are written incrementally
(upserted by `employer_name`), so the run is resumable — by default
already-labeled employers are skipped.

If the primary model is exhausted (HTTP 429 / quota errors) the client switches
to `claude-haiku-5-5` for the rest of the run; the `model` column records which
model produced each row.

At the end it prints the total token consumption (prompt / completion / total,
broken down by model).

## Interactive review app

```bash
uv run python labeling/app.py
# then open http://127.0.0.1:8000/
```

Options: `--db PATH`, `--host` (default `127.0.0.1`), `--port` (default 8000),
`-v`.

- Shows one card per **unlabeled employer** (one representative case each).
- Each card shows the employer name, case number, filing year, city/state, an
  NLRB case link, the union name, and the bargaining-unit description.
- Pick a 2-digit sector (industry name shown) or type a finer code. Selecting a
  sector shows example firms to confirm the choice.
- **Suggest** asks the LLM for the top 3 sectors with probabilities; clicking a
  suggestion fills the dropdown. An expandable "Show raw model response" reveals
  the full reply.
- **Save & next** writes manual labels (`model='manual'`) and loads a fresh
  batch.

## Labels table

```sql
llm_naics_labels (
    employer_name      VARCHAR PRIMARY KEY,
    naics_2            VARCHAR,   -- JSON: top-2 [{code,title,probability}, ...]
    naics_3            VARCHAR,   -- JSON: top-2 subsectors
    rationale          VARCHAR,
    query_text         VARCHAR,
    raw_response       VARCHAR,
    model              VARCHAR,
    prompt_tokens      BIGINT,
    completion_tokens  BIGINT,
    total_tokens       BIGINT,
    created_at         TIMESTAMP DEFAULT now(),
    updated_at         TIMESTAMP DEFAULT now()
)
```

`naics_2` and `naics_3` are JSON arrays of up to two candidates, each with the
code, its title, and the model's probability. The range sectors (`31-33`,
`44-45`, `48-49`) are kept as-is. Rows are keyed by `employer_name`, so one row
covers every case of that employer.

## Models

Both entry points call OpenCode Go's OpenAI-compatible endpoint
(`POST https://opencode.ai/zen/go/v1/chat/completions`) using `OPENCODE_GO_API`
from `.env` or the environment. The API requires an `x-opencode-session` header
and a browser-like `User-Agent`, which the client sets. Primary model is
`gemini-4-argon`; the pipeline falls back to `claude-haiku-5-5` on exhaustion.

## NAICS data

- `static/naics_sectors.json` — the 20 two-digit sectors, with 5–10 example
  firms per sector as a confirmation aid.
- `naics_subsectors.json` — the 96 three-digit subsectors (2022), each tagged
  with its parent sector.

Titles come from the U.S. Census Bureau 2022 NAICS manual. Example firm
attributions were checked against public NAICS references; conglomerates can
span several sectors, so they are illustrative.
