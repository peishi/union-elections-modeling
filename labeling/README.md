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
uv run python labeling/llm_pipeline.py --model claude  # start with claude-haiku-5-5
```

Options:

```bash
--db PATH          # database (default data/union_elections.duckdb)
--limit N          # process at most N employers
--overwrite        # re-query and replace already-labeled employers
--one-step         # single call with both catalogs (instead of two-step)
--workers N        # concurrent LLM requests (default 4)
--model NAME       # model to start with (default gemini-4-argon)
--fallback-model   # model if the start model is exhausted (default: the other one)
--base-url URL     # default https://opencode.ai/zen/go/v1
--max-tokens N     # default 1024
-v, --verbose
```

`--model` (and `--fallback-model`) accept the full IDs `gemini-4-argon` and
`claude-haiku-5-5`, or the aliases `gemini` / `gemini-4` and `claude` / `haiku`.
Whichever model you start with, the fallback defaults to the *other* model, e.g.
`--model claude` starts on `claude-haiku-5-5` and falls back to `gemini-4-argon`.
Pass `--fallback-model` explicitly to override that.

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

If the starting model is exhausted (HTTP 429 / quota errors) the client switches
to the fallback model for the rest of the run; the `model` column records which
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
    election_id        BIGINT,   -- representative case used for the label
    voting_unit_id     BIGINT,   -- its bargaining unit
    voting_unit_description VARCHAR,  -- unit text sent to the model
    created_at         TIMESTAMP DEFAULT now(),
    updated_at         TIMESTAMP DEFAULT now()
)
```

`naics_2` and `naics_3` are JSON arrays of up to two candidates, each with the
code, its title, and the model's probability. The range sectors (`31-33`,
`44-45`, `48-49`) are kept as-is. Rows are keyed by `employer_name`, so one row
covers every case of that employer.

The pipeline also records the representative `election_id`, its
`voting_unit_id`, and the `voting_unit_description` that was used as input.
Rows written before these columns existed keep `NULL`; re-run those employers
with `--overwrite` to populate them.

## Relating labels to filings

The transformation step (`transformations/build_analytical_view.py`) connects
labels to the rest of the database by exact `employer_name`:

- `analytical_elections` gains `naics_2`, `naics_3`, and `llm_to_naics` (the
  model that produced the label). Unlabeled employers are `NULL`.
- `employer_industry_labels` attaches a label to every filing for that
  employer, with `case_number`, `date_filed`, `city`, and `state`. It has one
  row per labeled employer and filing, so `employer_name` repeats.

Because both are views, labels added after the view is built show up
automatically. If the views were built before `llm_naics_labels` existed, rerun
`uv run python transformations/build_analytical_view.py` once to add the join.

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
