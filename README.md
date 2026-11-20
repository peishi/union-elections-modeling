# NLRB Elections Modeling

Machine learning on union election outcomes to try to learn the structure of
what factors play the largest role in success.

See additional background in this [Google Doc](https://docs.google.com/document/d/1Lqqyjd6rrg9wdLglbIX5ORfRwJgbsmGgaqrzT2W90w8/edit?usp=sharing)

The data source for this project is the labordata website: <https://labordata.bunkum.us/nlrb>

Specifically, we ingest the following datasets (composed of multiple tables) into a DuckDB instance.
- **NLRB cases** — representation elections, results, filings, and participants
  from <https://labordata.bunkum.us/nlrb>
- **OLMS union finances** — LM-2/LM-3/LM-4 annual reports from
  <https://labordata.bunkum.us/opdr>
- **Union name crosswalk** — precomputed entity resolution from election union
  names to OLMS file numbers from
  <https://labordata.bunkum.us/union_names_crosswalk>

## Project Design
The project proceeds in 3 steps
- **Ingestion**: Of 8 tables from the labordata website, keeping same schema
- **Transformation**: Joining raw data into a single flat SQL view for analysis
- **Modeling**: With feature engineering and initial model evaluation

Folder structure
```
union-elections-modeling/
├── ingestion/
|    ├── common.py
|    ├── ingest_nlrb_elections.py
|    ├── ingest_union_crosswalk.py
|    └── ingest_union_finances.py
├── transformations/
|    └── build_analytical_view.py
├── labeling/
|    ├── app.py
|    ├── llm.py
|    ├── llm_pipeline.py
|    ├── sampler.py
|    ├── store.py
|    ├── naics_subsectors.json
|    └── static/
|         ├── index.html
|         └── naics_sectors.json
└── modeling/
     ├── __init__.py
     ├── eda.ipynb
     ├── training.py
     └── training.ipynb
```

The `training.py` file contains the feature engineering pipelines 

## Industry labeling

The `labeling/` folder assigns 2022 NAICS codes to employers in
`analytical_elections` for training an industry classifier. Labels are stored
in `llm_naics_labels`, with each employer's top-2 two-digit sectors and top-2
three-digit subsectors (plus probabilities, the query text, and token usage).

- `labeling/llm_pipeline.py` batch-classifies every unique employer with an LLM
  and writes incrementally (resumable, token totals printed at the end):

```bash
uv run python labeling/llm_pipeline.py --limit 20   # test run
uv run python labeling/llm_pipeline.py              # all employers
```

- `labeling/app.py` is a local web UI for reviewing/correcting labels:

```bash
uv run python labeling/app.py
# open http://127.0.0.1:8000/
```

See `labeling/README.md` for details.

## Requirements

- Python 3.13 (pinned in `.python-version`)
- [uv](https://docs.astral.sh/uv/)

Install the environment:

```bash
uv sync
```

## Running just the final random forest model as a Docker container

The model pipeline (DictVectorizer -> RandomForestClassifier) is pickled after `modeling/training.py` and loaded by `modeling/predict.py`
to run "predictions" on any new data. NOTE: since the model is intended to be an explanatory model and not a predictive one,
it includes "turnout" as a feature variable even though turnout would be known only after election. 

Though one could imagine using the model to predict what the outcome probability is for different levels of turnout before an election
to get a sense of what level might be necessary to predict an election success.

For model training, you first have to ingest all of the data and transform it. See sections below for what is happening in
each of those steps. In commands below, just listing what you should run to get to a final model deployment.

```bash
# Ingest all the data
uv run python ingestion/ingest_nlrb_elections.py
uv run python ingestion/ingest_union_crosswalk.py
uv run python ingestion/ingest_union_crosswalk.py

# Transform data to a single view
uv run python transformations/build_analytical_view.py

# Train the model
uv run python modeling/training.py

# Build Docker image and run it
docker build -t nlrb_elections_model .
docker run -it --rm -p 8080:8080 nlrb_elections_model:latest
```

## Data Ingestion and Transformation

Run the steps in this order. Every ingestion script is idempotent and replaces
only the tables it owns, so reruns are safe.

```bash
# 1. NLRB elections, filings, and participants (2010 onward by default)
uv run python ingestion/ingest_nlrb_elections.py

# 2. OLMS union finance data (2010 onward by default)
uv run python ingestion/ingest_union_finances.py

# 3. Union-name crosswalk
uv run python ingestion/ingest_union_crosswalk.py

# 4. Build the analytical view
uv run python transformations/build_analytical_view.py
```

All output goes to `data/union_elections.duckdb`.

## What gets created

| Object | Script | Rows / description |
|---|---|---|
| `election` | `ingest_nlrb_elections.py` | one row per election |
| `election_result` | `ingest_nlrb_elections.py` | one row per election result |
| `election_mode` | `ingest_nlrb_elections.py` | additional detail on elections from FOIA |
| `tally` | `ingest_nlrb_elections.py` | per-option vote counts, keyed by `election_id` |
| `voting_unit` | `ingest_nlrb_elections.py` | bargaining-unit descriptions, keyed by `case_number` (→ `filing`) |
| `filing` | `ingest_nlrb_elections.py` | NLRB cases linked to the ingested elections |
| `participant` | `ingest_nlrb_elections.py` | participants for those cases |
| `lm_data` | `ingest_union_finances.py` | one row per OLMS annual report |
| `ar_disbursements_total` | `ingest_union_finances.py` | disbursement breakdowns by category per LM-2 report  |
| `union_names_crosswalk` | `ingest_union_crosswalk.py` | union name → OLMS report name + parent union + `f_num` |
| `analytical_elections` | `build_analytical_view.py` | view joining elections to matched union finances |

## Common options

Every ingestion script supports:

```bash
--db PATH          # target DuckDB file (default: data/union_elections.duckdb)
--workdir PATH     # keep downloaded CSVs instead of using a temp directory
--no-verify        # skip source row-count verification
--timeout SECONDS
--retries N
-v, --verbose
```

NLRB-specific:

```bash
--since 2010-01-01   # election-date cutoff (default)
--since all          # ingest every election in the source table
```

Finance-specific:

```bash
--since-year 2015    # earliest yr_covered to ingest (default: 2010)
```

Transformation-specific:

```bash
--db PATH            # database containing the source tables
--view NAME          # name of the view to create (default: analytical_elections)
```

## Analytical view

`analytical_elections` contains one row per completed `RC`/`RM` case and its
most recent election, plus the matched finance report. Only filings with a
`date_closed` are included, and earlier elections for the same case are dropped.
The union name used for finance matching comes from the primary union
`participant`, and a report is kept only when its coverage
period (`pd_covered_from` to `pd_covered_to`) contains the election date.

It includes:

- Filing and election fields relevant to modeling.
- **Vote totals from `tally`:** `votes_for_union` (the union option) and
  `votes_against` (the `'No union'` option). Elections whose tally has more
  than two options (contested multi-union ballots) or only a `'No union'`
  option are excluded, along with their filings. Elections whose tally total
  exceeds the election's `unit_size` (more votes than eligible voters) are
  also excluded — this is the rule that flagged the multi-unit cases where
  separate unit results appear to have been counted together or duplicated in
  the source. In the current 2010+ snapshot it removes 112 filings.
- **`voting_unit_description`:** the bargaining-unit description text for the
  case, taken from `voting_unit`. A filing usually has a single unit; for the
  ~1% of filings with several units (multiple locations or crafts of one
  employer) the distinct descriptions are concatenated, ordered by `unit_id`.
  This is intended as text input for industry classification.
- `lm_data` and `ar_disbursements_total` finance fields. The disbursement
  measure is `lm_data.ttl_disbursements`, the total reported on the filing.
  The itemized `ar_disbursements_total` categories are partial and reconcile
  with that total in only a minority of reports, so they are not summed.
- The resolved `election_union_name`, `pred_union_name`, and `pred_f_num`
  diagnostics, so matches can be audited.
- Additional feature columns to be tested in the future.
- `more_than_one_election`: boolean flag set when the case had more than one
  election in the source data (only the latest is retained in the view).

Example query:

```bash
uv run python - <<'PY'
import duckdb

con = duckdb.connect("data/union_elections.duckdb", read_only=True)
print(
    con.execute(
        """
        SELECT case_type, count(*) AS n,
               count(*) FILTER (WHERE rpt_id IS NOT NULL) AS with_finance
        FROM analytical_elections
        GROUP BY 1
        ORDER BY 2 DESC
        """
    ).fetchall()
)
PY
```

## Modeling

The `modeling/` folder contains the analysis scaffold:

| File | Purpose |
|---|---|
| `modeling/eda.ipynb` | Loads `analytical_elections` and profiles target variable, missingness, union finances, and election features. |
| `modeling/training.py` | Loads the view, calculates the regression and classification feature/target variables. Eventually will contain finished training pipeline. |
| `modeling/training.ipynb` | Tests various machine learning models to determine which model is best.  | 

Run the preparation script:

```bash
uv run python modeling/training.py
```

Run the exploratory data analysis:

```bash
uv run --with jupyter jupyter lab modeling/eda.ipynb
```

## Notes and caveats

- `election_mode` is still ingested but is not used by `analytical_elections`.
- `voting_unit` is not 1:1 with `filing`: a filing can have several bargaining
  units (multiple locations or crafts). `analytical_elections` keeps one row
  per filing, concatenates the units' descriptions, and excludes elections
  whose tally is impossible (votes > `unit_size`).
- `ar_disbursements_total` is effectively LM-2-only. For LM-3/LM-4 unions, use
  `lm_data.ttl_disbursements`.
- The `filing` and `participant` tables are limited to cases linked to the
  ingested elections (2010+ by default), not the full NLRB universe.
- `archive/` contains the earlier nlrb.gov spreadsheet scraper and is not part
  of the current pipeline.
