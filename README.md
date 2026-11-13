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
└── modeling/
     ├── __init__.py
     ├── eda.ipynb
     ├── training.py
     └── training.ipynb
```

The `training.py` file contains the feature engineering pipelines 

## Requirements

- Python 3.13 (pinned in `.python-version`)
- [uv](https://docs.astral.sh/uv/)

Install the environment:

```bash
uv sync
```

## Running the pipeline

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
  option are excluded, along with their filings.
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
- `ar_disbursements_total` is effectively LM-2-only. For LM-3/LM-4 unions, use
  `lm_data.ttl_disbursements`.
- The `filing` and `participant` tables are limited to cases linked to the
  ingested elections (2010+ by default), not the full NLRB universe.
- `archive/` contains the earlier nlrb.gov spreadsheet scraper and is not part
  of the current pipeline.
