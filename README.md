# union-elections-modeling

Machine learning on union election outcomes to try to learn the structure of
what factors play the largest role in success.

See notes about data sources and features in this [Google Doc](https://docs.google.com/document/d/1vmgNmOYvHBSOHspAy8-MRatB-HSxv6SZxCioY6TRNMY/edit?usp=sharing)

The project builds a single DuckDB database from three public sources:

- **NLRB cases** — representation elections, results, filings, and participants
  from <https://labordata.bunkum.us/nlrb>
- **OLMS union finances** — LM-2/LM-3/LM-4 annual reports from
  <https://labordata.bunkum.us/opdr>
- **Union name crosswalk** — precomputed entity resolution from election union
  names to OLMS file numbers from
  <https://labordata.bunkum.us/union_names_crosswalk>

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
| `election_mode` | `ingest_nlrb_elections.py` | ballot/tally details (subset of elections) |
| `filing` | `ingest_nlrb_elections.py` | NLRB cases linked to the ingested elections |
| `participant` | `ingest_nlrb_elections.py` | participants for those cases |
| `lm_data` | `ingest_union_finances.py` | one row per OLMS annual report |
| `ar_disbursements_total` | `ingest_union_finances.py` | disbursement totals per LM-2 report |
| `union_names_crosswalk` | `ingest_union_crosswalk.py` | union name → parent union + OLMS `f_num` |
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

`analytical_elections` contains one row per filing × election-mode record ×
matched finance report. Only `RC` and `RM` filings that are **completed**
(`date_closed` is not null) are included, and for each case only the **most
recent election** is kept (earlier elections for the same case are dropped).
The finance join uses the union name resolved through the crosswalk and keeps a
report only when its coverage period (`pd_covered_from` to `pd_covered_to`)
contains the election date.

It includes:

- Filing, election, and election-mode fields requested for modeling.
- `lm_data` and `ar_disbursements_total` finance fields. The disbursement
  measure is `lm_data.ttl_disbursements`, the total reported on the filing.
  The itemized `ar_disbursements_total` categories are partial and reconcile
  with that total in only a minority of reports, so they are not summed.
- The resolved `election_union_name`, `pred_union_name`, and `pred_f_num`
  diagnostics, so matches can be audited.
- 20 additional feature columns (employer name, region, eligible voters, vote
  counts, union affiliation, total receipts/disbursements, etc.).
- `more_than_one_election`: boolean flag set when the case had more than one
  election in the source data (only the latest is retained in the view).
- `mode_row_number` and `mode_row_count`: a small number of elections still
  have multiple `election_mode` rows; filter with `mode_row_number = 1` if one
  row per election is required.

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
| `modeling/eda.ipynb` | loads `analytical_elections` and profiles grain, missingness, finance coverage, and both targets |
| `modeling/training.py` | loads the view, collapses it to one row per election, and builds the regression and classification feature/target frames |

Run the preparation script:

```bash
uv run python modeling/training.py
```

The notebook needs a Jupyter kernel, which is not a project dependency:

```bash
uv run --with jupyter jupyter lab modeling/eda.ipynb
```

Targets:

- **Regression:** `vote_ratio = votes_for_labor_organization_1 / votes_cast_against_labor_org`.
  Elections with zero votes against have an undefined ratio and are dropped from
  the regression dataset.
- **Classification:** `success = reason_closed == 'Certific. of Representative'`.

## Notes and caveats

- `election_mode` covers only a subset of elections. The view uses left joins,
  so filings without an `election_mode` row are still included.
- `ar_disbursements_total` is effectively LM-2-only. For LM-3/LM-4 unions, use
  `lm_data.ttl_disbursements`.
- The `filing` and `participant` tables are limited to cases linked to the
  ingested elections (2010+ by default), not the full NLRB universe.
- `archive/` contains the earlier nlrb.gov spreadsheet scraper and is not part
  of the current pipeline.
