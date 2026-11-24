#!/usr/bin/env python3
"""Build the analytical election/finance view in DuckDB.

The view combines:

* ``filing``            one row per NLRB case (completed cases only)
* ``election``          the most recent election for each case
* ``tally``             per-option vote counts (union vs "No union")
* ``voting_unit``       bargaining-unit descriptions for the case
* ``participant``       the primary union participant for the case
* ``union_names_crosswalk``
                        election union name -> canonical parent + OLMS f_num
* ``lm_data``           OLMS annual reports for the matched file number
* ``ar_disbursements_total``
                        itemized disbursement totals for LM-2 reports

The finance rows are joined only when the report's coverage period
(``pd_covered_from`` .. ``pd_covered_to``) contains the election date.

Only ``RM`` and ``RC`` filings with a ``date_closed`` (i.e. completed cases)
are included, and only the most recent election per case is kept.

Vote totals come from ``tally`` (``votes_for_union`` / ``votes_against``).
Elections whose tally has more than two options (a contested multi-union
ballot), only a "No union" option, or an impossible tally (the sum of the
option votes exceeds the election's ``unit_size``) are excluded, along with
their filings.  The ``unit_size`` check drops cases where separate unit
results appear to have been counted together / duplicated in the source.

``voting_unit_description`` concatenates the distinct, non-empty
``voting_unit.description`` values for the case (ordered by ``unit_id``).  A
filing usually has a single unit; multi-unit cases (multiple locations or
crafts of one employer) contribute every unit's description.

The union name used for finance matching comes from the primary union
``participant``; ``election_mode`` is not used by this view.

The view is created in the same DuckDB file as the source tables.

Usage
-----
    uv run python transformations/build_analytical_view.py
    uv run python transformations/build_analytical_view.py --view analytical_elections
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Sequence

import duckdb

LOG = logging.getLogger("build-table")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "data" / "union_elections.duckdb"
DEFAULT_VIEW = "analytical_elections"

REQUIRED_TABLES = {
    "filing": "ingestion/ingest_nlrb_elections.py",
    "election": "ingestion/ingest_nlrb_elections.py",
    "tally": "ingestion/ingest_nlrb_elections.py",
    "voting_unit": "ingestion/ingest_nlrb_elections.py",
    "participant": "ingestion/ingest_nlrb_elections.py",
    "lm_data": "ingestion/ingest_union_finances.py",
    "ar_disbursements_total": "ingestion/ingest_union_finances.py",
    "union_names_crosswalk": "ingestion/ingest_union_crosswalk.py",
}


def view_sql(view_name: str, has_labels: bool = False) -> str:
    if has_labels:
        label_columns = (
            '    lab."naics_2" AS naics_2,\n'
            '    lab."naics_3" AS naics_3,\n'
            '    lab."model" AS llm_to_naics\n'
        )
        label_join = (
            'LEFT JOIN "llm_naics_labels" lab\n'
            '    ON lab."employer_name" = b."employer_name"\n'
        )
    else:
        label_columns = (
            '    CAST(NULL AS VARCHAR) AS naics_2,\n'
            '    CAST(NULL AS VARCHAR) AS naics_3,\n'
            '    CAST(NULL AS VARCHAR) AS llm_to_naics\n'
        )
        label_join = ""
    return f"""
CREATE OR REPLACE VIEW "{view_name}" AS
WITH crosswalk AS (
    SELECT
        upper(trim("union_name")) AS name_key,
        "pred_is_union",
        "pred_is_union_score",
        "pred_union_name",
        "pred_union_name_score",
        "pred_f_num",
        "pred_f_num_score"
    FROM "union_names_crosswalk"
    WHERE "union_name" IS NOT NULL
    QUALIFY row_number() OVER (
        PARTITION BY upper(trim("union_name"))
        ORDER BY
            "pred_f_num_score" DESC NULLS LAST,
            "pred_union_name_score" DESC NULLS LAST,
            "pred_is_union_score" DESC NULLS LAST
    ) = 1
),
primary_union AS (
    SELECT
        "case_number",
        "participant",
        "type",
        "subtype"
    FROM (
        SELECT
            "case_number",
            "participant",
            "type",
            "subtype",
            row_number() OVER (
                PARTITION BY "case_number"
                ORDER BY
                    CASE
                        WHEN "type" = 'Petitioner' THEN 1
                        WHEN "type" = 'Intervenor' THEN 2
                        WHEN "type" = 'Involved Party' THEN 3
                        WHEN "type" = 'Union' THEN 4
                        WHEN "type" = 'Union Involved' THEN 5
                        WHEN "type" = 'Charged Party / Respondent' THEN 6
                        WHEN "type" = 'Charging Party' THEN 7
                        ELSE 8
                    END,
                    "participant"
            ) AS rn
        FROM "participant"
        WHERE "subtype" = 'Union'
          AND "participant" IS NOT NULL
          AND upper(trim("participant")) <> 'N/A'
    ) ranked
    WHERE rn = 1
),
election_latest AS (
    SELECT
        *,
        row_number() OVER (
            PARTITION BY "case_number"
            ORDER BY "date" DESC NULLS LAST, "election_id" DESC
        ) AS election_rank,
        count(*) OVER (PARTITION BY "case_number") AS n_elections
    FROM "election"
),
tally_votes AS (
    SELECT
        "election_id",
        sum("votes") FILTER (WHERE "option" <> 'No union') AS votes_for_union,
        max("votes") FILTER (WHERE "option" = 'No union') AS votes_against
    FROM "tally"
    GROUP BY "election_id"
),
voting_unit_desc AS (
    SELECT
        "case_number",
        string_agg(
            "description", ' || ' ORDER BY "unit_id", "voting_unit_id"
        ) AS voting_unit_description
    FROM (
        SELECT
            "case_number",
            "unit_id",
            "voting_unit_id",
            "description",
            row_number() OVER (
                PARTITION BY "case_number", "description"
                ORDER BY "unit_id", "voting_unit_id"
            ) AS rn
        FROM "voting_unit"
        WHERE "description" IS NOT NULL
          AND trim("description") <> ''
    ) dedup
    WHERE rn = 1
    GROUP BY "case_number"
),
base AS (
    SELECT
        f."case_number",
        f."case_type",
        f."city",
        f."state",
        f."date_filed",
        f."date_closed",
        f."reason_closed",
        e."election_id",
        e."tally_type",
        e."unit_size",
        e."date" AS election_date,
        tv."votes_for_union",
        tv."votes_against",

        -- Feature columns from filing / election.
        f."name" AS employer_name,
        f."region_assigned",
        f."status" AS filing_status,
        f."number_of_eligible_voters" AS filing_number_of_eligible_voters,
        f."number_of_voters_on_petition_or_charge",
        f."certified_representative",
        e."ballot_type",
        e."voting_unit_id",
        vud."voting_unit_description",

        -- Case-level flags.
        COALESCE(e."n_elections", 0) > 1 AS more_than_one_election,

        -- Join keys and match diagnostics.
        pu."participant" AS election_union_name,
        pu."participant" AS union_participant,
        pu."type" AS union_participant_type
    FROM "filing" f
    LEFT JOIN election_latest e
        ON e."case_number" = f."case_number"
       AND e."election_rank" = 1
    LEFT JOIN tally_votes tv
        ON tv."election_id" = e."election_id"
    LEFT JOIN "voting_unit_desc" vud
        ON vud."case_number" = f."case_number"
    LEFT JOIN "primary_union" pu
        ON pu."case_number" = f."case_number"
    WHERE f."case_type" IN ('RM', 'RC')
      AND f."date_closed" IS NOT NULL
      AND e."election_id" NOT IN (
          SELECT t."election_id"
          FROM "tally" t
          JOIN "election" el ON el."election_id" = t."election_id"
          GROUP BY t."election_id", el."unit_size"
          HAVING count(*) > 2
              OR (count(*) = 1 AND max(t."option") = 'No union')
              OR (
                  el."unit_size" IS NOT NULL
                  AND sum(t."votes") > el."unit_size"
              )
      )
)
SELECT
    b."case_number",
    b."case_type",
    b."city",
    b."state",
    b."date_filed",
    b."date_closed",
    b."reason_closed",
    b."election_id",
    b."tally_type",
    b."unit_size",
    b."election_date",
    b."more_than_one_election",
    b."votes_for_union",
    b."votes_against",

    lm."rpt_id",
    lm."yr_covered",
    lm."pd_covered_from",
    lm."pd_covered_to",
    lm."f_num",
    lm."desig_num",
    lm."terminate",
    ad."representational",
    ad."political",
    ad."contributions",
    ad."general_overhead",
    ad."union_administration",

    -- Additional feature columns.
    b."employer_name",
    b."region_assigned",
    b."filing_status",
    b."filing_number_of_eligible_voters",
    b."number_of_voters_on_petition_or_charge",
    b."certified_representative",
    b."ballot_type",
    b."voting_unit_id",
    b."voting_unit_description",
    lm."union_name" AS finance_union_name,
    lm."aff_abbr" AS finance_aff_abbr,
    lm."members",
    lm."form_type",
    lm."ttl_receipts",
    lm."ttl_disbursements",

    -- Entity-resolution diagnostics.
    b."election_union_name",
    b."union_participant",
    b."union_participant_type",
    cw."pred_union_name",
    cw."pred_f_num",
    cw."pred_f_num_score",
    cw."pred_is_union",
    cw."pred_union_name_score",
    cw."pred_is_union_score",

    -- Industry labels (NULL until the employer has been labeled).
    {label_columns}FROM base b

LEFT JOIN crosswalk cw
    ON cw.name_key = upper(trim(b."election_union_name"))
{label_join}
LEFT JOIN "lm_data" lm
    ON lm."f_num" = cw."pred_f_num"
   AND b."election_date" IS NOT NULL
   AND b."election_date" >= lm."pd_covered_from"
   AND b."election_date" <= lm."pd_covered_to"
LEFT JOIN "ar_disbursements_total" ad
    ON ad."rpt_id" = lm."rpt_id"
"""


INDUSTRY_LABELS_VIEW = "employer_industry_labels"


def industry_labels_view_sql() -> str:
    """Attach each employer's LLM industry label to every filing it has."""
    return f"""
CREATE OR REPLACE VIEW "{INDUSTRY_LABELS_VIEW}" AS
SELECT
    l."employer_name",
    l."naics_2",
    l."naics_3",
    l."model",
    f."case_number",
    f."date_filed",
    f."city",
    f."state"
FROM "llm_naics_labels" l
JOIN "filing" f ON f."name" = l."employer_name"
"""


def table_exists(connection: duckdb.DuckDBPyConnection, name: str) -> bool:
    return bool(
        connection.execute(
            "SELECT count(*) FROM duckdb_tables() WHERE table_name = ?", [name]
        ).fetchone()[0]
    )


def check_tables(connection: duckdb.DuckDBPyConnection) -> None:
    present = {
        row[0]
        for row in connection.execute(
            "SELECT table_name FROM duckdb_tables()"
        ).fetchall()
    }
    missing = sorted(set(REQUIRED_TABLES) - present)
    if missing:
        lines = [
            f"  - {table} (run {REQUIRED_TABLES[table]})" for table in missing
        ]
        raise SystemExit(
            "Missing source table(s):\n" + "\n".join(lines)
        )


def report(
    connection: duckdb.DuckDBPyConnection, view_name: str, db_path: Path
) -> None:
    row = connection.execute(
        f"""
        SELECT
            count(*) AS rows,
            count(DISTINCT "case_number") AS filings,
            count(DISTINCT "election_id") AS elections,
            count(*) FILTER (
                WHERE "votes_for_union" IS NOT NULL
                   OR "votes_against" IS NOT NULL
            ) AS tally_rows,
            count(*) FILTER (WHERE "rpt_id" IS NOT NULL) AS finance_rows,
            count(DISTINCT "case_number") FILTER (
                WHERE "rpt_id" IS NOT NULL
            ) AS filings_with_finance,
            count(*) FILTER (
                WHERE "voting_unit_description" IS NOT NULL
            ) AS voting_unit_rows,
            count(*) FILTER (
                WHERE "llm_to_naics" IS NOT NULL
            ) AS labeled_rows,
            count(DISTINCT "employer_name") FILTER (
                WHERE "llm_to_naics" IS NOT NULL
            ) AS labeled_employers
        FROM "{view_name}"
        """
    ).fetchone()

    print(f"\nDatabase: {db_path}")
    print(f"View: {view_name}")
    if row:
        print(f"Rows: {row[0]:,}")
        print(f"Distinct filings: {row[1]:,}")
        print(f"Distinct elections: {row[2]:,}")
        print(f"Rows with tally vote data: {row[3]:,}")
        print(f"Rows with finance match: {row[4]:,}")
        print(f"Filings with at least one finance match: {row[5]:,}")
        print(f"Rows with voting unit description: {row[6]:,}")
        print(f"Rows with an industry label: {row[7]:,}")
        print(f"Distinct labeled employers: {row[8]:,}")

    fanout = connection.execute(
        f"""
        SELECT n_reports, count(*) AS elections
        FROM (
            SELECT "election_id", count(DISTINCT "rpt_id") AS n_reports
            FROM "{view_name}"
            WHERE "election_id" IS NOT NULL
            GROUP BY "election_id"
        )
        GROUP BY n_reports
        ORDER BY n_reports
        """
    ).fetchall()
    if fanout:
        print("Finance reports per election:")
        for n_reports, elections in fanout:
            print(f"  {n_reports:>2} report(s): {elections:,} elections")


def report_industry_labels(connection: duckdb.DuckDBPyConnection) -> None:
    row = connection.execute(
        f"""
        SELECT
            (SELECT count(*) FROM "llm_naics_labels"),
            (SELECT count(*) FROM "{INDUSTRY_LABELS_VIEW}"),
            (SELECT count(*) FROM "llm_naics_labels" l
              WHERE NOT EXISTS (
                  SELECT 1 FROM "filing" f
                  WHERE f."name" = l."employer_name"
              ))
        """
    ).fetchone()
    print(f"\nView: {INDUSTRY_LABELS_VIEW}")
    print(f"Labeled employers: {row[0]:,}")
    print(f"Labeled employer/filing rows: {row[1]:,}")
    print(f"Labels with no matching filing: {row[2]:,}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the analytical election/finance view.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="DuckDB database containing the source tables",
    )
    parser.add_argument(
        "--view",
        default=DEFAULT_VIEW,
        help="name of the view to create",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable debug logging"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )

    view_name = args.view.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", view_name):
        raise SystemExit(
            f"Invalid view name {view_name!r}: use letters, digits and underscores"
        )

    db_path = args.db.expanduser()
    if not db_path.exists():
        raise SystemExit(f"Database not found: {db_path}")

    connection = duckdb.connect(str(db_path))
    try:
        check_tables(connection)
        has_labels = table_exists(connection, "llm_naics_labels")
        LOG.info("Building view %s ...", view_name)
        connection.execute(view_sql(view_name, has_labels=has_labels))
        report(connection, view_name, db_path)

        if has_labels:
            LOG.info("Building view %s ...", INDUSTRY_LABELS_VIEW)
            connection.execute(industry_labels_view_sql())
            report_industry_labels(connection)
        else:
            LOG.info(
                "Skipping view %s (llm_naics_labels not found; "
                "run the labeling pipeline first).",
                INDUSTRY_LABELS_VIEW,
            )
    finally:
        connection.close()

    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
