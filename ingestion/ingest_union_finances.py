#!/usr/bin/env python3
"""Ingest OLMS union-finance data into DuckDB.

Source
------
https://labordata.bunkum.us/opdr mirrors the Department of Labor's Online
Public Disclosure Room (LM-2/LM-3/LM-4 annual reports).  This script uses the
instance's SQL/CSV endpoint, so only the needed rows are downloaded instead
of the full 938 MB database dump.

Tables written (in the shared DuckDB file):

    lm_data                 one row per annual report, keyed by rpt_id
    ar_disbursements_total  one row per report's disbursement totals

Filtering
---------
* ``lm_data`` keeps reports whose ``yr_covered`` is >= 2010 (inclusive).
* ``ar_disbursements_total`` keeps rows whose ``rpt_id`` appears in the
  filtered ``lm_data`` set.

Only these two tables are replaced; other tables in the database (for example
the NLRB election tables) are left untouched.

Usage
-----
    uv run python ingestion/ingest_union_finances.py
    uv run python ingestion/ingest_union_finances.py --since-year 2015
    uv run python ingestion/ingest_union_finances.py --db /tmp/opdr.duckdb

The script is idempotent: each run replaces ``lm_data`` and
``ar_disbursements_total``.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from pathlib import Path
from typing import Sequence

import duckdb

from common import (
    DEFAULT_DB,
    Column,
    HttpClient,
    TableSpec,
    inspect_csv,
    load_tables,
    query_url,
    source_count,
)

LOG = logging.getLogger("union-finances-ingest")

DEFAULT_BASE_URL = "https://labordata.bunkum.us/opdr"
DEFAULT_SINCE_YEAR = 2010


# ---------------------------------------------------------------------------
# Source schema
# ---------------------------------------------------------------------------


LM_DATA = TableSpec(
    name="lm_data",
    columns=(
        Column("union_name", "VARCHAR"),
        Column("aff_abbr", "VARCHAR"),
        Column("f_num", "BIGINT"),  # Note that this is Int on labordata but maybe should be string since has leading zeros
        Column("fye", "BIGINT"),
        Column("unit_name", "VARCHAR"),
        Column("est_date", "DATE"),
        Column("term_date", "DATE"),
        Column("shortage", "VARCHAR"),
        Column("maximum_bond", "BIGINT"),
        Column("pd_covered_from", "DATE"),
        Column("pd_covered_to", "DATE"),
        Column("constitution_bylaw", "VARCHAR"),
        Column("terminate", "VARCHAR"),
        Column("ttl_assets", "BIGINT"),
        Column("subsidiary", "VARCHAR"),
        Column("desiq_pre", "VARCHAR"),
        Column("desig_num", "BIGINT"),
        Column("desig_suf", "VARCHAR"),
        Column("desig_name", "VARCHAR"),
        Column("ttl_liabilities", "BIGINT"),
        Column("ttl_receipts", "BIGINT"),
        Column("ttl_disbursements", "BIGINT"),
        Column("members", "BIGINT"),
        Column("register_date", "TIMESTAMP"),
        Column("amended", "VARCHAR"),
        Column("hardship", "VARCHAR"),
        Column("has_trust", "VARCHAR"),
        Column("pac_funds", "VARCHAR"),
        Column("outside_audit", "VARCHAR"),
        Column("has_property_change", "VARCHAR"),
        Column("assets_pledged", "VARCHAR"),
        Column("contingent", "VARCHAR"),
        Column("next_election", "VARCHAR"),
        Column("has_liquidated_liabilities", "VARCHAR"),
        Column("has_extended_loan_credit", "VARCHAR"),
        Column("has_liquidated_receivables", "VARCHAR"),
        Column("has_subsidiary", "VARCHAR"),
        Column("num_attachments", "BIGINT"),
        Column("rpt_id", "BIGINT"),
        Column("yr_covered", "BIGINT"),
        Column("amendment", "BIGINT"),
        Column("receive_date", "TIMESTAMP"),
        Column("adr_id", "BIGINT"),
        Column("address_type", "BIGINT"),
        Column("mail_firstname", "VARCHAR"),
        Column("mail_lastname", "VARCHAR"),
        Column("build_num", "VARCHAR"),
        Column("street_adr", "VARCHAR"),
        Column("city", "VARCHAR"),
        Column("state", "VARCHAR"),
        Column("zip", "VARCHAR"),
        Column("voice", "VARCHAR"),
        Column("mod_date", "TIMESTAMP"),
        Column("mod_id", "BIGINT"),
        Column("record_kept", "VARCHAR"),
        Column("form_type", "VARCHAR"),
    ),
    order_by='"rpt_id"',
    primary_key="rpt_id",
)

AR_DISBURSEMENTS_TOTAL = TableSpec(
    name="ar_disbursements_total",
    columns=(
        Column("rpt_id", "BIGINT"),
        Column("representational", "BIGINT"),
        Column("political", "BIGINT"),
        Column("contributions", "BIGINT"),
        Column("general_overhead", "BIGINT"),
        Column("union_administration", "BIGINT"),
        Column("withheld", "BIGINT"),
        Column("members", "BIGINT"),
        Column("supplies", "BIGINT"),
        Column("fees", "BIGINT"),
        Column("administration", "BIGINT"),
        Column("direct_taxes", "BIGINT"),
        Column("strike_benefits", "BIGINT"),
        Column("per_capita_tax", "BIGINT"),
        Column("to_officers", "BIGINT"),
        Column("investments", "BIGINT"),
        Column("benefits", "BIGINT"),
        Column("loans_made", "BIGINT"),
        Column("loans_payment", "BIGINT"),
        Column("affiliates", "BIGINT"),
        Column("other_disbursements", "BIGINT"),
        Column("to_employees", "BIGINT"),
        Column("less_total_disbursed", "BIGINT"),
        Column("withheld_not_disbursed", "BIGINT"),
        Column("officer_deductions", "BIGINT"),
        Column("employee_deductions", "BIGINT"),
        Column("inv_purch_reinvestments", "BIGINT"),
        Column("all_other_rep_activities", "BIGINT"),
        Column("all_other_pol_activities", "BIGINT"),
        Column("all_other_contributions", "BIGINT"),
        Column("all_other_gen_overhead", "BIGINT"),
        Column("all_other_union_admin", "BIGINT"),
        Column("off_admin_expense", "BIGINT"),
        Column("edu_pub_expense", "BIGINT"),
        Column("pro_fees", "BIGINT"),
    ),
    order_by='"rpt_id"',
    primary_key="rpt_id",
)

TABLE_SPECS: tuple[TableSpec, ...] = (LM_DATA, AR_DISBURSEMENTS_TOTAL)


def _table_filter(table: str, since_year: int) -> str:
    """Return the WHERE clause that ties ``table`` to the ingested reports."""
    if table == "lm_data":
        return 'WHERE "yr_covered" >= CAST(:since_year AS INTEGER)'

    if table == "ar_disbursements_total":
        return (
            'WHERE "rpt_id" IN (\n'
            '    SELECT "rpt_id" FROM "lm_data"\n'
            '    WHERE "yr_covered" >= CAST(:since_year AS INTEGER)\n'
            ")"
        )

    raise ValueError(f"Unknown table: {table}")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def report(connection: duckdb.DuckDBPyConnection, db_path: Path) -> None:
    lm_range = connection.execute(
        'SELECT min("yr_covered"), max("yr_covered"), count(*), '
        'count(DISTINCT "rpt_id") FROM "lm_data"'
    ).fetchone()
    disbursements = connection.execute(
        'SELECT count(*), count(DISTINCT "rpt_id") FROM "ar_disbursements_total"'
    ).fetchone()
    orphans = connection.execute(
        'SELECT count(*) FROM "ar_disbursements_total" d '
        'WHERE NOT EXISTS ('
        '  SELECT 1 FROM "lm_data" l WHERE l."rpt_id" = d."rpt_id"'
        ") "
    ).fetchone()

    print(f"\nDatabase: {db_path}")
    if lm_range:
        print(
            f"lm_data years: {lm_range[0]} .. {lm_range[1]} "
            f"({lm_range[2]:,} reports, {lm_range[3]:,} distinct rpt_id)"
        )
    if disbursements:
        print(
            f"ar_disbursements_total: {disbursements[0]:,} rows, "
            f"{disbursements[1]:,} distinct rpt_id"
        )
    if orphans:
        print(f"Disbursement rows without lm_data: {orphans[0]:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Ingest OLMS LM-2/LM-3/LM-4 data from labordata.bunkum.us "
            "into DuckDB."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="target DuckDB database file",
    )
    parser.add_argument(
        "--since-year",
        type=int,
        default=DEFAULT_SINCE_YEAR,
        metavar="YEAR",
        help="earliest yr_covered value to ingest (inclusive)",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Datasette database URL",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="HTTP timeout in seconds",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=5,
        help="number of attempts for transient HTTP failures",
    )
    parser.add_argument(
        "--workdir",
        type=Path,
        help="keep downloaded CSVs in this directory instead of a temporary one",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="skip comparing downloaded row counts with the source",
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

    client = HttpClient(timeout=args.timeout, retries=args.retries)
    specs = TABLE_SPECS
    since_year: int = args.since_year
    params = {"since_year": str(since_year)}

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.workdir:
        workdir = args.workdir.expanduser()
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="union-finances-ingest-")
        workdir = Path(temporary.name)

    expected: dict[str, int] = {}
    try:
        for spec in specs:
            sql = spec.select_sql(_table_filter(spec.name, since_year))
            destination = workdir / f"{spec.name}.csv"
            LOG.info("Downloading %s ...", spec.name)
            client.download(
                query_url(args.base_url, sql, params, "csv"), destination
            )
            rows = inspect_csv(destination, spec.column_names)
            if args.no_verify:
                expected[spec.name] = rows
            else:
                LOG.info("Verifying %s against source ...", spec.name)
                expected[spec.name] = source_count(
                    client, args.base_url, sql, params
                )
                if rows != expected[spec.name]:
                    raise RuntimeError(
                        f"{spec.name}: downloaded {rows:,} rows but source "
                        f"reports {expected[spec.name]:,}"
                    )
            LOG.info("  %s rows", f"{rows:,}")

        LOG.info("Loading into %s ...", args.db)
        counts = load_tables(args.db.expanduser(), specs, workdir)
    finally:
        if temporary is not None:
            temporary.cleanup()

    print()
    print(f"{'table':<24}{'rows':>10}{'source':>10}")
    for spec in specs:
        print(
            f"{spec.name:<24}{counts[spec.name]:>10,}"
            f"{expected[spec.name]:>10,}"
        )

    mismatches = [
        spec.name for spec in specs if counts[spec.name] != expected[spec.name]
    ]
    if mismatches:
        LOG.error("Row-count mismatch after load: %s", ", ".join(mismatches))
        return 1

    connection = duckdb.connect(str(args.db.expanduser()), read_only=True)
    try:
        report(connection, args.db)
    finally:
        connection.close()

    LOG.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
