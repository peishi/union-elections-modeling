#!/usr/bin/env python3
"""Ingest NLRB representation-election data into DuckDB.

Source
------
https://labordata.bunkum.us/nlrb is a Datasette instance backed by the NLRB's
case data (scraped from nlrb.gov).  This script uses the instance's SQL/CSV
endpoint, so only the needed rows are downloaded (~20 MB by default) instead
of the full 444 MB database dump.

Tables written (all in the target DuckDB file):

    election         one row per election
    election_result  one row per election result
    election_mode    election-mode / tally details (may be multiple per case)
    filing           one row per NLRB case referenced by the ingested elections

By default the cutoff is the *election date* (2010-01-01), not the filing
date.  Pass ``--since all`` to ingest every election in the source table.

Usage
-----
    uv run python ingestion/ingest_nlrb_elections.py
    uv run python ingestion/ingest_nlrb_elections.py --since all
    uv run python ingestion/ingest_nlrb_elections.py --db /tmp/nlrb.duckdb --workdir csv/

The script is idempotent: each run replaces the four tables in the target
database.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from datetime import date
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

LOG = logging.getLogger("nlrb-ingest")

DEFAULT_BASE_URL = "https://labordata.bunkum.us/nlrb"
DEFAULT_SINCE = "2010-01-01"


def _table_filter(table: str, since: str | None) -> str:
    """Return the WHERE clause that ties ``table`` to the ingested elections."""
    if table == "election":
        return "WHERE \"date\" >= :since" if since else ""

    if table == "election_result":
        if since:
            return (
                'WHERE "election_id" IN (\n'
                '    SELECT "election_id" FROM "election" WHERE "date" >= :since\n'
                ")"
            )
        return 'WHERE "election_id" IN (SELECT "election_id" FROM "election")'

    # election_mode, filing and participant are keyed by case_number.  Using
    # case_number (rather than election_id) keeps the 94 election_mode rows
    # whose election_id is NULL but whose case is in range.
    if since:
        return (
            'WHERE "case_number" IN (\n'
            '    SELECT "case_number" FROM "election" WHERE "date" >= :since\n'
            ")"
        )
    return 'WHERE "case_number" IN (SELECT "case_number" FROM "election")'


TABLE_SPECS: tuple[TableSpec, ...] = (
    TableSpec(
        name="election",
        columns=(
            Column("election_id", "BIGINT"),
            Column("case_number", "VARCHAR"),
            Column("voting_unit_id", "BIGINT"),
            Column("date", "DATE"),
            Column("tally_type", "VARCHAR"),
            Column("ballot_type", "VARCHAR"),
            Column("unit_size", "BIGINT"),
            Column("created_at", "TIMESTAMP"),
        ),
        order_by='"election_id"',
        primary_key="election_id",
    ),
    TableSpec(
        name="election_result",
        columns=(
            Column("election_id", "BIGINT"),
            Column("total_ballots_counted", "BIGINT"),
            Column("void_ballots", "BIGINT"),
            Column("challenged_ballots", "BIGINT"),
            Column("challenges_are_determinative", "VARCHAR"),
            Column("runoff_required", "VARCHAR"),
            Column("union_to_certify", "VARCHAR"),
            Column("created_at", "TIMESTAMP"),
        ),
        order_by='"election_id"',
        primary_key="election_id",
    ),
    TableSpec(
        name="election_mode",
        columns=(
            Column("election_id", "BIGINT"),
            Column("case_number", "VARCHAR"),
            Column("name", "VARCHAR"),
            Column("city", "VARCHAR"),
            Column("state", "VARCHAR"),
            Column("status", "VARCHAR"),
            Column("date_filed", "DATE"),
            Column("date_closed", "DATE"),
            Column("reason_closed", "VARCHAR"),
            Column("election_mode", "VARCHAR"),
            Column("date_ballot_mailed", "DATE"),
            Column("date_ballot_counted", "DATE"),
            Column("date_election_scheduled", "DATE"),
            Column("date_tally_scheduled", "DATE"),
            Column("date_tallied", "DATE"),
            Column("tally_type", "VARCHAR"),
            Column("ballot_type", "VARCHAR"),
            Column("unit_id", "VARCHAR"),
            Column("ballots_impounded", "VARCHAR"),
            Column("number_of_eligible_voters", "BIGINT"),
            Column("number_of_void_ballots", "BIGINT"),
            Column("labor_organization_1_name", "VARCHAR"),
            Column("votes_for_labor_organization_1", "BIGINT"),
            Column("labor_organization_2_name", "VARCHAR"),
            Column("votes_for_labor_organization_2", "BIGINT"),
            Column("labor_organization_3_name", "VARCHAR"),
            Column("votes_for_labor_organization_3", "BIGINT"),
            Column("votes_cast_against_labor_org", "BIGINT"),
            Column("number_of_valid_votes_counted", "BIGINT"),
            Column("number_of_challenged_ballots", "BIGINT"),
            Column("challenges_are_determinative", "VARCHAR"),
            Column("runoff_required", "VARCHAR"),
            Column("union_to_certify", "VARCHAR"),
            Column("unit_involved_in_petition", "VARCHAR"),
            Column("bargaining_unit_determined", "VARCHAR"),
        ),
        order_by='"case_number", "election_id"',
    ),
    TableSpec(
        name="filing",
        columns=(
            Column("case_number", "VARCHAR"),
            Column("name", "VARCHAR"),
            Column("case_type", "VARCHAR"),
            Column("url", "VARCHAR"),
            Column("city", "VARCHAR"),
            Column("state", "VARCHAR"),
            Column("date_filed", "DATE"),
            Column("region_assigned", "VARCHAR"),
            Column("status", "VARCHAR"),
            Column("date_closed", "DATE"),
            Column("reason_closed", "VARCHAR"),
            Column("number_of_eligible_voters", "BIGINT"),
            Column("number_of_voters_on_petition_or_charge", "BIGINT"),
            Column("certified_representative", "VARCHAR"),
            Column("created_at", "TIMESTAMP"),
            Column("updated_at", "TIMESTAMP"),
            Column("last_checked_at", "TIMESTAMP"),
        ),
        order_by='"case_number"',
        primary_key="case_number",
    ),
    TableSpec(
        name="participant",
        columns=(
            Column("case_number", "VARCHAR"),
            Column("participant", "VARCHAR"),
            Column("type", "VARCHAR"),
            Column("subtype", "VARCHAR"),
            Column("address", "VARCHAR"),
            Column("address_1", "VARCHAR"),
            Column("address_2", "VARCHAR"),
            Column("city", "VARCHAR"),
            Column("state", "VARCHAR"),
            Column("zip", "VARCHAR"),
            Column("phone_number", "VARCHAR"),
            Column("created_at", "TIMESTAMP"),
        ),
        order_by='"case_number", "participant"',
    ),
)


def report(
    connection: duckdb.DuckDBPyConnection, db_path: Path
) -> None:  # pragma: no cover - presentation only
    date_range = connection.execute(
        'SELECT min("date"), max("date"), count(DISTINCT "case_number") '
        'FROM "election"'
    ).fetchone()
    orphans = connection.execute(
        'SELECT count(*) FROM "election" e '
        'WHERE NOT EXISTS ('
        '  SELECT 1 FROM "filing" f WHERE f."case_number" = e."case_number"'
        ") "
    ).fetchone()
    print(f"\nDatabase: {db_path}")
    if date_range:
        print(
            f"Election dates: {date_range[0]} .. {date_range[1]}"
            f" ({date_range[2]:,} distinct case numbers)"
        )
    if orphans:
        print(f"Elections with no filing row: {orphans[0]:,}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_since(value: str) -> str | None:
    text = value.strip()
    if text.lower() in {"all", "*", "none", ""}:
        return None
    try:
        return date.fromisoformat(text).isoformat()
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"expected an ISO date (YYYY-MM-DD) or 'all', got {value!r}"
        ) from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest NLRB election data from labordata.bunkum.us into DuckDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path(DEFAULT_DB),
        help="target DuckDB database file",
    )
    parser.add_argument(
        "--since",
        type=parse_since,
        default=parse_since(DEFAULT_SINCE),
        metavar="DATE|all",
        help="earliest election date to ingest, or 'all'",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="Datasette database URL",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
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
    since: str | None = args.since

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.workdir:
        workdir = args.workdir.expanduser()
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="nlrb-ingest-")
        workdir = Path(temporary.name)

    expected: dict[str, int] = {}
    params = {"since": since} if since else None
    try:
        for spec in specs:
            sql = spec.select_sql(_table_filter(spec.name, since))
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
    print(f"{'table':<18}{'rows':>10}{'source':>10}")
    for spec in specs:
        print(
            f"{spec.name:<18}{counts[spec.name]:>10,}"
            f"{expected[spec.name]:>10,}"
        )

    mismatches = [
        spec.name
        for spec in specs
        if counts[spec.name] != expected[spec.name]
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
