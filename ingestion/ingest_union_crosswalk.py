#!/usr/bin/env python3
"""Ingest the labor-data union-name crosswalk into DuckDB.

Source
------
https://labordata.bunkum.us/union_names_crosswalk is a Datasette database
containing a precomputed entity-resolution table: an arbitrary union name is
mapped to a canonical parent name and an OLMS file number (``f_num``), with
confidence scores.

The table is written to the shared DuckDB file as ``union_names_crosswalk``.
It is used by ``transformations/build_table.py`` to connect election union
names to the OLMS finance data.

Usage
-----
    uv run python ingestion/ingest_union_crosswalk.py
    uv run python ingestion/ingest_union_crosswalk.py --db /tmp/crosswalk.duckdb
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

LOG = logging.getLogger("union-crosswalk-ingest")

DEFAULT_BASE_URL = "https://labordata.bunkum.us/union_names_crosswalk"


CROSSWALK = TableSpec(
    name="union_names_crosswalk",
    columns=(
        Column("union_name", "VARCHAR"),
        Column("pred_is_union", "BIGINT"),
        Column("pred_is_union_score", "DOUBLE"),
        Column("pred_union_name", "VARCHAR"),
        Column("pred_union_name_score", "DOUBLE"),
        Column("pred_f_num", "BIGINT"),
        Column("pred_f_num_score", "DOUBLE"),
    ),
    order_by='"union_name"',
)

TABLE_SPECS: tuple[TableSpec, ...] = (CROSSWALK,)


def report(connection: duckdb.DuckDBPyConnection, db_path: Path) -> None:
    row = connection.execute(
        'SELECT count(*), count(DISTINCT "union_name"), '
        'count(DISTINCT "pred_union_name"), count(DISTINCT "pred_f_num") '
        'FROM "union_names_crosswalk"'
    ).fetchone()
    print(f"\nDatabase: {db_path}")
    if row:
        print(
            f"union_names_crosswalk: {row[0]:,} rows, "
            f"{row[1]:,} names, {row[2]:,} parent unions, "
            f"{row[3]:,} file numbers"
        )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest the union-name crosswalk into DuckDB.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB,
        help="target DuckDB database file",
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

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.workdir:
        workdir = args.workdir.expanduser()
        workdir.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="union-crosswalk-ingest-")
        workdir = Path(temporary.name)

    expected: dict[str, int] = {}
    try:
        for spec in specs:
            sql = spec.select_sql()
            destination = workdir / f"{spec.name}.csv"
            LOG.info("Downloading %s ...", spec.name)
            client.download(query_url(args.base_url, sql, None, "csv"), destination)
            rows = inspect_csv(destination, spec.column_names)
            if args.no_verify:
                expected[spec.name] = rows
            else:
                LOG.info("Verifying %s against source ...", spec.name)
                expected[spec.name] = source_count(
                    client, args.base_url, sql, None
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
