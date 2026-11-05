"""Shared helpers for the labor-data ingestion scripts.

Both ingestion scripts stream CSV from a Datasette SQL endpoint, validate the
download against the source row count, and load the result into DuckDB using
an explicit schema.  This module holds the pieces that are not specific to a
particular dataset.
"""

from __future__ import annotations

import csv
import json
import logging
import random
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import duckdb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "data" / "union_elections.duckdb"
USER_AGENT = "labor-data-ingest/1.0"

RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
RETRYABLE_ERRORS = (
    urllib.error.URLError,
    TimeoutError,
    ConnectionError,
    OSError,
)

LOG = logging.getLogger("labor-data-ingest")


# ---------------------------------------------------------------------------
# Source schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Column:
    name: str
    sql_type: str  # BIGINT, VARCHAR, DATE, TIMESTAMP


@dataclass(frozen=True)
class TableSpec:
    name: str
    columns: Sequence[Column]
    order_by: str
    primary_key: str | None = None

    @property
    def column_names(self) -> list[str]:
        return [column.name for column in self.columns]

    def ddl(self) -> str:
        lines = [f'    "{c.name}" {c.sql_type}' for c in self.columns]
        if self.primary_key:
            lines.append(f'    PRIMARY KEY ("{self.primary_key}")')
        return f'CREATE TABLE "{self.name}" (\n' + ",\n".join(lines) + "\n)"

    def select_sql(self, where: str = "") -> str:
        columns = ", ".join(f'"{name}"' for name in self.column_names)
        sql = f'SELECT {columns}\nFROM "{self.name}"'
        if where:
            sql += f"\n{where}"
        return f"{sql}\nORDER BY {self.order_by}"

    def insert_sql(self) -> str:
        expressions = ", ".join(_cast_expression(column) for column in self.columns)
        return (
            f'INSERT INTO "{self.name}"\n'
            f"SELECT {expressions}\n"
            f"FROM read_csv(?, header = true, all_varchar = true)"
        )


def _cast_expression(column: Column) -> str:
    quoted = f'"{column.name}"'
    if column.sql_type == "VARCHAR":
        # Preserve empty strings; the source uses '' rather than NULL for
        # missing text values.
        return quoted
    # Empty strings in numeric/date/timestamp columns mean NULL.
    return f"NULLIF({quoted}, '')::{column.sql_type}"


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------


class HttpClient:
    def __init__(self, timeout: float, retries: int) -> None:
        self.timeout = timeout
        self.retries = max(1, retries)

    @staticmethod
    def _sleep(attempt: int, retry_after: str | None) -> None:
        if retry_after:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = 0.0
            if delay > 0:
                time.sleep(min(delay, 60))
                return
        time.sleep(min(2 ** (attempt - 1), 30) + random.uniform(0, 0.5))

    def download(self, url: str, destination: Path) -> None:
        """Download ``url`` to ``destination``, retrying transient failures."""
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "text/csv"}
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    with destination.open("wb") as output:
                        shutil.copyfileobj(response, output, length=1024 * 1024)
                return
            except urllib.error.HTTPError as exc:
                body = exc.read(1000).decode("utf-8", "replace").strip()
                last_error = RuntimeError(
                    f"HTTP {exc.code} from {url}" + (f": {body}" if body else "")
                )
                if exc.code in RETRYABLE_STATUS and attempt < self.retries:
                    LOG.warning(
                        "HTTP %s from %s; retrying (%d/%d)",
                        exc.code,
                        url,
                        attempt,
                        self.retries,
                    )
                    self._sleep(attempt, exc.headers.get("Retry-After"))
                    continue
                raise last_error from exc
            except RETRYABLE_ERRORS as exc:
                last_error = exc
                if attempt < self.retries:
                    LOG.warning(
                        "Network error (%s); retrying (%d/%d)",
                        exc,
                        attempt,
                        self.retries,
                    )
                    self._sleep(attempt, None)
                    continue
                raise RuntimeError(f"Failed to download {url}: {exc}") from exc
        raise RuntimeError(f"Failed to download {url}: {last_error}")

    def get_json(self, url: str) -> object:
        last_error: Exception | None = None
        for attempt in range(1, self.retries + 1):
            request = urllib.request.Request(
                url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"}
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                body = exc.read(1000).decode("utf-8", "replace").strip()
                last_error = RuntimeError(
                    f"HTTP {exc.code} from {url}" + (f": {body}" if body else "")
                )
                if exc.code in RETRYABLE_STATUS and attempt < self.retries:
                    LOG.warning(
                        "HTTP %s from %s; retrying (%d/%d)",
                        exc.code,
                        url,
                        attempt,
                        self.retries,
                    )
                    self._sleep(attempt, exc.headers.get("Retry-After"))
                    continue
                raise last_error from exc
            except RETRYABLE_ERRORS as exc:
                last_error = exc
                if attempt < self.retries:
                    LOG.warning(
                        "Network error (%s); retrying (%d/%d)",
                        exc,
                        attempt,
                        self.retries,
                    )
                    self._sleep(attempt, None)
                    continue
                raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc
        raise RuntimeError(f"Failed to fetch {url}: {last_error}")


def query_url(
    base_url: str,
    sql: str,
    params: Mapping[str, str] | None = None,
    fmt: str = "csv",
) -> str:
    query: dict[str, str] = {"sql": sql}
    if params:
        query.update(params)
    if fmt == "csv":
        query["_stream"] = "on"
        query["_size"] = "max"
    elif fmt == "json":
        query["_shape"] = "array"
    return f"{base_url.rstrip('/')}/-/query.{fmt}?{urllib.parse.urlencode(query)}"


def source_count(
    client: HttpClient,
    base_url: str,
    sql: str,
    params: Mapping[str, str] | None = None,
) -> int:
    count_sql = f"SELECT COUNT(*) AS n FROM (\n{sql}\n)"
    payload = client.get_json(query_url(base_url, count_sql, params, "json"))
    try:
        return int(payload[0]["n"])  # type: ignore[index]
    except (TypeError, KeyError, IndexError, ValueError) as exc:
        raise RuntimeError(f"Unexpected count response: {payload!r}") from exc


# ---------------------------------------------------------------------------
# CSV validation and DuckDB loading
# ---------------------------------------------------------------------------


def inspect_csv(path: Path, expected_columns: Sequence[str]) -> int:
    """Validate the header and return the number of data rows."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration as exc:
            raise RuntimeError(f"{path} is empty") from exc
        if header != list(expected_columns):
            raise RuntimeError(
                f"Unexpected columns in {path}.\n"
                f"  expected: {list(expected_columns)}\n"
                f"  received: {header}"
            )
        return sum(1 for _ in reader)


def load_tables(
    db_path: Path, specs: Sequence[TableSpec], workdir: Path
) -> dict[str, int]:
    """Replace ``specs`` in ``db_path`` from the CSVs in ``workdir``."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(db_path))
    try:
        connection.execute("BEGIN TRANSACTION")
        for spec in reversed(specs):
            connection.execute(f'DROP TABLE IF EXISTS "{spec.name}"')
        for spec in specs:
            connection.execute(spec.ddl())
            connection.execute(
                spec.insert_sql(), [str(workdir / f"{spec.name}.csv")]
            )
        connection.execute("COMMIT")
    except Exception:
        try:
            connection.execute("ROLLBACK")
        except Exception:
            pass
        raise

    try:
        counts: dict[str, int] = {}
        for spec in specs:
            row = connection.execute(
                f'SELECT COUNT(*) FROM "{spec.name}"'
            ).fetchone()
            counts[spec.name] = int(row[0])
    finally:
        connection.close()
    return counts
