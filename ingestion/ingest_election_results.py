#!/usr/bin/env python3
"""
Download NLRB election-report spreadsheets and load them into DuckDB.

Source index: https://www.nlrb.gov/reports/agency-performance/election-reports

For each fiscal year (FY 2020 onward by default):
  * if the year's page has a "Total FY Cases Closed" .xlsx, ingest only that;
  * otherwise (the year is still in progress / no total posted yet), ingest
    every monthly "Cases Closed" .xlsx on the page.
Using one or the other, never both, avoids counting a case twice.

Each workbook: everything above the header row is ignored. The header row is
located by scanning the top of the sheet for the expected column names (it is
row 18 in FY 2020-2021 and row 28 in later years); if no row matches, the file
is skipped and reported as a failure. The data below the header goes into the
DuckDB table `election_results`.

Install:  pip install requests beautifulsoup4 pandas openpyxl duckdb
Run:      python ingest_election_results.py
          python ingest_election_results.py --fy-min 2023 --rebuild
"""
from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urljoin, urlparse

import duckdb
import pandas as pd
import requests
from bs4 import BeautifulSoup

INDEX_URL = "https://www.nlrb.gov/reports/agency-performance/election-reports"
USER_AGENT = "Mozilla/5.0 (compatible; nlrb-election-ingest-script/3.0)"
FY_LABEL_RE = re.compile(r"^\s*FY\s*(\d{4})\s*$", re.IGNORECASE)
DEFAULT_FY_MIN = 2020  # no spreadsheets exist before FY 2020

MONTH_NAMES = [
    "january", "february", "march", "april", "may", "june", "july",
    "august", "september", "october", "november", "december",
]
MONTH_RE = re.compile(r"(" + "|".join(MONTH_NAMES) + r")[\s_\-]*(\d{4})")

HEADER_SCAN_ROWS = 60  # how many rows from the top to search for the header row
TABLE = "election_results"

# (header text in the spreadsheet, database column, DuckDB type)
SCHEMA: list[tuple[str, str, str]] = [
    ("Region", "region", "VARCHAR"),
    ("Case Number", "case_number", "VARCHAR"),
    ("Case ID", "case_id", "VARCHAR"),
    ("Case Name", "case_name", "VARCHAR"),
    ("Case Type", "case_type", "VARCHAR"),
    ("Dispute Unit City", "dispute_unit_city", "VARCHAR"),
    ("Dispute Unit State", "dispute_unit_state", "VARCHAR"),
    ("Election Held Date", "election_held_date", "DATE"),
    ("Num Eligible Voters", "num_eligible_voters", "INTEGER"),
    ("Valid Votes Against", "votes_against", "INTEGER"),
    ("Votes for Labor Org 1", "votes_for", "INTEGER"),
    ("Labor Org 1 Name", "labor_org_name", "VARCHAR"),
    ("Stipulated / Consent / Directed", "stipulated_consent_directed", "VARCHAR"),
    ("Cert of Rep (Win)", "cert_of_rep_win", "VARCHAR"),
    ("Cert of Results (Loss)", "cert_of_results_loss", "VARCHAR"),
    ("Closed Date", "closed_date", "DATE"),
    ("Closed Reason", "closed_reason", "VARCHAR"),
]
DATA_COLUMNS = [col for _, col, _ in SCHEMA]
META_DDL = [
    ("fiscal_year", "INTEGER"),
    ("report_kind", "VARCHAR"),   # 'fy_total' or 'monthly'
    ("report_month", "DATE"),     # first day of the month (monthly files only)
    ("source_url", "VARCHAR"),
    ("source_file", "VARCHAR"),
    ("ingested_at", "TIMESTAMP"),
]

log = logging.getLogger("nlrb")


# --------------------------------------------------------------------------
# HTTP + discovery
# --------------------------------------------------------------------------
def make_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT})
    return s


def http_get(session: requests.Session, url: str, retries: int = 3) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_exc = exc
            wait = 2 ** attempt
            log.warning("GET %s failed (%s); retry %d/%d in %ds", url, exc, attempt, retries, wait)
            time.sleep(wait)
    raise RuntimeError(f"Giving up on {url}: {last_exc}")


def discover_fiscal_year_pages(session: requests.Session, index_url: str) -> dict[int, str]:
    """Return {fiscal_year: page_url} using only links labelled 'FY 20XX'."""
    soup = BeautifulSoup(http_get(session, index_url).text, "html.parser")
    pages: dict[int, str] = {}
    for a in soup.find_all("a", href=True):
        m = FY_LABEL_RE.match(a.get_text(strip=True))
        if m:
            pages[int(m.group(1))] = urljoin(index_url, a["href"])
    return dict(sorted(pages.items(), reverse=True))


def discover_xlsx_links(session: requests.Session, page_url: str) -> dict[str, str]:
    """Return {xlsx_url: link_text} for every .xlsx linked from a FY page."""
    soup = BeautifulSoup(http_get(session, page_url).text, "html.parser")
    found: dict[str, str] = {}
    for a in soup.find_all("a", href=True):
        url = urljoin(page_url, a["href"])
        if urlparse(url).path.lower().endswith(".xlsx"):
            found.setdefault(url, a.get_text(" ", strip=True))
    return found


def select_files(links: dict[str, str]) -> list[tuple[str, str, date | None]]:
    """
    Decide what to ingest for one fiscal year.
    Returns [(url, report_kind, report_month)]: the FY total file if there is
    one, otherwise every monthly file (oldest month first).
    """
    totals, monthly = [], []
    for url, text in links.items():
        name = Path(unquote(urlparse(url).path)).name
        blob = f"{text} {name}".lower()
        if "total" in blob:
            totals.append((url, "fy_total", None))
            continue
        m = MONTH_RE.search(blob)
        month = date(int(m.group(2)), MONTH_NAMES.index(m.group(1)) + 1, 1) if m else None
        monthly.append((url, "monthly", month))

    if totals:
        return totals
    return sorted(monthly, key=lambda t: (t[2] is None, t[2] or date.min, t[0]))


def download_file(session: requests.Session, url: str, dest: Path) -> bool:
    """Download url to dest. Returns True if a download happened."""
    if dest.exists() and dest.stat().st_size > 0:
        log.debug("Already have %s", dest)
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    resp = http_get(session, url)
    if not resp.content.startswith(b"PK"):  # .xlsx files are zip archives
        raise ValueError(f"{url} did not return an .xlsx file")
    tmp = dest.with_suffix(dest.suffix + ".part")
    tmp.write_bytes(resp.content)
    tmp.replace(dest)
    return True


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------
def _norm(text) -> str:
    """Lowercase and collapse whitespace so header comparison is forgiving."""
    if text is None or (not isinstance(text, str) and pd.isna(text)):
        return ""
    return re.sub(r"\s+", " ", str(text)).strip().lower()


def _to_text(value):
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    text = str(value).strip()
    return text or None


def find_header_row(raw: pd.DataFrame, expected: list[str]) -> int | None:
    """Return the 0-based index of the row whose cells match the expected headers."""
    for i in range(min(HEADER_SCAN_ROWS, len(raw))):
        row = [_norm(c) for c in raw.iloc[i, : len(expected)]]
        if row == expected:
            return i
    return None


def parse_sheet(raw: pd.DataFrame, label: str) -> pd.DataFrame | None:
    """
    raw = sheet read with header=None. Finds the header row (see find_header_row),
    ignores everything above it, and returns a typed DataFrame, or None if no
    row in the first HEADER_SCAN_ROWS rows matches the expected columns.
    """
    expected = [_norm(h) for h, _, _ in SCHEMA]
    idx = find_header_row(raw, expected)
    if idx is None:
        log.warning(
            "%s: no row in the first %d rows matches the expected headers.\n    expected: %s",
            label, HEADER_SCAN_ROWS, expected,
        )
        return None
    log.info("  %s: header found on row %d", label, idx + 1)

    body = raw.iloc[idx + 1:, : len(expected)].copy()
    body.columns = DATA_COLUMNS
    body = body.dropna(how="all")
    body = body[body["case_number"].notna()]  # drop footnotes / blank trailers

    for _, col, typ in SCHEMA:
        if typ == "VARCHAR":
            body[col] = body[col].map(_to_text)
            continue
        before = body[col].notna().sum()
        if typ == "DATE":
            conv = pd.to_datetime(body[col], errors="coerce").dt.date
            conv = conv.astype(object).where(pd.notna(conv), None)
        else:  # INTEGER
            conv = pd.to_numeric(body[col], errors="coerce").astype("Int64")
        lost = before - conv.notna().sum()
        if lost:
            bad = body.loc[conv.isna() & body[col].notna(), col].head(3).tolist()
            log.warning("%s: %d value(s) in %s could not be read as %s (e.g. %s) -> NULL",
                        label, lost, col, typ, bad)
        body[col] = conv
    return body.reset_index(drop=True)


def read_workbook(path: Path) -> list[pd.DataFrame]:
    # sheet_name=None reads every sheet and returns {sheet_name: DataFrame}
    sheets = pd.read_excel(path, sheet_name=None, header=None, dtype=object, engine="openpyxl")
    frames = []
    for sheet, raw in sheets.items():
        df = parse_sheet(raw, f"{path.name}[{sheet}]")
        if df is not None and not df.empty:
            frames.append(df)
    return frames


# --------------------------------------------------------------------------
# DuckDB
# --------------------------------------------------------------------------
def init_db(con: duckdb.DuckDBPyConnection, rebuild: bool) -> None:
    con.execute("DROP VIEW IF EXISTS election_results_monthly")  # left by an earlier version
    if rebuild:
        con.execute(f"DROP TABLE IF EXISTS {TABLE}")

    cols = [f"{c} {t}" for c, t in META_DDL] + [f"{c} {t}" for _, c, t in SCHEMA]
    exists = con.execute(
        "SELECT count(*) FROM information_schema.tables WHERE table_name = ?", [TABLE]
    ).fetchone()[0]
    if exists:
        have = {r[0] for r in con.execute(f"DESCRIBE {TABLE}").fetchall()}
        want = {c for c, _ in META_DDL} | set(DATA_COLUMNS)
        if have != want:
            raise SystemExit(
                f"Table '{TABLE}' already exists with a different layout (probably from an earlier "
                f"version of this script). Re-run with --rebuild to drop and recreate it."
            )
    else:
        con.execute(f"CREATE TABLE {TABLE} ({', '.join(cols)})")


def load_dataframe(
    con, df: pd.DataFrame, *, fiscal_year: int, report_kind: str,
    report_month: date | None, source_url: str, source_file: str,
) -> None:
    con.register("incoming", df)
    try:
        con.execute(
            f"""
            INSERT INTO {TABLE} BY NAME
            SELECT ?::INTEGER AS fiscal_year, ?::VARCHAR AS report_kind,
                   ?::DATE AS report_month, ?::VARCHAR AS source_url,
                   ?::VARCHAR AS source_file, ?::TIMESTAMP AS ingested_at, *
            FROM incoming
            """,
            [fiscal_year, report_kind, report_month, source_url, source_file,
             datetime.now(timezone.utc).replace(tzinfo=None)],
        )
    finally:
        con.unregister("incoming")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default="data/union_elections.duckdb", help="DuckDB file to write")
    ap.add_argument("--raw-dir", default="data/nlrb_raw", help="where to save downloaded .xlsx files")
    ap.add_argument("--fy-min", type=int, default=DEFAULT_FY_MIN, help="earliest fiscal year (default 2020)")
    ap.add_argument("--fy-max", type=int, help="latest fiscal year")
    ap.add_argument("--delay", type=float, default=0.5, help="seconds to pause between requests")
    ap.add_argument("--rebuild", action="store_true", help="drop and recreate the table first")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    raw_dir = Path(args.raw_dir)
    session = make_session()

    fy_pages = discover_fiscal_year_pages(session, INDEX_URL)
    fy_pages = {
        fy: url for fy, url in fy_pages.items()
        if fy >= args.fy_min and (args.fy_max is None or fy <= args.fy_max)
    }
    if not fy_pages:
        log.error("No fiscal-year links found on %s (page layout may have changed).", INDEX_URL)
        return 1
    log.info("Fiscal years to process: %s", ", ".join(f"FY{fy}" for fy in fy_pages))

    con = duckdb.connect(args.db)
    init_db(con, args.rebuild)

    failures: list[str] = []
    files_loaded = rows_loaded = 0

    for fy, page_url in fy_pages.items():
        time.sleep(args.delay)
        try:
            selected = select_files(discover_xlsx_links(session, page_url))
        except Exception as exc:  # noqa: BLE001
            log.error("FY%d: could not read %s: %s", fy, page_url, exc)
            failures.append(f"FY{fy} page")
            continue
        if not selected:
            log.warning("FY%d: no .xlsx files found at %s", fy, page_url)
            failures.append(f"FY{fy} (no xlsx files)")
            continue
        kind = selected[0][1]
        log.info("FY%d: using %s (%d file%s)", fy,
                 "FY total file" if kind == "fy_total" else "monthly files",
                 len(selected), "" if len(selected) == 1 else "s")

        # If a year switches from monthly files to a total file between runs,
        # remove the monthly rows so nothing is double counted.
        keep = [f"FY{fy}/{Path(unquote(urlparse(u).path)).name}" for u, _, _ in selected]
        con.execute(
            f"DELETE FROM {TABLE} WHERE fiscal_year = ? AND NOT list_contains(?, source_file)",
            [fy, keep],
        )

        for url, kind, month in selected:
            filename = Path(unquote(urlparse(url).path)).name
            rel = f"FY{fy}/{filename}"
            dest = raw_dir / f"FY{fy}" / filename
            try:
                if download_file(session, url, dest):
                    time.sleep(args.delay)
                frames = read_workbook(dest)
                if not frames:
                    raise ValueError("no sheet had the expected header row")

                con.execute("BEGIN")
                con.execute(f"DELETE FROM {TABLE} WHERE source_file = ?", [rel])  # idempotent re-runs
                n = 0
                for df in frames:
                    load_dataframe(con, df, fiscal_year=fy, report_kind=kind,
                                   report_month=month, source_url=url, source_file=rel)
                    n += len(df)
                con.execute("COMMIT")
                files_loaded += 1
                rows_loaded += n
                log.info("  loaded %-45s %6d rows", rel, n)
            except Exception as exc:  # noqa: BLE001
                try:
                    con.execute("ROLLBACK")
                except duckdb.Error:
                    pass
                log.error("  FAILED %s: %s", rel, exc)
                failures.append(rel)

    total = con.execute(f"SELECT count(*) FROM {TABLE}").fetchone()[0]
    con.close()
    log.info("Done. %d file(s), %d rows this run; %d rows total in %s", files_loaded, rows_loaded, total, args.db)
    if failures:
        log.warning("%d failure(s): %s", len(failures), ", ".join(failures))
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())