#!/usr/bin/env python3
"""Batch-label unique employers in ``analytical_elections`` with NAICS codes.

By default each employer is classified in **two steps** to keep prompts small:

1. ask for the two most likely 2-digit sectors (20-sector catalog);
2. ask for the two most likely 3-digit subsectors, listing only the subsectors
   that fall under the sectors chosen in step 1.

Both steps use the employer name, filing year, city, state and bargaining unit
description.  The static catalogs come first in each prompt and the employer
details last, so the cacheable prefix is shared across calls (this helps the
``claude-haiku-5-5`` fallback, which supports prompt caching; ``gemini-4-argon``
does not cache).  Pass ``--one-step`` for the original single-call prompt.

Results are written incrementally to ``llm_naics_labels`` (one row per
employer, upserted).  By default already-labeled employers are skipped, so the
run is resumable; ``--overwrite`` re-queries and replaces existing rows.  The
old interactive ``naics_labels`` table is dropped.

Primary model is ``gemini-4-argon``; on exhaustion the client switches to
``claude-haiku-5-5`` and records the model that produced each row.  Token
consumption is accumulated from the API responses and printed at the end.

Usage
-----
    uv run python labeling/llm_pipeline.py --limit 20        # small test run
    uv run python labeling/llm_pipeline.py                    # all employers
    uv run python labeling/llm_pipeline.py --overwrite        # replace labels
    uv run python labeling/llm_pipeline.py --one-step         # single-call mode
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import duckdb

from llm import DEFAULT_BASE_URL, DEFAULT_FALLBACK_MODEL, DEFAULT_MODEL, LLMClient

LOG = logging.getLogger("naics-pipeline")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "data" / "union_elections.duckdb"
SECTORS_FILE = Path(__file__).resolve().parent / "static" / "naics_sectors.json"
SUBSECTORS_FILE = Path(__file__).resolve().parent / "naics_subsectors.json"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS "llm_naics_labels" (
    "employer_name"      VARCHAR PRIMARY KEY,
    "naics_2"            VARCHAR,
    "naics_3"            VARCHAR,
    "rationale"          VARCHAR,
    "query_text"         VARCHAR,
    "raw_response"       VARCHAR,
    "model"              VARCHAR,
    "prompt_tokens"      BIGINT,
    "completion_tokens"  BIGINT,
    "total_tokens"       BIGINT,
    "created_at"         TIMESTAMP DEFAULT now(),
    "updated_at"         TIMESTAMP DEFAULT now()
)
"""

UPSERT_SQL = """
INSERT OR REPLACE INTO "llm_naics_labels" (
    "employer_name", "naics_2", "naics_3", "rationale", "query_text",
    "raw_response", "model", "prompt_tokens", "completion_tokens",
    "total_tokens", "updated_at"
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, now())
"""


# ---------------------------------------------------------------------------
# Catalog helpers
# ---------------------------------------------------------------------------


def load_catalogs() -> dict[str, Any]:
    with SECTORS_FILE.open(encoding="utf-8") as handle:
        sectors = json.load(handle)["sectors"]
    with SUBSECTORS_FILE.open(encoding="utf-8") as handle:
        subsectors = json.load(handle)["subsectors"]
    return {
        "sector_titles": {str(s["code"]): str(s["title"]) for s in sectors},
        "sectors": sectors,
        "subsector_titles": {str(s["code"]): str(s["title"]) for s in subsectors},
        "subsectors": subsectors,
    }


def _employer_block(employer: dict[str, Any]) -> str:
    location = ", ".join(
        part for part in (employer.get("city"), employer.get("state")) if part
    ) or "unknown"
    year = employer.get("filing_year") or "unknown"
    description = employer.get("voting_unit_description") or "unknown"
    return (
        f"Name: {employer['employer_name']}\n"
        f"Filing year: {year}\n"
        f"Location: {location}\n"
        f"Bargaining unit description: {description}"
    )


def build_step1_prompt(employer: dict[str, Any], catalogs: dict[str, Any]) -> str:
    """First step: choose the two most likely 2-digit sectors."""
    sector_lines = "\n".join(
        f'{s["code"]}: {s["title"]}' for s in catalogs["sectors"]
    )
    return (
        "You are classifying a U.S. employer into the two most likely 2022 "
        "NAICS 2-digit sectors.\n\n"
        "Two-digit sectors (choose from these):\n"
        f"{sector_lines}\n\n"
        "Return ONLY a JSON object with exactly this shape:\n"
        '{"naics_2": [{"code": "<sector>", "probability": <0..1>}, '
        '{"code": "<sector>", "probability": <0..1>}], '
        '"rationale": "<one short sentence>"}\n'
        "The two probabilities must sum to 1. No markdown, no extra text.\n\n"
        "--- Employer ---\n"
        f"{_employer_block(employer)}\n"
    )


def build_step2_prompt(
    employer: dict[str, Any],
    sector_codes: Sequence[str],
    catalogs: dict[str, Any],
) -> str:
    """Second step: choose the two most likely subsectors within the sectors."""
    allowed = [
        s for s in catalogs["subsectors"] if s["sector"] in sector_codes
    ]
    subsector_lines = "\n".join(f'{s["code"]} {s["title"]}' for s in allowed)
    return (
        "You are classifying a U.S. employer into the two most likely 2022 "
        "NAICS 3-digit subsectors.\n\n"
        f"Chosen 2-digit sectors: {', '.join(sector_codes)}\n"
        "Three-digit subsectors to choose from:\n"
        f"{subsector_lines}\n\n"
        "Return ONLY a JSON object with exactly this shape:\n"
        '{"naics_3": [{"code": "<subsector>", "probability": <0..1>}, '
        '{"code": "<subsector>", "probability": <0..1>}], '
        '"rationale": "<one short sentence>"}\n'
        "The two probabilities must sum to 1. No markdown, no extra text.\n\n"
        "--- Employer ---\n"
        f"{_employer_block(employer)}\n"
    )


def build_one_step_prompt(
    employer: dict[str, Any], catalogs: dict[str, Any]
) -> str:
    """Original single-call prompt: both levels at once (for --one-step)."""
    sector_lines = "\n".join(
        f'{s["code"]}: {s["title"]}' for s in catalogs["sectors"]
    )
    subsector_lines = "\n".join(
        f'{s["code"]} {s["title"]} [sector {s["sector"]}]'
        for s in catalogs["subsectors"]
    )
    return (
        "You are classifying a U.S. employer into the North American Industry "
        "Classification System (NAICS 2022).\n\n"
        "Two-digit sectors (choose from these):\n"
        f"{sector_lines}\n\n"
        "Three-digit subsectors (choose from these):\n"
        f"{subsector_lines}\n\n"
        "Give the two most likely 2-digit sectors and the two most likely "
        "3-digit subsectors. Each 3-digit code must be a subsector of one of "
        "your chosen 2-digit sectors.\n\n"
        "Return ONLY a JSON object with exactly this shape:\n"
        '{\n'
        '  "naics_2": [{"code": "<sector>", "probability": <0..1>}, '
        '{"code": "<sector>", "probability": <0..1>}],\n'
        '  "naics_3": [{"code": "<subsector>", "probability": <0..1>}, '
        '{"code": "<subsector>", "probability": <0..1>}],\n'
        '  "rationale": "<one short sentence>"\n'
        '}\n'
        "The two probabilities in each list must sum to 1. No markdown, no "
        "extra text.\n\n"
        "--- Employer ---\n"
        f"{_employer_block(employer)}\n"
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def _parse_probability(value: Any) -> float | None:
    if value is None:
        return None
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    if 1.0 < probability <= 100.0:
        probability = probability / 100.0
    return max(0.0, min(1.0, probability))


def _normalise_level(
    items: Any, titles: dict[str, str]
) -> list[dict[str, Any]]:
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return []
    seen: dict[str, dict[str, Any]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        code = str(item.get("code", "")).strip()
        if code not in titles:
            continue
        candidate = {
            "code": code,
            "title": titles[code],
            "probability": _parse_probability(item.get("probability")),
        }
        if code not in seen:
            seen[code] = candidate
    ranked = sorted(
        seen.values(),
        key=lambda c: c["probability"] if c["probability"] is not None else -1.0,
        reverse=True,
    )
    return ranked[:2]


def parse_response(
    content: str, catalogs: dict[str, Any]
) -> tuple[list[dict], list[dict], str]:
    text = content.strip()
    text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    start, end = text.find("{"), text.rfind("}")
    payload: dict[str, Any] = {}
    if start != -1 and end > start:
        try:
            payload = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            payload = {}
    naics_2 = _normalise_level(payload.get("naics_2"), catalogs["sector_titles"])
    naics_3 = _normalise_level(
        payload.get("naics_3"), catalogs["subsector_titles"]
    )
    rationale = str(payload.get("rationale", "")).strip()
    return naics_2, naics_3, rationale


# ---------------------------------------------------------------------------
# Database access
# ---------------------------------------------------------------------------


class LabelWriter:
    """Serialises DuckDB writes from worker threads."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path).expanduser()
        if not self.db_path.exists():
            raise SystemExit(f"Database not found: {self.db_path}")
        self._lock = threading.Lock()
        self._connection = duckdb.connect(str(self.db_path))
        with self._lock:
            self._connection.execute('DROP TABLE IF EXISTS "naics_labels"')
            self._connection.execute(SCHEMA_SQL)

    def upsert(self, row: Sequence[Any]) -> None:
        with self._lock:
            self._connection.execute(UPSERT_SQL, list(row))

    def close(self) -> None:
        with self._lock:
            self._connection.close()


EMPLOYERS_SQL = """
WITH picked AS (
    SELECT
        a."employer_name",
        a."city",
        a."state",
        EXTRACT(YEAR FROM a."date_filed") AS "filing_year",
        a."voting_unit_description"
    FROM "analytical_elections" a
    WHERE a."employer_name" IS NOT NULL
    QUALIFY row_number() OVER (
        PARTITION BY a."employer_name"
        ORDER BY (a."voting_unit_description" IS NOT NULL) DESC,
                 a."case_number"
    ) = 1
)
SELECT "employer_name", "city", "state", "filing_year",
       "voting_unit_description"
FROM picked
ORDER BY "employer_name"
"""


def fetch_employers(
    db_path: Path, overwrite: bool, limit: int | None
) -> list[dict[str, Any]]:
    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        existing: set[str] = set()
        if not overwrite:
            try:
                existing = {
                    row[0]
                    for row in connection.execute(
                        'SELECT "employer_name" FROM "llm_naics_labels"'
                    ).fetchall()
                }
            except duckdb.CatalogException:
                existing = set()
        rows = connection.execute(EMPLOYERS_SQL).fetchall()
    finally:
        connection.close()

    employers = [
        {
            "employer_name": row[0],
            "city": row[1],
            "state": row[2],
            "filing_year": row[3],
            "voting_unit_description": row[4],
        }
        for row in rows
        if overwrite or row[0] not in existing
    ]
    if limit is not None:
        employers = employers[:limit]
    return employers


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------


def _is_fatal(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(
        word in message
        for word in ("exhaust", "quota", "usage limit", "insufficient", "http 429", "http 402")
    )


def _record(
    writer: LabelWriter,
    employer: dict[str, Any],
    naics_2: list[dict],
    naics_3: list[dict],
    rationale: str,
    query_text: str,
    raw_response: str,
    model: str,
    usage: dict[str, int],
) -> None:
    writer.upsert(
        (
            employer["employer_name"],
            json.dumps(naics_2),
            json.dumps(naics_3),
            rationale,
            query_text,
            raw_response,
            model,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            usage.get("total_tokens", 0),
        )
    )


def _sum_usage(*usages: dict[str, int]) -> dict[str, int]:
    total = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for usage in usages:
        for key in total:
            total[key] += int(usage.get(key, 0))
    return total


def process_one(
    client: LLMClient,
    writer: LabelWriter,
    employer: dict[str, Any],
    catalogs: dict[str, Any],
    max_tokens: int,
    stop_event: threading.Event,
    one_step: bool,
) -> dict[str, Any] | None:
    if stop_event.is_set():
        return None
    try:
        if one_step:
            prompt = build_one_step_prompt(employer, catalogs)
            result = client.chat(
                [{"role": "user", "content": prompt}], max_tokens=max_tokens
            )
            naics_2, naics_3, rationale = parse_response(result.content, catalogs)
            _record(
                writer, employer, naics_2, naics_3, rationale, prompt,
                result.content, result.model, result.usage,
            )
            model = result.model
            usage = result.usage
        else:
            step1_prompt = build_step1_prompt(employer, catalogs)
            step1 = client.chat(
                [{"role": "user", "content": step1_prompt}],
                max_tokens=max_tokens,
            )
            naics_2, _, rationale_1 = parse_response(step1.content, catalogs)
            sector_codes = [c["code"] for c in naics_2][:2] or sorted(
                catalogs["sector_titles"]
            )
            step2_prompt = build_step2_prompt(employer, sector_codes, catalogs)
            step2 = client.chat(
                [{"role": "user", "content": step2_prompt}],
                max_tokens=max_tokens,
            )
            _, naics_3, rationale_2 = parse_response(step2.content, catalogs)
            usage = _sum_usage(step1.usage, step2.usage)
            model = (
                step2.model
                if step1.model == step2.model
                else f"{step1.model} -> {step2.model}"
            )
            rationale = " | ".join(x for x in (rationale_1, rationale_2) if x)
            query_text = f"STEP 1:\n{step1_prompt}\n\nSTEP 2:\n{step2_prompt}"
            raw_response = json.dumps(
                {"step1": step1.content, "step2": step2.content}
            )
            _record(
                writer, employer, naics_2, naics_3, rationale, query_text,
                raw_response, model, usage,
            )
    except Exception as exc:  # noqa: BLE001 - logged and counted
        LOG.error("failed for %s: %s", employer["employer_name"], exc)
        if client.fallback_active and _is_fatal(exc):
            stop_event.set()
        return None

    return {
        "employer": employer["employer_name"],
        "model": model,
        "usage": usage,
        "parsed": bool(naics_2 and naics_3),
    }


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch-label employers with NAICS 2- and 3-digit codes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--limit", type=int, default=None, help="process at most N employers")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="re-query and replace existing labels",
    )
    parser.add_argument(
        "--one-step",
        action="store_true",
        help="use a single call with both catalogs instead of two steps",
    )
    parser.add_argument("--workers", type=int, default=4, help="concurrent LLM requests")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--fallback-model", default=DEFAULT_FALLBACK_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--max-tokens", type=int, default=1024)
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

    catalogs = load_catalogs()
    employers = fetch_employers(args.db, args.overwrite, args.limit)
    if not employers:
        LOG.info("No employers to label (use --overwrite to re-query).")
        return 0
    mode = "one-step" if args.one_step else "two-step"
    LOG.info(
        "Labeling %s employers (%s) with %s workers ...",
        len(employers),
        mode,
        args.workers,
    )

    client = LLMClient(
        model=args.model,
        fallback_model=args.fallback_model,
        base_url=args.base_url,
    )
    writer = LabelWriter(args.db)
    stop_event = threading.Event()
    counter = {"done": 0, "parsed": 0, "failed": 0}
    counter_lock = threading.Lock()

    def work(employer: dict[str, Any]) -> None:
        outcome = process_one(
            client, writer, employer, catalogs, args.max_tokens, stop_event,
            args.one_step,
        )
        with counter_lock:
            counter["done"] += 1
            if outcome is None:
                counter["failed"] += 1
            elif outcome["parsed"]:
                counter["parsed"] += 1
            done = counter["done"]
        if done % 25 == 0 or done == len(employers):
            LOG.info("progress: %s/%s", done, len(employers))

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = [executor.submit(work, employer) for employer in employers]
            for future in as_completed(futures):
                future.result()
    finally:
        writer.close()

    totals = client.total_usage()
    print("\n=== NAICS labeling run summary ===")
    print(f"Mode: {mode}")
    print(f"Employers processed: {counter['done']}")
    print(f"  parsed successfully: {counter['parsed']}")
    print(f"  failed: {counter['failed']}")
    print(f"API calls: {client.calls}")
    print(
        f"Token consumption: prompt={totals['prompt_tokens']:,} "
        f"completion={totals['completion_tokens']:,} "
        f"total={totals['total_tokens']:,}"
    )
    for model, bucket in sorted(client.usage_totals.items()):
        print(
            f"  {model}: prompt={bucket['prompt_tokens']:,} "
            f"completion={bucket['completion_tokens']:,} "
            f"total={bucket['total_tokens']:,}"
        )
    if stop_event.is_set():
        print("Run stopped early because the API appears exhausted.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
