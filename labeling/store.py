"""DuckDB storage for NAICS labels.

The ``llm_naics_labels`` table is keyed by ``employer_name`` and stores the two
most likely 2-digit sectors and 3-digit subsectors as JSON arrays.  The batch
pipeline (``llm_pipeline.py``) populates it; the interactive review app can also
write a manual 2-digit label (``model='manual'``).
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Iterable, Mapping

import duckdb

from sampler import sample_unlabeled

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "data" / "union_elections.duckdb"

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


class LabelStore:
    """Thread-safe wrapper around the labels table in the project database."""

    def __init__(self, db_path: Path | str = DEFAULT_DB) -> None:
        self.db_path = Path(db_path).expanduser()
        if not self.db_path.exists():
            raise SystemExit(f"Database not found: {self.db_path}")
        self._lock = threading.Lock()
        self._connection = duckdb.connect(str(self.db_path))
        with self._lock:
            self._ensure_tables()

    def _ensure_tables(self) -> None:
        views = {
            row[0]
            for row in self._connection.execute(
                "SELECT view_name FROM duckdb_views() WHERE NOT internal"
            ).fetchall()
        }
        tables = {
            row[0]
            for row in self._connection.execute(
                "SELECT table_name FROM duckdb_tables()"
            ).fetchall()
        }
        if "analytical_elections" not in views | tables:
            raise SystemExit(
                "analytical_elections is missing; run "
                "transformations/build_analytical_view.py first."
            )
        self._connection.execute(SCHEMA_SQL)

    def sample(self, n: int) -> list[dict[str, object]]:
        """Return up to ``n`` unlabeled employers (one case each)."""
        with self._lock:
            return sample_unlabeled(self._connection, n)

    def save_many(self, labels: Iterable[Mapping[str, object]]) -> int:
        """Persist manual 2-digit labels (``model='manual'``), one row per employer."""
        items = list(labels)
        if not items:
            return 0
        with self._lock:
            self._connection.execute("BEGIN TRANSACTION")
            try:
                for item in items:
                    naics_2 = json.dumps(
                        [
                            {
                                "code": item["naics"],
                                "title": item.get("naics_title"),
                                "probability": None,
                            }
                        ]
                    )
                    self._connection.execute(
                        UPSERT_SQL,
                        [
                            item["employer_name"],
                            naics_2,
                            None,
                            None,
                            None,
                            None,
                            "manual",
                            0,
                            0,
                            0,
                        ],
                    )
                self._connection.execute("COMMIT")
            except Exception:
                self._connection.execute("ROLLBACK")
                raise
        return len(items)

    def progress(self) -> dict[str, int]:
        with self._lock:
            row = self._connection.execute(
                """
                SELECT
                    (SELECT count(DISTINCT "employer_name")
                       FROM "llm_naics_labels"),
                    (SELECT count(DISTINCT "employer_name")
                       FROM "analytical_elections"
                      WHERE "employer_name" IS NOT NULL)
                """
            ).fetchone()
        return {"labeled": int(row[0]), "total": int(row[1])}

    def close(self) -> None:
        with self._lock:
            self._connection.close()
