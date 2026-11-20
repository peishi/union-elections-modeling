"""Sample unlabeled employers from ``analytical_elections``.

The sampler works at the employer level: it picks one representative case per
employer that has not yet been labeled, preferring a case that has a voting
unit description.  The outer ``ORDER BY random()`` randomizes which employers
are returned, so repeated calls give a fresh batch.
"""

from __future__ import annotations

import duckdb

SAMPLE_COLUMNS = (
    "case_number",
    "employer_name",
    "city",
    "state",
    "filing_year",
    "nlrb_url",
    "election_union_name",
    "voting_unit_description",
)

SAMPLE_SQL = """
WITH picked AS (
    SELECT
        a."case_number",
        a."employer_name",
        a."city",
        a."state",
        EXTRACT(YEAR FROM a."date_filed") AS "filing_year",
        f."url" AS "nlrb_url",
        a."election_union_name",
        a."voting_unit_description"
    FROM "analytical_elections" a
    LEFT JOIN "filing" f ON f."case_number" = a."case_number"
    WHERE a."employer_name" IS NOT NULL
      AND NOT EXISTS (
          SELECT 1
          FROM "llm_naics_labels" l
          WHERE l."employer_name" = a."employer_name"
      )
    QUALIFY row_number() OVER (
        PARTITION BY a."employer_name"
        ORDER BY (a."voting_unit_description" IS NOT NULL) DESC,
                 a."case_number"
    ) = 1
)
SELECT "case_number", "employer_name", "city", "state", "filing_year",
       "nlrb_url", "election_union_name", "voting_unit_description"
FROM picked
ORDER BY random()
LIMIT ?
"""


def sample_unlabeled(
    connection: duckdb.DuckDBPyConnection, n: int
) -> list[dict[str, object]]:
    """Return up to ``n`` unlabeled employers (one case each)."""
    rows = connection.execute(SAMPLE_SQL, [n]).fetchall()
    return [dict(zip(SAMPLE_COLUMNS, row)) for row in rows]
