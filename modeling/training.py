#!/usr/bin/env python3
"""Load the analytical election view and prepare modeling datasets.

Targets
-------
Regression
    ``vote_ratio = votes_for_labor_organization_1 / votes_cast_against_labor_org``

    Elections with zero votes against have an undefined ratio; those rows are
    left as NaN and dropped when the regression dataset is built.

Classification
    ``success = reason_closed == "Certific. of Representative"``

No model is fitted here yet.  This module only loads the view, collapses it to
one row per election, and builds the feature/target frames.

Usage
-----
    uv run python modeling/training.py
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "data" / "union_elections.duckdb"
VIEW_NAME = "analytical_elections"
SUCCESS_REASON = "Certific. of Representative"

# Candidate predictors.  Categorical columns are left as strings here; encoding
# is a modeling step.
FEATURES = [
    # filing / election
    "case_type",
    "state",
    "region_assigned",
    "filing_status",
    "ballot_type",
    "unit_size",
    "filing_number_of_eligible_voters",
    "number_of_voters_on_petition_or_charge",
    "election_mode",
    # union finance
    "ttl_receipts",
    "ttl_disbursements",
    "representational",
    "political",
    "contributions",
    "general_overhead",
    "union_administration",
    "finance_aff_abbr",
    "form_type",
]

CATEGORICAL_FEATURES = [
    "case_type",
    "state",
    "region_assigned",
    "filing_status",
    "ballot_type",
    "election_mode",
    "finance_aff_abbr",
    "form_type",
]


def load_view(db_path: Path = DEFAULT_DB, view: str = VIEW_NAME) -> pd.DataFrame:
    """Read the analytical view into a DataFrame."""
    connection = duckdb.connect(str(db_path), read_only=True)
    try:
        return connection.execute(f'SELECT * FROM "{view}"').fetchdf()
    finally:
        connection.close()


def dedupe_elections(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the view to one row per election.

    The view can hold several ``election_mode`` records (and, rarely, several
    finance reports) per election.  Prefer the first election-mode row and keep
    the first remaining row per ``election_id``.
    """
    return (
        df.sort_values(["election_id", "mode_row_number"], na_position="last")
        .drop_duplicates(subset="election_id", keep="first")
        .reset_index(drop=True)
    )


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add the regression and classification targets."""
    df = df.copy()

    against = df["votes_cast_against_labor_org"].replace(0, np.nan)
    df["vote_ratio"] = df["votes_for_labor_organization_1"] / against

    df["success"] = (
        df["reason_closed"]
        .map(lambda value: pd.NA if pd.isna(value) else int(value == SUCCESS_REASON))
        .astype("Int64")
    )
    return df


def regression_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Feature frame and ``vote_ratio`` target, dropping undefined ratios."""
    data = add_targets(df).dropna(subset=["vote_ratio"])
    return data[FEATURES].copy(), data["vote_ratio"].copy()


def classification_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Feature frame and binary ``success`` target."""
    data = add_targets(df).dropna(subset=["success"])
    return data[FEATURES].copy(), data["success"].astype(int).copy()


def main() -> int:
    elections = dedupe_elections(load_view())

    X_reg, y_reg = regression_dataset(elections)
    X_clf, y_clf = classification_dataset(elections)

    print(f"Elections in view: {len(elections):,}")
    print(
        f"Regression: {len(X_reg):,} rows, "
        f"vote_ratio {y_reg.min():.2f} .. {y_reg.max():.2f}"
    )
    print(
        f"Classification: {len(X_clf):,} rows, "
        f"success rate {y_clf.mean():.1%}"
    )
    print()
    print("Feature columns:", ", ".join(FEATURES))
    print("Categorical columns:", ", ".join(CATEGORICAL_FEATURES))
    print()
    print(X_reg.head())

    # TODO: encode categoricals and fit/evaluate regression and classification.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
