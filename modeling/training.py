#!/usr/bin/env python3
"""Load the analytical election view and prepare modeling datasets.

Targets
-------
Regression
    ``net_margin = votes_for_union - votes_against / <total_votes_cast>```

Classification
    ``success = reason_closed == "Certific. of Representative"``

Usage
-----
    uv run python modeling/training.py
"""

from __future__ import annotations

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer

import duckdb
import numpy as np
import pandas as pd
import datetime

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
    "unit_size",
    "turnout",
    "days_to_election",
    # union finance
    "log_members"
    "log_rep_per_mem",
    "log_admin_per_mem",
    "finance_aff_abbr",
]

CATEGORICAL_FEATURES = [
    "case_type",
    "state",
    "finance_aff_abbr",
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

    The view is already one row per election, but the finance join can rarely
    produce more than one row.  Prefer a row with a finance match.
    """
    return (
        df.sort_values(["election_id", "rpt_id"], na_position="last")
        .drop_duplicates(subset="election_id", keep="first")
        .reset_index(drop=True)
    )


def preprocess_filter_elections(df: pd.DataFrame) -> pd.DataFrame:
    """Remove elections that we don't want
    
    Should consider only single labor organization elections which is the vast majority.
    """
    relevant = df['ballot_type'].isin(['Single Labor Organization', 'Revised Single Labor Org'])
    df = df[relevant]

    # Note that there are 2 cases with filing_status = 'Open' but with non-null date_closed. Include anyway.

    # NAs in vote cols are actually 0s
    vote_cols = ['votes_for_union', 'votes_against']
    df[vote_cols] = df[vote_cols].where(pd.notna(df[vote_cols]), 0)

    # Remove any rows where total votes is 0
    df['total_votes'] = df['votes_for_union'] + df['votes_against']
    df = df[df['total_votes'] > 0]

    return df


def calc_finance_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Calculated columns like spending per member by category"""
    # First only select rows where the relevant columns are not null. These are LM-2 reports
    exist_finance = df[['ttl_disbursements', 'representational', 'political', 'members']].apply(
        lambda row: all([pd.notna(x) for x in row]), axis=1
    )
    # Prevent divide by zero errors
    nonzero = (df['ttl_disbursements'] != 0) & (df['members'] != 0)
    fin_calc = df[exist_finance & nonzero].copy()

    fin_calc['rep_per_mem'] = fin_calc['representational'] / fin_calc['members']
    fin_calc['admin_per_mem'] = fin_calc['union_administration'] / fin_calc['ttl_disbursements']

    # Choosing cutoffs based on EDA of which unions are reporting spending at different levels
    cutoff_rep = fin_calc['rep_per_mem'] > 5000
    cutoff_admin = fin_calc['admin_per_mem'] > 10000
    fin_calc = fin_calc[ (~cutoff_rep) & (~cutoff_admin) ]

    # Take log to get better fit. Add 1 to prevent log blowing up. Means error of $1/mem of spending which is negligible.
    fin_calc['log_rep_per_mem'] = np.log(fin_calc['rep_per_mem'] + 1)
    fin_calc['log_admin_per_mem'] = np.log(fin_calc['admin_per_mem'] + 1)

    # Log number of members
    fin_calc['log_members'] = np.log(fin_calc['members'])

    # Merge to original
    cols_to_merge = ['log_rep_per_mem', 'log_admin_per_mem', 'case_number']
    df = df.merge(fin_calc[cols_to_merge], how='left', on='case_number')

    return df


def calc_election_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Election calculations: days from filing to election, turnout"""
    # Days to election
    df['days_to_election'] = int(
        (df['election_date'] - df['date_filed']) / datetime.timedelta(days=1)
    )
    invalid = df['days_to_election'] < 0
    df.loc[invalid, 'days_to_election'] = None
    df.loc[~invalid, 'log_days_to_election'] = np.log(df.loc[~invalid, 'days_to_election'] + 0.1)  # 0.1 day as buffer for 0s

    # Turnout
    nonzero = df['filing_number_of_eligible_voters'] > 0
    df.loc[nonzero, 'turnout'] = df.loc[nonzero, 'total_votes'] / df.loc[nonzero, 'filing_number_of_eligible_voters']
    invalid = df['turnout'] > 1.0
    df.loc[invalid, 'turnout'] = None

    return df


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Add the regression and classification targets."""
    df = df.copy()

    df["net_margin"] = (df["votes_for_union"] - df["votes_against"]) / df["total_votes"]

    df["success"] = (
        df["reason_closed"]
        .map(lambda value: pd.NA if pd.isna(value) else int(value == SUCCESS_REASON))
        .astype("Int64")
    )
    return df


def regression_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Feature frame and ``vote_ratio`` target, dropping undefined."""
    data = add_targets(df).dropna(subset=["net_margin"])
    return data[FEATURES + ["net_margin"]].copy()


def classification_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Feature frame and binary ``success`` target."""
    data = add_targets(df).dropna(subset=["success"])
    return data[FEATURES + "success"].copy()


def main() -> int:
    elections = dedupe_elections(load_view())
    elections = preprocess_filter_elections(elections)

    elections = calc_finance_cols(elections)
    elections = calc_election_cols(elections)

    data_reg = regression_dataset(elections)
    data_clf = classification_dataset(elections)

    print()
    print("Feature columns:", ", ".join(FEATURES))
    print("Categorical columns:", ", ".join(CATEGORICAL_FEATURES))
    print()

    
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
