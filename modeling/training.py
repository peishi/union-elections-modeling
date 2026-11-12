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
    "log_members",
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
        df = connection.execute(f'SELECT * FROM "{view}"').fetchdf()
        print(f"Loaded data has {len(df)} rows and {len(df.columns)} columns")
        return df
    finally:
        connection.close()


def dedupe_elections(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the view to one row per election.

    The view is already one row per election, but the finance join can rarely
    produce more than one row.  Prefer a row with a finance match.
    """
    df = (
        df.sort_values(["election_id", "rpt_id"], na_position="last")
        .drop_duplicates(subset="election_id", keep="first")
        .reset_index(drop=True)
    )
    print(f"Dropped duplicates by election id. Have {len(df)} rows")
    return df


def preprocess_filter_elections(df: pd.DataFrame) -> pd.DataFrame:
    """Remove elections that we don't want
    
    Should consider only single labor organization elections which is the vast majority.
    """
    relevant = df['ballot_type'].isin(['Single Labor Organization', 'Revised Single Labor Org'])
    print(f"Dropping {len(df) - sum(relevant)} rows where elections are not for single labor organization.")
    df = df[relevant]

    # Note that there are 2 cases with filing_status = 'Open' but with non-null date_closed. Include anyway.

    # NAs in vote cols are actually 0s
    vote_cols = ['votes_for_union', 'votes_against']
    df[vote_cols] = df[vote_cols].where(pd.notna(df[vote_cols]), 0)

    # Remove any rows where total votes is 0
    df['total_votes'] = df['votes_for_union'] + df['votes_against']
    print(f"Dropping {len(df) - sum(df['total_votes'] > 0)} rows where total votes is not >0.")
    df = df[df['total_votes'] > 0]

    print(f"Dropping {sum(df['ttl_disbursements'].isna())} rows where ttl_disbursements is null (no finance info)")
    df = df.dropna(subset=['ttl_disbursements'], axis=0)
    print(f"Dropping {sum(df['unit_size'].isna())} rows where unit_size is null (cannot calculate turnout)")
    df = df.dropna(subset=['unit_size'], axis=0)
    print(f"Dropping {sum(df['unit_size'].isna())} rows where members is null (cannot calculate per member spending)")
    df = df.dropna(subset=['members'], axis=0)

    return df


def calc_finance_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Calculated columns like spending per member by category
    
    Code the rows missing the spending data as zeros since they are smaller unions.
    Another idea is to calculate an average fraction of total disbursements the larger unions
    spend on representational and then just use that ratio for total of smaller unions
    """
    # Remove any zeros to prevent divide by zero
    nonzero = (df['ttl_disbursements'] != 0) & (df['members'] != 0)
    print(f"Dropping {sum(~nonzero)} rows where total disbursements or members is zero.")
    df = df[nonzero]

    # Log number of members
    df['log_members'] = np.log(df['members'])

    # Select rows where category breakdowns exist
    exist_by_cat = df[['representational', 'union_administration']].apply(
        lambda row: all([pd.notna(x) for x in row]), axis=1
    )
    fin_calc = df[exist_by_cat].copy()
    fin_calc['rep_per_mem'] = fin_calc['representational'] / fin_calc['members']
    fin_calc['admin_per_mem'] = fin_calc['union_administration'] / fin_calc['members']

    # Choosing cutoffs based on EDA of which unions are reporting spending at different levels
    cutoff_rep = fin_calc['rep_per_mem'] > 5000
    cutoff_admin = fin_calc['admin_per_mem'] > 10000
    fin_calc = fin_calc[ (~cutoff_rep) & (~cutoff_admin) ]

    # Calculate a median representational or administrative spending as fraction of total spending to use for smaller locals which do not report by category
    med_rep_frac = np.median(fin_calc['representational'] / fin_calc['ttl_disbursements'])
    med_admin_frac = np.median(fin_calc['union_administration'] / fin_calc['ttl_disbursements'])

    # Impute the value by using the median times the total disbursements
    no_fin_cat = df[~exist_by_cat].copy()
    no_fin_cat['rep_per_mem'] = no_fin_cat['ttl_disbursements'] * med_rep_frac / no_fin_cat['members']
    no_fin_cat['admin_per_mem'] = no_fin_cat['ttl_disbursements'] * med_admin_frac / no_fin_cat['members']

    all_fin = pd.concat([fin_calc, no_fin_cat], axis=0)
    
    # Take log to get better fit. Add 1 to prevent log blowing up. Means error of $1/mem of spending which is negligible.
    all_fin['log_rep_per_mem'] = np.log(all_fin['rep_per_mem'] + 1)
    all_fin['log_admin_per_mem'] = np.log(all_fin['admin_per_mem'] + 1)

    # Merge to original
    cols_to_merge = ['log_rep_per_mem', 'log_admin_per_mem', 'case_number']
    df = df.merge(all_fin[cols_to_merge], how='left', on='case_number')

    df = df.dropna(subset=cols_to_merge, axis=0)

    return df


def calc_election_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Election calculations: days from filing to election, turnout"""
    # Days to election
    df['days_to_election'] = (df['election_date'] - df['date_filed']).apply(
        lambda x: int( x / datetime.timedelta(days=1) ) if pd.notna(x) else x
    )
    invalid = df['days_to_election'] < 0
    df.loc[invalid, 'days_to_election'] = None
    df.loc[~invalid, 'log_days_to_election'] = np.log(df.loc[~invalid, 'days_to_election'] + 0.1)  # 0.1 day as buffer for 0s

    # Turnout
    # Drop rows where eligible != unit size then otherwise use unit size for divisor.
    mismatch = df[['unit_size', 'filing_number_of_eligible_voters']].apply(
        lambda x: x['unit_size'] != x['filing_number_of_eligible_voters'] if all(x.notna()) else False, axis=1
    )
    df = df[~mismatch]
    nonzero = df['unit_size'] > 0
    df.loc[nonzero, 'turnout'] = df.loc[nonzero, 'total_votes'] / df.loc[nonzero, 'unit_size']
    invalid = df['turnout'] > 1.0
    df.loc[invalid, 'turnout'] = None

    df = df.dropna(subset=['turnout', 'days_to_election'], axis=0)

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
    return data[FEATURES + ["success"]].copy()


def get_linear_regression(df: pd.DataFrame) -> pd.DataFrame:
    """"""
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)

    print(f"Test: {len(df_test)} rows, Train: {len(df_train)} rows, Validation: {len(df_val)}")

    y_train =  df_train['net_margin'].values
    y_val = df_val['net_margin'].values
    y_test = df_test['net_margin'].values

    X_train, X_val, X_test = df_train[FEATURES], df_val[FEATURES], df_test[FEATURES]

    # Just to be sure
    del df_train['net_margin']
    del df_val['net_margin']
    del df_test['net_margin']

    


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

    get_linear_regression(data_reg)

    
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
