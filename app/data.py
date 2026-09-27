import numpy as np
import pandas as pd

from .config import (RAW_X_PATH, RAW_Y_PATH, PROCESSED_PATH, LOAN_TERM,
                     TRAIN_END, VALID_END, TEST_END, FEATURES, TARGETS)

MONTHS = {m: i for i, m in enumerate(
    ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], start=1)}

EMP_LENGTH_YEARS = {
    '< 1 year': 0, '1 year': 1, '10+ years': 10,
    **{f'{n} years': n for n in range(2, 10)}
}


def load_raw(x_path=RAW_X_PATH, y_path=RAW_Y_PATH):
    df = pd.concat([pd.read_csv(x_path, low_memory=False), pd.read_csv(y_path)], axis=1)
    return df.rename(columns={'y': 'default'})


def add_issue_period(df):
    df['issue_period'] = df['issue_date_year'] * 100 + df['issue_date_month'].map(MONTHS)
    return df


def build_targets(df):
    # total received = principal + interest + late fees + post charge-off recoveries
    df['recoveries'] = (df['paym_rec_for_tot_amnt_fund'] - df['princ_rec']
                        - df['interest_rec'] - df['late_fees_rec']).clip(lower=0)
    df['ead_default'] = (df['funded_amnt'] - df['princ_rec']).clip(lower=0)

    resolved = df['remaining_princ_for_tot_amnt_fund'] < 1
    has_lgd = (df['default'] == 1) & resolved & (df['ead_default'] > 0)

    df['lgd'] = np.nan
    df.loc[has_lgd, 'lgd'] = (1 - df.loc[has_lgd, 'recoveries'] / df.loc[has_lgd, 'ead_default']).clip(0, 1)
    df.loc[df['default'] == 0, ['ead_default', 'recoveries']] = 0.0

    # a non-default with principal outstanding has no final label yet
    return df[resolved | (df['default'] == 1)]


def build_features(df):
    df['emp_length_years'] = df['emp_length'].map(EMP_LENGTH_YEARS)

    earliest = df['earliest_cr_line_year'] + (df['earliest_cr_line_month'].map(MONTHS) - 1) / 12
    issued   = df['issue_date_year'] + (df['issue_date_month'].map(MONTHS) - 1) / 12
    df['credit_history_years'] = (issued - earliest).round(2)

    df['dept_paym_income_ratio'] = df['dept_paym_income_ratio'].where(df['dept_paym_income_ratio'] >= 0)
    return df


def assign_split(period):
    return np.select([period <= TRAIN_END, period <= VALID_END, period <= TEST_END],
                     ['train', 'valid', 'test'], default='out_of_scope')


def prepare(df):
    df = df[df['loan_term_months'] == LOAN_TERM].copy()
    df = add_issue_period(df)
    df = build_targets(df)
    df = build_features(df)

    df['split'] = assign_split(df['issue_period'])
    df = df[df['split'] != 'out_of_scope']

    return df[FEATURES + TARGETS + ['issue_period', 'split']].reset_index(drop=True)


def load_processed(path=PROCESSED_PATH):
    return pd.read_parquet(path)


def main():
    try:
        raw = load_raw()
    except FileNotFoundError:
        print(f"[ERROR] Raw data not found: {RAW_X_PATH}, {RAW_Y_PATH}")
        return

    df = prepare(raw)
    df.to_parquet(PROCESSED_PATH, index=False)

    summary = df.groupby('split').agg(
        loans=('default', 'size'),
        default_rate=('default', 'mean'),
        lgd_obs=('lgd', 'count'),
        lgd_mean=('lgd', 'mean'),
        first_issue=('issue_period', 'min'),
        last_issue=('issue_period', 'max')
    ).reindex(['train', 'valid', 'test'])

    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nSaved {len(df):,} loans x {df.shape[1]} columns at {PROCESSED_PATH}")


if __name__ == "__main__":
    main()
