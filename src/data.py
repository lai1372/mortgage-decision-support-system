"""Load and inspect the supplied synthetic mortgage observations.

Week 7 teaching scaffold. The audit provides evidence for learners to
interpret; it does not select features or complete the assessed data analysis.
"""
from pathlib import Path
import pandas as pd

DATA_FILE = Path(__file__).resolve().parents[1] / 'data' / 'mortgage_data.csv'
TARGET = 'repayment_difficulty_12m'


def load_data(path=DATA_FILE):
    """Load CSV and confirm the identifier and observed target exist."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'Mortgage dataset not found: {path}')
    df = pd.read_csv(path)
    missing = {'application_id', TARGET} - set(df.columns)
    if missing:
        raise ValueError(f'Required columns missing: {sorted(missing)}')
    if df[TARGET].isna().any() or not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError('Expected a complete binary repayment_difficulty_12m target')
    return df


def print_audit(df):
    """Show initial data-quality and mortgage consistency checks."""
    print('Rows and columns:', df.shape)
    print('\nData types:\n', df.dtypes.to_string())
    missing = pd.DataFrame({
        'count': df.isna().sum(),
        'percent': (df.isna().mean() * 100).round(2),
    })
    print('\nMissing values:\n', missing.to_string())
    target_counts = df[TARGET].value_counts().sort_index()
    target_summary = pd.DataFrame({
        'count': target_counts,
        'percent': (target_counts / len(df) * 100).round(2),
    })
    print('\nTarget distribution:\n', target_summary.to_string())
    print('\nDuplicate application IDs:', df['application_id'].duplicated().sum())

    numeric = df.select_dtypes(include='number')
    print('\nNumeric summary:\n', numeric.describe().T[['min', '50%', 'max']].to_string())

    print('\nPotential IQR outliers:')
    for column in numeric.columns:
        q1, q3 = numeric[column].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = ((numeric[column] < q1 - 1.5 * iqr) |
                    (numeric[column] > q3 + 1.5 * iqr)).sum()
        print(f'{column}: {int(outliers)}')

    loan_to_income = df['requested_loan_gbp'] / df['annual_income_gbp']
    q1, q3 = loan_to_income.quantile([0.25, 0.75])
    iqr_multiplier = 3.5
    upper_fence = q3 + iqr_multiplier * (q3 - q1)
    high_loan_to_income = df.loc[loan_to_income > upper_fence, [
        'application_id', 'annual_income_gbp', 'requested_loan_gbp'
    ]].copy()
    high_loan_to_income['loan_to_income_ratio'] = loan_to_income[
        high_loan_to_income.index
    ]
    print(f'\nHigh loan-to-income applications (above {iqr_multiplier} * IQR):')
    if high_loan_to_income.empty:
        print('None')
    else:
        print(high_loan_to_income.to_string(index=False, formatters={
            'annual_income_gbp': '{:,.0f}'.format,
            'requested_loan_gbp': '{:,.0f}'.format,
            'loan_to_income_ratio': '{:.2f}x'.format,
        }))

    range_checks = {
        'credit_score outside 300-850':
            ((df['credit_score'] < 300) | (df['credit_score'] > 850)).sum(),
        'loan_to_value_pct outside 0-100':
            ((df['loan_to_value_pct'] < 0) | (df['loan_to_value_pct'] > 100)).sum(),
        'debt_to_income_ratio outside 0-1':
            ((df['debt_to_income_ratio'] < 0) | (df['debt_to_income_ratio'] > 1)).sum(),
        'credit_utilisation_pct outside 0-100':
            ((df['credit_utilisation_pct'] < 0) |
             (df['credit_utilisation_pct'] > 100)).sum(),
    }
    print('\nRange checks:')
    for check, count in range_checks.items():
        print(f'{check}: {int(count)}')

    ltv_difference = (
        df['loan_to_value_pct'] -
        df['requested_loan_gbp'] / df['property_value_gbp'] * 100
    ).abs()
    loan_balance_difference = (
        df['requested_loan_gbp'] + df['deposit_gbp'] - df['property_value_gbp']
    ).abs()
    print('\nMortgage consistency checks:')
    print('Loan plus deposit differs from property value by more than GBP 1:',
          int((loan_balance_difference > 1).sum()))
    print('Reported LTV differs from calculated LTV by more than 1 percentage point:',
          int((ltv_difference > 1).sum()))
    print('Categorical values:')
    for column in ('employment_type', 'application_channel'):
        print(f'{column}: {df[column].value_counts(dropna=False).to_dict()}')


if __name__ == '__main__':
    print_audit(load_data())
