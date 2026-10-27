"""Clean the mortgage dataset without training a model."""
from pathlib import Path

import pandas as pd

from src.data import DATA_FILE


CLEANED_DATA_FILE = DATA_FILE.with_name('mortgage_data_cleaned.csv')
NUMERIC_COLUMNS = (
    'annual_income_gbp',
    'property_value_gbp',
    'deposit_gbp',
    'requested_loan_gbp',
    'loan_to_value_pct',
    'monthly_debt_commitments_gbp',
    'debt_to_income_ratio',
    'credit_score',
    'employment_years',
    'savings_gbp',
    'credit_utilisation_pct',
    'interest_rate_pct',
)
STRING_COLUMNS = ('employment_type', 'application_channel')
BINARY_COLUMNS = ('previous_default_flag', 'repayment_difficulty_12m')
SELECTED_COLUMNS = (
    'application_id',
    'credit_score',
    'debt_to_income_ratio',
    'loan_to_value_pct',
    'repayment_difficulty_12m',
)


def convert_numeric_columns(df):
    """Convert required numeric fields from strings to numeric values."""
    cleaned = df.copy()
    available_columns = [
        column for column in NUMERIC_COLUMNS if column in cleaned.columns
    ]
    cleaned[available_columns] = cleaned[available_columns].apply(
        pd.to_numeric, errors='coerce'
    )
    return cleaned


def remove_whitespace(df):
    """Strip leading and trailing whitespace from column names and text values."""
    cleaned = df.copy()
    cleaned.columns = cleaned.columns.str.strip()
    text_columns = cleaned.select_dtypes(include=['object', 'string']).columns
    for column in text_columns:
        cleaned[column] = cleaned[column].str.strip()
    return cleaned


def remove_duplicate_ids(df):
    """Remove duplicate application IDs, keeping the first record."""
    if 'application_id' not in df.columns:
        raise ValueError('Required column missing: application_id')
    return df.drop_duplicates(subset='application_id', keep='first').reset_index(
        drop=True
    )


def select_cleaned_columns(df):
    """Keep the identifier, selected predictors, and observed target."""
    missing_columns = set(SELECTED_COLUMNS) - set(df.columns)
    if missing_columns:
        raise ValueError(f'Required output columns missing: {sorted(missing_columns)}')
    return df.loc[:, SELECTED_COLUMNS].copy()


def convert_string_columns(df):
    """Convert required categorical fields to string values."""
    cleaned = df.copy()
    available_columns = [
        column for column in STRING_COLUMNS if column in cleaned.columns
    ]
    cleaned[available_columns] = cleaned[available_columns].astype('string')
    return cleaned


def convert_binary_columns(df):
    """Convert required flags to numeric values and validate 0/1 values."""
    cleaned = df.copy()
    available_columns = [
        column for column in BINARY_COLUMNS if column in cleaned.columns
    ]
    cleaned[available_columns] = cleaned[available_columns].apply(
        pd.to_numeric, errors='coerce'
    )
    invalid_values = {
        column: sorted(cleaned.loc[
            cleaned[column].notna() & ~cleaned[column].isin({0, 1}), column
        ].unique().tolist())
        for column in available_columns
    }
    invalid_values = {
        column: values for column, values in invalid_values.items() if values
    }
    if invalid_values:
        raise ValueError(f'Binary columns must contain only 0 or 1: {invalid_values}')
    return cleaned


def validate_cleaned_columns(df):
    """Confirm required numeric, string, and binary columns are valid."""
    required_columns = NUMERIC_COLUMNS + STRING_COLUMNS + BINARY_COLUMNS
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        raise ValueError(f'Required cleaning columns missing: {sorted(missing_columns)}')
    if not all(pd.api.types.is_numeric_dtype(df[column]) for column in NUMERIC_COLUMNS):
        raise ValueError('Numeric fields were not converted successfully')
    if not all(pd.api.types.is_string_dtype(df[column]) for column in STRING_COLUMNS):
        raise ValueError('String fields were not converted successfully')
    if not all(df[column].isin({0, 1}).all() for column in BINARY_COLUMNS):
        raise ValueError('Binary fields must contain only 0 or 1')


def fill_missing_with_column_medians(df):
    """Replace missing numeric values with each column's median."""
    cleaned = remove_whitespace(df)
    cleaned = remove_duplicate_ids(cleaned)
    cleaned = convert_numeric_columns(cleaned)
    cleaned = convert_string_columns(cleaned)
    cleaned = convert_binary_columns(cleaned)
    numeric_columns = cleaned.select_dtypes(include='number').columns

    for column in numeric_columns:
        if cleaned[column].isna().any():
            cleaned[column] = cleaned[column].fillna(cleaned[column].median())

    return cleaned


def save_cleaned_data(input_path=DATA_FILE, output_path=CLEANED_DATA_FILE):
    """Load, clean, and save the dataset without fitting a model."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    raw_data = pd.read_csv(input_path)
    cleaned_data = fill_missing_with_column_medians(raw_data)
    validate_cleaned_columns(cleaned_data)
    cleaned_data = select_cleaned_columns(cleaned_data)
    cleaned_data.to_csv(output_path, index=False)
    return cleaned_data


if __name__ == '__main__':
    raw_data = pd.read_csv(DATA_FILE)
    cleaned_data = save_cleaned_data()
    print('Missing values before cleaning:', int(raw_data.isna().sum().sum()))
    print('Missing values after cleaning:', int(cleaned_data.isna().sum().sum()))
    print('Numeric fields validated successfully.')
    print('String fields validated successfully.')
    print('Binary fields validated successfully.')
    print('Whitespace cleaned successfully.')
    print('Duplicate application IDs handled successfully.')
    print('Selected columns validated successfully.')
    print('All data cleaning functions completed successfully.')
    print(f'Cleaned data saved to: {CLEANED_DATA_FILE}')
