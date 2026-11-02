"""Run the supplied baseline only; assessed advisory rules are NOT provided."""
from pathlib import Path
import pandas as pd
import joblib
from src.data_cleaning import CLEANED_DATA_FILE
from src.model import BASELINE_FEATURES, MODEL_FILE, run_baseline
from src.visualisations import plot_all_visualizations
from sklearn.model_selection import train_test_split
from src.data import TARGET


def report(name, result):
    print(f'\n{name}')
    for metric in ('accuracy', 'precision_1', 'recall_1', 'f1_1', 'roc_auc'):
        print(f'{metric}: {result[metric]:.3f}')
    print('Confusion matrix [actual rows 0,1; predicted columns 0,1]:')
    print(result['confusion_matrix'])


def main():
    if not CLEANED_DATA_FILE.is_file():
        raise FileNotFoundError(f'Cleaned data not found: {CLEANED_DATA_FILE}. Run: python -m src.data_cleaning')
    df = pd.read_csv(CLEANED_DATA_FILE)
    print(f'Loaded cleaned data: {df.shape[0]} rows, {df.shape[1]} columns')
    print(f'Columns: {list(df.columns)}')
    reference_results, baseline_results = run_baseline(df)
    print('\nTutor-selected baseline features:', BASELINE_FEATURES)
    report('Majority-class reference', reference_results)
    report('Two-feature logistic-regression baseline', baseline_results)
    print(f'\nTrained baseline saved: {MODEL_FILE}')
    
    # Generate visualizations
    X = df[BASELINE_FEATURES].copy()
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42,
    )
    
    pipeline = joblib.load(MODEL_FILE)
    y_scores = pipeline.predict_proba(X_test)[:, 1]
    
    plot_all_visualizations(
        results_dict=baseline_results,
        cm=baseline_results['confusion_matrix'],
        y_test=y_test,
        y_scores=y_scores,
        model_name='Logistic Regression'
    )
    
    print('LEARNER: justify/extend model; create decision rules, application integration, tests, and improvement evidence.')


if __name__ == '__main__':
    main()
