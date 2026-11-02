# Mortgage Decision Support System

A logistic regression model to predict mortgage repayment difficulty based on applicant financial characteristics.

## Setup

### Prerequisites
- Python 3.9+

### Create Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
Creates an isolated Python environment for this project to avoid dependency conflicts.

### Install Dependencies
```bash
pip install -r requirements.txt
```
Installs required packages (pandas, scikit-learn, matplotlib, joblib).

## Usage

### 1. Clean the Data
```bash
python3 -m src.data_cleaning
```
**What it does:** Preprocesses raw data, handles missing values, removes duplicates, and selects relevant features.

**Output:** `data/mortgage_data_cleaned.csv`

---

### 2. Train the Model
```bash
python3 -m src.main
```
**What it does:** Trains a logistic regression model on the cleaned data using an 80/20 train/test split. Generates evaluation visualisations.

**Outputs:**
- `models/baseline_model.joblib` — Trained model (pickle file)
- `outputs/logistic_regression_roc_curve.png` — ROC curve visualisation
- `outputs/logistic_regression_confusion_matrix.png` — Confusion matrix heatmap
- `outputs/logistic_regression_metrics_table.png` — Performance metrics table

---

### 3. Score an Applicant
```bash
python3 -m src.applicant_example
```
**What it does:** Loads a trained model and predicts repayment risk for a new mortgage application.

**Output:** Risk score between 0.0 and 1.0 (higher = greater risk)

---

### 4. Run Tests
```bash
python3 -m unittest tests.test_model -v
```
**What it does:** Runs automated tests to verify data quality, model performance, and saved model integrity.

**Output:** Test results (16 tests total)

---

### Command Explanation
- `python3 -m` — Runs a Python module as a script
- `pip install -r` — Installs packages listed in requirements.txt
- `unittest -v` — Runs tests with verbose output (shows each test)
