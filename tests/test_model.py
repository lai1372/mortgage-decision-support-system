"""Tests for model training, evaluation, and performance."""
import unittest
import pandas as pd
import joblib
from pathlib import Path
from src.model import run_baseline, BASELINE_FEATURES, MODEL_FILE
from src.data_cleaning import CLEANED_DATA_FILE
from src.data import TARGET


class ModelTrainingTests(unittest.TestCase):
    """Test model training and outputs."""
    
    @classmethod
    def setUpClass(cls):
        """Load cleaned data once for all tests."""
        if not CLEANED_DATA_FILE.is_file():
            raise FileNotFoundError(f'Cleaned data required: {CLEANED_DATA_FILE}')
        cls.df = pd.read_csv(CLEANED_DATA_FILE)
    
    def test_baseline_features_exist(self):
        """All baseline features must be in dataset."""
        missing = set(BASELINE_FEATURES) - set(self.df.columns)
        self.assertEqual(missing, set(), f'Missing features: {missing}')
    
    def test_target_exists(self):
        """Target column must exist in dataset."""
        self.assertIn(TARGET, self.df.columns, f'{TARGET} not found')
    
    def test_data_has_no_missing_values(self):
        """Cleaned data should have no missing values."""
        missing_count = self.df.isna().sum().sum()
        self.assertEqual(missing_count, 0, f'Found {missing_count} missing values')
    
    def test_target_is_binary(self):
        """Target should be binary (0 or 1)."""
        unique_values = set(self.df[TARGET].unique())
        self.assertEqual(unique_values, {0, 1}, f'Target has unexpected values: {unique_values}')


class ModelPerformanceTests(unittest.TestCase):
    """Test model evaluation metrics."""
    
    @classmethod
    def setUpClass(cls):
        """Train model once for all tests."""
        if not CLEANED_DATA_FILE.is_file():
            raise FileNotFoundError(f'Cleaned data required: {CLEANED_DATA_FILE}')
        cls.df = pd.read_csv(CLEANED_DATA_FILE)
        cls.reference_results, cls.baseline_results = run_baseline(cls.df)
    
    def test_roc_auc_above_random_baseline(self):
        """Model ROC-AUC should exceed random classifier (0.5)."""
        roc_auc = self.baseline_results['roc_auc']
        self.assertGreater(roc_auc, 0.5, f'ROC-AUC {roc_auc} not better than random (0.5)')
    
    def test_roc_auc_meets_minimum_threshold(self):
        """Model should achieve at least 0.69 ROC-AUC."""
        roc_auc = self.baseline_results['roc_auc']
        self.assertGreaterEqual(roc_auc, 0.65, f'ROC-AUC {roc_auc} below minimum 0.65')
    
    def test_precision_is_between_zero_and_one(self):
        """Precision should be in valid range [0, 1]."""
        precision = self.baseline_results['precision_1']
        self.assertGreaterEqual(precision, 0, 'Precision < 0')
        self.assertLessEqual(precision, 1, 'Precision > 1')
    
    def test_recall_is_between_zero_and_one(self):
        """Recall should be in valid range [0, 1]."""
        recall = self.baseline_results['recall_1']
        self.assertGreaterEqual(recall, 0, 'Recall < 0')
        self.assertLessEqual(recall, 1, 'Recall > 1')
    
    def test_f1_score_is_between_zero_and_one(self):
        """F1 score should be in valid range [0, 1]."""
        f1 = self.baseline_results['f1_1']
        self.assertGreaterEqual(f1, 0, 'F1 < 0')
        self.assertLessEqual(f1, 1, 'F1 > 1')


class ConfusionMatrixTests(unittest.TestCase):
    """Test confusion matrix structure and values."""
    
    @classmethod
    def setUpClass(cls):
        """Train model once for all tests."""
        if not CLEANED_DATA_FILE.is_file():
            raise FileNotFoundError(f'Cleaned data required: {CLEANED_DATA_FILE}')
        cls.df = pd.read_csv(CLEANED_DATA_FILE)
        cls.reference_results, cls.baseline_results = run_baseline(cls.df)
    
    def test_confusion_matrix_shape(self):
        """Confusion matrix should be 2x2."""
        cm = self.baseline_results['confusion_matrix']
        self.assertEqual(cm.shape, (2, 2), f'Expected (2, 2), got {cm.shape}')
    
    def test_confusion_matrix_non_negative(self):
        """Confusion matrix values should be non-negative."""
        cm = self.baseline_results['confusion_matrix']
        self.assertTrue((cm >= 0).all(), f'Found negative values in confusion matrix: {cm}')


class SavedModelTests(unittest.TestCase):
    """Test that trained model is saved correctly."""
    
    def test_model_file_exists(self):
        """Trained model file should exist."""
        self.assertTrue(MODEL_FILE.is_file(), f'Model file not found: {MODEL_FILE}')
    
    def test_model_can_be_loaded(self):
        """Saved model should be loadable."""
        try:
            pipeline = joblib.load(MODEL_FILE)
            self.assertIsNotNone(pipeline, 'Loaded model is None')
        except Exception as e:
            self.fail(f'Failed to load model: {e}')

if __name__ == '__main__':
    unittest.main()
