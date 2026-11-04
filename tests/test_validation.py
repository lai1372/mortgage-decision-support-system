"""Examples of testing generic helper functions; not mortgage policy tests."""
import unittest
from src.decisions import advisory_decision, validate_application_fields, validate_score

class GenericValidationTests(unittest.TestCase):
    def test_valid_score(self):
        self.assertEqual(validate_score(0.4), 0.4)

    def test_score_outside_range(self):
        with self.assertRaises(ValueError):
            validate_score(1.5)

    def test_required_value_present(self):
        self.assertTrue(validate_application_fields({'credit_score': 680}, ['credit_score']))

    def test_required_value_missing(self):
        with self.assertRaises(ValueError):
            validate_application_fields({}, ['credit_score'])

    def test_decision_thresholds_and_unclassified_band(self):
        self.assertEqual(advisory_decision(0.22), 'low risk')
        self.assertEqual(advisory_decision(0.30), 'low risk')
        self.assertEqual(advisory_decision(0.31), 'unclassified')
        self.assertEqual(advisory_decision(0.59), 'unclassified')
        self.assertEqual(advisory_decision(0.60), 'high risk')
        self.assertEqual(advisory_decision(0.71), 'high risk')

    def test_decision_rejects_invalid_score(self):
        with self.assertRaises(ValueError):
            advisory_decision(1.1)

if __name__ == '__main__':
    unittest.main()
