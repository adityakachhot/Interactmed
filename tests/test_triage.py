"""
test_triage.py
==============
Unit tests verifying the independent TriageClassifier contract:
Input:  clinical_finding (str)
Output: TriageResult with category in {'Green', 'Yellow', 'Red'} and confidence in [0.0, 1.0]
"""

import unittest
from src.triage.classifier import TriageClassifier, TriageResult


class TestTriageClassifier(unittest.TestCase):
    def setUp(self):
        self.classifier = TriageClassifier()

    def test_green_normal_finding(self):
        text = "Lungs are clear bilaterally. No acute cardiopulmonary abnormalities."
        res: TriageResult = self.classifier.classify(text)

        self.assertEqual(res.category, "Green")
        self.assertTrue(0.0 <= res.confidence <= 1.0)
        self.assertEqual(res.urgency_level, "Routine")
        self.assertEqual(len(res.emergency_flags), 0)

    def test_yellow_moderate_finding(self):
        text = "Moderate cardiomegaly noted. Mild vascular engorgement without edema."
        res: TriageResult = self.classifier.classify(text)

        self.assertEqual(res.category, "Yellow")
        self.assertTrue(0.0 <= res.confidence <= 1.0)
        self.assertIn("cardiomegaly", res.emergency_flags)

    def test_red_critical_finding(self):
        text = "Large right apical pneumothorax with visible pleural line."
        res: TriageResult = self.classifier.classify(text)

        self.assertEqual(res.category, "Red")
        self.assertTrue(0.0 <= res.confidence <= 1.0)
        self.assertEqual(res.urgency_level, "Immediate (Stat)")
        self.assertIn("pneumothorax", res.emergency_flags)

    def test_negation_safety(self):
        # Explicit negation must not trigger Red
        text = "No pneumothorax or acute focal consolidation."
        res: TriageResult = self.classifier.classify(text)

        self.assertNotEqual(res.category, "Red")


if __name__ == "__main__":
    unittest.main()
