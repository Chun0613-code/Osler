import unittest

from eicu_aki_transition_extract import classify_aki_action


class EicuAkiRouterTests(unittest.TestCase):
    def test_classifies_core_aki_action_terms(self):
        self.assertIn("fluids", classify_aki_action("normal saline bolus"))
        self.assertIn("vasopressor", classify_aki_action("norepinephrine infusion"))
        self.assertIn("diuretics", classify_aki_action("furosemide IV"))
        self.assertIn("renal_replacement", classify_aki_action("CRRT"))
        self.assertIn("nephrotoxin", classify_aki_action("vancomycin IVPB"))

    def test_unknown_terms_do_not_create_actions(self):
        self.assertEqual(classify_aki_action("acetaminophen tablet"), ())


if __name__ == "__main__":
    unittest.main()
