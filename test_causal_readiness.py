import unittest

from osler_jepa.causal_readiness import causal_source_registry, default_causal_specs


class CausalReadinessTests(unittest.TestCase):
    def test_source_registry_includes_fail_closed_trial_sources(self):
        sources = {entry["source"] for entry in causal_source_registry()}

        self.assertIn("BioLINCC", sources)
        self.assertIn("Vivli", sources)
        self.assertIn("YODA", sources)
        self.assertIn("observational EHR", sources)

    def test_default_specs_cover_current_chapter_b_diseases(self):
        specs = default_causal_specs()
        diseases = {spec.disease for spec in specs}

        self.assertIn("DKA", diseases)
        self.assertIn("sepsis", diseases)
        self.assertIn("AKI", diseases)
        self.assertIn("respiratory_failure", diseases)
        self.assertIn("cardiovascular_instability", diseases)
        self.assertIn("acute_neuro", diseases)
        self.assertIn("hepatic_failure", diseases)
        self.assertIn("coagulopathy_heme", diseases)


if __name__ == "__main__":
    unittest.main()
