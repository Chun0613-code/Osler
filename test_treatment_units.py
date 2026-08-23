import unittest

from osler_jepa.treatment_units import (
    canonical_dose,
    canonical_rate,
    route_category,
)


class TreatmentUnitTests(unittest.TestCase):
    def test_canonical_dose_preserves_dimensions(self):
        self.assertEqual(canonical_dose(2.0, "g"), (2000.0, "mass_mg"))
        self.assertEqual(canonical_dose(20.0, "mEq."), (20.0, "charge_meq"))
        self.assertEqual(canonical_dose(5.0, "Units"), (5.0, "drug_units"))
        self.assertEqual(canonical_dose(0.5, "L"), (500.0, "volume_ml"))

    def test_canonical_rate_normalizes_time_without_mixing_weighted_rates(self):
        self.assertEqual(
            canonical_rate(2.0, "mL/min"),
            (120.0, "volume_ml_per_hour"),
        )
        self.assertEqual(
            canonical_rate(10.0, "mcg/min"),
            (0.6, "mass_mg_per_hour"),
        )
        self.assertEqual(
            canonical_rate(0.1, "mcg/kg/min"),
            (0.1, "mass_mcg_per_kg_min"),
        )

    def test_unknown_units_are_fail_closed(self):
        value, dimension = canonical_dose(1.0, "bag")
        self.assertIsNone(dimension)
        self.assertNotEqual(value, value)

    def test_inputevents_are_intravenous_by_construction(self):
        self.assertEqual(route_category(None, "mimic_inputevents"), "iv")


if __name__ == "__main__":
    unittest.main()
