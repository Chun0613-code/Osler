import unittest

from teacher_anchored_joint_jepa_audit import (
    _validate_resume_seed_extension,
)


class TeacherAnchoredResumePolicyTests(unittest.TestCase):
    def test_resume_may_add_new_seeds(self):
        _validate_resume_seed_extension(
            [307, 331, 353],
            [307, 331, 353, 379],
            [{"seed": 307}, {"seed": 331}, {"seed": 353}],
        )

    def test_resume_cannot_drop_a_previous_seed(self):
        with self.assertRaisesRegex(ValueError, "cannot remove"):
            _validate_resume_seed_extension(
                [307, 331, 353],
                [307, 331],
                [{"seed": 307}, {"seed": 331}, {"seed": 353}],
            )

    def test_completed_seed_must_be_requested(self):
        with self.assertRaisesRegex(ValueError, "absent from this run"):
            _validate_resume_seed_extension(
                [307, 331],
                [307, 331],
                [{"seed": 307}, {"seed": 999}],
            )


if __name__ == "__main__":
    unittest.main()
