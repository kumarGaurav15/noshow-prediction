import unittest
import pandas as pd
from noshow.generate_data import generate


class TestGenerator(unittest.TestCase):
    def test_same_seed_same_data(self):
        a = generate(300, 120, seed=1)
        b = generate(300, 120, seed=1)
        pd.testing.assert_frame_equal(a, b)

    def test_different_seed_different_data(self):
        a = generate(300, 120, seed=1)
        b = generate(300, 120, seed=2)
        self.assertFalse(a["no_show"].equals(b["no_show"]))

    def test_leakage_trap_columns_exist_in_raw_and_target_rate_sane(self):
        d = generate(600, 220, seed=3)
        for c in ["reminder_confirmed", "minutes_late_at_checkin", "rescheduled_after_booking"]:
            self.assertIn(c, d.columns)
        self.assertTrue(0.10 < d["no_show"].mean() < 0.45)


if __name__ == "__main__":
    unittest.main()