import unittest
import numpy as np
import pandas as pd
from noshow.features import (FEATURES, FORBIDDEN, CATEGORICAL, add_features, clean, chronological_split)


def make_df(n=100):
    booked = pd.Timestamp("2024-01-01") + pd.to_timedelta(np.arange(n), unit="D")
    df = pd.DataFrame({
        "appointment_id": [f"A{i}" for i in range(n)],
        "booked_date": booked,
        "appointment_date": booked + pd.Timedelta(days=5),
        "lead_time_days": 5,
        "prior_appointments": 2, "prior_no_shows": 1,
        "no_show": np.tile([0, 1], n // 2),
    })
    for c in CATEGORICAL:
        df[c] = "x"
    return df


class TestFeatures(unittest.TestCase):
    def test_no_forbidden_columns_in_features(self):
        self.assertEqual(set(FEATURES) & set(FORBIDDEN), set())

    def test_lead_time_computed_from_dates(self):
        d = pd.DataFrame({"booked_date": ["2024-03-01"], "appointment_date": ["2024-03-11"],
                          "prior_appointments": [0], "prior_no_shows": [0]})
        self.assertEqual(int(add_features(d).loc[0, "lead_time_days"]), 10)

    def test_smoothed_rate_is_not_extreme_for_small_history(self):
        d = pd.DataFrame({"booked_date": ["2024-03-01"], "appointment_date": ["2024-03-11"],
                          "prior_appointments": [1], "prior_no_shows": [1]})
        r = add_features(d).loc[0, "prior_no_show_rate"]
        self.assertTrue(0 < r < 1)

    def test_clean_drops_impossible_dates(self):
        df = make_df(10)
        df.loc[3, ["booked_date", "appointment_date"]] = [pd.Timestamp("2024-02-01"), pd.Timestamp("2024-01-01")]
        out, rep = clean(df)
        self.assertEqual(rep["appointment_before_booking_dropped"], 1)
        self.assertEqual(len(out), 9)

    def test_split_is_chronological_with_purge(self):
        tr, va, te, info = chronological_split(make_df(100))
        self.assertLess(tr["booked_date"].max(), va["booked_date"].min())
        self.assertLess(va["booked_date"].max(), te["booked_date"].min())
        self.assertLess(tr["appointment_date"].max(), va["booked_date"].min())
        self.assertLess(va["appointment_date"].max(), te["booked_date"].min())
        self.assertGreater(info["purged_rows"], 0)
        self.assertEqual(len(set(tr.appointment_id) & set(te.appointment_id)), 0)


if __name__ == "__main__":
    unittest.main()