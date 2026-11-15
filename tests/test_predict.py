import copy
import unittest
from pathlib import Path
from noshow.config import MODEL_PATH
from noshow.predict import load_bundle, predict_one, validate

HIGH = {"booked_date": "2025-06-02", "appointment_date": "2025-07-14", "appt_hour": 16,
        "clinic_type": "screening", "booking_channel": "online", "sms_reminder_enabled": 0,
        "is_follow_up": 0, "age_band": "18-34", "distance_band": ">15km",
        "prior_appointments": 4, "prior_no_shows": 2}
LOW = {"booked_date": "2025-06-05", "appointment_date": "2025-06-12", "appt_hour": 10,
       "clinic_type": "specialist", "booking_channel": "front_desk", "sms_reminder_enabled": 1,
       "is_follow_up": 1, "age_band": "65+", "distance_band": "<5km",
       "prior_appointments": 6, "prior_no_shows": 0}


@unittest.skipUnless(Path(MODEL_PATH).exists(), "run python -m noshow.train first")
class TestPredict(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = load_bundle()

    def test_probability_in_range_and_deterministic(self):
        a = predict_one(self.bundle, HIGH)
        b = predict_one(self.bundle, HIGH)
        self.assertTrue(0 <= a["probability_no_show"] <= 1)
        self.assertEqual(a["probability_no_show"], b["probability_no_show"])

    def test_different_inputs_give_different_predictions(self):
        hi = predict_one(self.bundle, HIGH)["probability_no_show"]
        lo = predict_one(self.bundle, LOW)["probability_no_show"]
        self.assertGreater(hi, lo)

    def test_missing_required_field_rejected(self):
        r = copy.deepcopy(HIGH); del r["age_band"]
        with self.assertRaises(ValueError):
            predict_one(self.bundle, r)

    def test_unseen_category_rejected(self):
        r = copy.deepcopy(HIGH); r["clinic_type"] = "cardiology_ward"
        with self.assertRaises(ValueError):
            validate(r, self.bundle)

    def test_logically_invalid_inputs_rejected(self):
        r = copy.deepcopy(HIGH); r["prior_no_shows"] = 9
        with self.assertRaises(ValueError):
            validate(r, self.bundle)
        r = copy.deepcopy(HIGH); r["appointment_date"] = "2025-05-01"
        with self.assertRaises(ValueError):
            validate(r, self.bundle)

    def test_optional_fields_can_be_missing(self):
        r = copy.deepcopy(HIGH); del r["distance_band"]; del r["booking_channel"]
        out = predict_one(self.bundle, r)
        self.assertTrue(len(out["warnings"]) >= 2)

    def test_output_contains_explanation_and_disclaimer(self):
        out = predict_one(self.bundle, HIGH)
        self.assertTrue(len(out["top_factors"]) > 0)
        self.assertIn("Decision support", out["disclaimer"])


if __name__ == "__main__":
    unittest.main()