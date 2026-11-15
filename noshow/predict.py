"""Inference: load the saved artifact, validate input, predict, explain.

Only information known at booking time is accepted. Nothing here is hard-coded:
every probability comes from the saved model + calibrator.
"""
import pandas as pd
import joblib
from noshow.config import MODEL_PATH
from noshow.features import add_features, FEATURES

REQUIRED = ["booked_date", "appointment_date", "appt_hour", "clinic_type",
            "sms_reminder_enabled", "is_follow_up", "age_band",
            "prior_appointments", "prior_no_shows"]
OPTIONAL = ["booking_channel", "distance_band"]

# Feature groups for "what pushed this prediction" (correlated features explained together)
GROUPS = {
    "Booking lead time": ["lead_time_days"],
    "Appointment time/day": ["appt_hour", "appt_weekday", "appt_month"],
    "Past attendance history": ["prior_appointments", "prior_no_shows", "prior_no_show_rate"],
    "SMS reminder opt-in": ["sms_reminder_enabled"],
    "Follow-up visit": ["is_follow_up"],
    "Clinic type": ["clinic_type"],
    "Booking channel": ["booking_channel"],
    "Age band": ["age_band"],
    "Distance band": ["distance_band"],
}

DISCLAIMER = ("Decision support only. This is an estimated probability, not a certainty. "
              "Staff should review before sending any reminder. It must not be used to deny, "
              "delay or reprioritise care, and it says nothing about clinical need.")


def load_bundle(path=None):
    return joblib.load(path or MODEL_PATH)


def _int(record, key, errors, lo, hi):
    v = record.get(key)
    try:
        if isinstance(v, bool):
            raise ValueError
        if isinstance(v, float) and not v.is_integer():
            raise ValueError
        v = int(v)
    except (ValueError, TypeError):
        errors.append(f"{key} must be a whole number")
        return None
    if v < lo or v > hi:
        errors.append(f"{key} must be between {lo} and {hi}")
        return None
    return v


def validate(record, bundle):
    """Return a cleaned dict, or raise ValueError(list_of_error_messages)."""
    if not isinstance(record, dict):
        raise ValueError(["input must be a JSON object"])
    errors, out = [], {}

    for k in REQUIRED:
        if record.get(k) in (None, ""):
            errors.append(f"missing required field: {k}")
    if errors:
        raise ValueError(errors)

    for k in ["booked_date", "appointment_date"]:
        try:
            out[k] = pd.to_datetime(str(record[k]), format="%Y-%m-%d")
        except (ValueError, TypeError):
            errors.append(f"{k} must be a date in YYYY-MM-DD format")

    for k, lo, hi in [("appt_hour", 0, 23), ("sms_reminder_enabled", 0, 1), ("is_follow_up", 0, 1),
                      ("prior_appointments", 0, 1000), ("prior_no_shows", 0, 1000)]:
        v = _int(record, k, errors, lo, hi)
        if v is not None:
            out[k] = v

    for c in ["clinic_type", "age_band"] + OPTIONAL:
        v = record.get(c)
        if v in (None, "") and c in OPTIONAL:
            out[c] = None
            continue
        if v not in bundle["categories"][c]:
            errors.append(f"{c} must be one of {bundle['categories'][c]}")
        else:
            out[c] = v

    if not errors:
        if out["appointment_date"] < out["booked_date"]:
            errors.append("appointment_date cannot be before booked_date")
        if out["prior_no_shows"] > out["prior_appointments"]:
            errors.append("prior_no_shows cannot exceed prior_appointments")
    if errors:
        raise ValueError(errors)
    return out


def to_frame(clean, bundle):
    row, notes = dict(clean), []
    for c in OPTIONAL:
        if row.get(c) is None:
            row[c] = bundle["reference"][c]
            notes.append(f"{c} not provided; the most common training value was used")
    X = add_features(pd.DataFrame([row]))
    return X[FEATURES], notes


def _score(bundle, X):
    raw = bundle["pipeline"].predict_proba(X[FEATURES])[:, 1]
    return bundle["calibrator"].predict(raw)


def explain(bundle, X, p0, top_k=4):
    """Occlusion explanation: replace one feature group with a typical value and see how the
    calibrated probability changes. Positive = that group raised the predicted risk."""
    ref, rows = bundle["reference"], []
    for name, cols in GROUPS.items():
        X2 = X.copy()
        for c in cols:
            X2[c] = ref[c]
        delta = p0 - float(_score(bundle, X2)[0])
        if name == "Past attendance history":
            value = f"{int(X.iloc[0]['prior_no_shows'])} no-shows in {int(X.iloc[0]['prior_appointments'])} past appointments"
        else:
            value = ", ".join(f"{c}={X.iloc[0][c]}" for c in cols[:1])
        rows.append({"factor": name, "value": value, "effect_pct_points": round(delta * 100, 1),
                     "direction": "raises risk" if delta > 0 else "lowers risk"})
    rows.sort(key=lambda r: abs(r["effect_pct_points"]), reverse=True)
    return rows[:top_k]


def predict_one(bundle, record):
    clean = validate(record, bundle)
    X, notes = to_frame(clean, bundle)
    p = float(_score(bundle, X)[0])
    thr = float(bundle["threshold"])
    meta = bundle["metadata"]

    warnings = list(notes)
    if int(X.iloc[0]["lead_time_days"]) > meta["train_lead_time_max"]:
        warnings.append("Lead time is longer than anything seen in training; treat the estimate with extra caution.")
    if clean["appt_hour"] < 8 or clean["appt_hour"] > 17:
        warnings.append("Appointment hour is outside the 8:00-17:59 range seen in training.")
    if clean["prior_appointments"] == 0:
        warnings.append("No attendance history: the estimate relies on population patterns only.")

    band = "high" if p >= 2 * thr else ("elevated" if p >= thr else "lower")
    flag = p >= thr
    return {
        "probability_no_show": round(p, 4),
        "threshold": round(thr, 3),
        "risk_band": band,
        "suggest_reminder_for_human_review": bool(flag),
        "recommendation": ("Consider this appointment for a staff-reviewed reminder."
                           if flag else "Below the review threshold; standard reminder process applies."),
        "top_factors": explain(bundle, X, p),
        "warnings": warnings,
        "uncertainty": (f"Calibrated probability (test ECE {meta['calibration']['ece_calibrated_test']}). "
                        f"Model ranking power is modest (test ROC-AUC {meta['test_metrics']['roc_auc']}), "
                        "so individual outcomes remain uncertain."),
        "model": meta["model"],
        "disclaimer": DISCLAIMER,
    }