"""Cleaning, feature engineering and chronological splitting.

Everything used as a model input must be known when the appointment is booked.
"""
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "no_show"
NUMERIC = ["lead_time_days", "appt_hour", "appt_weekday", "appt_month",
           "prior_appointments", "prior_no_shows", "prior_no_show_rate",
           "sms_reminder_enabled", "is_follow_up"]
CATEGORICAL = ["clinic_type", "booking_channel", "age_band", "distance_band"]
FEATURES = NUMERIC + CATEGORICAL

# Never allowed as model inputs: post-booking information, the target, identifiers.
FORBIDDEN = ["reminder_confirmed", "minutes_late_at_checkin", "rescheduled_after_booking",
             "no_show", "appointment_id", "patient_token"]

# Smoothing for the prior no-show rate: (no_shows + a*p) / (appointments + a)
SMOOTH_A, SMOOTH_PRIOR = 2.0, 0.20


def add_features(df):
    """Derive features from booking-time fields. Works for training data and a single API record."""
    out = df.copy()
    out["booked_date"] = pd.to_datetime(out["booked_date"])
    out["appointment_date"] = pd.to_datetime(out["appointment_date"])
    out["lead_time_days"] = (out["appointment_date"] - out["booked_date"]).dt.days
    out["appt_weekday"] = out["appointment_date"].dt.dayofweek
    out["appt_month"] = out["appointment_date"].dt.month
    out["prior_no_show_rate"] = ((out["prior_no_shows"] + SMOOTH_A * SMOOTH_PRIOR)
                                 / (out["prior_appointments"] + SMOOTH_A))
    return out


def clean(raw):
    """Data-quality handling. Returns (clean_df, report_dict)."""
    df = raw.copy()
    rep = {"rows_in": int(len(df))}
    df["booked_date"] = pd.to_datetime(df["booked_date"])
    df["appointment_date"] = pd.to_datetime(df["appointment_date"])

    rep["duplicate_ids_dropped"] = int(df.duplicated("appointment_id").sum())
    df = df.drop_duplicates("appointment_id")

    impossible = df["appointment_date"] < df["booked_date"]
    rep["appointment_before_booking_dropped"] = int(impossible.sum())
    df = df[~impossible]

    derived = (df["appointment_date"] - df["booked_date"]).dt.days
    rep["lead_time_mismatch_recomputed"] = int((derived != df["lead_time_days"]).sum())
    rep["lead_time_gt_120_days"] = int((derived > 120).sum())
    rep["missing_values"] = {c: int(df[c].isna().sum()) for c in CATEGORICAL if df[c].isna().any()}
    rep["no_show_rate"] = round(float(df[TARGET].mean()), 4)
    rep["rows_out"] = int(len(df))
    return df.reset_index(drop=True), rep


def chronological_split(df, train_frac=0.6, val_frac=0.2, purge=True):
    """Split by booking date. With purge=True, rows in an earlier fold whose appointment date falls
    on/after the next fold's first booking date are removed, because their outcome would not yet be
    known at that moment (prevents future leakage across the boundary)."""
    d = df.sort_values(["booked_date", "appointment_id"]).reset_index(drop=True)
    i1, i2 = int(len(d) * train_frac), int(len(d) * (train_frac + val_frac))
    tr, va, te = d.iloc[:i1], d.iloc[i1:i2], d.iloc[i2:]
    info = {"val_start": str(va["booked_date"].min().date()),
            "test_start": str(te["booked_date"].min().date())}
    if purge:
        n0 = len(tr) + len(va)
        tr = tr[tr["appointment_date"] < va["booked_date"].min()]
        va = va[va["appointment_date"] < te["booked_date"].min()]
        info["purged_rows"] = int(n0 - len(tr) - len(va))
    info.update(n_train=len(tr), n_val=len(va), n_test=len(te))
    return tr, va, te, info


def build_preprocessor():
    num = Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())])
    cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                    ("oh", OneHotEncoder(handle_unknown="ignore"))])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])