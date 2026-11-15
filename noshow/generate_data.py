"""Synthetic hospital appointment generator (no real patients, no clinical data).

Prediction point: the moment an appointment is BOOKED.
Target: no_show = 1 if the patient did not attend.

The raw file deliberately includes three post-booking columns
(reminder_confirmed, minutes_late_at_checkin, rescheduled_after_booking).
They are leakage traps: the model must NOT use them.
"""
import argparse
import numpy as np
import pandas as pd
from noshow.config import DATA_PATH, SEED

AGE = ["0-17", "18-34", "35-49", "50-64", "65+"]
AGE_P = [0.12, 0.28, 0.24, 0.22, 0.14]
AGE_EFF = {"0-17": 0.10, "18-34": 0.35, "35-49": 0.10, "50-64": -0.15, "65+": -0.30}
DIST = ["<5km", "5-15km", ">15km"]
DIST_P = [0.45, 0.35, 0.20]
DIST_EFF = {"<5km": -0.15, "5-15km": 0.0, ">15km": 0.40}
CLINIC = ["primary_care", "specialist", "screening", "physiotherapy"]
CLINIC_P = [0.45, 0.25, 0.15, 0.15]
CLINIC_EFF = {"primary_care": 0.0, "specialist": -0.20, "screening": 0.30, "physiotherapy": 0.10}
CHANNEL = ["phone", "online", "front_desk"]
CHANNEL_P = [0.45, 0.30, 0.25]
CHANNEL_EFF = {"phone": -0.15, "online": 0.25, "front_desk": -0.30}


def generate(n_appointments=6000, n_patients=2200, seed=SEED, start="2024-01-01", days=540):
    rng = np.random.default_rng(seed)

    # Patient-level attributes + a hidden "tendency to miss" (makes history predictive)
    pat_age = rng.choice(AGE, n_patients, p=AGE_P)
    pat_dist = rng.choice(DIST, n_patients, p=DIST_P)
    pat_eff = rng.normal(0, 0.8, n_patients)
    weights = rng.dirichlet(np.full(n_patients, 0.7))
    pid = rng.choice(n_patients, n_appointments, p=weights)

    # Booking date, lead time, appointment date (no weekends)
    booked = pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, days, n_appointments), unit="D")
    lead = np.clip(rng.gamma(2.0, 7.0, n_appointments).round().astype(int), 0, 90)
    appt = booked + pd.to_timedelta(lead, unit="D")
    wd = appt.day_of_week.values
    appt = appt + pd.to_timedelta(np.where(wd == 5, 2, np.where(wd == 6, 1, 0)), unit="D")
    lead = (appt - booked).days.values

    df = pd.DataFrame({
        "appointment_id": [f"A{i:06d}" for i in range(n_appointments)],
        "patient_token": [f"P{p:05d}" for p in pid],
        "booked_date": booked.normalize(),
        "appointment_date": appt.normalize(),
        "appt_hour": rng.choice([8, 9, 10, 11, 12, 13, 14, 15, 16, 17], n_appointments,
                                p=[.08, .13, .15, .13, .08, .08, .12, .11, .07, .05]),
        "clinic_type": rng.choice(CLINIC, n_appointments, p=CLINIC_P),
        "booking_channel": rng.choice(CHANNEL, n_appointments, p=CHANNEL_P),
        "sms_reminder_enabled": rng.binomial(1, 0.65, n_appointments),
        "is_follow_up": rng.binomial(1, 0.35, n_appointments),
        "age_band": pat_age[pid],
        "distance_band": pat_dist[pid],
        "lead_time_days": lead,
    })

    # True no-show probability (the hidden "rules" a model should learn)
    wd = df["appointment_date"].dt.day_of_week.values
    h = df["appt_hour"].values
    logit = (-1.55 + 0.022 * np.minimum(lead, 60)
             + df["age_band"].map(AGE_EFF).values
             + df["distance_band"].map(DIST_EFF).values
             + df["clinic_type"].map(CLINIC_EFF).values
             + df["booking_channel"].map(CHANNEL_EFF).values
             - 0.45 * df["sms_reminder_enabled"].values
             - 0.20 * df["is_follow_up"].values
             + np.where(h < 9, 0.20, 0) + np.where((h >= 12) & (h <= 13), 0.15, 0)
             + np.where(h >= 16, 0.25, 0)
             + np.where(wd == 0, 0.20, 0) + np.where(wd == 4, 0.15, 0)
             + pat_eff[pid])
    p = 1 / (1 + np.exp(-logit))
    df["no_show"] = rng.binomial(1, p)

    # Booking-time history: count only appointments whose DATE is before this booking date,
    # because only those outcomes are known when the booking is made.
    prior_n = np.zeros(n_appointments, dtype=int)
    prior_ns = np.zeros(n_appointments, dtype=int)
    for _, idx in df.groupby("patient_token").indices.items():
        sub = df.iloc[idx]
        order = np.argsort(sub["appointment_date"].values, kind="stable")
        dates = sub["appointment_date"].values[order]
        cum = np.cumsum(sub["no_show"].values[order])
        k = np.searchsorted(dates, sub["booked_date"].values, side="left")
        prior_n[idx] = k
        prior_ns[idx] = np.where(k > 0, cum[np.maximum(k - 1, 0)], 0)
    df["prior_appointments"] = prior_n
    df["prior_no_shows"] = prior_ns

    # LEAKAGE TRAPS: only known after booking / after the outcome
    att = df["no_show"].values == 0
    df["reminder_confirmed"] = np.where(att, rng.binomial(1, 0.60, n_appointments),
                                        rng.binomial(1, 0.08, n_appointments))
    df["minutes_late_at_checkin"] = np.where(att, rng.integers(0, 25, n_appointments), np.nan)
    df["rescheduled_after_booking"] = np.where(att, rng.binomial(1, 0.05, n_appointments),
                                               rng.binomial(1, 0.30, n_appointments))

    # Realistic data-quality problems for the cleaning step
    m = rng.random(n_appointments)
    df.loc[m < 0.04, "distance_band"] = np.nan                         # ~4% missing
    df.loc[(m >= 0.04) & (m < 0.06), "booking_channel"] = np.nan       # ~2% missing
    bad = rng.choice(n_appointments, int(0.004 * n_appointments), replace=False)
    df.loc[bad, "lead_time_days"] = rng.choice([-3, 400, 999], len(bad))   # corrupted values
    swap = rng.choice(n_appointments, int(0.003 * n_appointments), replace=False)
    df.loc[swap, ["booked_date", "appointment_date"]] = \
        df.loc[swap, ["appointment_date", "booked_date"]].values           # impossible date order

    return df.sort_values("booked_date").reset_index(drop=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--out", default=str(DATA_PATH))
    a = ap.parse_args()
    d = generate(a.n, max(300, int(a.n * 0.37)), a.seed)
    d.to_csv(a.out, index=False)
    print(f"wrote {len(d)} rows to {a.out}; no-show rate = {d.no_show.mean():.3f}")