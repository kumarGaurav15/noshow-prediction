# Hospital Appointment No-Show Prediction

Decision-support ML tool that estimates the probability an appointment will be missed, so staff can
plan **human-reviewed** reminders. It never predicts diagnosis, urgency or clinical outcomes.
Uses **synthetic data only**.

- **Prediction unit:** one appointment
- **Target:** `no_show` (1 = patient did not attend)
- **Prediction point:** the moment the appointment is booked (only booking-time information is used)

## Setup (Windows PowerShell)
```
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Reproduce everything
```
python -m noshow.generate_data      # writes data/appointments_synthetic.csv (seed 42)
python -m noshow.train              # split, baselines, 3 models, calibration, threshold, saves model
python -m pytest -q                 # tests
python -m app.api                   # API + dashboard at http://127.0.0.1:5000
```

Outputs: `models/noshow_model.joblib`, and `reports/` (model_comparison.csv, metrics.json,
cost_sensitivity.csv, subgroup_metrics.csv, permutation_importance.csv, figures/).

## Example API call
```
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d "{\"booked_date\":\"2025-06-02\",\"appointment_date\":\"2025-07-14\",\"appt_hour\":16,\"clinic_type\":\"screening\",\"booking_channel\":\"online\",\"sms_reminder_enabled\":0,\"is_follow_up\":0,\"age_band\":\"18-34\",\"distance_band\":\">15km\",\"prior_appointments\":4,\"prior_no_shows\":2}"
```
Endpoints: `GET /health`, `GET /schema`, `POST /predict`, `GET /` (staff form).

## Method summary
- Chronological split by booking date (60/20/20) with a purge step so no outcome unknown at the
  boundary leaks backwards.
- Baselines: majority class, and "flag if any prior no-show". Candidates: logistic regression,
  random forest, gradient boosting (class_weight="balanced"), tuned on validation PR-AUC.
- Probabilities calibrated (Platt) on validation. Threshold chosen on validation by business cost
  (missed no-show = 5, unnecessary reminder = 1; editable in `noshow/config.py`).
- Test set used once for final reporting.

## Leakage control
Raw data contains `reminder_confirmed`, `minutes_late_at_checkin`, `rescheduled_after_booking`
(known only after booking/outcome). They are excluded and a test asserts no forbidden column is a feature.
No diagnosis, treatment or clinical variables exist in the data.

## Known limitations
- Data is synthetic; real hospitals need re-validation on local, de-identified data.
- Ranking power is modest (see reports/metrics.json). Value is mainly prioritisation.
- Age and distance bands can proxy for socioeconomic factors; see reports/subgroup_metrics.csv.
- Intended only for human-reviewed reminders, never automated denial of care.