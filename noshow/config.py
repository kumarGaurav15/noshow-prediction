"""Central configuration. Change costs here to re-tune the decision threshold."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "appointments_synthetic.csv"
MODEL_PATH = ROOT / "models" / "noshow_model.joblib"
REPORT_DIR = ROOT / "reports"

SEED = 42

# Business cost assumptions (relative units, editable).
# FN = a no-show we did not flag (wasted slot, partly recoverable by a reminder)
# FP = an unnecessary reminder call/SMS review (staff minutes)
COST_FN = 5.0
COST_FP = 1.0

# Chronological split by booking date (test = the remaining 20%)
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20