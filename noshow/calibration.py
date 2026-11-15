"""Probability calibration, fitted on the validation set (never the test set)."""
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class Calibrator:
    """Maps a model's raw score to a calibrated probability (Platt scaling by default)."""

    def __init__(self, method="sigmoid"):
        self.method = method

    def fit(self, scores, y):
        scores = np.asarray(scores)
        if self.method == "isotonic":
            self.m_ = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(scores, y)
        else:
            self.m_ = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(scores), y)
        return self

    def predict(self, scores):
        scores = np.asarray(scores)
        if self.method == "isotonic":
            return self.m_.predict(scores)
        return self.m_.predict_proba(_logit(scores))[:, 1]


def expected_calibration_error(y, p, bins=10):
    """Average gap between predicted probability and actual rate, weighted by bin size."""
    y, p = np.asarray(y), np.asarray(p)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mk = idx == b
        if mk.any():
            ece += mk.mean() * abs(y[mk].mean() - p[mk].mean())
    return float(ece)