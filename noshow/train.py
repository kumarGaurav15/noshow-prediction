"""Training pipeline: baselines, 3 candidate models, calibration, cost-based threshold, saved artifact.

Run from the project root:  python -m noshow.train
"""
import json
import numpy as np
import pandas as pd
import joblib
import sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             precision_recall_fscore_support, confusion_matrix, precision_recall_curve)
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline

from noshow.config import (DATA_PATH, MODEL_PATH, REPORT_DIR, SEED, COST_FN, COST_FP,
                           TRAIN_FRAC, VAL_FRAC)
from noshow.features import (TARGET, NUMERIC, CATEGORICAL, FEATURES, FORBIDDEN, clean,
                             add_features, chronological_split, build_preprocessor)
from noshow.calibration import Calibrator, expected_calibration_error

FIG = REPORT_DIR / "figures"


def make_model(name, **params):
    if name == "logreg":
        est = LogisticRegression(class_weight="balanced", max_iter=2000, **params)
    elif name == "random_forest":
        est = RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1, **params)
    else:
        est = HistGradientBoostingClassifier(class_weight="balanced", random_state=SEED, **params)
    return Pipeline([("prep", build_preprocessor()), ("model", est)])


GRIDS = {
    "logreg": {"C": [0.1, 1.0, 10.0]},
    "random_forest": {"n_estimators": [300], "max_depth": [6, 10, None], "min_samples_leaf": [5, 20]},
    "hist_gb": {"learning_rate": [0.05, 0.1], "max_depth": [3, None], "max_leaf_nodes": [8, 15]},
}


def metrics(y, score, pred):
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {"precision": round(float(p), 4), "recall": round(float(r), 4), "f1": round(float(f), 4),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
            "pr_auc": round(float(average_precision_score(y, score)), 4),
            "roc_auc": round(float(roc_auc_score(y, score)), 4)}


def best_f1_threshold(y, score):
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-12)
    return float(t[np.argmax(f1)])


def total_cost(y, pred):
    y, pred = np.asarray(y), np.asarray(pred)
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    return fn * COST_FN + fp * COST_FP


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # 1) Data, cleaning, features, chronological split
    raw = pd.read_csv(DATA_PATH)
    df, clean_report = clean(raw)
    df = add_features(df)
    train, val, test, split_info = chronological_split(df, TRAIN_FRAC, VAL_FRAC)
    assert not set(FEATURES) & set(FORBIDDEN), "Forbidden column in feature list!"
    print("cleaning:", clean_report)
    print("split:", split_info)
    Xtr, ytr = train[FEATURES], train[TARGET]
    Xva, yva = val[FEATURES], val[TARGET]
    Xte, yte = test[FEATURES], test[TARGET]

    rows = []

    def record(name, split, m):
        rows.append({"model": name, "split": split, **m})

    # 2) Baselines
    dummy = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    for split, X, y in [("val", Xva, yva), ("test", Xte, yte)]:
        record("baseline_majority", split, metrics(y, dummy.predict_proba(X)[:, 1], np.zeros(len(y), dtype=int)))
        record("baseline_rule_prior_noshow>=1", split,
               metrics(y, X["prior_no_show_rate"].values, (X["prior_no_shows"].values >= 1).astype(int)))

    # 3) Candidates: small grid on VALIDATION (PR-AUC), then threshold = best F1 on validation
    best = {}
    for name, grid in GRIDS.items():
        top = (-1, None)
        for params in ParameterGrid(grid):
            pipe = make_model(name, **params).fit(Xtr, ytr)
            ap = average_precision_score(yva, pipe.predict_proba(Xva)[:, 1])
            if ap > top[0]:
                top = (ap, params)
        pipe = make_model(name, **top[1]).fit(Xtr, ytr)
        s_va, s_te = pipe.predict_proba(Xva)[:, 1], pipe.predict_proba(Xte)[:, 1]
        thr = best_f1_threshold(yva, s_va)
        record(name, "val", metrics(yva, s_va, (s_va >= thr).astype(int)))
        record(name, "test", metrics(yte, s_te, (s_te >= thr).astype(int)))
        best[name] = {"pipe": pipe, "params": top[1], "val_ap": top[0]}
        print(f"{name}: best params {top[1]}  val PR-AUC {top[0]:.4f}")

    comp = pd.DataFrame(rows)
    comp.to_csv(REPORT_DIR / "model_comparison.csv", index=False)
    print(comp.to_string(index=False))

    # 4) Final model = best validation PR-AUC
    final_name = max(best, key=lambda k: best[k]["val_ap"])
    pipe = best[final_name]["pipe"]
    print("FINAL MODEL:", final_name, best[final_name]["params"])
    s_va, s_te = pipe.predict_proba(Xva)[:, 1], pipe.predict_proba(Xte)[:, 1]

    # 5) Calibration (fit on validation only)
    cal = Calibrator("sigmoid").fit(s_va, yva)
    p_va, p_te = cal.predict(s_va), cal.predict(s_te)
    calib = {"brier_raw_test": round(brier_score_loss(yte, s_te), 4),
             "brier_calibrated_test": round(brier_score_loss(yte, p_te), 4),
             "ece_raw_test": round(expected_calibration_error(yte, s_te), 4),
             "ece_calibrated_test": round(expected_calibration_error(yte, p_te), 4)}
    print("calibration:", calib)
    plt.figure(figsize=(5, 5))
    for lab, p in [("raw (class-weighted)", s_te), ("calibrated", p_te)]:
        fp_, mp_ = calibration_curve(yte, p, n_bins=10, strategy="quantile")
        plt.plot(mp_, fp_, marker="o", label=lab)
    plt.plot([0, 1], [0, 1], "k--", label="perfect")
    plt.xlabel("predicted probability"); plt.ylabel("observed no-show rate")
    plt.title("Calibration (test set)"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "calibration.png", dpi=120); plt.close()

    # 6) Cost-based threshold chosen on validation
    grid_t = np.linspace(0.02, 0.9, 89)
    val_costs = [total_cost(yva, (p_va >= t).astype(int)) for t in grid_t]
    thr = float(grid_t[int(np.argmin(val_costs))])
    plt.figure(figsize=(6, 4))
    plt.plot(grid_t, val_costs); plt.axvline(thr, color="r", ls="--", label=f"chosen = {thr:.2f}")
    plt.xlabel("threshold"); plt.ylabel(f"cost (FN={COST_FN}, FP={COST_FP})")
    plt.title("Validation cost vs threshold"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG / "cost_vs_threshold.png", dpi=120); plt.close()

    # 7) FINAL one-time test evaluation
    pred_te = (p_te >= thr).astype(int)
    final = metrics(yte, p_te, pred_te)
    cost_model = total_cost(yte, pred_te)
    cost_all = total_cost(yte, np.ones(len(yte), dtype=int))
    cost_none = total_cost(yte, np.zeros(len(yte), dtype=int))
    top20 = p_te >= np.quantile(p_te, 0.80)
    top20_recall = float(yte[top20].sum() / yte.sum())
    top20_precision = float(yte[top20].mean())
    print("FINAL TEST @ threshold", round(thr, 3), final)
    print(f"cost: model={cost_model}  remind_all={cost_all}  remind_none={cost_none}")
    print(f"top-20% flagged: precision={top20_precision:.3f} recall={top20_recall:.3f}")

    # 7b) Cost sensitivity: how the best policy changes with the assumed cost of a missed no-show
    sens = []
    yv, yt = yva.values, yte.values
    for fn_c in [1.5, 2, 3, 5, 8]:
        ts = np.linspace(0.02, 0.9, 89)
        vc = [((yv == 1) & (p_va < t)).sum() * fn_c + ((yv == 0) & (p_va >= t)).sum() * COST_FP for t in ts]
        t_b = float(ts[int(np.argmin(vc))])
        pr = (p_te >= t_b).astype(int)
        sens.append({"cost_fn": fn_c, "threshold": round(t_b, 2), "flag_rate": round(float(pr.mean()), 3),
                     "precision": round(float(yt[pr == 1].mean()) if pr.sum() else 0, 3),
                     "recall": round(float(pr[yt == 1].mean()), 3),
                     "cost_model": float(((yt == 1) & (pr == 0)).sum() * fn_c + ((yt == 0) & (pr == 1)).sum() * COST_FP),
                     "cost_remind_all": float((yt == 0).sum() * COST_FP),
                     "cost_remind_none": float(yt.sum() * fn_c)})
    sens_df = pd.DataFrame(sens)
    sens_df.to_csv(REPORT_DIR / "cost_sensitivity.csv", index=False)
    print("cost sensitivity:")
    print(sens_df.to_string(index=False))

    cm = np.array([[final["tn"], final["fp"]], [final["fn"], final["tp"]]])
    plt.figure(figsize=(4, 4)); plt.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    plt.xticks([0, 1], ["pred attend", "pred no-show"]); plt.yticks([0, 1], ["attended", "no-show"])
    plt.title("Confusion matrix (test)"); plt.tight_layout()
    plt.savefig(FIG / "confusion_matrix.png", dpi=120); plt.close()

    # 8) Subgroup check (bias discussion)
    sub = []
    for col in ["age_band", "distance_band"]:
        for g, idx in test.reset_index(drop=True).groupby(col).groups.items():
            yy, pp = yte.values[idx], pred_te[idx]
            sub.append({"attribute": col, "group": g, "n": len(idx),
                        "no_show_rate": round(float(yy.mean()), 3), "flag_rate": round(float(pp.mean()), 3),
                        "recall": round(float(pp[yy == 1].mean()) if yy.sum() else 0, 3),
                        "precision": round(float(yy[pp == 1].mean()) if pp.sum() else 0, 3)})
    pd.DataFrame(sub).to_csv(REPORT_DIR / "subgroup_metrics.csv", index=False)

    # 9) Global importance (validation set, not test)
    imp = permutation_importance(pipe, Xva, yva, scoring="average_precision", n_repeats=10, random_state=SEED)
    imp_df = pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean}).sort_values("importance")
    imp_df.to_csv(REPORT_DIR / "permutation_importance.csv", index=False)
    plt.figure(figsize=(6, 5)); plt.barh(imp_df["feature"], imp_df["importance"])
    plt.xlabel("drop in PR-AUC when shuffled"); plt.title("Permutation importance (validation)")
    plt.tight_layout(); plt.savefig(FIG / "permutation_importance.png", dpi=120); plt.close()

    # 10) Save artifact
    reference = {c: float(Xtr[c].median()) for c in NUMERIC}
    reference.update({c: Xtr[c].mode().iloc[0] for c in CATEGORICAL})
    bundle = {"pipeline": pipe, "calibrator": cal, "threshold": thr, "features": FEATURES,
              "numeric": NUMERIC, "categorical": CATEGORICAL,
              "categories": {c: sorted(Xtr[c].dropna().unique().tolist()) for c in CATEGORICAL},
              "reference": reference,
              "metadata": {"model": final_name, "params": best[final_name]["params"], "seed": SEED,
                           "cost_fn": COST_FN, "cost_fp": COST_FP, "sklearn": sklearn.__version__,
                           "split": split_info, "test_metrics": final, "calibration": calib,
                           "train_lead_time_max": int(Xtr["lead_time_days"].max())}}
    joblib.dump(bundle, MODEL_PATH)
    with open(REPORT_DIR / "metrics.json", "w") as f:
        json.dump({"final_model": final_name, "params": best[final_name]["params"],
                   "threshold": thr, "test": final, "calibration": calib,
                   "cost": {"model": cost_model, "remind_all": cost_all, "remind_none": cost_none},
                   "top20": {"precision": top20_precision, "recall": top20_recall},
                   "split": split_info, "cleaning": clean_report}, f, indent=2, default=float)
    print("saved", MODEL_PATH)


if __name__ == "__main__":
    main()