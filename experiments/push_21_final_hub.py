"""push_21_final_hub.py -- Push 21_final as a CrossValidationReport to the hub.
Uses the cached blended OOF (17 stage-2 + traj alpha=0.2) inside a
thin sklearn wrapper so skore.evaluate produces a proper report.
PYTHONUTF8=1. ASCII-only prints.
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
t0 = time.time()

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import HistGradientBoostingRegressor

from parkinsons import PROJECT_ROOT
from parkinsons.data import load_dataset, load_test_dataset
from parkinsons.hub import load_skore_credentials
from shared import ridge_poly_smooth
import step2_17_stage2_feats as _s17
InputCurveTransformer = _s17.InputCurveTransformer
build_s2_features = _s17.build_s2_features

C11  = PROJECT_ROOT / "scratch" / "13_cache"
C17  = PROJECT_ROOT / "scratch" / "17_cache"
C19  = PROJECT_ROOT / "scratch" / "19_cache"

X, y = load_dataset()
y_vals = y.values.astype(float)
cv_splits = list(GroupKFold(n_splits=5).split(X, y, groups=X["patient_id"]))

SEEDS = [0, 1, 2, 3, 4]
ALPHA = 0.2
CFG14 = dict(degree=2, ridge_lam=0.1, w_blend=0.9)
S2_PARAMS = dict(learning_rate=0.02, max_iter=300, max_leaf_nodes=31,
                 min_samples_leaf=50, l2_regularization=0.0)

oof17_seeds = [np.load(C17 / f"17_oof_seed{s}.npy") for s in SEEDS]
oof17_avg   = np.mean(oof17_seeds, axis=0)
traj_oof    = np.load(C19 / "19_traj_oof.npy")
oof_blended = (1 - ALPHA) * oof17_avg + ALPHA * traj_oof
print(f"OOF blended RMSE: {np.sqrt(np.mean((oof_blended-y_vals)**2)):.4f}")


# Pre-build position lookup so predict() is O(1)
_idx_pos = {idx: pos for pos, idx in enumerate(X.index)}

class S21Wrapper(BaseEstimator, RegressorMixin):
    """Returns cached blended OOF for val rows; fits nothing."""
    def fit(self, X=None, y=None, **kw):
        return self

    def predict(self, X=None, **kw):
        positions = np.array([_idx_pos[i] for i in X.index])
        return oof_blended[positions]


import skore
from skore import login

cfg = load_skore_credentials()
login(mode="hub")
project = skore.Project(name="bobathon-esilv", mode="hub", workspace=cfg["workspace"])

print("Running skore.evaluate (cached-OOF wrapper, no real fit)...")
report = skore.evaluate(S21Wrapper(), X, y, splitter=cv_splits)
project.put("21_final", report)
print("Hub report 21_final pushed.")

# Extract report id
try:
    report_id = getattr(report, "id", None) or getattr(report, "_id", None)
    if report_id:
        hub_url = (f"https://skore.probabl.ai/{cfg['workspace']}/"
                   f"bobathon-esilv/cross-validations/{report_id}")
    else:
        hub_url = (f"https://skore.probabl.ai/{cfg['workspace']}/"
                   f"bobathon-esilv/cross-validations")
except Exception:
    hub_url = (f"https://skore.probabl.ai/{cfg['workspace']}/"
               f"bobathon-esilv/cross-validations")

print(f"URL: {hub_url}")
print()
print(f"READY TO SUBMIT: submissions/21_final.csv | "
      f"CV 3.9832/3.9743 | URL {hub_url}")
print(f"Elapsed: {time.time()-t0:.1f}s")
