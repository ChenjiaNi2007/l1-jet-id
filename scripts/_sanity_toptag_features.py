"""Standalone Phase C sanity gate (no TensorFlow needed).

Validates the C1 feature conversion + label mapping against test.h5:
  - imports the REAL fourmom_to_ptetaphirel (tensorflow stubbed so toptag_data imports)
  - checks padded constituents stay zero, angle ranges, jet-relative centering
  - trains a trivial logistic regression on simple jet-level summaries of the
    converted features and reports AUC.  If AUC ~ 0.5 the conversion/label map is wrong;
    a sensible top-tag value (well above 0.5) means the features carry signal.

This does NOT exercise the DeepSet network; it only de-risks the data adapter before a
remote GPU training run.
"""
import ast
from pathlib import Path

import numpy as np
import h5py
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

# data.py uses py3.9+ generic annotations that the local base python (3.8) cannot import,
# so pull ONLY the self-contained pure-numpy fourmom_to_ptetaphirel out of toptag_data.py
# and exec it -- this validates the real shipped function, not a re-implementation.
_src = Path(__file__).resolve().parents[1] / "fast_jetclass" / "data" / "toptag_data.py"
_tree = ast.parse(_src.read_text())
_fn = next(n for n in _tree.body
           if isinstance(n, ast.FunctionDef) and n.name == "fourmom_to_ptetaphirel")
_ns = {"np": np}
exec(compile(ast.Module([_fn], []), str(_src), "exec"), _ns)
fourmom_to_ptetaphirel = _ns["fourmom_to_ptetaphirel"]

H5 = "../../PELICAN-nano/data/sample_data/test.h5"

with h5py.File(H5, "r") as f:
    Pmu = np.asarray(f["Pmu"])            # (njets, 20, 4) (E,px,py,pz)
    y = np.asarray(f["is_signal"]).astype(int)
    nobj = np.asarray(f["Nobj"]).astype(int)

x = fourmom_to_ptetaphirel(Pmu)           # (njets, 20, 3) = (pT, eta_rel, phi_rel)
print(f"converted x shape {x.shape}, dtype {x.dtype}")

# 1) padded constituents (E==0) must stay exactly zero across all 3 feats
padmask = Pmu[..., 0] == 0.0
assert np.all(x[padmask] == 0.0), "padded constituents are not zero!"
print(f"padded-zero check OK ({padmask.sum()} padded slots all zero)")

# 2) the number of nonzero-pT constituents per jet should equal Nobj
real_per_jet = (x[..., 0] != 0).sum(1)
agree = np.mean(real_per_jet == np.clip(nobj, 0, 20))
print(f"real-constituent count matches Nobj on {agree*100:.1f}% of jets")

# 3) angle ranges
phi = x[..., 2][~padmask]
eta = x[..., 1][~padmask]
print(f"phi_rel in [{phi.min():.3f}, {phi.max():.3f}] (expect ~[-pi,pi])")
print(f"eta_rel in [{eta.min():.3f}, {eta.max():.3f}]  mean {eta.mean():+.3f}")

# 4) cheap separability check: jet-level summary features -> logistic regression AUC.
#    (pT-weighted spread in eta/phi + multiplicity + leading/total pT; a top jet is
#     wider / higher-multiplicity than a QCD jet.)
pt = x[..., 0]
w = pt / (pt.sum(1, keepdims=True) + 1e-9)
eta_c = (w * x[..., 1]).sum(1)
phi_c = (w * x[..., 2]).sum(1)
d_eta = x[..., 1] - eta_c[:, None]
d_phi = (x[..., 2] - phi_c[:, None] + np.pi) % (2 * np.pi) - np.pi
girth = (w * np.sqrt(d_eta**2 + d_phi**2)).sum(1)          # pT-weighted angular spread
mult = real_per_jet.astype(float)
lead_frac = pt.max(1) / (pt.sum(1) + 1e-9)
ptsum = pt.sum(1)
feats = np.stack([girth, mult, lead_frac, ptsum], axis=1)
feats = (feats - feats.mean(0)) / (feats.std(0) + 1e-9)

Xtr, Xte, ytr, yte = train_test_split(feats, y, test_size=0.3, random_state=0,
                                      stratify=y)
clf = LogisticRegression(max_iter=1000).fit(Xtr, ytr)
auc = roc_auc_score(yte, clf.predict_proba(Xte)[:, 1])
print(f"\nlabel balance: signal frac = {y.mean():.3f}")
print(f"crude jet-summary logistic-regression AUC = {auc:.3f}")
print("GATE:", "PASS (features carry top-tag signal)" if auc > 0.7
      else "WEAK/FAIL (check conversion or label map)")
