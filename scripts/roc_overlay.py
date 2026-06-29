"""C6 — ROC overlay: DeepSet vs nanoPELICAN(float) vs nanoPELICAN(firmware).

One ROC axis, one curve per model, on the SAME held-out top-tag test set
(PELICAN-nano test.h5). Writes:
  <out-dir>/roc_overlay.png
  <out-dir>/roc_summary.csv   (model, AUC, inv_eps_b@0.3, accuracy, n)

Score-file conventions (every file is aligned 1:1 to test.h5 jet order):
  * DeepSet  : y_pred.dat from `deepsets_test` (float32, flattened njets x 2 softmax
               probabilities). The signal score is column 1 (p_top).
  * nPELICAN : one signal LOGIT per line (higher = more top-like). Float build or
               firmware csim over the same test jets. (ROC/AUC/1-eps_B are rank-based,
               so a raw logit and its sigmoid give identical curves.)

Labels come from test.h5 `is_signal` (0/1), read in file order — the same order the
DeepSet adapter and the firmware `full_*.dat` export use, so all rows correspond.

Pure numpy/sklearn/matplotlib/h5py — does NOT import tensorflow or fast_jetclass, so it
runs in any environment with those four packages.

Example (run after C5 training/test + nPELICAN logit export):
  python roc_overlay.py \
      --labels-h5 ../../PELICAN-nano/data/sample_data/test.h5 \
      --deepset   ../trained_deepsets/deepsets_8bit_20const_toptag/kfolding1/plots_123/y_pred.dat \
      --npelican-float ../../results/npelican_float_logits_test.dat \
      --npelican-fw    ../../results/npelican_fw_logits_test.dat \
      --out-dir ../../results
"""
import argparse
import csv
import os

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


def inv_eps_b_at(y, scores, eps_s=0.3):
    """1 / background-efficiency at the threshold giving signal-efficiency eps_s.

    Field-standard top-tagging discrimination scalar (higher = better). Matches
    nPELICAN-fpga/equivariance/equiv_common.inv_eps_b_at so the numbers are comparable.
    """
    y = np.asarray(y).astype(int)
    s = np.asarray(scores, dtype=np.float64)
    sig, bkg = s[y == 1], s[y == 0]
    if len(sig) == 0 or len(bkg) == 0:
        return float("nan")
    thr = np.quantile(sig, 1.0 - eps_s, method="lower")
    eps_b = float((bkg >= thr).mean())
    return float("inf") if eps_b <= 0.0 else 1.0 / eps_b


def load_scores(path, njets):
    """Load a score file and return a per-jet signal score aligned to test.h5 order."""
    arr = np.fromfile(path, dtype=np.float32)
    if arr.size == njets:
        return arr.astype(np.float64)                 # one logit/score per jet
    if arr.size == 2 * njets:
        return arr.reshape(njets, 2)[:, 1].astype(np.float64)  # softmax p_top
    # fall back to whitespace text if it was not a raw float32 blob
    arr = np.loadtxt(path, dtype=np.float64).reshape(-1)
    if arr.size == njets:
        return arr
    if arr.size == 2 * njets:
        return arr.reshape(njets, 2)[:, 1]
    raise ValueError(f"{path}: {arr.size} values, expected {njets} or {2*njets}")


def accuracy(y, scores, two_col_argmax=False, raw=None):
    """Top-class accuracy. DeepSet: argmax of the 2 softmax cols. nPELICAN: logit>0."""
    if two_col_argmax and raw is not None and raw.size == 2 * len(y):
        pred = raw.reshape(len(y), 2).argmax(1)
    else:
        pred = (np.asarray(scores) > 0).astype(int)
    return float((pred == np.asarray(y).astype(int)).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels-h5", required=True,
                   help="test.h5 providing is_signal (jet order = score-file order).")
    p.add_argument("--deepset", default=None, help="DeepSet y_pred.dat (njets x 2).")
    p.add_argument("--npelican-float", default=None, help="nPELICAN float logits (1/jet).")
    p.add_argument("--npelican-fw", default=None, help="nPELICAN firmware logits (1/jet).")
    p.add_argument("--out-dir", default="results")
    a = p.parse_args()

    with h5py.File(a.labels_h5, "r") as f:
        y = np.asarray(f["is_signal"]).astype(int)
    njets = len(y)
    print(f"labels: {njets} jets, signal frac {y.mean():.3f}")

    # (display name, path, is_deepset)
    entries = [("DeepSet (8-bit)", a.deepset, True),
               ("nanoPELICAN (float)", a.npelican_float, False),
               ("nanoPELICAN (firmware)", a.npelican_fw, False)]

    os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    plt.figure(figsize=(6, 5))
    for name, path, is_ds in entries:
        if not path:
            continue
        if not os.path.exists(path):
            print(f"  WARN: {name}: {path} missing, skipping")
            continue
        raw = np.fromfile(path, dtype=np.float32)
        score = load_scores(path, njets)
        auc = roc_auc_score(y, score)
        ieb = inv_eps_b_at(y, score, eps_s=0.3)
        acc = accuracy(y, score, two_col_argmax=is_ds, raw=raw)
        fpr, tpr, _ = roc_curve(y, score)
        # background rejection vs signal efficiency (standard top-tag axes)
        with np.errstate(divide="ignore"):
            rej = np.where(fpr > 0, 1.0 / fpr, np.inf)
        plt.plot(tpr, rej, label=f"{name}: AUC={auc:.3f}, 1/eps_B@0.3={ieb:.0f}")
        rows.append({"model": name, "AUC": f"{auc:.4f}",
                     "inv_eps_b@0.3": f"{ieb:.3f}", "accuracy": f"{acc:.4f}",
                     "n": njets})
        print(f"  {name}: AUC={auc:.4f}  1/eps_B@0.3={ieb:.2f}  acc={acc:.4f}")

    if not rows:
        raise SystemExit("No model score files provided/found; nothing to plot.")

    plt.xlabel(r"Signal efficiency $\epsilon_S$ (top)")
    plt.ylabel(r"Background rejection $1/\epsilon_B$")
    plt.yscale("log")
    plt.xlim(0, 1)
    plt.legend(fontsize=8)
    plt.title("Top-tagging ROC: DeepSet vs nanoPELICAN")
    plt.tight_layout()
    png = os.path.join(a.out_dir, "roc_overlay.png")
    plt.savefig(png, dpi=130)
    plt.close()

    csv_path = os.path.join(a.out_dir, "roc_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "AUC", "inv_eps_b@0.3",
                                          "accuracy", "n"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {png}\nwrote {csv_path}")


if __name__ == "__main__":
    main()
