# Drop-in dataset for binary top-tagging on the PELICAN-nano h5 files.
#
# Subclasses HLS4MLData150 so standardisation, kfolding, constituent shuffling, and
# show_details are inherited verbatim. Only data acquisition (no Zenodo download) and the
# feature build (4-momenta -> jet-relative pT/eta/phi) are overridden.

from pathlib import Path

import h5py
import numpy as np

from fast_jetclass.data.data import HLS4MLData150
from fast_jetclass.data import standardization


def fourmom_to_ptetaphirel(Pmu: np.ndarray) -> np.ndarray:
    """(njets, N, 4) in (E, px, py, pz)  ->  (njets, N, 3) = (pT, eta_rel, phi_rel).

    Angles are relative to the per-jet axis (sum of the constituent 4-vectors).
    Zero-padded constituents (E == 0) stay exactly zero.
    """
    E, px, py, pz = (Pmu[..., i] for i in range(4))
    mask = E != 0.0
    pt = np.sqrt(px**2 + py**2)
    eta = np.arcsinh(np.divide(pz, pt, out=np.zeros_like(pz), where=pt > 0))
    phi = np.arctan2(py, px)
    # jet axis from the summed 4-vector (over the constituent axis)
    jpx, jpy, jpz = px.sum(1), py.sum(1), pz.sum(1)
    jpt = np.sqrt(jpx**2 + jpy**2)
    jeta = np.arcsinh(np.divide(jpz, jpt, out=np.zeros_like(jpz), where=jpt > 0))[:, None]
    jphi = np.arctan2(jpy, jpx)[:, None]
    eta_rel = eta - jeta
    phi_rel = (phi - jphi + np.pi) % (2 * np.pi) - np.pi  # wrap to (-pi, pi]
    out = np.stack([pt, eta_rel, phi_rel], axis=-1)
    out[~mask] = 0.0
    return out.astype(np.float32)


def _select_leading_pt(Pmu, nconst):
    """Keep the nconst highest-pT constituents per jet (pT^2 = px^2 + py^2), padding
    with zeros if fewer are available.

    Matches nanoPELICAN's leading-``--nobj`` cap so both models see the same particles.
    Zero-padded slots (all-zero 4-vectors) have pT=0 and sort last. A no-op when Pmu is
    already pT-sorted and exactly nconst-wide; the DeepSet is permutation-invariant so any
    reordering among the kept constituents is irrelevant.
    """
    n = Pmu.shape[1]
    if n < nconst:
        return np.pad(Pmu, ((0, 0), (0, nconst - n), (0, 0)))
    pt2 = Pmu[..., 1] ** 2 + Pmu[..., 2] ** 2            # (njets, n)
    idx = np.argsort(-pt2, axis=1)[:, :nconst]            # leading-nconst per jet
    return np.take_along_axis(Pmu, idx[..., None], axis=1)


class TopTagData(HLS4MLData150):
    """Binary top-tagging data from a PELICAN-nano h5 file.

    Reads ``Pmu`` (njets, 20, 4) in (E, px, py, pz) and ``is_signal`` (njets,) {0,1},
    converts to jet-relative (pT, eta_rel, phi_rel), trims/pads to ``nconst``, and reuses
    the base class' robust/minmax/standard standardisation (fit on TRAIN, reused for
    val/test via the inherited normparams pkl).
    """

    def __init__(self, root, nconst, feats, norm, train, kfolds, seed=None,
                 h5_path=None):
        self.h5_path = h5_path  # consumed by _get_raw_data / _get_processed_data below
        super().__init__(root, nconst, feats, norm, train, kfolds, seed)

    def _get_raw_data(self):
        # Skip the Zenodo download entirely; raw data is the configured h5 file.
        return Path(self.h5_path).parent

    def _get_processed_data(self):
        with h5py.File(self.h5_path, "r") as f:
            Pmu = np.asarray(f["Pmu"])  # (njets, 20, 4) in (E, px, py, pz)
            y = np.asarray(f["is_signal"]).astype(int)

        # Keep the leading-nconst constituents by pT BEFORE building features, so the
        # DeepSet sees the same particles nanoPELICAN's --nobj cap keeps (leading pT).
        # Works for any Pmu width (e.g. toptag is 200-wide, sample_data was 20-wide) and
        # is a no-op when Pmu is already pT-sorted and exactly nconst-wide.
        Pmu = _select_leading_pt(Pmu, self.nconst)  # (njets, nconst, 4)
        x = fourmom_to_ptetaphirel(Pmu)  # jet axis from the kept nconst; (njets, nconst, 3)

        self.y = np.eye(2, dtype=np.float32)[y]  # binary one-hot, argmax-compatible

        # Fit normalisation on TRAIN, reuse for val/test (mirrors the base pkl cache).
        if self.train:
            self.norm_params = standardization.fit_standardisation(self.norm, x)
            self._save_norm_parameters()
        else:
            self.norm_params = self._get_normalisation_params()  # inherited: loads pkl

        self.x = standardization.apply_standardisation(self.norm, x, self.norm_params)

        if self.seed and self.train:
            self.shuffle_constituents(self.seed)
