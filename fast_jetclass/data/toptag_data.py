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

        x = fourmom_to_ptetaphirel(Pmu)  # (njets, 20, 3)

        # Restrict / pad the constituent axis to nconst (h5 is 20-wide).
        if self.nconst <= x.shape[1]:
            x = x[:, : self.nconst, :]
        else:
            x = np.pad(x, ((0, 0), (0, self.nconst - x.shape[1]), (0, 0)))

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
