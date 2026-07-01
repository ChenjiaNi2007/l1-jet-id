# DeepSet↔nanoPELICAN comparison — see the full handoff in the other repo

This DeepSet config trio is the l1-jet-id side of the DeepSet-vs-nanoPELICAN comparison.

**Full state + runbook:** `nPELICAN-fpga/comparison/HANDOFF.md` (branch `deepset-equivariance`).

TL;DR for this repo (branch `toptag-comparison`):
- Configs now point at **`data/toptag`** (nanoPELICAN's dataset), not `sample_data` — the two
  models were on different datasets, which is why nanoPELICAN looked like 0.70 vs its real 0.95.
- `fast_jetclass/data/toptag_data.py` selects the **leading-20 by pT** from toptag's 200-wide
  `Pmu` (matches nanoPELICAN's `--nobj=20`).
- **Next:** get GPU TF working on the pod, then
  `./deepsets_train --config configs/deepsets_8bit_20const_toptag/deepsets_8bit_20const_toptag.yml --gpu 0`
  (use `--gpu 0`, NEVER `--gpu ""`), then test with
  `--h5_path ../../PELICAN-nano/data/toptag/test.h5`, then regenerate the equivariance overlay
  (`nPELICAN-fpga/equivariance/{run_sweep_deepset.py, overlay_deepset.py}`) and the C6 ROC.
