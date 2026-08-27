#!/usr/bin/env python
"""Create an UNTRAINED model in the layout `deepsets_synth` expects.

Purpose: get resource/latency numbers without a GPU. Mirrors exactly what
`fast_jetclass/deepsets/train.py` writes -- `hyperparameters.json` at the model root
(saved BEFORE `input_size` is injected, same as train.py) and a TF-format model in
`kfolding1/` -- so the normal `deepsets_synth` path runs unmodified.

WHAT IS VALID with random weights:
  * latency, II, pipeline style, every datapath bit width  -- weight-independent
  * the relative effect of synthparams changes (ReuseFactor, Precision.maximum,
    Strategy) -- these are datapath-width questions
WHAT IS NOT:
  * the accuracy ratio (hls4ml vs QKeras) -- meaningless on an untrained model
  * LUT/DSP as a FINAL number. hls4ml feeds the count of exactly-zero weights into
    `multiplier_limit = DIV_ROUNDUP(k*n_chan*n_filt, RF) - nzeros/RF`, and a trained
    8-bit model quantises some weights to exactly 0 while a randomly-initialised one
    essentially never does. So this run is an UPPER BOUND on LUT/DSP.

Usage:
  python _untrained_stub.py --config_file configs/<cfg>/<cfg>.yml [--nfeats 3]
"""
import argparse
import yaml

import tensorflow as tf

from fast_jetclass.util import util
from fast_jetclass.deepsets import util as dsutil


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--config_file", type=str, required=True)
    parser.add_argument(
        "--nfeats", type=int, default=3,
        help="Features per constituent; 3 for feats='ptetaphi'.",
    )
    args = parser.parse_args()

    with open(args.config_file, "r") as stream:
        config = yaml.load(stream, Loader=yaml.Loader)

    # train.py saves the hyperparameters BEFORE build_model() injects input_size --
    # synthesize.py's functionalize_for_synth() passes input_size itself, so an
    # input_size key here would be a duplicate kwarg. Keep the order.
    outdir = util.make_output_directory("trained_deepsets", config["outdir"])
    util.save_hyperparameters_file(config, outdir)

    nconst = config["data_hyperparams"]["nconst"]
    input_size = (1, nconst, args.nfeats)
    config["model_hyperparams"].update({"input_size": input_size})

    model = dsutil.choose_deepsets(config["model_type"], config["model_hyperparams"])
    model.build(input_size)
    model(tf.zeros((1, nconst, args.nfeats)))  # materialise the subclassed call graph

    kfold_dir = util.make_output_directory(outdir, "kfolding1")
    model.save(kfold_dir, save_format="tf")
    print(f"\nUNTRAINED model saved to {kfold_dir}")
    print("Resource numbers from this are an UPPER BOUND -- see module docstring.")


if __name__ == "__main__":
    main()
