"""Run the notebook's experiment without an interactive kernel."""

import argparse
import os
from pathlib import Path

# Configure logging before TensorFlow is imported. Do not hide Python errors.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("quick", "full"), default="quick")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--threads", type=int, default=1,
                        help="TensorFlow CPU threads; adjust for your hardware")
    args = parser.parse_args()
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(args.threads)
    tf.config.threading.set_inter_op_parallelism_threads(args.threads)
    from hedging import Config, run_experiment
    config = Config()
    if args.profile == "quick":
        config = config.quick()
    output = args.output or Path("results") / args.profile
    run_experiment(config, output, profile=args.profile)


if __name__ == "__main__":
    main()
