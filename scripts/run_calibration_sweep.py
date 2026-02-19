"""Sweep over calibration set sizes for a single task."""

import json
from pathlib import Path

from absl import app, flags
from ml_collections import config_flags

from tabpfn_misspec import evaluate_calibrated_misspecification

_CONFIG = config_flags.DEFINE_config_file(
    "config", "configs/experiment.py", "Path to experiment config."
)
_OUTPUT_DIR = flags.DEFINE_string("output_dir", "results", "Output directory.")


def main(_):
    cfg = _CONFIG.value
    out_dir = Path(_OUTPUT_DIR.value)
    out_dir.mkdir(parents=True, exist_ok=True)

    calib_sizes = [10, 50, 200, 1000]
    all_results = {}

    for n_calib in calib_sizes:
        print(f"\n--- num_calibration = {n_calib} ---")
        all_results[n_calib] = evaluate_calibrated_misspecification(
            task_name=cfg.task,
            misspec_type=cfg.misspec_type,
            misspec_kwargs=dict(cfg.misspec_kwargs),
            num_simulations=cfg.num_simulations,
            num_calibration=n_calib,
            num_posterior_samples=cfg.num_posterior_samples,
            num_observations=cfg.num_observations,
            num_synthetic=cfg.num_synthetic,
            seed=cfg.seed,
            use_prior_transform=cfg.use_prior_transform,
        )

    out_file = out_dir / f"{cfg.task}_{cfg.misspec_type}_sweep.json"
    serialized = {
        str(n): [r.to_dict() for r in results]
        for n, results in all_results.items()
    }
    with open(out_file, "w") as f:
        json.dump(serialized, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    app.run(main)
