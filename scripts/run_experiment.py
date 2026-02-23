"""CLI entry point for misspecification experiments."""

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

    print(f"Task: {cfg.task}, Misspec: {cfg.misspec_type}, kwargs: {cfg.misspec_kwargs}")
    print(f"num_calibration={cfg.num_calibration}")

    results = evaluate_calibrated_misspecification(
        task_name=cfg.task,
        misspec_type=cfg.misspec_type,
        misspec_kwargs=dict(cfg.misspec_kwargs),
        num_simulations=cfg.num_simulations,
        num_calibration=cfg.num_calibration,
        num_posterior_samples=cfg.num_posterior_samples,
        num_observations=cfg.num_observations,
        num_synthetic=cfg.num_synthetic,
        seed=cfg.seed,
        use_prior_transform=cfg.use_prior_transform,
        skip_methods=list(cfg.get("skip_methods", [])),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{cfg.task}_{cfg.misspec_type}_calibrated_n{cfg.num_calibration}.json"
    with open(out_file, "w") as f:
        json.dump([r.to_dict() for r in results], f, indent=2)
    print(f"Results saved to {out_file}")


if __name__ == "__main__":
    app.run(main)
