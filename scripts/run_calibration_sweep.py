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
_CALIB_SIZES = flags.DEFINE_list(
    "calib_sizes",
    None,
    "Comma-separated list of calibration sizes (e.g. '10,50,200,1000'). "
    "Default None uses [10, 50, 200, 1000].",
)


def main(_):
    cfg = _CONFIG.value
    out_dir = Path(_OUTPUT_DIR.value)
    out_dir.mkdir(parents=True, exist_ok=True)

    calib_sizes = (
        [int(n) for n in _CALIB_SIZES.value]
        if _CALIB_SIZES.value is not None
        else [10, 50, 200, 1000]
    )
    seeds = list(getattr(cfg, "seeds", [cfg.seed]))
    all_results = {}

    for n_calib in calib_sizes:
        all_results[n_calib] = []
        for seed in seeds:
            print(f"\n{'#' * 60}")
            print(f"### n_calib={n_calib}  seed={seed}")
            print(f"{'#' * 60}")
            artifacts_path = (
                out_dir
                / f"{cfg.task}_{cfg.misspec_type}"
                / "artifacts"
                / f"ncalib{n_calib}_seed{seed}"
            )
            results = evaluate_calibrated_misspecification(
                task_name=cfg.task,
                misspec_type=cfg.misspec_type,
                misspec_kwargs=dict(cfg.misspec_kwargs),
                num_sim_mixed=cfg.num_sim_mixed,
                num_calibration=n_calib,
                num_posterior_samples=cfg.num_posterior_samples,
                num_observations=cfg.num_observations,
                num_context=cfg.num_context,
                seed=seed,
                use_prior_transform=cfg.use_prior_transform,
                artifacts_dir=artifacts_path,
                skip_methods=list(cfg.get("skip_methods", [])),
                batch_size=cfg.batch_size,
                cache_data=cfg.get("cache_data", False),
                use_cache=cfg.get("use_cache", True),
                augment_M=cfg.get("augment_M", 1),
                metrics_to_compute=list(cfg.get("metrics_to_compute", ("c2st", "mmd"))),
                train_batch_size=cfg.get("train_batch_size", 1024),
            )
            for r in results:
                r.seed = seed
            all_results[n_calib].extend(results)

    out_file = out_dir / f"{cfg.task}_{cfg.misspec_type}_sweep.json"
    serialized = {str(n): [r.to_dict() for r in results] for n, results in all_results.items()}
    with open(out_file, "w") as f:
        json.dump(serialized, f, indent=2)
    print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    app.run(main)
