"""Run npepfn_ythetaonly_npepfn for all tasks, loading cached results for other methods."""

import importlib.util
import json
import sys
from pathlib import Path

import ml_collections

# All methods except the new one
SKIP = [
    "npepfn_misspec",
    "npepfn_calib",
    "npepfn_mixed",
    "npe_sbi",
    "npepfn_y_fmpe",
    "npepfn_y_npepfn",
    "npepfn_y_fmpe_concat",
    "npepfn_y_npepfn_concat",
]

TASKS = [
    "gaussian_mixture",
    "two_moons",
    "slcp",
    "sir",
    "lotka_volterra",
]


def main():
    for task in TASKS:
        print(f"\n{'=' * 60}")
        print(f"  {task}")
        print(f"{'=' * 60}")

        # Import the task config
        config_path = Path(f"configs/{task}.py")
        spec = importlib.util.spec_from_file_location("cfg", config_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        cfg = mod.get_config()

        # Override skip_methods and use_cache
        cfg.skip_methods = SKIP
        cfg.use_cache = True

        from scripts.run_calibration_sweep import _OUTPUT_DIR
        from tabpfn_misspec import evaluate_calibrated_misspecification

        out_dir = Path("results")
        calib_sizes = [10, 50, 200, 1000]
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
                    skip_methods=list(cfg.skip_methods),
                    batch_size=cfg.batch_size,
                    cache_data=cfg.get("cache_data", False),
                    use_cache=cfg.get("use_cache", True),
                    augment_M=cfg.get("augment_M", 1),
                    metrics_to_compute=list(
                        cfg.get("metrics_to_compute", ("c2st", "mmd"))
                    ),
                )
                for r in results:
                    r.seed = seed
                all_results[n_calib].extend(results)

        out_file = out_dir / f"{cfg.task}_{cfg.misspec_type}_sweep.json"
        serialized = {
            str(n): [r.to_dict() for r in res] for n, res in all_results.items()
        }
        with open(out_file, "w") as f:
            json.dump(serialized, f, indent=2)
        print(f"\nResults saved to {out_file}")


if __name__ == "__main__":
    main()
