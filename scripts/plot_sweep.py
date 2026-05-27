"""Generate plots from saved sweep JSON files."""

import json
import math
import re
from pathlib import Path

# Load matplotlib before torch: torch's bundled libs pull in the system
# libstdc++.so.6 (CXXABI 1.3.13), which then masks the conda env's newer
# libstdc++ that matplotlib's C extension requires (CXXABI 1.3.15).
import matplotlib  # noqa: F401

import torch
from absl import app, flags

from tabpfn_misspec.plotting import (
    plot_calibration_comparison,
    plot_calibration_comparison_seeds,
    plot_posterior_pairplot,
    plot_sweep_figure,
    plot_y_distributional,
)

_INPUT_DIR = flags.DEFINE_string("input_dir", "results", "Directory containing sweep JSON files.")
_OUTPUT_DIR = flags.DEFINE_string("output_dir", "results", "Output directory for plots.")


def main(_):
    input_dir = Path(_INPUT_DIR.value)
    sweep_files = sorted(input_dir.glob("*_sweep.json"))

    if not sweep_files:
        print(f"No *_sweep.json files found in {input_dir}")
        return

    # Map each {task}_{misspec} combo (== artifacts dir name) to the real sbibm
    # task name, so the pairplot loop can regenerate reference posteriors when
    # a saved reference_*.pt is missing.
    combo_to_task = {}

    for sweep_file in sweep_files:
        # Extract task name from filename: {task}_{misspec_type}_sweep.json
        task_name = sweep_file.stem.rsplit("_sweep", 1)[0]
        print(f"\nPlotting {task_name} from {sweep_file}")

        with open(sweep_file) as f:
            raw = json.load(f)

        results_by_n_calib = {int(k): v for k, v in raw.items()}

        # Detect whether results contain seed information
        sample_result = next(iter(raw.values()))[0]
        has_seeds = "seed" in sample_result
        combo_to_task[task_name] = sample_result.get("task_name", task_name)

        if has_seeds:
            plot_fn = plot_calibration_comparison_seeds
        else:
            plot_fn = plot_calibration_comparison

        plot_fn(results_by_n_calib, metric="c2st", output_dir=_OUTPUT_DIR.value, task_name=task_name)
        plot_fn(results_by_n_calib, metric="mmd", output_dir=_OUTPUT_DIR.value, task_name=task_name)
        # Optional metrics: only plot those present and non-NaN in the results.
        for metric in ("log_prob", "sbc_ks", "tarp_ece"):
            present = any(
                math.isfinite(r.get(metric, float("nan")))
                for results in results_by_n_calib.values()
                for r in results
            )
            if present:
                plot_fn(
                    results_by_n_calib, metric=metric,
                    output_dir=_OUTPUT_DIR.value, task_name=task_name,
                )

        # Two-panel figure for papers
        plot_sweep_figure(results_by_n_calib, output_dir=_OUTPUT_DIR.value, task_name=task_name)

    # --- Posterior pairplots and y-diagnostics from artifacts ---
    output_dir = Path(_OUTPUT_DIR.value)
    for artifacts_root in sorted(input_dir.glob("*/artifacts")):
        task_name = artifacts_root.parent.name
        real_task = combo_to_task.get(task_name)
        task_obj = None  # lazily built only if a reference_*.pt is missing
        for run_dir in sorted(artifacts_root.iterdir()):
            if not run_dir.is_dir():
                continue
            run_tag = run_dir.name  # e.g. "ncalib10_seed42"
            seed_m = re.search(r"seed(\d+)", run_tag)
            seed = int(seed_m.group(1)) if seed_m else None

            # Discover observation indices from any saved posterior .pt (works
            # even when reference_*.pt is absent).
            obs_indices = set()
            for pt_file in run_dir.glob("*_obs*.pt"):
                m = re.search(r"_obs(\d+)\.pt$", pt_file.name)
                if m is not None:
                    obs_indices.add(int(m.group(1)))

            for obs_idx in sorted(obs_indices):
                # Reference posterior: prefer the saved .pt, else regenerate it
                # from the task (closed-form / sbibm-shipped, no models needed).
                ref_path = run_dir / f"reference_seed{seed}_obs{obs_idx}.pt"
                if ref_path.exists():
                    ref_samples = torch.load(ref_path, weights_only=True).numpy()
                elif real_task is not None:
                    if task_obj is None:
                        from tabpfn_misspec.tasks import get_task

                        task_obj = get_task(real_task)
                    ref_samples = task_obj.get_reference_posterior_samples(obs_idx).numpy()
                else:
                    continue  # no reference available, skip

                # Load method posteriors for this obs
                samples_by_method = {}
                for pt_file in run_dir.glob(f"*_obs{obs_idx}.pt"):
                    name = pt_file.name
                    if name.startswith("reference_"):
                        continue
                    # Extract method name: {method}_seed{S}_obs{O}.pt
                    method = name.rsplit("_seed", 1)[0]
                    samples_by_method[method] = torch.load(pt_file, weights_only=True).numpy()

                if samples_by_method:
                    out_path = output_dir / task_name / "pairplots" / f"{run_tag}_obs{obs_idx}.pdf"
                    plot_posterior_pairplot(samples_by_method, ref_samples, output_path=out_path)

            # Distributional y-diagnostics
            for y_diag_file in sorted(run_dir.glob("y_dist_diag_seed*.pt")):
                data = torch.load(y_diag_file, weights_only=True)
                theta_diag = data["theta_diag"].numpy()
                y_true = data["y_true"].numpy()
                y_tilde = data["y_tilde"].numpy()
                out_path = output_dir / task_name / "y_diagnostics" / f"{run_tag}_dist.pdf"
                plot_y_distributional(theta_diag, y_true, y_tilde, output_path=out_path)


if __name__ == "__main__":
    app.run(main)
