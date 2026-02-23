"""Generate plots from saved sweep JSON files."""

import json
import re
from pathlib import Path

import torch
from absl import app, flags

from tabpfn_misspec.plotting import (
    plot_calibration_comparison,
    plot_calibration_comparison_seeds,
    plot_posterior_pairplot,
    plot_sweep_figure,
    plot_y_diagnostics,
)

_INPUT_DIR = flags.DEFINE_string("input_dir", "results", "Directory containing sweep JSON files.")
_OUTPUT_DIR = flags.DEFINE_string("output_dir", "results", "Output directory for plots.")


def main(_):
    input_dir = Path(_INPUT_DIR.value)
    sweep_files = sorted(input_dir.glob("*_sweep.json"))

    if not sweep_files:
        print(f"No *_sweep.json files found in {input_dir}")
        return

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

        if has_seeds:
            plot_fn = plot_calibration_comparison_seeds
        else:
            plot_fn = plot_calibration_comparison

        plot_fn(results_by_n_calib, metric="c2st", output_dir=_OUTPUT_DIR.value, task_name=task_name)
        plot_fn(results_by_n_calib, metric="mmd", output_dir=_OUTPUT_DIR.value, task_name=task_name)

        # Two-panel figure for papers
        plot_sweep_figure(results_by_n_calib, output_dir=_OUTPUT_DIR.value, task_name=task_name)

    # --- Posterior pairplots and y-diagnostics from artifacts ---
    output_dir = Path(_OUTPUT_DIR.value)
    for artifacts_root in sorted(input_dir.glob("*/artifacts")):
        task_name = artifacts_root.parent.name
        for run_dir in sorted(artifacts_root.iterdir()):
            if not run_dir.is_dir():
                continue
            run_tag = run_dir.name  # e.g. "ncalib10_seed42"

            # Discover observation indices from reference files
            ref_files = sorted(run_dir.glob("reference_seed*_obs*.pt"))
            for ref_file in ref_files:
                m = re.search(r"_obs(\d+)\.pt$", ref_file.name)
                if m is None:
                    continue
                obs_idx = int(m.group(1))
                ref_samples = torch.load(ref_file, weights_only=True).numpy()

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

            # Y-diagnostics
            for y_diag_file in sorted(run_dir.glob("y_diag_seed*.pt")):
                data = torch.load(y_diag_file, weights_only=True)
                y_pred = data["y_pred"].numpy()
                y_true = data["y_true"].numpy()
                out_path = output_dir / task_name / "y_diagnostics" / f"{run_tag}.pdf"
                plot_y_diagnostics(y_pred, y_true, output_path=out_path)


if __name__ == "__main__":
    app.run(main)
