"""Generate plots from saved sweep JSON files."""

import json
from pathlib import Path

from absl import app, flags

from tabpfn_misspec.plotting import plot_calibration_comparison

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

        plot_calibration_comparison(results_by_n_calib, metric="c2st", output_dir=_OUTPUT_DIR.value, task_name=task_name)
        plot_calibration_comparison(results_by_n_calib, metric="mmd", output_dir=_OUTPUT_DIR.value, task_name=task_name)


if __name__ == "__main__":
    app.run(main)
