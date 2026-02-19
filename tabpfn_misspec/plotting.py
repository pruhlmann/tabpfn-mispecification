"""Plotting utilities for calibration comparison experiments."""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def plot_calibration_comparison(
    results_by_n_calib, metric="c2st", output_dir="results", task_name=None
):
    """Line plot comparing methods across calibration set sizes.

    Args:
        results_by_n_calib: dict mapping num_calibration → list[EvalResult].
            Each EvalResult must have a `method` field.
        metric: "c2st" or "mmd".
        output_dir: Directory to save the figure.
    """
    # Collect metric values per method across calibration sizes
    # method -> (list of x values, list of y values)
    method_data = defaultdict(lambda: ([], []))

    for n_calib in sorted(results_by_n_calib.keys()):
        results = results_by_n_calib[n_calib]
        # Group by method, average over observations
        method_values = defaultdict(list)
        for r in results:
            val = r[metric] if isinstance(r, dict) else getattr(r, metric)
            method = r["method"] if isinstance(r, dict) else r.method
            method_values[method].append(val)

        for method, values in method_values.items():
            xs, ys = method_data[method]
            xs.append(n_calib)
            ys.append(sum(values) / len(values))

    fig, ax = plt.subplots(figsize=(8, 5))

    for method, (xs, ys) in sorted(method_data.items()):
        # npepfn_misspec has no calibration data — plot as horizontal line
        if method == "npepfn_misspec":
            ax.axhline(y=ys[0], linestyle="--", label=method, alpha=0.7)
        else:
            ax.plot(xs, ys, marker="o", label=method)

    ax.set_xlabel("Number of calibration samples")
    ax.set_ylabel(metric.upper())
    ax.set_xscale("log")
    title = f"{metric.upper()} vs calibration set size"
    if task_name:
        title = f"{task_name}: {title}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{task_name}_" if task_name else ""
    out_path = out_dir / f"{prefix}calibration_comparison_{metric}.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
