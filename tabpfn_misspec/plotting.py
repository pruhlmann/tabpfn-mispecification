"""Plotting utilities for calibration comparison experiments."""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


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


def plot_calibration_comparison_seeds(
    results_by_n_calib, metric="c2st", output_dir="results", task_name=None
):
    """Line plots with mean +/- std across seeds, per-observation and averaged.

    Args:
        results_by_n_calib: dict mapping num_calibration -> list[dict].
            Each dict must have "method", "seed", "num_observation", and the metric key.
        metric: "c2st" or "mmd".
        output_dir: Directory to save figures.
        task_name: Optional prefix for filenames and titles.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{task_name}_" if task_name else ""

    # Build lookup: (n_calib, method, obs) -> [metric values across seeds]
    lookup = defaultdict(list)
    observations = set()
    for n_calib, results in results_by_n_calib.items():
        for r in results:
            obs = r["num_observation"]
            observations.add(obs)
            lookup[(n_calib, r["method"], obs)].append(r[metric])

    sorted_n = sorted(results_by_n_calib.keys())
    observations = sorted(observations)

    # Collect all methods
    methods = sorted({r["method"] for results in results_by_n_calib.values() for r in results})

    # --- Per-observation plots ---
    for obs in observations:
        fig, ax = plt.subplots(figsize=(8, 5))
        for method in methods:
            xs, means, stds = [], [], []
            for n_calib in sorted_n:
                vals = lookup[(n_calib, method, obs)]
                if not vals:
                    continue
                xs.append(n_calib)
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            means, stds = np.array(means), np.array(stds)

            if method == "npepfn_misspec":
                ax.axhline(y=means[0], linestyle="--", label=method, alpha=0.7)
                ax.axhspan(means[0] - stds[0], means[0] + stds[0], alpha=0.1)
            else:
                ax.plot(xs, means, marker="o", label=method)
                ax.fill_between(xs, means - stds, means + stds, alpha=0.2)

        ax.set_xlabel("Number of calibration samples")
        ax.set_ylabel(metric.upper())
        ax.set_xscale("log")
        title = f"{metric.upper()} vs calibration size (obs {obs})"
        if task_name:
            title = f"{task_name}: {title}"
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

        out_path = out_dir / f"{prefix}calibration_seeds_{metric}_obs{obs}.pdf"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")

    # --- Averaged plot: pool all (seed, obs) pairs per (n_calib, method) ---
    fig, ax = plt.subplots(figsize=(8, 5))
    for method in methods:
        xs, means, stds = [], [], []
        for n_calib in sorted_n:
            # Pool across all observations and seeds
            vals = []
            for obs in observations:
                vals.extend(lookup[(n_calib, method, obs)])
            if not vals:
                continue
            xs.append(n_calib)
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        means, stds = np.array(means), np.array(stds)

        if method == "npepfn_misspec":
            ax.axhline(y=means[0], linestyle="--", label=method, alpha=0.7)
            ax.axhspan(means[0] - stds[0], means[0] + stds[0], alpha=0.1)
        else:
            ax.plot(xs, means, marker="o", label=method)
            ax.fill_between(xs, means - stds, means + stds, alpha=0.2)

    ax.set_xlabel("Number of calibration samples")
    ax.set_ylabel(metric.upper())
    ax.set_xscale("log")
    title = f"{metric.upper()} vs calibration size (averaged)"
    if task_name:
        title = f"{task_name}: {title}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)

    out_path = out_dir / f"{prefix}calibration_seeds_{metric}_avg.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")
