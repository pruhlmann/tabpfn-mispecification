"""Plotting utilities for calibration comparison experiments."""

from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, pearsonr


# -- Method display names and visual style (Tol colorblind-safe palette) ------

METHOD_STYLE = {
    "npepfn_mixed": dict(label="NPE-PFN (mixed)", color="#228833", marker="o", ls="-"),
    "npepfn_calib": dict(label="NPE-PFN (calib.)", color="#EE6677", marker="^", ls="-"),
    "npepfn_misspec": dict(label="NPE-PFN (misspec.)", color="#4477AA", marker="s", ls="--"),
    "npe_sbi": dict(label="NPE (sbi)", color="#CCBB44", marker="D", ls="-"),
    "npepfn_y_fmpe": dict(label="NPE-PFN + FMPE", color="#AA3377", marker="v", ls="-."),
}

# Plotting order: "ours" first, then baselines
METHOD_ORDER = ["npepfn_mixed", "npepfn_calib", "npe_sbi", "npepfn_y_fmpe", "npepfn_misspec"]

_FALLBACK_COLORS = ["#66CCEE", "#CC6677", "#882255", "#117733", "#332288"]
_FALLBACK_MARKERS = ["P", "X", "h", "*", "p"]

METRIC_LABEL = {
    "c2st": r"C2ST $\downarrow$",
    "mmd": r"MMD $\downarrow$",
}


def _style(method):
    if method in METHOD_STYLE:
        return METHOD_STYLE[method]
    i = hash(method) % len(_FALLBACK_COLORS)
    return dict(label=method, color=_FALLBACK_COLORS[i], marker=_FALLBACK_MARKERS[i], ls="-")


def _apply_rc():
    """Set publication-quality rcParams (NeurIPS / ICML style)."""
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{amsmath}",
            "font.size": 8,
            "axes.labelsize": 9,
            "legend.fontsize": 7,
            "legend.handlelength": 1.8,
            "legend.columnspacing": 1.0,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "lines.linewidth": 1.5,
            "lines.markersize": 4,
            "axes.linewidth": 0.6,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "savefig.dpi": 300,
            "savefig.pad_inches": 0.03,
        }
    )


def _build_lookup(results_by_n_calib):
    """Build (n_calib, method, obs, metric) -> [values] lookup from raw results."""
    lookup = defaultdict(list)
    observations = set()
    methods = set()
    for n_calib, results in results_by_n_calib.items():
        for r in results:
            obs = r["num_observation"] if isinstance(r, dict) else r.num_observation
            method = r["method"] if isinstance(r, dict) else r.method
            observations.add(obs)
            methods.add(method)
            for m in ("c2st", "mmd"):
                val = r[m] if isinstance(r, dict) else getattr(r, m)
                lookup[(n_calib, method, obs, m)].append(val)
    sorted_n = sorted(results_by_n_calib.keys())
    observations = sorted(observations)
    ordered_methods = [m for m in METHOD_ORDER if m in methods]
    ordered_methods += sorted(methods - set(METHOD_ORDER))
    return lookup, sorted_n, observations, ordered_methods


def _plot_metric(ax, lookup, sorted_n, observations, methods, metric, obs_filter=None):
    """Plot a single metric on *ax*.

    Args:
        obs_filter: If None, pool all observations. If an int, use only that observation.
    """
    obs_list = [obs_filter] if obs_filter is not None else observations
    for method in methods:
        s = _style(method)
        xs, means, stds = [], [], []
        for n_calib in sorted_n:
            vals = []
            for obs in obs_list:
                vals.extend(lookup[(n_calib, method, obs, metric)])
            vals = [v for v in vals if np.isfinite(v)]
            if not vals:
                continue
            xs.append(n_calib)
            means.append(np.mean(vals))
            stds.append(np.std(vals))
        if not xs:
            continue
        means, stds = np.array(means), np.array(stds)

        if method == "npepfn_misspec":
            ax.axhline(means[0], color=s["color"], ls="--", lw=1.0, alpha=0.8, label=s["label"])
            ax.axhspan(means[0] - stds[0], means[0] + stds[0], color=s["color"], alpha=0.05)
        else:
            ax.plot(xs, means, color=s["color"], marker=s["marker"], ls=s["ls"], label=s["label"])
            ax.fill_between(xs, means - stds, means + stds, color=s["color"], alpha=0.12)

    ax.set_xscale("log")
    ax.set_ylabel(METRIC_LABEL.get(metric, metric.upper()))


def _make_two_panel(lookup, sorted_n, observations, methods, obs_filter=None):
    """Create a two-panel (C2ST + MMD) figure. Returns (fig, out_suffix)."""
    fig, (ax_c2st, ax_mmd) = plt.subplots(1, 2, figsize=(5.5, 2.6))

    _plot_metric(ax_c2st, lookup, sorted_n, observations, methods, "c2st", obs_filter=obs_filter)
    _plot_metric(ax_mmd, lookup, sorted_n, observations, methods, "mmd", obs_filter=obs_filter)

    # Shared x-label
    fig.text(0.5, 0.01, r"$n_{\mathrm{calib}}$", ha="center", fontsize=9)

    # Single shared legend below
    handles, labels = ax_c2st.get_legend_handles_labels()
    ncol = min(3, len(methods))
    fig.legend(handles, labels, loc="lower center", ncol=ncol, frameon=False, bbox_to_anchor=(0.5, -0.14))

    fig.subplots_adjust(wspace=0.35)
    return fig


# -- Public API ---------------------------------------------------------------


def plot_calibration_comparison(results_by_n_calib, metric="c2st", output_dir="results", task_name=None):
    """Simple single-panel plot (no seed information)."""
    _apply_rc()
    lookup, sorted_n, observations, methods = _build_lookup(results_by_n_calib)

    fig, ax = plt.subplots(figsize=(3.25, 2.4))
    _plot_metric(ax, lookup, sorted_n, observations, methods, metric)
    ax.set_xlabel(r"$n_{\mathrm{calib}}$")
    ax.legend(frameon=False)

    out_dir = Path(output_dir) / task_name if task_name else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"calibration_comparison_{metric}.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_calibration_comparison_seeds(results_by_n_calib, metric="c2st", output_dir="results", task_name=None):
    """Single-panel plot with mean +/- std across seeds (averaged over obs)."""
    _apply_rc()
    lookup, sorted_n, observations, methods = _build_lookup(results_by_n_calib)

    fig, ax = plt.subplots(figsize=(3.25, 2.4))
    _plot_metric(ax, lookup, sorted_n, observations, methods, metric)
    ax.set_xlabel(r"$n_{\mathrm{calib}}$")
    ax.legend(frameon=False)

    out_dir = Path(output_dir) / task_name if task_name else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"calibration_seeds_{metric}_avg.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_sweep_figure(results_by_n_calib, output_dir="results", task_name=None):
    """Two-panel figure (C2ST + MMD) suitable for a conference paper.

    Produces:
    - ``sweep_avg.pdf``: averaged over all observations and seeds.
    - ``sweep_obs{i}.pdf``: one per observation, averaged over seeds only.
    """
    _apply_rc()
    lookup, sorted_n, observations, methods = _build_lookup(results_by_n_calib)

    out_dir = Path(output_dir) / task_name if task_name else Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Averaged over all observations
    fig = _make_two_panel(lookup, sorted_n, observations, methods)
    out_path = out_dir / "sweep_avg.pdf"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")

    # Per observation
    for obs in observations:
        fig = _make_two_panel(lookup, sorted_n, observations, methods, obs_filter=obs)
        out_path = out_dir / f"sweep_obs{obs}.pdf"
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {out_path}")


def plot_posterior_pairplot(samples_by_method, ref_samples, param_names=None, output_path=None):
    """Corner-style pairplot: KDE on diagonal, scatter on lower triangle.

    Args:
        samples_by_method: dict mapping method name -> (N, D) numpy array.
        ref_samples: (N, D) numpy array of reference posterior samples.
        param_names: Optional list of parameter names (length D).
        output_path: If set, save figure to this path.
    """
    _apply_rc()
    D = ref_samples.shape[1]
    if param_names is None:
        param_names = [rf"$\theta_{{{i + 1}}}$" for i in range(D)]

    fig, axes = plt.subplots(D, D, figsize=(1.5 * D, 1.5 * D))
    if D == 1:
        axes = np.array([[axes]])

    # Ordered methods present in samples_by_method
    methods = [m for m in METHOD_ORDER if m in samples_by_method]
    methods += sorted(set(samples_by_method) - set(METHOD_ORDER))

    ref_color = "#999999"
    max_scatter = 500

    for i in range(D):
        for j in range(D):
            ax = axes[i, j]
            if j > i:
                ax.set_visible(False)
                continue

            if i == j:
                # Diagonal: KDE
                ref_col = ref_samples[:, i]
                lo = float(ref_col.min())
                hi = float(ref_col.max())
                for method in methods:
                    col = samples_by_method[method][:, i]
                    lo = min(lo, float(col.min()))
                    hi = max(hi, float(col.max()))
                grid = np.linspace(lo, hi, 200)

                kde_ref = gaussian_kde(ref_col)
                ax.plot(grid, kde_ref(grid), color=ref_color, lw=1.2, label="Reference")

                for method in methods:
                    s = _style(method)
                    col = samples_by_method[method][:, i]
                    kde_m = gaussian_kde(col)
                    ax.plot(grid, kde_m(grid), color=s["color"], lw=1.0, label=s["label"])
                ax.set_yticks([])
            else:
                # Lower triangle: scatter
                idx = np.random.choice(len(ref_samples), size=min(max_scatter, len(ref_samples)), replace=False)
                ax.scatter(
                    ref_samples[idx, j],
                    ref_samples[idx, i],
                    c=ref_color,
                    s=3,
                    alpha=0.15,
                    rasterized=True,
                    label="Reference",
                )
                for method in methods:
                    s = _style(method)
                    samp = samples_by_method[method]
                    idx_m = np.random.choice(len(samp), size=min(max_scatter, len(samp)), replace=False)
                    ax.scatter(
                        samp[idx_m, j], samp[idx_m, i], c=s["color"], s=3, alpha=0.15, rasterized=True, label=s["label"]
                    )

            if i == D - 1:
                ax.set_xlabel(param_names[j])
            else:
                ax.set_xticklabels([])
            if j == 0 and i != 0:
                ax.set_ylabel(param_names[i])
            elif j != 0:
                ax.set_yticklabels([])

    # Shared legend from diagonal[0,0]
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=6)
    fig.subplots_adjust(hspace=0.1, wspace=0.1)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved {output_path}")
    plt.close(fig)
    return fig


def plot_synthetic_y_scatter(y_pred, y_true, output_path=None):
    """Spatial scatter of predicted vs true y samples, overlaid.

    Args:
        y_pred: (N, D) numpy array of predicted y values.
        y_true: (N, D) numpy array of true y values.
        output_path: If set, save figure to this path.
    """
    _apply_rc()
    D = y_true.shape[1]
    max_pts = 500
    true_color = "#999999"
    pred_color = "#AA3377"

    if D == 2:
        fig, ax = plt.subplots(figsize=(3.0, 3.0))
        idx_t = np.random.choice(len(y_true), size=min(max_pts, len(y_true)), replace=False)
        idx_p = np.random.choice(len(y_pred), size=min(max_pts, len(y_pred)), replace=False)
        ax.scatter(y_true[idx_t, 0], y_true[idx_t, 1], c=true_color, s=4, alpha=0.3, rasterized=True, label="True")
        ax.scatter(y_pred[idx_p, 0], y_pred[idx_p, 1], c=pred_color, s=4, alpha=0.3, rasterized=True, label="Predicted")
        ax.set_xlabel(r"$y_1$")
        ax.set_ylabel(r"$y_2$")
        ax.legend(frameon=False, fontsize=6, markerscale=2)
        fig.tight_layout()
    else:
        # Lower-triangle pairwise grid
        fig, axes = plt.subplots(D, D, figsize=(1.5 * D, 1.5 * D))
        if D == 1:
            axes = np.array([[axes]])
        idx_t = np.random.choice(len(y_true), size=min(max_pts, len(y_true)), replace=False)
        idx_p = np.random.choice(len(y_pred), size=min(max_pts, len(y_pred)), replace=False)

        for i in range(D):
            for j in range(D):
                ax = axes[i, j]
                if j > i:
                    ax.set_visible(False)
                    continue
                if i == j:
                    # Diagonal: overlaid KDE
                    lo = min(float(y_true[:, i].min()), float(y_pred[:, i].min()))
                    hi = max(float(y_true[:, i].max()), float(y_pred[:, i].max()))
                    grid = np.linspace(lo, hi, 200)
                    kde_t = gaussian_kde(y_true[:, i])
                    kde_p = gaussian_kde(y_pred[:, i])
                    ax.plot(grid, kde_t(grid), color=true_color, lw=1.2, label="True")
                    ax.plot(grid, kde_p(grid), color=pred_color, lw=1.0, label="Predicted")
                    ax.set_yticks([])
                else:
                    ax.scatter(
                        y_true[idx_t, j], y_true[idx_t, i], c=true_color, s=3, alpha=0.15, rasterized=True, label="True"
                    )
                    ax.scatter(
                        y_pred[idx_p, j],
                        y_pred[idx_p, i],
                        c=pred_color,
                        s=3,
                        alpha=0.15,
                        rasterized=True,
                        label="Predicted",
                    )

                if i == D - 1:
                    ax.set_xlabel(rf"$y_{{{j + 1}}}$")
                else:
                    ax.set_xticklabels([])
                if j == 0 and i != 0:
                    ax.set_ylabel(rf"$y_{{{i + 1}}}$")
                elif j != 0:
                    ax.set_yticklabels([])

        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=6)
        fig.subplots_adjust(hspace=0.1, wspace=0.1)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved {output_path}")
    plt.close(fig)
    return fig


def plot_y_diagnostics(y_pred, y_true, output_path=None):
    """Scatter of predicted vs true y with y=x diagonal and Pearson r.

    Args:
        y_pred: (N, dim_y) numpy array of predicted y values.
        y_true: (N, dim_y) numpy array of true y values.
        output_path: If set, save figure to this path.
    """
    _apply_rc()
    dim_y = y_true.shape[1]
    color = METHOD_STYLE["npepfn_y_fmpe"]["color"]

    fig, axes = plt.subplots(1, dim_y, figsize=(2.5 * dim_y, 2.5), squeeze=False)
    for d in range(dim_y):
        ax = axes[0, d]
        ax.scatter(y_true[:, d], y_pred[:, d], c=color, s=4, alpha=0.3, rasterized=True)
        lo = min(float(y_true[:, d].min()), float(y_pred[:, d].min()))
        hi = max(float(y_true[:, d].max()), float(y_pred[:, d].max()))
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        r, _ = pearsonr(y_true[:, d], y_pred[:, d])
        ax.annotate(f"$r = {r:.3f}$", xy=(0.05, 0.92), xycoords="axes fraction", fontsize=7)
        ax.set_xlabel(rf"$y_{{{d + 1}}}$ (true)")
        ax.set_ylabel(rf"$y_{{{d + 1}}}$ (predicted)")

    fig.tight_layout()
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved {output_path}")
    plt.close(fig)
    return fig
