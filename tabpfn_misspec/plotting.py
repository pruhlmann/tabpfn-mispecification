"""Plotting utilities for calibration comparison experiments."""

from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


# -- Method display names and visual style ------------------------------------
# Three color groups:
#   syn + cal: purple/magenta gradient, different linestyles
#   syn only:  blue/teal gradient, different linestyles
#   other:     distinct individual colors

METHOD_STYLE = {
    # --- Synthetic-y methods: TabPFN generates y, then run inference on it ---
    "npepfn_y_fmpe_concat": dict(
        label=r"$\tilde y \sim \mathrm{TabPFN}(\cdot \mid \theta_{\mathrm{sim}}, x_{\mathrm{sim}})$;  FMPE on $\tilde y \cup$ calib",
        color="#7B2D8E", marker="v", ls="-",
    ),
    "npepfn_y_npepfn_concat": dict(
        label=r"$\tilde y \sim \mathrm{TabPFN}(\cdot \mid \theta_{\mathrm{sim}}, x_{\mathrm{sim}})$;  $\theta \sim \mathrm{TabPFN}(\cdot \mid y_{\mathrm{obs}}, D_{\mathrm{cal}})$,  $D_{\mathrm{cal}} = (\theta_{\mathrm{sim}}, \tilde y) \cup (\theta_{\mathrm{cal}}, y_{\mathrm{cal}})$",
        color="#C77CFF", marker="h", ls="--",
    ),
    "npepfn_y_fmpe": dict(
        label=r"$\tilde y \sim \mathrm{TabPFN}(\cdot \mid \theta_{\mathrm{sim}}, x_{\mathrm{sim}})$;  FMPE on $\tilde y$",
        color="#0B3D91", marker="v", ls="-",
    ),
    "npepfn_y_npepfn": dict(
        label=r"$\tilde y \sim \mathrm{TabPFN}(\cdot \mid \theta_{\mathrm{sim}}, x_{\mathrm{sim}})$;  $\theta \sim \mathrm{TabPFN}(\cdot \mid y_{\mathrm{obs}}, D_{\mathrm{cal}})$,  $D_{\mathrm{cal}} = (\theta_{\mathrm{sim}}, \tilde y)$",
        color="#4DA6FF", marker="h", ls="--",
    ),
    "npepfn_ythetaonly_npepfn": dict(
        label=r"$\tilde y \sim \mathrm{TabPFN}(\cdot \mid \theta_{\mathrm{prior}})$;  $\theta \sim \mathrm{TabPFN}(\cdot \mid y_{\mathrm{obs}}, D_{\mathrm{cal}})$,  $D_{\mathrm{cal}} = (\theta_{\mathrm{prior}}, \tilde y)$",
        color="#99D6FF", marker="d", ls="-.",
    ),
    # --- TabPFN in-context inference on real data (no synth y) ---
    "npepfn_mixed": dict(
        label=r"$\theta \sim \mathrm{TabPFN}(\cdot \mid y_{\mathrm{obs}}, D_{\mathrm{cal}})$,  $D_{\mathrm{cal}} = (\theta_{\mathrm{sim}}, x_{\mathrm{sim}}) \cup (\theta_{\mathrm{cal}}, y_{\mathrm{cal}})$",
        color="#228833", marker="o", ls="-",
    ),
    "npepfn_calib": dict(
        label=r"$\theta \sim \mathrm{TabPFN}(\cdot \mid y_{\mathrm{obs}}, D_{\mathrm{cal}})$,  $D_{\mathrm{cal}} = (\theta_{\mathrm{cal}}, y_{\mathrm{cal}})$",
        color="#EE6677", marker="^", ls="-",
    ),
    "npepfn_misspec": dict(
        label=r"$\theta \sim \mathrm{TabPFN}(\cdot \mid y_{\mathrm{obs}}, D_{\mathrm{cal}})$,  $D_{\mathrm{cal}} = (\theta_{\mathrm{sim}}, x_{\mathrm{sim}})$",
        color="#CCBB44", marker="s", ls="--",
    ),
    # --- Trained density estimators ---
    "npe_sbi": dict(
        label=r"NPE trained on calib $(\theta, y)$",
        color="#FF8C00", marker="D", ls="-",
    ),
    "mf_npe": dict(
        label=r"NPE pretrained on misspec. sims $\to$ fine-tuned on calib",
        color="#AA3377", marker="P", ls="-",
    ),
    "fmcpe": dict(
        label=r"FMCPE:  NPE proposal $+\; y \to x \;+\; \theta$ transform",
        color="#117733", marker="X", ls="-",
    ),
}

# Plotting order: "ours" first, then baselines
METHOD_ORDER = [
    "npepfn_y_fmpe_concat",
    "npepfn_y_npepfn_concat",
    "npepfn_y_fmpe",
    "npepfn_y_npepfn",
    "npepfn_ythetaonly_npepfn",
    "npepfn_mixed",
    "npepfn_calib",
    "npe_sbi",
    "mf_npe",
    "fmcpe",
    "npepfn_misspec",
]

_FALLBACK_COLORS = ["#66CCEE", "#CC6677", "#882255", "#117733", "#332288"]
_FALLBACK_MARKERS = ["P", "X", "h", "*", "p"]

METRIC_LABEL = {
    "c2st": r"C2ST $\downarrow$",
    "mmd": r"MMD $\downarrow$",
    "log_prob": r"Log Prob $\uparrow$",
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
            for m in ("c2st", "mmd", "log_prob"):
                val = r.get(m, float("nan")) if isinstance(r, dict) else getattr(r, m, float("nan"))
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

    # X-label on each axis (cleaner than shared fig.text with long legend below)
    ax_c2st.set_xlabel(r"$n_{\mathrm{calib}}$")
    ax_mmd.set_xlabel(r"$n_{\mathrm{calib}}$")

    # Single shared legend below, with enough gap to avoid overlapping x-labels.
    # Long explicit labels => single column. bbox_inches='tight' on savefig will
    # expand the figure box to include the legend.
    handles, labels = ax_c2st.get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", ncol=1, frameon=False,
        bbox_to_anchor=(0.5, -0.05), borderaxespad=0.0,
    )

    fig.subplots_adjust(wspace=0.35, bottom=0.20)
    return fig


# -- Public API ---------------------------------------------------------------


def plot_calibration_comparison(
    results_by_n_calib, metric="c2st", output_dir="results", task_name=None
):
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


def plot_calibration_comparison_seeds(
    results_by_n_calib, metric="c2st", output_dir="results", task_name=None
):
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
                idx = np.random.choice(
                    len(ref_samples), size=min(max_scatter, len(ref_samples)), replace=False
                )
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
                    idx_m = np.random.choice(
                        len(samp), size=min(max_scatter, len(samp)), replace=False
                    )
                    ax.scatter(
                        samp[idx_m, j],
                        samp[idx_m, i],
                        c=s["color"],
                        s=3,
                        alpha=0.15,
                        rasterized=True,
                        label=s["label"],
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


def plot_y_distributional(theta_diag, y_true, y_tilde, output_path=None):
    """Distributional comparison of true vs predicted y for K fixed thetas.

    Args:
        theta_diag: (K, dim_theta) numpy array of diagnostic theta values.
        y_true: (K, N_test, dim_y) numpy array of true y samples.
        y_tilde: (K, N_test, dim_y) numpy array of predicted y samples.
        output_path: If set, save figure to this path.
    """
    _apply_rc()
    K, N_test, dim_y = y_true.shape
    true_color = "#999999"
    pred_color = "#AA3377"

    fig, axes = plt.subplots(K, dim_y, figsize=(2.5 * dim_y, 1.8 * K), squeeze=False)
    for k in range(K):
        for d in range(dim_y):
            ax = axes[k, d]
            col_true = y_true[k, :, d]
            col_pred = y_tilde[k, :, d]
            lo = min(float(col_true.min()), float(col_pred.min()))
            hi = max(float(col_true.max()), float(col_pred.max()))
            grid = np.linspace(lo, hi, 200)
            try:
                kde_t = gaussian_kde(col_true)
                ax.plot(
                    grid,
                    kde_t(grid),
                    color=true_color,
                    lw=1.2,
                    label="True" if k == 0 and d == 0 else None,
                )
            except np.linalg.LinAlgError:
                pass
            try:
                kde_p = gaussian_kde(col_pred)
                ax.plot(
                    grid,
                    kde_p(grid),
                    color=pred_color,
                    lw=1.0,
                    label="Predicted" if k == 0 and d == 0 else None,
                )
            except np.linalg.LinAlgError:
                pass
            ax.set_yticks([])
            if k == K - 1:
                ax.set_xlabel(rf"$y_{{{d + 1}}}$")
            else:
                ax.set_xticklabels([])
            if d == 0:
                ax.set_ylabel(rf"$\theta_{{{k + 1}}}$")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False, fontsize=6)
    fig.subplots_adjust(hspace=0.3, wspace=0.2)

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, bbox_inches="tight")
        print(f"Saved {output_path}")
    plt.close(fig)
    return fig
