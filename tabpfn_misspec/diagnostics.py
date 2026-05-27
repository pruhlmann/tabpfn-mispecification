"""Calibration diagnostics: simulation-based calibration (SBC) ranks and TARP coverage.

These operate purely on drawn posterior samples, so they are decoupled from each
method's posterior interface (TabPFN, sbi, FMCPE all differ). The convention used
throughout this module is that posterior samples have shape ``(N, L, dim)``: for
each of the ``N`` SBC test points, ``L`` posterior draws conditioned on ``x_i``.

References:
    - Talts et al. 2018, "Validating Bayesian Inference Algorithms with
      Simulation-Based Calibration" (https://arxiv.org/abs/1804.06788)
    - Lemos et al. 2023, "Sampling-Based Accuracy Testing of Posterior
      Estimators for General Inference" (https://arxiv.org/abs/2302.03026)
"""

import torch
from scipy.stats import kstest

from sbi.diagnostics.tarp import _run_tarp, check_tarp, get_tarp_references


def compute_sbc_ranks(post_samples: torch.Tensor, thetas: torch.Tensor) -> torch.Tensor:
    """SBC ranks of the true theta among posterior draws, per dimension.

    Args:
        post_samples: Posterior draws, shape ``(N, L, dim)``.
        thetas: True parameters, shape ``(N, dim)``.

    Returns:
        Integer ranks, shape ``(N, dim)``, each in ``[0, L]``.
    """
    return (post_samples < thetas.unsqueeze(1)).sum(dim=1)


def sbc_uniformity(ranks: torch.Tensor, num_posterior_samples: int) -> float:
    """Mean per-dimension KS statistic of normalized ranks against Uniform(0, 1).

    0 = perfectly calibrated marginals; larger = more miscalibrated.
    """
    u = ranks.float().cpu().numpy() / float(num_posterior_samples)
    stats = [kstest(u[:, d], "uniform").statistic for d in range(u.shape[1])]
    return float(sum(stats) / len(stats))


def tarp_curve(post_samples: torch.Tensor, thetas: torch.Tensor):
    """TARP expected-coverage-probability curve from drawn samples.

    Args:
        post_samples: Posterior draws, shape ``(N, L, dim)``.
        thetas: True parameters, shape ``(N, dim)``.

    Returns:
        (ecp, alpha) tensors as returned by sbi's ``_run_tarp``.
    """
    # sbi's _run_tarp expects samples shaped (L, N, dim).
    post_lnd = post_samples.transpose(0, 1).contiguous()
    references = get_tarp_references(thetas)
    ecp, alpha = _run_tarp(post_lnd, thetas, references)
    return ecp, alpha


def run_method_diagnostics(
    method,
    draw_fn,
    theta_sbc,
    x_sbc,
    num_posterior_samples,
    artifacts_dir=None,
    seed=42,
):
    """Run SBC + TARP for one method and return scalar summaries.

    Args:
        method: Method name (used for artifact filenames).
        draw_fn: Callable ``(x_sbc, L) -> Tensor[N, L, dim]`` drawing L posterior
            samples for each of the N SBC test observations.
        theta_sbc: True parameters, shape ``(N, dim)``.
        x_sbc: SBC test observations, shape ``(N, dim_x)``.
        num_posterior_samples: L, samples drawn per test point.
        artifacts_dir: If set, save raw ranks / TARP curve as ``sbc_{method}_seed{seed}.pt``.

    Returns:
        dict with ``sbc_ks`` and ``tarp_ece`` (mean |ecp - alpha|).
    """
    post_samples = draw_fn(x_sbc, num_posterior_samples).cpu()
    thetas = theta_sbc.cpu()

    ranks = compute_sbc_ranks(post_samples, thetas)
    sbc_ks = sbc_uniformity(ranks, num_posterior_samples)

    ecp, alpha = tarp_curve(post_samples, thetas)
    tarp_ece = float((ecp - alpha).abs().mean())
    tarp_atc, tarp_ks_pval = check_tarp(ecp, alpha)
    tarp_atc, tarp_ks_pval = float(tarp_atc), float(tarp_ks_pval)  # scipy returns numpy floats

    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "ranks": ranks,
                "num_posterior_samples": num_posterior_samples,
                "ecp": ecp,
                "alpha": alpha,
                "sbc_ks": sbc_ks,
                "tarp_ece": tarp_ece,
                "tarp_atc": tarp_atc,
                "tarp_ks_pval": tarp_ks_pval,
            },
            artifacts_dir / f"sbc_{method}_seed{seed}.pt",
        )

    return {"sbc_ks": sbc_ks, "tarp_ece": tarp_ece}
