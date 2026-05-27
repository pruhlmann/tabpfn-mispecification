"""Tests for SBC + TARP calibration diagnostics.

Uses a conjugate-Gaussian synthetic setup where the exact posterior is known, so
a correctly-dispersed "posterior" must yield uniform SBC ranks and TARP coverage
on the diagonal, while an over-confident one must be detectably miscalibrated.
"""

import torch

from tabpfn_misspec.diagnostics import (
    compute_sbc_ranks,
    run_method_diagnostics,
    sbc_uniformity,
    tarp_curve,
)


def _conjugate_gaussian_samples(scale, N=600, L=600, dim=2, seed=0):
    """Build (theta, post_samples) for prior N(0,1), likelihood x = theta + N(0,1).

    Exact posterior is N(x/2, 1/2). ``scale`` multiplies the posterior std:
    scale=1 is calibrated, scale<1 is over-confident.
    """
    g = torch.Generator().manual_seed(seed)
    theta = torch.randn(N, dim, generator=g)
    x = theta + torch.randn(N, dim, generator=g)
    mu = x / 2.0
    s = (0.5**0.5) * scale
    post = mu.unsqueeze(1) + s * torch.randn(N, L, dim, generator=g)
    return theta, post


def test_sbc_tarp_calibrated():
    theta, post = _conjugate_gaussian_samples(scale=1.0)
    L = post.shape[1]
    ranks = compute_sbc_ranks(post, theta)
    assert ranks.shape == theta.shape
    sbc_ks = sbc_uniformity(ranks, L)
    ecp, alpha = tarp_curve(post, theta)
    tarp_ece = float((ecp - alpha).abs().mean())

    assert sbc_ks < 0.1, f"calibrated SBC KS too high: {sbc_ks}"
    assert tarp_ece < 0.1, f"calibrated TARP error too high: {tarp_ece}"


def test_sbc_tarp_detects_overconfidence():
    theta_c, post_c = _conjugate_gaussian_samples(scale=1.0)
    theta_o, post_o = _conjugate_gaussian_samples(scale=0.3)
    L = post_c.shape[1]

    ks_c = sbc_uniformity(compute_sbc_ranks(post_c, theta_c), L)
    ks_o = sbc_uniformity(compute_sbc_ranks(post_o, theta_o), L)
    ece_c = float((lambda e, a: (e - a).abs().mean())(*tarp_curve(post_c, theta_c)))
    ece_o = float((lambda e, a: (e - a).abs().mean())(*tarp_curve(post_o, theta_o)))

    assert ks_o > 0.15, f"overconfident SBC KS not flagged: {ks_o}"
    assert ks_o > ks_c
    assert ece_o > ece_c


def test_run_method_diagnostics_orchestrator(tmp_path):
    """Exercise the orchestrator + artifact saving with a fixed sample block."""
    theta, post = _conjugate_gaussian_samples(scale=1.0)
    x_sbc = theta + 1.0  # arbitrary; draw_fn ignores it here

    def draw_fn(xs, L):
        assert xs.shape[0] == theta.shape[0]
        return post

    out = run_method_diagnostics(
        "dummy", draw_fn, theta, x_sbc, post.shape[1], tmp_path, seed=7
    )
    assert set(out) == {"sbc_ks", "tarp_ece"}
    assert (tmp_path / "sbc_dummy_seed7.pt").exists()
    saved = torch.load(tmp_path / "sbc_dummy_seed7.pt", weights_only=True)
    assert saved["ranks"].shape == theta.shape
    assert "ecp" in saved and "alpha" in saved
