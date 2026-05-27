"""Tests for the custom-task registry and the GaussianLinearHD task."""

import pytest
import torch

from tabpfn_misspec.simulators import get_misspecified_simulator
from tabpfn_misspec.tasks import GaussianLinearHD, get_task


def test_gaussian_linear_hd_interface():
    task = GaussianLinearHD()

    assert task.name == "gaussian_linear_hd"
    assert task.dim_parameters > 20
    assert task.dim_data > 5
    assert task.num_observations >= 1

    prior = task.get_prior_dist()
    theta = prior.sample((16,))
    assert theta.shape == (16, task.dim_parameters)

    simulator = task.get_simulator()
    y = simulator(theta)
    assert y.shape == (16, task.dim_data)

    y_obs_a = task.get_observation(1)
    y_obs_b = task.get_observation(1)
    assert y_obs_a.shape == (1, task.dim_data)
    assert torch.equal(y_obs_a, y_obs_b)

    ref = task.get_reference_posterior_samples(1)
    assert ref.shape == (task.num_posterior_samples, task.dim_parameters)


def test_get_task_dispatch():
    custom = get_task("gaussian_linear_hd")
    assert isinstance(custom, GaussianLinearHD)

    sbibm_task = get_task("gaussian_linear")
    assert sbibm_task.name == "gaussian_linear"


def test_linear_misspec_factory():
    task = get_task("gaussian_linear_hd")
    sim = get_misspecified_simulator("gaussian_linear_hd", "linear_misspec", sigma_x=0.5)

    theta = task.get_prior_dist().sample((32,))
    x = sim(theta)
    assert x.shape == (32, task.dim_x)


def test_nonlinear_theta_factory():
    task = get_task("gaussian_linear_hd")
    sim = get_misspecified_simulator(
        "gaussian_linear_hd", "nonlinear_theta", sigma_x=0.5, alpha=0.1,
    )

    theta = task.get_prior_dist().sample((32,))
    x = sim(theta)
    assert x.shape == (32, task.dim_x)


def test_nonlinear_theta_reduces_to_linear_when_alpha_zero():
    """alpha=0 must give the same draw as linear_misspec under fixed seed."""
    task = get_task("gaussian_linear_hd")
    theta = task.get_prior_dist().sample((16,))

    torch.manual_seed(0)
    sim_lin = get_misspecified_simulator(
        "gaussian_linear_hd", "linear_misspec", sigma_x=0.5,
    )
    x_lin = sim_lin(theta)

    torch.manual_seed(0)
    sim_nl = get_misspecified_simulator(
        "gaussian_linear_hd", "nonlinear_theta", sigma_x=0.5, alpha=0.0,
    )
    x_nl = sim_nl(theta)

    assert torch.allclose(x_lin, x_nl)


def test_misspec_matrices_close_to_truth():
    """A ~ C and b ~ d so the dominant misspec is the noise scale, not the map."""
    task = GaussianLinearHD(misspec_matrix_eps=0.01)
    rel_A = torch.linalg.norm(task.A - task.C) / torch.linalg.norm(task.C)
    rel_b = torch.linalg.norm(task.b - task.d) / torch.linalg.norm(task.d)
    assert rel_A < 0.05
    assert rel_b < 0.05


def test_reference_posterior_matches_closed_form():
    """Empirical mean/cov of cached reference samples ~ analytic posterior moments."""
    task = GaussianLinearHD(num_posterior_samples=20000)
    ref = task.get_reference_posterior_samples(1)

    emp_mean = ref.mean(dim=0)
    expected_mean = task._post_means[1]
    assert torch.allclose(emp_mean, expected_mean, atol=0.05)

    emp_cov = torch.cov(ref.T)
    expected_cov = task._post_cov
    # Frobenius-norm relative error
    rel = torch.linalg.norm(emp_cov - expected_cov) / torch.linalg.norm(expected_cov)
    assert rel < 0.1


def test_gaussian_linear_hd_reference_log_prob():
    """Oracle log_prob is finite and maximized at the true-posterior samples."""
    task = GaussianLinearHD()
    ref = task.get_reference_posterior_samples(1)
    lp = task.reference_log_prob(ref, 1)
    assert lp.shape == (ref.shape[0],)
    assert torch.isfinite(lp).all()
    # Shifting samples away from the posterior mode lowers the mean log-density.
    lp_shifted = task.reference_log_prob(ref + 3.0, 1)
    assert lp.mean() > lp_shifted.mean()


def test_gaussian_mixture_hd_reference_log_prob():
    task = get_task("gaussian_mixture_hd")
    ref = task.get_reference_posterior_samples(1)
    lp = task.reference_log_prob(ref, 1)
    assert lp.shape == (ref.shape[0],)
    assert torch.isfinite(lp).all()
    lp_shifted = task.reference_log_prob(ref + 20.0, 1)
    assert lp.mean() > lp_shifted.mean()


def test_lotka_volterra_hd_interface():
    task = get_task("lotka_volterra_hd")
    assert task.name == "lotka_volterra_hd"
    assert task.dim_parameters == 25
    assert task.dim_data == 50
    assert task.num_observations == 3


def test_lotka_volterra_hd_param_packing():
    from tabpfn_misspec.lotka_volterra_hd import _theta_to_p

    theta = torch.arange(25, dtype=torch.float).unsqueeze(0)
    p = _theta_to_p(theta)
    assert p.shape == (1, 30)
    assert torch.equal(p[0, :5], torch.arange(5, dtype=torch.float))
    A = p[0, 5:].reshape(5, 5)
    assert torch.equal(A.diag(), torch.ones(5))
    # Off-diagonals filled in row-major skip-diagonal order from theta[5:]
    assert A[0, 1] == 5
    assert A[0, 4] == 8
    assert A[4, 3] == 24


def test_lotka_volterra_hd_simulator_shape():
    task = get_task("lotka_volterra_hd")
    theta = task.get_prior_dist().sample((4,))
    assert theta.shape == (4, 25)
    x = task.get_simulator()(theta)
    assert x.shape == (4, 50)


def test_wrong_noise_scale_factory():
    sim = get_misspecified_simulator(
        "lotka_volterra_hd", "wrong_noise_scale", scale=0.5,
    )
    theta = get_task("lotka_volterra_hd").get_prior_dist().sample((2,))
    x = sim(theta)
    assert x.shape == (2, 50)


def test_slice_mcmc_matches_sbibm_gaussian_linear_uniform_reference():
    """Stage-1 (Slice-MCMC) of the GT pipeline against the shipped reference.

    Stage 1 is the only stage we replaced — NUTS -> SliceSamplerVectorized.
    Stages 2 (NSF) and 3 (rejection) are unchanged sbibm code; we exercise
    them in test_gt_full_pipeline_smoke below.

    Target: gaussian_linear_uniform (dim=10, BoxUniform prior, Gaussian
    likelihood). Reference samples ship with the package. Uses uniform-prior
    rather than gaussian_linear because the latter's identity bijection
    triggers a latent typo bug in sbibm.utils.torch.get_log_abs_det_jacobian
    (the helper hot-patches that bug, but we still want to exercise the
    bounded-transform branch here).

    Posterior is approximately N(y, 0.1 I) truncated to [-1,1]^10, so std ~ 0.32.
    """
    import sbibm

    import numpy as np
    import sbibm.utils.nflows  # noqa: F401  -- import for side effects (patch)
    from sbi.samplers.mcmc.slice_numpy import SliceSamplerVectorized

    from tabpfn_misspec import _gt_pipeline  # noqa: F401  -- applies bug patch

    task = sbibm.get_task("gaussian_linear_uniform")
    num_observation = 1
    num_samples = 2_000
    num_warmup = 500

    torch_log_prob_fn = task._get_log_prob_fn(
        num_observation=num_observation,
        implementation="experimental",
        posterior=True,
    )

    def np_log_prob_fn(params_np):
        params = torch.as_tensor(np.atleast_2d(params_np), dtype=torch.float32)
        with torch.no_grad():
            log_p = torch_log_prob_fn(params)
        out = log_p.detach().cpu().numpy().astype(np.float64).reshape(-1)
        out[~np.isfinite(out)] = -np.inf
        return out

    init_np = task.get_true_parameters(num_observation=num_observation).numpy().reshape(
        1, task.dim_parameters,
    ).astype(np.float64)

    sampler = SliceSamplerVectorized(
        log_prob_fn=np_log_prob_fn,
        init_params=init_np,
        num_chains=1,
        thin=1,
        tuning=50,
        verbose=False,
        init_width=0.01,
        num_workers=1,
    )
    chain = sampler.run(num_warmup + num_samples)[:, num_warmup:, :]
    samples = torch.as_tensor(chain.reshape(-1, task.dim_parameters), dtype=torch.float32)

    ref = task.get_reference_posterior_samples(num_observation=num_observation)[:num_samples]

    assert samples.shape == (num_samples, task.dim_parameters)
    assert torch.isfinite(samples).all()

    diff_mean = (samples.mean(dim=0) - ref.mean(dim=0)).abs().max()
    assert diff_mean < 0.1, f"slice-MCMC mean diff vs reference: {diff_mean:.3f}"

    rel_cov = (
        torch.linalg.norm(torch.cov(samples.T) - torch.cov(ref.T))
        / torch.linalg.norm(torch.cov(ref.T))
    )
    assert rel_cov < 0.3, f"slice-MCMC cov rel-error vs reference: {rel_cov:.3f}"


@pytest.mark.slow
def test_gt_full_pipeline_smoke():
    """Smoke test of all three stages (Slice + NSF + rejection) plumbed together.

    Intentionally tiny num_samples: the rejection step's acceptance rate on
    gaussian_linear_uniform's BoxUniform prior is structurally low (~1e-4),
    so even 50 accepts is ~30s of stage-3 work after the ~80s slice + ~20s NSF.
    Asserts only shape + finiteness — statistical agreement is covered by the
    stage-1 test above.
    """
    import sbibm

    from tabpfn_misspec._gt_pipeline import run_slice_nsf_rejection_pipeline

    task = sbibm.get_task("gaussian_linear_uniform")
    num_samples = 50

    samples = run_slice_nsf_rejection_pipeline(
        task=task,
        num_samples=num_samples,
        num_observation=1,
        num_warmup=200,
        num_chains=1,
        tuning=30,
        batch_size=2_000,
        num_batches_without_new_max=10,
    )

    assert samples.shape == (num_samples, task.dim_parameters)
    assert torch.isfinite(samples).all()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
