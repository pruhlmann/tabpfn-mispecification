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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
