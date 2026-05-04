"""Task registry: custom tasks first, with sbibm fallback.

Custom tasks must implement the same duck-typed interface the rest of the
pipeline relies on (see plan in repo docs):

  Methods:    get_prior_dist, get_simulator, get_observation,
              get_reference_posterior_samples
  Properties: dim_data, dim_parameters, name, num_observations

Register a new task by adding it to ``_CUSTOM_TASKS``.
"""

import sbibm
import torch
from torch.distributions import Independent, Normal


def get_task(task_name):
    factory = _CUSTOM_TASKS.get(task_name)
    if factory is not None:
        return factory()
    return sbibm.get_task(task_name)


class GaussianLinearHD:
    """High-dim Gaussian linear task with closed-form posterior.

    True model:    y = C @ theta + d + eps_y,   eps_y ~ N(0, sigma_y^2 I_{dim_y})
    Prior:         theta ~ N(0, I_{dim_theta})

    The matrices ``A``, ``b`` for the misspecified linear simulator are
    near-copies of ``C``, ``d`` perturbed by ``misspec_matrix_eps``, so the
    dominant source of misspecification is the noise scale ``sigma_x`` (set
    on the ``linear_misspec`` factory) differing from ``sigma_y``.

    All matrices and the per-observation ``theta_star`` / ``y_obs`` are
    deterministic given the seeds, so repeated instantiation is identical.
    """

    name = "gaussian_linear_hd"

    def __init__(
        self,
        dim_theta=25,
        dim_y=8,
        sigma_y=0.1,
        num_observations=1,
        num_posterior_samples=10000,
        matrix_seed=0,
        obs_seed=1,
        misspec_matrix_eps=0.01,
    ):
        self.dim_parameters = dim_theta
        self.dim_data = dim_y
        self.dim_x = dim_y
        self.sigma_y = sigma_y
        self.num_observations = num_observations
        self.num_posterior_samples = num_posterior_samples

        g = torch.Generator().manual_seed(matrix_seed)
        self.C = torch.randn(dim_y, dim_theta, generator=g)
        self.d = torch.randn(dim_y, generator=g)
        # A, b are small perturbations of C, d so the misspec is dominated by
        # the noise-scale mismatch (sigma_x vs sigma_y), not the linear map.
        self.A = self.C + misspec_matrix_eps * torch.randn(
            dim_y, dim_theta, generator=g
        )
        self.b = self.d + misspec_matrix_eps * torch.randn(dim_y, generator=g)

        # Closed-form posterior: prior N(0, I), likelihood N(C theta + d, sigma_y^2 I)
        prec = torch.eye(dim_theta) + (self.C.T @ self.C) / (sigma_y ** 2)
        self._post_cov = torch.linalg.inv(prec)
        self._post_chol = torch.linalg.cholesky(self._post_cov)

        g_obs = torch.Generator().manual_seed(obs_seed)
        self._theta_star = {}
        self._y_obs = {}
        self._post_means = {}
        self._ref_samples = {}
        for k in range(1, num_observations + 1):
            theta_k = torch.randn(dim_theta, generator=g_obs)
            eps_k = sigma_y * torch.randn(dim_y, generator=g_obs)
            y_k = self.C @ theta_k + self.d + eps_k
            self._theta_star[k] = theta_k
            self._y_obs[k] = y_k.unsqueeze(0)
            self._post_means[k] = (
                self._post_cov @ self.C.T @ (y_k - self.d) / (sigma_y ** 2)
            )
            g_post = torch.Generator().manual_seed(matrix_seed + 7 + k)
            z = torch.randn(num_posterior_samples, dim_theta, generator=g_post)
            self._ref_samples[k] = self._post_means[k] + z @ self._post_chol.T

    def get_prior_dist(self):
        loc = torch.zeros(self.dim_parameters)
        scale = torch.ones(self.dim_parameters)
        return Independent(Normal(loc, scale), 1)

    def get_simulator(self):
        C, d, sigma_y = self.C, self.d, self.sigma_y
        dim_y = self.dim_data

        def simulator(theta):
            return theta @ C.T + d + sigma_y * torch.randn(theta.shape[0], dim_y)

        return simulator

    def get_observation(self, obs_idx):
        return self._y_obs[obs_idx]

    def get_reference_posterior_samples(self, obs_idx):
        return self._ref_samples[obs_idx]


_CUSTOM_TASKS = {
    "gaussian_linear_hd": GaussianLinearHD,
}
