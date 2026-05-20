"""Misspecified simulator registry for sbibm tasks."""

import math
import os
import warnings
from contextlib import contextmanager

import sbibm
from sbibm.tasks.gaussian_mixture.task import GaussianMixture
import torch
import pyro
import pyro.distributions as pdist

from tabpfn_misspec.tasks import get_task

# Suppress diffeqtorch Python warnings (e.g. "JULIA_SYSIMAGE_DIFFEQTORCH not set")
warnings.filterwarnings("ignore", module="diffeqtorch")


@contextmanager
def suppress_julia_output():
    """Redirect fd-level stderr to /dev/null to silence Julia runtime warnings."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    old_stderr = os.dup(2)
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(old_stderr, 2)
        os.close(devnull)
        os.close(old_stderr)


def _additive_noise(task_name, noise_std=0.5):
    """Wrap the true simulator with additive Gaussian noise."""
    task = sbibm.get_task(task_name)
    simulator = task.get_simulator()

    def misspecified_simulator(theta):
        x = simulator(theta)
        return x + noise_std * torch.randn_like(x)

    return misspecified_simulator


def _scale_shift(task_name, scale=2.0, shift=1.0):
    """Wrap the true simulator with an affine transformation."""
    task = sbibm.get_task(task_name)
    simulator = task.get_simulator()

    def misspecified_simulator(theta):
        x = simulator(theta)
        return scale * x + shift

    return misspecified_simulator


def _one_gaussian(task_name, mean=0.0, std=1.0):
    """Replace the true simulator with a single Gaussian, ignoring theta."""
    if task_name != "gaussian_mixture":
        raise ValueError(
            f"_one_gaussian misspecification is only defined for 'gaussian_mixture', got '{task_name}'"
        )
    task: GaussianMixture = sbibm.get_task("gaussian_mixture")
    dim_data = task.dim_data

    # Select loc and scales according to mixture index
    def misspecified_simulator(theta):
        loc = task.simulator_params["mixture_locs_factor"][0] * theta
        scale = task.simulator_params["mixture_scales"].mean()

        return pyro.sample("data", pdist.Normal(loc=loc, scale=scale).to_event(1))

    return misspecified_simulator


def _weekend_delay(task_name, delay_fraction=0.05):
    """SIR simulator with weekend reporting delay (Ward et al., 2022).

    Applies the misspecification to the raw daily ODE solution before
    sbibm's subsampling and Binomial observation noise.  A fraction of
    Saturday/Sunday infection counts is shifted to the following Monday.

    Args:
        delay_fraction: Fraction of weekend counts deferred to Monday.
    """
    if task_name != "sir":
        raise ValueError(f"_weekend_delay is only defined for 'sir', got '{task_name}'")
    task = sbibm.get_task("sir")
    N = task.N
    total_count = task.total_count

    def misspecified_simulator(theta):
        num_samples = theta.shape[0]

        us = []
        with suppress_julia_output():
            for i in range(num_samples):
                u, t = task.de(task.u0, task.tspan, theta[i, :])

                if u.shape != torch.Size([3, int(task.dim_data_raw / 3)]):
                    u = float("nan") * torch.ones((3, int(task.dim_data_raw / 3)))
                    u = u.double()
                us.append(u.reshape(1, 3, -1))
        us = torch.cat(us).float()  # (num_samples, 3, 161)

        # Extract infected compartment: (num_samples, 161)
        infected = us[:, 1, :]

        # Apply weekend delay on daily resolution
        num_days = infected.shape[1]
        sat_idx = list(range(5, num_days, 7))  # day 0 = Monday convention
        sun_idx = list(range(6, num_days, 7))
        mon_idx = list(range(7, num_days, 7))
        # Pair each weekend with the following Monday; drop unpaired weekends
        n_pairs = min(len(sat_idx), len(sun_idx), len(mon_idx))
        sat_idx = sat_idx[:n_pairs]
        sun_idx = sun_idx[:n_pairs]
        mon_idx = mon_idx[:n_pairs]

        missed_sat = infected[:, sat_idx] * delay_fraction
        missed_sun = infected[:, sun_idx] * delay_fraction
        infected[:, sat_idx] -= missed_sat
        infected[:, sun_idx] -= missed_sun
        infected[:, mon_idx] += missed_sat + missed_sun

        # Subsample every 17 days (same as sbibm) and apply Binomial noise
        nan_mask = torch.isnan(infected.reshape(num_samples, -1)).any(dim=1)
        sub = infected[:, ::17]  # (num_samples, 10)
        data = float("nan") * torch.ones((num_samples, 10))
        ok = ~nan_mask
        if ok.any():
            probs = (sub[ok, :] / N).clamp(0.0, 1.0)
            data[ok, :] = pyro.sample(
                "data",
                pdist.Binomial(total_count=total_count, probs=probs).to_event(1),
            )
        return data

    return misspecified_simulator


def _carrying_capacity(task_name, K=100):
    """Lotka-Volterra with density-dependent prey growth.

    Replaces `dx/dt = α·x - β·x·y` with `dx/dt = α·x·(1 - x/K) - β·x·y`.
    The carrying capacity K is fixed and not inferred.
    """
    if task_name != "lotka_volterra":
        raise ValueError(
            f"_carrying_capacity is only defined for 'lotka_volterra', got '{task_name}'"
        )
    from diffeqtorch import DiffEq

    task = sbibm.get_task("lotka_volterra")

    de = DiffEq(
        f="""
        function f(du, u, p, t)
            x, y = u
            alpha, beta, gamma, delta, K = p
            du[1] = alpha * x * (1 - x / K) - beta * x * y
            du[2] = -gamma * y + delta * x * y
        end
        """,
        saveat=task.saveat,
        debug=False,
    )

    dim_data_raw = task.dim_data_raw  # 2 * num_timepoints
    dim_data = task.dim_data  # 20

    def misspecified_simulator(theta):
        num_samples = theta.shape[0]
        # Append K to each parameter vector
        K_col = torch.full((num_samples, 1), float(K))
        p = torch.cat([theta, K_col], dim=1)

        us = []
        with suppress_julia_output():
            for i in range(num_samples):
                u, t = de(task.u0, task.tspan, p[i, :])
                if u.shape != torch.Size([2, int(dim_data_raw / 2)]):
                    u = float("nan") * torch.ones((2, int(dim_data_raw / 2)))
                    u = u.double()
                us.append(u.reshape(1, 2, -1))
        us = torch.cat(us).float()  # (num_samples, 2, num_timepoints)

        # Subsample every 21 steps, flatten to (num_samples, dim_data)
        us = us[:, :, ::21].reshape(num_samples, -1)

        nan_mask = torch.isnan(us).any(dim=1)
        data = float("nan") * torch.ones((num_samples, dim_data))
        ok = ~nan_mask
        if ok.any():
            data[ok, :] = pyro.sample(
                "data",
                pdist.LogNormal(
                    loc=torch.log(us[ok, :].clamp(1e-10, 10000.0)),
                    scale=0.1,
                ).to_event(1),
            )
        return data

    return misspecified_simulator


def _diagonal_covariance(task_name):
    """SLCP simulator that ignores the correlation parameter (forces rho=0).

    The model assumes a full covariance matrix, but the true data has
    independent dimensions — a common simplifying assumption in practice.
    """
    if task_name != "slcp":
        raise ValueError(f"_diagonal_covariance is only defined for 'slcp', got '{task_name}'")

    num_data = 4

    def misspecified_simulator(theta):
        num_samples = theta.shape[0]

        m = torch.stack(
            (theta[:, 0], theta[:, 1])
        ).T
        if m.dim() == 1:
            m.unsqueeze_(0)

        s1 = theta[:, 2] ** 2
        s2 = theta[:, 3] ** 2
        # Ignore theta[:, 4] — force rho=0

        S = torch.zeros((num_samples, 2, 2))
        S[:, 0, 0] = s1 ** 2 + 1e-6
        S[:, 1, 1] = s2 ** 2 + 1e-6

        data_dist = pdist.MultivariateNormal(
            m.unsqueeze(1).float(), S.unsqueeze(1).float()
        ).expand((num_samples, num_data))

        return pyro.sample("data", data_dist).reshape(num_samples, -1)

    return misspecified_simulator


def _heavy_tail_likelihood(task_name, df=3):
    """SLCP simulator with Student-t observations instead of Gaussian.

    Replaces MVN(m, S) with a multivariate Student-t(df, m, S).
    The model assumes Gaussian, but the real data has heavier tails.
    """
    if task_name != "slcp":
        raise ValueError(
            f"_heavy_tail_likelihood is only defined for 'slcp', got '{task_name}'"
        )

    num_data = 4

    def misspecified_simulator(theta):
        num_samples = theta.shape[0]

        m = torch.stack((theta[:, 0], theta[:, 1])).T
        if m.dim() == 1:
            m.unsqueeze_(0)

        s1 = theta[:, 2] ** 2
        s2 = theta[:, 3] ** 2
        rho = torch.tanh(theta[:, 4])

        S = torch.empty((num_samples, 2, 2))
        S[:, 0, 0] = s1 ** 2 + 1e-6
        S[:, 0, 1] = rho * s1 * s2
        S[:, 1, 0] = rho * s1 * s2
        S[:, 1, 1] = s2 ** 2 + 1e-6

        # Cholesky for scale_tril required by MultivariateStudentT
        L = torch.linalg.cholesky(S)

        # Sample num_data independent Student-t draws per sample
        samples = []
        for i in range(num_data):
            dist = pdist.MultivariateStudentT(
                df=df, loc=m.float(), scale_tril=L.float()
            )
            samples.append(pyro.sample(f"data_{i}", dist))
        # Stack to (num_samples, num_data, 2) then flatten to (num_samples, 8)
        return torch.stack(samples, dim=1).reshape(num_samples, -1)

    return misspecified_simulator


def _heteroscedastic(task_name, alpha=0.5):
    """SLCP simulator with input-dependent variance.

    Scales the covariance by (1 + alpha * ||m||), so variance grows
    with the magnitude of the mean. The model assumes homoscedastic noise.
    """
    if task_name != "slcp":
        raise ValueError(
            f"_heteroscedastic is only defined for 'slcp', got '{task_name}'"
        )

    num_data = 4

    def misspecified_simulator(theta):
        num_samples = theta.shape[0]

        m = torch.stack((theta[:, 0], theta[:, 1])).T
        if m.dim() == 1:
            m.unsqueeze_(0)

        s1 = theta[:, 2] ** 2
        s2 = theta[:, 3] ** 2
        rho = torch.tanh(theta[:, 4])

        S = torch.empty((num_samples, 2, 2))
        S[:, 0, 0] = s1 ** 2
        S[:, 0, 1] = rho * s1 * s2
        S[:, 1, 0] = rho * s1 * s2
        S[:, 1, 1] = s2 ** 2

        # Scale covariance by (1 + alpha * ||m||)
        m_norm = torch.norm(m, dim=1, keepdim=True).unsqueeze(2)  # (N, 1, 1)
        S = S * (1.0 + alpha * m_norm)

        S[:, 0, 0] += 1e-6
        S[:, 1, 1] += 1e-6

        data_dist = pdist.MultivariateNormal(
            m.unsqueeze(1).float(), S.unsqueeze(1).float()
        ).expand((num_samples, num_data))

        return pyro.sample("data", data_dist).reshape(num_samples, -1)

    return misspecified_simulator


def _heavy_tail_radius(task_name, df=2):
    """Two Moons simulator with Student-t radius instead of Gaussian.

    Inlines the sbibm Two Moons logic (rotation by -pi/4, abs shift) but
    replaces r ~ Normal(0.1, 0.01) with r ~ StudentT(df, 0.1, 0.01).
    """
    if task_name != "two_moons":
        raise ValueError(f"_heavy_tail_radius is only defined for 'two_moons', got '{task_name}'")

    ang = torch.tensor([-math.pi / 4.0])
    c = torch.cos(ang)
    s = torch.sin(ang)
    shift = 0.2

    def misspecified_simulator(theta):
        n = theta.shape[0]
        a = pdist.Uniform(-math.pi / 2.0, math.pi / 2.0).sample((n, 1))
        r = pdist.StudentT(df, 0.1, 0.01).sample((n, 1))
        p = torch.cat([torch.cos(a) * r + 0.25, torch.sin(a) * r], dim=1)

        # Rotate theta by -pi/4 and apply abs-shift (sbibm _map_fun)
        z0 = (c * theta[:, 0] - s * theta[:, 1]).reshape(-1, 1)
        z1 = (s * theta[:, 0] + c * theta[:, 1]).reshape(-1, 1)
        return p + (1.0 + shift) * torch.cat([-torch.abs(z0), z1], dim=1)

    return misspecified_simulator


def _wrong_noise_scale(task_name, scale=0.5):
    """LV-HD with wrong observation-noise scale (true scale = 0.1).

    Same Julia ODE solve as the true simulator; only the LogNormal scale on
    the final observation step changes.
    """
    if task_name != "lotka_volterra_hd":
        raise ValueError(
            f"_wrong_noise_scale is only defined for 'lotka_volterra_hd', got '{task_name}'"
        )
    from tabpfn_misspec.lotka_volterra_hd import _theta_to_p

    task = get_task("lotka_volterra_hd")
    de = task.de
    u0, tspan = task.u0, task.tspan
    dim_data_raw, dim_data = task.dim_data_raw, task.dim_data

    def misspecified_simulator(theta):
        num_samples = theta.shape[0]
        p = _theta_to_p(theta)

        us = []
        with suppress_julia_output():
            for n in range(num_samples):
                u, _t = de(u0, tspan, p[n, :])
                if u.shape != torch.Size([5, int(dim_data_raw / 5)]):
                    u = float("nan") * torch.ones(
                        (5, int(dim_data_raw / 5))
                    ).double()
                us.append(u.reshape(1, 5, -1))
        us = torch.cat(us).float()[:, :, ::21].reshape(num_samples, -1)

        data = float("nan") * torch.ones((num_samples, dim_data))
        ok = ~torch.isnan(us).any(dim=1)
        if ok.any():
            data[ok] = pyro.sample(
                "data",
                pdist.LogNormal(
                    loc=torch.log(us[ok].clamp(1e-10, 1e4)),
                    scale=scale,
                ).to_event(1),
            )
        return data

    return misspecified_simulator


def _linear_misspec(task_name, sigma_x=0.1):
    """Linear misspecified simulator: x = A @ theta + b + eps_x.

    Reads task.A and task.b from the custom task; intended for
    ``gaussian_linear_hd`` (and any future linear-Gaussian custom task that
    exposes the same attributes).
    """
    task = get_task(task_name)
    A, b = task.A, task.b
    dim_x = A.shape[0]

    def misspecified_simulator(theta):
        return theta @ A.T + b + sigma_x * torch.randn(theta.shape[0], dim_x)

    return misspecified_simulator


def _nonlinear_theta(task_name, sigma_x=0.5, alpha=0.1):
    """Quadratic-in-theta misspecified simulator.

    x = theta @ A.T + b + alpha * (theta**2) @ A.T + sigma_x * eps_x

    Reduces to ``_linear_misspec`` when alpha == 0. Reuses ``task.A`` and
    ``task.b`` (the same near-copy of ``C``, ``d`` used by the linear
    misspec), so the only structural difference vs ``linear_misspec`` is
    the elementwise quadratic term in theta.
    """
    task = get_task(task_name)
    A, b = task.A, task.b
    dim_x = A.shape[0]

    def misspecified_simulator(theta):
        return (
            theta @ A.T
            + b
            + alpha * (theta ** 2) @ A.T
            + sigma_x * torch.randn(theta.shape[0], dim_x)
        )

    return misspecified_simulator


def _ellipse_modes(task_name, axis_ratio=0.5, sigma_x=None):
    """Misspecified Gaussian-mixture simulator: modes on an ELLIPSE.

    Same model as the well-specified mixture (x = theta + c_z + eps_x) but the
    anchors c_k lie on an ellipse with semi-axes (rho, rho*axis_ratio) instead
    of a circle of radius rho. axis_ratio == 1.0 recovers the circle.
    """
    if task_name not in ("gaussian_mixture_hd", "gaussian_mixture_2d"):
        raise ValueError(
            f"_ellipse_modes is only defined for the gaussian_mixture tasks, got '{task_name}'"
        )
    task = get_task(task_name)
    D, n_modes, rho, phis = task.dim_x, task.n_modes, task.rho, task.phis
    sigma_x = task.sigma if sigma_x is None else sigma_x

    anchors = torch.zeros(n_modes, D)
    anchors[:, 0] = rho * torch.cos(phis)
    anchors[:, 1] = (rho * axis_ratio) * torch.sin(phis)

    def misspecified_simulator(theta):
        num_samples = theta.shape[0]
        z = torch.randint(0, n_modes, (num_samples,))
        return theta + anchors[z] + sigma_x * torch.randn(num_samples, D)

    return misspecified_simulator


_REGISTRY = {
    # Generic (any task)
    "additive_noise": _additive_noise,
    "scale_shift": _scale_shift,
    # Task-specific entries use (task_name, misspec_type) keys:
    # ("two_moons", "wrong_likelihood"): _two_moons_wrong_likelihood,
    ("gaussian_mixture", "one_gaussian"): _one_gaussian,
    ("two_moons", "heavy_tail_radius"): _heavy_tail_radius,
    ("slcp", "diagonal_covariance"): _diagonal_covariance,
    ("slcp", "heavy_tail_likelihood"): _heavy_tail_likelihood,
    ("slcp", "heteroscedastic"): _heteroscedastic,
    ("sir", "weekend_delay"): _weekend_delay,
    ("lotka_volterra", "carrying_capacity"): _carrying_capacity,
    ("gaussian_linear_hd", "linear_misspec"): _linear_misspec,
    ("gaussian_linear_hd", "nonlinear_theta"): _nonlinear_theta,
    ("gaussian_mixture_hd", "ellipse_modes"): _ellipse_modes,
    ("gaussian_mixture_2d", "ellipse_modes"): _ellipse_modes,
    ("lotka_volterra_hd", "wrong_noise_scale"): _wrong_noise_scale,
}


def get_misspecified_simulator(task_name, misspec_type, **kwargs):
    """Get a misspecified simulator for a given sbibm task.

    Args:
        task_name: Name of the sbibm task (e.g. "two_moons").
        misspec_type: Type of misspecification (e.g. "additive_noise").
        **kwargs: Passed to the misspecification factory.

    Returns:
        Callable that takes theta (N, dim_params) and returns x (N, dim_data).
    """
    factory = _REGISTRY.get((task_name, misspec_type)) or _REGISTRY.get(misspec_type)
    if factory is None:
        raise ValueError(
            f"Unknown misspec_type '{misspec_type}' for task '{task_name}'. "
            f"Available: {list(list_misspec_types(task_name))}"
        )
    print(f"  Simulator: {misspec_type} ({task_name})")
    return factory(task_name, **kwargs)


def list_misspec_types(task_name=None):
    """Return available misspecification types.

    Args:
        task_name: If given, include task-specific types for this task
            in addition to generic types. If None, return all types.

    Returns:
        Set of misspec_type strings.
    """
    types = set()
    for key in _REGISTRY:
        if isinstance(key, tuple):
            t_name, m_type = key
            if task_name is None or t_name == task_name:
                types.add(m_type)
        else:
            types.add(key)
    return types
