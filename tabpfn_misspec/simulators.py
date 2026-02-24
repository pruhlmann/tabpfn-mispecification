"""Misspecified simulator registry for sbibm tasks."""

import math

import sbibm
from sbibm.tasks.gaussian_mixture.task import GaussianMixture
import torch
import pyro
import pyro.distributions as pdist


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
        raise ValueError(f"_one_gaussian misspecification is only defined for 'gaussian_mixture', got '{task_name}'")
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

    def misspecified_simulator(theta):
        n = theta.shape[0]
        a = pdist.Uniform(-math.pi / 2.0, math.pi / 2.0).sample((n, 1))
        r = pdist.StudentT(df, 0.1, 0.01).sample((n, 1))
        p = torch.cat([torch.cos(a) * r + 0.25, torch.sin(a) * r], dim=1)

        # Rotate theta by -pi/4 and apply abs-shift (sbibm _map_fun)
        z0 = (c * theta[:, 0] - s * theta[:, 1]).reshape(-1, 1)
        z1 = (s * theta[:, 0] + c * theta[:, 1]).reshape(-1, 1)
        return p + torch.cat([-torch.abs(z0), z1], dim=1)

    return misspecified_simulator


_REGISTRY = {
    # Generic (any task)
    "additive_noise": _additive_noise,
    "scale_shift": _scale_shift,
    # Task-specific entries use (task_name, misspec_type) keys:
    # ("two_moons", "wrong_likelihood"): _two_moons_wrong_likelihood,
    ("gaussian_mixture", "one_gaussian"): _one_gaussian,
    ("two_moons", "heavy_tail_radius"): _heavy_tail_radius,
    ("sir", "weekend_delay"): _weekend_delay,
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
    print("Loading misspecified simulator:", factory, "for task:", task_name)
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
