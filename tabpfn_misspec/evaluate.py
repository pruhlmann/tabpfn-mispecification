"""Core evaluation loop for misspecification experiments."""

import time
from dataclasses import asdict, dataclass

import sbibm
import torch

from npe_pfn import TabPFN_Based_NPE_PFN

from tabpfn_misspec.metrics import c2st, mmd
from tabpfn_misspec.simulators import get_misspecified_simulator


def get_parameter_transform(prior):
    """Return (forward, inverse, unbounded_prior) for mapping bounded priors to R^n.

    For priors with bounded support (e.g. Independent(Uniform)), returns a logit
    transform to R^n, its sigmoid inverse, and an unbounded prior for the
    transformed space. For unbounded priors, returns identity functions and the
    original prior.
    """
    try:
        low = prior.base_dist.low
        high = prior.base_dist.high
    except AttributeError:
        return lambda theta: theta, lambda z: z, prior

    if not (torch.isfinite(low).all() and torch.isfinite(high).all()):
        return lambda theta: theta, lambda z: z, prior

    dim = low.shape[0]
    unbounded_prior = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(dim), 100.0 * torch.ones(dim)), 1
    )

    def forward(theta):
        normalized = (theta - low) / (high - low)
        return torch.logit(normalized.clamp(1e-6, 1 - 1e-6))

    def inverse(z):
        return torch.sigmoid(z) * (high - low) + low

    return forward, inverse, unbounded_prior


@dataclass
class EvalResult:
    task_name: str
    misspec_type: str
    misspec_kwargs: dict
    num_observation: int
    num_simulations: int
    c2st: float
    mmd: float
    method: str = "npepfn_misspec"

    def to_dict(self):
        return asdict(self)


def evaluate_misspecification(
    task_name,
    misspec_type,
    misspec_kwargs=None,
    num_simulations=1000,
    num_posterior_samples=1000,
    num_observations=1,
    seed=42,
    use_prior_transform=True,
):
    """Run misspecification evaluation on an sbibm task.

    Args:
        task_name: sbibm task name (e.g. "two_moons").
        misspec_type: Misspecification type (e.g. "additive_noise").
        misspec_kwargs: Kwargs for the misspecified simulator.
        num_simulations: Number of (theta, x) training pairs.
        num_posterior_samples: Samples to draw from the approximate posterior.
        num_observations: Number of sbibm observations to evaluate.
        seed: Random seed.

    Returns:
        List of EvalResult, one per observation.
    """
    if misspec_kwargs is None:
        misspec_kwargs = {}

    torch.manual_seed(seed)

    task = sbibm.get_task(task_name)
    prior = task.get_prior_dist()
    simulator = get_misspecified_simulator(task_name, misspec_type, **misspec_kwargs)

    if use_prior_transform:
        forward, inverse, est_prior = get_parameter_transform(prior)
    else:
        forward, inverse, est_prior = lambda t: t, lambda z: z, prior

    # Generate training data from misspecified simulator
    theta = prior.sample((num_simulations,))
    x = simulator(theta)

    estimator = TabPFN_Based_NPE_PFN(prior=est_prior)
    estimator.append_simulations(forward(theta), x)

    # Evaluate on each observation
    results = []
    for obs_idx in range(1, num_observations + 1):
        y_obs = task.get_observation(obs_idx)
        ref_samples = task.get_reference_posterior_samples(obs_idx)

        posterior_samples = inverse(estimator.sample(
            sample_shape=torch.Size([num_posterior_samples]),
            x=y_obs,
        ))

        result = EvalResult(
            task_name=task_name,
            misspec_type=misspec_type,
            misspec_kwargs=misspec_kwargs,
            num_observation=obs_idx,
            num_simulations=num_simulations,
            c2st=float(c2st(posterior_samples, ref_samples)),
            mmd=float(mmd(posterior_samples, ref_samples)),
        )
        results.append(result)
        print(
            f"  obs {obs_idx}: C2ST={result.c2st:.3f}, MMD={result.mmd:.4f}"
        )

    return results


def evaluate_calibrated_misspecification(
    task_name,
    misspec_type,
    misspec_kwargs=None,
    num_simulations=1000,
    num_calibration=50,
    num_posterior_samples=1000,
    num_observations=1,
    num_synthetic=1000,
    seed=42,
    use_prior_transform=True,
):
    """Run calibrated misspecification evaluation comparing 5 methods.

    Methods:
        npepfn_misspec: NPE-PFN on misspecified simulations only.
        npepfn_calib: NPE-PFN on calibration (true) data only.
        npepfn_mixed: NPE-PFN with mixed sim + calibration context (NaN masking).
        npe_sbi: Standard NPE from sbi trained on calibration data.
        npepfn_y_fmpe: TabPFN y-corrector + FMPE posterior.

    Args:
        task_name: sbibm task name (e.g. "two_moons").
        misspec_type: Misspecification type (e.g. "additive_noise").
        misspec_kwargs: Kwargs for the misspecified simulator.
        num_simulations: Number of (theta, x) simulation pairs.
        num_calibration: Number of (theta, y) calibration pairs.
        num_posterior_samples: Samples to draw from each posterior.
        num_observations: Number of sbibm observations to evaluate.
        num_synthetic: Number of synthetic (theta, ỹ) pairs for y-corrector method.
        seed: Random seed.

    Returns:
        List of EvalResult with method field set per method.
    """
    from tabpfn_misspec.baselines import evaluate_npe_sbi, evaluate_npepfn_calib_only, evaluate_npepfn_y_fmpe
    from tabpfn_misspec.calibrated import build_calibrated_estimator, sample_calibrated

    if misspec_kwargs is None:
        misspec_kwargs = {}

    torch.manual_seed(seed)

    task = sbibm.get_task(task_name)
    prior = task.get_prior_dist()
    true_simulator = task.get_simulator()
    misspec_simulator = get_misspecified_simulator(task_name, misspec_type, **misspec_kwargs)

    if use_prior_transform:
        forward, inverse, est_prior = get_parameter_transform(prior)
    else:
        forward, inverse, est_prior = lambda t: t, lambda z: z, prior

    # --- Shared data generation ---
    # Simulation set: (theta_sim, x_sim) from misspecified simulator
    theta_sim = prior.sample((num_simulations,))
    x_sim = misspec_simulator(theta_sim)

    # Calibration set: (theta_calib, y_calib) from true simulator + x_calib from misspecified
    theta_calib = prior.sample((num_calibration,))
    y_calib = true_simulator(theta_calib)
    x_calib = misspec_simulator(theta_calib)

    # Transform theta to unbounded space for TabPFN-based methods
    theta_sim_t = forward(theta_sim)
    theta_calib_t = forward(theta_calib)

    dim_x = x_sim.shape[1]
    all_results = []

    # --- Method 1: NPE-PFN (misspecified) ---
    print("Running npepfn_misspec...")
    estimator_misspec = TabPFN_Based_NPE_PFN(prior=est_prior)
    estimator_misspec.append_simulations(theta_sim_t, x_sim)
    for obs_idx in range(1, num_observations + 1):
        y_obs = task.get_observation(obs_idx)
        ref_samples = task.get_reference_posterior_samples(obs_idx)
        t0 = time.perf_counter()
        posterior_samples = inverse(estimator_misspec.sample(
            sample_shape=torch.Size([num_posterior_samples]), x=y_obs,
        ))
        t_inference = time.perf_counter() - t0
        result = EvalResult(
            task_name=task_name, misspec_type=misspec_type,
            misspec_kwargs=misspec_kwargs, num_observation=obs_idx,
            num_simulations=num_simulations,
            c2st=float(c2st(posterior_samples, ref_samples)),
            mmd=float(mmd(posterior_samples, ref_samples)),
            method="npepfn_misspec",
        )
        all_results.append(result)
        print(f"  [npepfn_misspec] obs {obs_idx}: C2ST={result.c2st:.3f}, MMD={result.mmd:.4f}, inference={t_inference:.1f}s")

    # --- Method 2: NPE-PFN (calibration only) ---
    print("Running npepfn_calib...")
    all_results.extend(evaluate_npepfn_calib_only(
        task, theta_calib_t, y_calib, inverse, est_prior,
        num_posterior_samples, num_observations,
        task_name, misspec_type, misspec_kwargs,
    ))

    # --- Method 3: NPE-PFN (mixed, ours) ---
    print("Running npepfn_mixed...")
    estimator_mixed = build_calibrated_estimator(
        theta_sim_t, x_sim, theta_calib_t, x_calib, y_calib, est_prior,
    )
    for obs_idx in range(1, num_observations + 1):
        y_obs = task.get_observation(obs_idx)
        ref_samples = task.get_reference_posterior_samples(obs_idx)
        t0 = time.perf_counter()
        posterior_samples = inverse(sample_calibrated(
            estimator_mixed, y_obs, dim_x, num_posterior_samples,
        ))
        t_inference = time.perf_counter() - t0
        result = EvalResult(
            task_name=task_name, misspec_type=misspec_type,
            misspec_kwargs=misspec_kwargs, num_observation=obs_idx,
            num_simulations=num_simulations,
            c2st=float(c2st(posterior_samples, ref_samples)),
            mmd=float(mmd(posterior_samples, ref_samples)),
            method="npepfn_mixed",
        )
        all_results.append(result)
        print(f"  [npepfn_mixed] obs {obs_idx}: C2ST={result.c2st:.3f}, MMD={result.mmd:.4f}, inference={t_inference:.1f}s")

    # --- Method 4: NPE (sbi) ---
    print("Running npe_sbi...")
    all_results.extend(evaluate_npe_sbi(
        task, prior, theta_calib, y_calib, num_posterior_samples, num_observations,
        task_name, misspec_type, misspec_kwargs,
    ))

    # --- Method 5: TabPFN y-corrector + FMPE ---
    print("Running npepfn_y_fmpe...")
    all_results.extend(evaluate_npepfn_y_fmpe(
        task, prior, theta_calib, x_calib, y_calib,
        misspec_simulator, num_synthetic, num_posterior_samples, num_observations,
        task_name, misspec_type, misspec_kwargs,
    ))

    return all_results
