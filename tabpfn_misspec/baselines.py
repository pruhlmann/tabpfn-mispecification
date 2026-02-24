"""Baseline methods for calibration comparison experiments."""

import time

import torch
import sbibm

from npe_pfn import TabPFN_Based_NPE_PFN
from sbi.inference import FMPE

from tabpfn_misspec.calibrated import build_y_predictor, generate_synthetic_y
from tabpfn_misspec.evaluate import EvalResult
from tabpfn_misspec.metrics import c2st, mmd


def _save_posterior(artifacts_dir, method, obs_idx, seed, samples):
    """Save posterior samples to a .pt file."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.save(samples, artifacts_dir / f"{method}_seed{seed}_obs{obs_idx}.pt")


def evaluate_npepfn_calib_only(
    task,
    theta_calib_t,
    y_calib,
    inverse,
    unbounded_prior,
    num_posterior_samples,
    num_observations,
    task_name,
    misspec_type,
    misspec_kwargs,
    seed=42,
    artifacts_dir=None,
):
    """Baseline: NPE-PFN with only calibration (theta, y) as context.

    Uses the standard TabPFN_Based_NPE_PFN with calibration data only —
    no simulation data, no NaN masking.

    Args:
        theta_calib_t: Calibration parameters in transformed (unbounded) space.
        inverse: Callable to map samples back to original space.
        unbounded_prior: Prior in transformed space (for rejection sampling).

    Returns:
        List of EvalResult with method="npepfn_calib".
    """
    estimator = TabPFN_Based_NPE_PFN(prior=unbounded_prior)
    estimator.append_simulations(theta_calib_t, y_calib)

    results = []
    for obs_idx in range(1, num_observations + 1):
        y_obs = task.get_observation(obs_idx)
        ref_samples = task.get_reference_posterior_samples(obs_idx)

        t0 = time.perf_counter()
        posterior_samples = inverse(
            estimator.sample(
                sample_shape=torch.Size([num_posterior_samples]),
                x=y_obs,
            )
        )
        t_inference = time.perf_counter() - t0

        if artifacts_dir is not None:
            _save_posterior(artifacts_dir, "npepfn_calib", obs_idx, seed, posterior_samples)

        result = EvalResult(
            task_name=task_name,
            misspec_type=misspec_type,
            misspec_kwargs=misspec_kwargs,
            num_observation=obs_idx,
            num_simulations=len(theta_calib_t),
            c2st=float(c2st(posterior_samples, ref_samples)),
            mmd=float(mmd(posterior_samples, ref_samples)),
            method="npepfn_calib",
        )
        results.append(result)
        print(
            f"  [npepfn_calib] obs {obs_idx}: C2ST={result.c2st:.3f}, MMD={result.mmd:.4f}, inference={t_inference:.1f}s"
        )

    return results


def evaluate_npe_sbi(
    task,
    prior,
    theta_calib,
    y_calib,
    num_posterior_samples,
    num_observations,
    task_name,
    misspec_type,
    misspec_kwargs,
    seed=42,
    artifacts_dir=None,
):
    """Baseline: standard NPE from sbi trained on calibration data.

    Returns:
        List of EvalResult with method="npe_sbi".
    """
    from sbi.inference import NPE

    inference = NPE(prior=prior)
    inference.append_simulations(theta_calib, y_calib)
    t0 = time.perf_counter()
    density_estimator = inference.train()
    t_train = time.perf_counter() - t0
    posterior = inference.build_posterior(density_estimator)
    print(f"  [npe_sbi] training={t_train:.1f}s")

    results = []
    for obs_idx in range(1, num_observations + 1):
        y_obs = task.get_observation(obs_idx)
        ref_samples = task.get_reference_posterior_samples(obs_idx)

        t0 = time.perf_counter()
        posterior_samples = posterior.sample((num_posterior_samples,), x=y_obs)
        t_inference = time.perf_counter() - t0

        if artifacts_dir is not None:
            _save_posterior(artifacts_dir, "npe_sbi", obs_idx, seed, posterior_samples)

        result = EvalResult(
            task_name=task_name,
            misspec_type=misspec_type,
            misspec_kwargs=misspec_kwargs,
            num_observation=obs_idx,
            num_simulations=len(theta_calib),
            c2st=float(c2st(posterior_samples, ref_samples)),
            mmd=float(mmd(posterior_samples, ref_samples)),
            method="npe_sbi",
        )
        results.append(result)
        print(f"  [npe_sbi] obs {obs_idx}: C2ST={result.c2st:.3f}, MMD={result.mmd:.4f}, inference={t_inference:.1f}s")

    return results


def evaluate_npepfn_y_fmpe(
    task,
    prior,
    theta_calib,
    x_calib,
    y_calib,
    misspec_simulator,
    num_synthetic,
    num_posterior_samples,
    num_observations,
    task_name,
    misspec_type,
    misspec_kwargs,
    seed=42,
    artifacts_dir=None,
):
    """Two-stage method: TabPFN y-corrector + FMPE posterior.

    Stage 1: Build a TabPFN y-predictor from calibration data.
    Stage 2: Generate synthetic (theta, ỹ) pairs via the y-predictor.
    Stage 3: Train FMPE on synthetic data for posterior inference.

    Args:
        task: sbibm task.
        prior: Original prior distribution.
        theta_calib: Calibration parameters in original space.
        x_calib: Misspecified simulator outputs for calibration params.
        y_calib: True simulator outputs for calibration params.
        misspec_simulator: Misspecified simulator callable.
        num_synthetic: Number of synthetic (theta, ỹ) pairs to generate.
        num_posterior_samples: Samples to draw from each posterior.
        num_observations: Number of sbibm observations to evaluate.
        task_name: sbibm task name.
        misspec_type: Misspecification type string.
        misspec_kwargs: Misspecification kwargs dict.

    Returns:
        List of EvalResult with method="npepfn_y_fmpe".
    """
    # Stage 1: build y-predictor from calibration data
    y_predictor = build_y_predictor(theta_calib, x_calib, y_calib)
    train_features = torch.cat([theta_calib, x_calib], dim=1)

    # Save y-diagnostics: predicted vs true y on calibration set
    if artifacts_dir is not None:
        y_pred = generate_synthetic_y(y_predictor, theta_calib, x_calib, train_features=train_features)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "y_pred": y_pred,
                "y_true": y_calib,
                "theta_calib": theta_calib,
                "x_calib": x_calib,
            },
            artifacts_dir / f"y_diag_seed{seed}.pt",
        )

    # Stage 2: generate synthetic dataset
    theta_syn = prior.sample((num_synthetic,))
    x_syn = misspec_simulator(theta_syn)
    t0 = time.perf_counter()
    y_tilde = generate_synthetic_y(y_predictor, theta_syn, x_syn, train_features=train_features)
    t_gen = time.perf_counter() - t0
    print(f"  [npepfn_y_fmpe] generated {num_synthetic} synthetic samples in {t_gen:.1f}s")

    # Stage 3: train FMPE
    fmpe = FMPE(prior=prior)
    fmpe.append_simulations(theta_syn, y_tilde)
    t0 = time.perf_counter()
    density_estimator = fmpe.train()
    t_train = time.perf_counter() - t0
    posterior = fmpe.build_posterior(density_estimator)
    print(f"  [npepfn_y_fmpe] FMPE training={t_train:.1f}s")

    # Stage 4: evaluate
    results = []
    for obs_idx in range(1, num_observations + 1):
        y_obs = task.get_observation(obs_idx)
        ref_samples = task.get_reference_posterior_samples(obs_idx)

        t0 = time.perf_counter()
        posterior_samples = posterior.sample((num_posterior_samples,), x=y_obs)
        t_inference = time.perf_counter() - t0

        if artifacts_dir is not None:
            _save_posterior(artifacts_dir, "npepfn_y_fmpe", obs_idx, seed, posterior_samples)

        result = EvalResult(
            task_name=task_name,
            misspec_type=misspec_type,
            misspec_kwargs=misspec_kwargs,
            num_observation=obs_idx,
            num_simulations=num_synthetic,
            c2st=float(c2st(posterior_samples, ref_samples)),
            mmd=float(mmd(posterior_samples, ref_samples)),
            method="npepfn_y_fmpe",
        )
        results.append(result)
        print(
            f"  [npepfn_y_fmpe] obs {obs_idx}: C2ST={result.c2st:.3f}, MMD={result.mmd:.4f}, inference={t_inference:.1f}s"
        )

    return results
