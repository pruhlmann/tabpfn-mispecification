"""Baseline methods for calibration comparison experiments."""

import time

import torch

from npe_pfn import TabPFN_Based_NPE_PFN
from sbi.inference import FMPE

from tabpfn_misspec.evaluate import EvalResult, _compute_sample_metrics, _info, _obs_line, _save_metrics


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
    forward=None,
    log_abs_det_jac=None,
    metrics_to_compute=("c2st", "mmd", "log_prob"),
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
    compute_log_prob = "log_prob" in metrics_to_compute
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

        sample_metrics = _compute_sample_metrics(
            posterior_samples, ref_samples, metrics_to_compute
        )
        log_prob_val = float("nan")
        if compute_log_prob and forward is not None and log_abs_det_jac is not None:
            try:
                ref_t = forward(ref_samples)
                lp = estimator.log_prob(ref_t, x=y_obs) + log_abs_det_jac(ref_t)
                log_prob_val = float(lp.mean())
            except Exception:
                pass
        sample_metrics["log_prob"] = log_prob_val
        if artifacts_dir is not None:
            _save_metrics(artifacts_dir, "npepfn_calib", obs_idx, seed, sample_metrics)
        result = EvalResult(
            task_name=task_name,
            misspec_type=misspec_type,
            misspec_kwargs=misspec_kwargs,
            num_observation=obs_idx,
            num_context_size=len(theta_calib_t),
            c2st=sample_metrics.get("c2st", float("nan")),
            mmd=sample_metrics.get("mmd", float("nan")),
            log_prob=log_prob_val,
            method="npepfn_calib",
        )
        results.append(result)
        _obs_line("npepfn_calib", obs_idx, {"c2st": result.c2st, "mmd": result.mmd, "log_prob": result.log_prob}, f"({t_inference:.1f}s)")

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
    metrics_to_compute=("c2st", "mmd", "log_prob"),
):
    """Baseline: standard NPE from sbi trained on calibration data.

    Returns:
        List of EvalResult with method="npe_sbi".
    """
    from sbi.inference import NPE

    compute_log_prob = "log_prob" in metrics_to_compute
    inference = NPE(prior=prior)
    inference.append_simulations(theta_calib, y_calib)
    t0 = time.perf_counter()
    density_estimator = inference.train()
    t_train = time.perf_counter() - t0
    posterior = inference.build_posterior(density_estimator)
    _info(f"Training completed ({t_train:.1f}s)")

    results = []
    for obs_idx in range(1, num_observations + 1):
        y_obs = task.get_observation(obs_idx)
        ref_samples = task.get_reference_posterior_samples(obs_idx)

        t0 = time.perf_counter()
        posterior_samples = posterior.sample((num_posterior_samples,), x=y_obs)
        t_inference = time.perf_counter() - t0

        if artifacts_dir is not None:
            _save_posterior(artifacts_dir, "npe_sbi", obs_idx, seed, posterior_samples)

        sample_metrics = _compute_sample_metrics(
            posterior_samples, ref_samples, metrics_to_compute
        )
        log_prob_val = float("nan")
        if compute_log_prob:
            try:
                lp = posterior.log_prob(ref_samples, x=y_obs)
                log_prob_val = float(lp.mean())
            except Exception:
                pass
        sample_metrics["log_prob"] = log_prob_val
        if artifacts_dir is not None:
            _save_metrics(artifacts_dir, "npe_sbi", obs_idx, seed, sample_metrics)
        result = EvalResult(
            task_name=task_name,
            misspec_type=misspec_type,
            misspec_kwargs=misspec_kwargs,
            num_observation=obs_idx,
            num_context_size=len(theta_calib),
            c2st=sample_metrics.get("c2st", float("nan")),
            mmd=sample_metrics.get("mmd", float("nan")),
            log_prob=log_prob_val,
            method="npe_sbi",
        )
        results.append(result)
        _obs_line("npe_sbi", obs_idx, {"c2st": result.c2st, "mmd": result.mmd, "log_prob": result.log_prob}, f"({t_inference:.1f}s)")

    return results


def evaluate_npepfn_y_fmpe(
    task,
    prior,
    theta_syn,
    y_tilde,
    theta_calib,
    y_calib,
    num_posterior_samples,
    num_observations,
    task_name,
    misspec_type,
    misspec_kwargs,
    seed=42,
    artifacts_dir=None,
    concat_calib=False,
    metrics_to_compute=("c2st", "mmd", "log_prob"),
):
    """Train FMPE on pre-generated synthetic (theta, ỹ) data.

    Args:
        task: sbibm task.
        prior: Original prior distribution.
        theta_syn: Synthetic parameters, shape (N_syn, dim_theta).
        y_tilde: Synthetic y values, shape (N_syn, dim_y).
        theta_calib: Calibration parameters in original space.
        y_calib: True simulator outputs for calibration params.
        concat_calib: If True, concatenate calibration data with synthetic data.

    Returns:
        List of EvalResult.
    """
    compute_log_prob = "log_prob" in metrics_to_compute
    method_name = "npepfn_y_fmpe_concat" if concat_calib else "npepfn_y_fmpe"

    theta_train = theta_syn
    y_train = y_tilde
    if concat_calib:
        theta_train = torch.cat([theta_syn, theta_calib], dim=0)
        y_train = torch.cat([y_tilde, y_calib], dim=0)

    fmpe = FMPE(prior=prior)
    fmpe.append_simulations(theta_train.float(), y_train.float())
    t0 = time.perf_counter()
    density_estimator = fmpe.train()
    t_train = time.perf_counter() - t0
    posterior = fmpe.build_posterior(density_estimator)
    _info(f"FMPE training completed ({t_train:.1f}s, n_train={len(theta_train)})")

    results = []
    for obs_idx in range(1, num_observations + 1):
        y_obs = task.get_observation(obs_idx)
        ref_samples = task.get_reference_posterior_samples(obs_idx)

        t0 = time.perf_counter()
        posterior_samples = posterior.sample((num_posterior_samples,), x=y_obs)
        t_inference = time.perf_counter() - t0

        if artifacts_dir is not None:
            _save_posterior(artifacts_dir, method_name, obs_idx, seed, posterior_samples)

        sample_metrics = _compute_sample_metrics(
            posterior_samples, ref_samples, metrics_to_compute
        )
        log_prob_val = float("nan")
        if compute_log_prob:
            try:
                lp = posterior.log_prob(ref_samples, x=y_obs)
                log_prob_val = float(lp.mean())
            except Exception:
                pass
        sample_metrics["log_prob"] = log_prob_val
        if artifacts_dir is not None:
            _save_metrics(artifacts_dir, method_name, obs_idx, seed, sample_metrics)
        result = EvalResult(
            task_name=task_name,
            misspec_type=misspec_type,
            misspec_kwargs=misspec_kwargs,
            num_observation=obs_idx,
            num_context_size=len(theta_train),
            c2st=sample_metrics.get("c2st", float("nan")),
            mmd=sample_metrics.get("mmd", float("nan")),
            log_prob=log_prob_val,
            method=method_name,
        )
        results.append(result)
        _obs_line(method_name, obs_idx, {"c2st": result.c2st, "mmd": result.mmd, "log_prob": result.log_prob}, f"({t_inference:.1f}s)")

    return results


def evaluate_npepfn_y_npepfn(
    task,
    theta_syn,
    y_tilde,
    theta_calib_t,
    y_calib,
    inverse,
    unbounded_prior,
    forward,
    num_posterior_samples,
    num_observations,
    task_name,
    misspec_type,
    misspec_kwargs,
    seed=42,
    artifacts_dir=None,
    concat_calib=False,
    log_abs_det_jac=None,
    metrics_to_compute=("c2st", "mmd", "log_prob"),
    method_name=None,
):
    """NPE-PFN trained on pre-generated synthetic (theta, ỹ) data.

    Args:
        task: sbibm task.
        theta_syn: Synthetic parameters in original space, shape (N_syn, dim_theta).
        y_tilde: Synthetic y values, shape (N_syn, dim_y).
        theta_calib_t: Calibration parameters in transformed space.
        y_calib: True simulator outputs for calibration params.
        inverse: Callable to map samples back to original space.
        unbounded_prior: Prior in transformed space.
        forward: Callable to map theta to transformed space.
        concat_calib: If True, concatenate calibration data with synthetic data.
        method_name: Override for the method name in results.

    Returns:
        List of EvalResult.
    """
    compute_log_prob = "log_prob" in metrics_to_compute
    if method_name is None:
        method_name = "npepfn_y_npepfn_concat" if concat_calib else "npepfn_y_npepfn"

    theta_syn_t = forward(theta_syn)
    theta_train = theta_syn_t
    y_train = y_tilde
    if concat_calib:
        theta_train = torch.cat([theta_syn_t, theta_calib_t], dim=0)
        y_train = torch.cat([y_tilde, y_calib], dim=0)

    estimator = TabPFN_Based_NPE_PFN(prior=unbounded_prior)
    estimator.append_simulations(theta_train, y_train)

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
            _save_posterior(artifacts_dir, method_name, obs_idx, seed, posterior_samples)

        sample_metrics = _compute_sample_metrics(
            posterior_samples, ref_samples, metrics_to_compute
        )
        log_prob_val = float("nan")
        if compute_log_prob and log_abs_det_jac is not None:
            try:
                ref_t = forward(ref_samples)
                lp = estimator.log_prob(ref_t, x=y_obs) + log_abs_det_jac(ref_t)
                log_prob_val = float(lp.mean())
            except Exception:
                pass
        sample_metrics["log_prob"] = log_prob_val
        if artifacts_dir is not None:
            _save_metrics(artifacts_dir, method_name, obs_idx, seed, sample_metrics)
        result = EvalResult(
            task_name=task_name,
            misspec_type=misspec_type,
            misspec_kwargs=misspec_kwargs,
            num_observation=obs_idx,
            num_context_size=len(theta_train),
            c2st=sample_metrics.get("c2st", float("nan")),
            mmd=sample_metrics.get("mmd", float("nan")),
            log_prob=log_prob_val,
            method=method_name,
        )
        results.append(result)
        _obs_line(method_name, obs_idx, {"c2st": result.c2st, "mmd": result.mmd, "log_prob": result.log_prob}, f"({t_inference:.1f}s)")

    return results
