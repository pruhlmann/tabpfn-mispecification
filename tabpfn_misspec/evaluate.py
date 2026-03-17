"""Core evaluation loop for misspecification experiments."""

import json
import time
import warnings
from dataclasses import asdict, dataclass
from typing import List

import sbibm
import torch

from npe_pfn import TabPFN_Based_NPE_PFN
from tabpfn_misspec.metrics import c2st, mmd
from tabpfn_misspec.simulators import get_misspecified_simulator, suppress_julia_output

# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------
_W = 60  # header width


def _header(step, total, method):
    tag = f" [{step}/{total}] {method} "
    print(f"\n{'=' * _W}")
    print(f"{tag:=^{_W}}")
    print(f"{'=' * _W}")


def _info(msg):
    print(f"  {msg}")


def _detail(msg):
    print(f"    {msg}")


def _obs_line(method, obs_idx, metrics_dict, extra=""):
    parts = [f"obs {obs_idx}:"]
    for k, v in metrics_dict.items():
        if k == "mmd":
            parts.append(f"{k.upper()}={v:.4f}")
        else:
            parts.append(f"{k.upper()}={v:.3f}" if isinstance(v, float) else f"{k}={v}")
    if extra:
        parts.append(extra)
    _detail(" ".join(parts))


def _save_y_distributional_diag(
    y_predictor,
    prior,
    misspec_simulator,
    true_simulator,
    artifacts_dir,
    seed,
    K=5,
    N_test=1000,
    batched_mode=True,
    batch_size=None,
):
    """Save distributional y-diagnostic data: K thetas x N_test samples each."""
    torch.manual_seed(seed + 9999)  # separate seed for diagnostic
    theta_diag = prior.sample((K,))
    y_true_all = []
    y_tilde_all = []
    for k in range(K):
        theta_k = theta_diag[k].unsqueeze(0).expand(N_test, -1)
        y_true_k = true_simulator(theta_k)
        x_k = misspec_simulator(theta_k)
        from tabpfn_misspec.calibrated import generate_synthetic_y as _gen_syn_y

        y_tilde_k = _gen_syn_y(
            y_predictor,
            theta_k,
            x_k,
            batched_mode=batched_mode,
            batch_size=batch_size,
        )
        y_true_all.append(y_true_k)
        y_tilde_all.append(y_tilde_k)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "theta_diag": theta_diag,
            "y_true": torch.stack(y_true_all),  # (K, N_test, dim_y)
            "y_tilde": torch.stack(y_tilde_all),  # (K, N_test, dim_y)
        },
        artifacts_dir / f"y_dist_diag_seed{seed}.pt",
    )


def _save_posterior(artifacts_dir, method, obs_idx, seed, samples):
    """Save posterior samples to a .pt file."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    torch.save(samples, artifacts_dir / f"{method}_seed{seed}_obs{obs_idx}.pt")


def _save_metrics(artifacts_dir, method, obs_idx, seed, metrics_dict):
    """Save metrics to a JSON file alongside the posterior."""
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / f"{method}_seed{seed}_obs{obs_idx}_metrics.json"
    with open(path, "w") as f:
        json.dump(metrics_dict, f)


def _load_metrics(artifacts_dir, method, obs_idx, seed):
    """Load cached metrics JSON. Returns dict or None if not found."""
    path = artifacts_dir / f"{method}_seed{seed}_obs{obs_idx}_metrics.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_parameter_transform(prior):
    """Return (forward, inverse, unbounded_prior, log_abs_det_jac) for mapping bounded priors to R^n.

    For priors with bounded support (e.g. Independent(Uniform)), returns a logit
    transform to R^n, its sigmoid inverse, and an unbounded prior for the
    transformed space. For unbounded priors, returns identity functions and the
    original prior.
    """
    try:
        low = prior.base_dist.low
        high = prior.base_dist.high
    except AttributeError:
        return lambda theta: theta, lambda z: z, prior, lambda z: torch.zeros(z.shape[0])

    if not (torch.isfinite(low).all() and torch.isfinite(high).all()):
        return lambda theta: theta, lambda z: z, prior, lambda z: torch.zeros(z.shape[0])

    dim = low.shape[0]
    unbounded_prior = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(dim), 100.0 * torch.ones(dim)), 1
    )

    def forward(theta):
        normalized = (theta - low) / (high - low)
        return torch.logit(normalized.clamp(1e-6, 1 - 1e-6))

    def inverse(z):
        return torch.sigmoid(z) * (high - low) + low

    def log_abs_det_jac(z):
        return (torch.log_sigmoid(z) + torch.log_sigmoid(-z) + torch.log(high - low)).sum(dim=-1)

    return forward, inverse, unbounded_prior, log_abs_det_jac


@dataclass
class EvalResult:
    task_name: str
    misspec_type: str
    misspec_kwargs: dict
    num_observation: int
    num_context_size: int
    c2st: float
    mmd: float
    log_prob: float = float("nan")
    method: str = "npepfn_misspec"
    seed: int = 42

    def to_dict(self):
        return asdict(self)


def _npepfn_log_prob(estimator, theta, x, log_abs_det_jac):
    """Compute log_prob for a TabPFN-based estimator, handling device transfer."""
    device = estimator._theta_train.device
    lp = estimator.log_prob(theta.to(device), x=x.to(device))
    return lp.cpu() + log_abs_det_jac(theta)


def _compute_sample_metrics(posterior_samples, ref_samples, metrics_to_compute):
    """Compute sample-based metrics, returning a dict of metric_name -> float."""
    out = {}
    if "c2st" in metrics_to_compute:
        out["c2st"] = float(c2st(posterior_samples, ref_samples))
    if "mmd" in metrics_to_compute:
        out["mmd"] = float(mmd(posterior_samples, ref_samples))
    return out


def _load_cached_results(
    artifacts_dir,
    method,
    seed,
    task,
    num_observations,
    task_name,
    misspec_type,
    misspec_kwargs,
    num_context_size,
    metrics_to_compute=("c2st", "mmd"),
):
    """Load cached metrics (or recompute from posteriors if metrics not cached)."""
    results = []
    for obs_idx in range(1, num_observations + 1):
        # Try loading pre-computed metrics first
        cached_metrics = _load_metrics(artifacts_dir, method, obs_idx, seed)
        if cached_metrics is not None:
            _info(f"Loading cached metrics for {method}, obs = {obs_idx}...")
            result = EvalResult(
                task_name=task_name,
                misspec_type=misspec_type,
                misspec_kwargs=misspec_kwargs,
                num_observation=obs_idx,
                num_context_size=num_context_size,
                c2st=cached_metrics.get("c2st", float("nan")),
                mmd=cached_metrics.get("mmd", float("nan")),
                log_prob=cached_metrics.get("log_prob", float("nan")),
                method=method,
            )
            results.append(result)
            _obs_line(method, obs_idx, {"c2st": result.c2st, "mmd": result.mmd}, "(cached)")
            continue

        # Fall back to recomputing from posterior samples
        path = artifacts_dir / f"{method}_seed{seed}_obs{obs_idx}.pt"
        if not path.exists():
            return None  # incomplete cache
        _info(f"Loading cached posteriors for {method}, obs = {obs_idx} (recomputing metrics)...")
        posterior_samples = torch.load(path, weights_only=True)
        ref_samples = task.get_reference_posterior_samples(obs_idx)
        sample_metrics = _compute_sample_metrics(posterior_samples, ref_samples, metrics_to_compute)
        # Save metrics so next time we skip recomputation
        _save_metrics(artifacts_dir, method, obs_idx, seed, sample_metrics)
        result = EvalResult(
            task_name=task_name,
            misspec_type=misspec_type,
            misspec_kwargs=misspec_kwargs,
            num_observation=obs_idx,
            num_context_size=num_context_size,
            c2st=sample_metrics.get("c2st", float("nan")),
            mmd=sample_metrics.get("mmd", float("nan")),
            method=method,
        )
        results.append(result)
        _obs_line(method, obs_idx, {"c2st": result.c2st, "mmd": result.mmd}, "(cached)")
    return results


def evaluate_calibrated_misspecification(
    task_name,
    misspec_type,
    misspec_kwargs=None,
    num_sim_mixed=1000,
    num_calibration=50,
    num_posterior_samples=1000,
    num_observations=1,
    num_context=2000,
    seed=42,
    use_prior_transform=True,
    artifacts_dir=None,
    skip_methods=None,
    batch_size=None,
    cache_data=False,
    use_cache=True,
    augment_M=1,
    metrics_to_compute=("c2st", "mmd", "log_prob"),
) -> List[EvalResult]:
    """Run calibrated misspecification evaluation comparing multiple methods.

    Context sizing:
        - sim-only (npepfn_misspec): num_context simulation pairs.
        - syn-only (npepfn_y_fmpe, npepfn_y_npepfn, npepfn_ythetaonly_npepfn):
          num_context synthetic pairs.
        - syn+cal (npepfn_y_fmpe_concat, npepfn_y_npepfn_concat): num_context
          total, calibration first, completed with synthetic data.
        - cal-only (npepfn_calib, npe_sbi): num_calibration (swept externally).
        - npepfn_mixed: num_sim_mixed + num_calibration (unchanged).

    Args:
        task_name: sbibm task name (e.g. "two_moons").
        misspec_type: Misspecification type (e.g. "additive_noise").
        misspec_kwargs: Kwargs for the misspecified simulator.
        num_sim_mixed: Number of (theta, x) simulation pairs for npepfn_mixed.
        num_calibration: Number of (theta, y) calibration pairs.
        num_posterior_samples: Samples to draw from each posterior.
        num_observations: Number of sbibm observations to evaluate.
        num_context: Fixed context size for sim-only, syn-only, and syn+cal methods.
        seed: Random seed.
        artifacts_dir: If set, save posterior samples as .pt files.
        skip_methods: List of method names to skip. Cached results are used if
            available in artifacts_dir.
        batch_size: Size of the batch for NPE-PFN `sample_batched` method
        cache_data: If True, cache simulation/calibration data to disk to avoid
            rerunning expensive simulators across sweep runs.
        augment_M: Number of times to call the misspecified simulator per calibration
            theta. M=1 means no augmentation. M>1 replicates theta/y and generates
            M independent x samples per theta.
    Returns:
        List of EvalResult with method field set per method.
    """
    from tabpfn_misspec.baselines import (
        evaluate_npe_sbi,
        evaluate_npepfn_calib_only,
        evaluate_npepfn_y_fmpe,
        evaluate_npepfn_y_npepfn,
    )
    from tabpfn_misspec.calibrated import (
        build_calibrated_estimator,
        build_y_predictor,
        build_y_predictor_theta_only,
        generate_synthetic_y,
        generate_synthetic_y_theta_only,
        sample_calibrated,
    )

    try:
        import npe_pfn

        _info(f"npe_pfn: {npe_pfn.__file__}")
    except ImportError:
        _info("npe_pfn not found.")

    if misspec_kwargs is None:
        misspec_kwargs = {}
    skip_methods = set(skip_methods or [])
    metrics_to_compute = set(metrics_to_compute)
    compute_log_prob = "log_prob" in metrics_to_compute
    if skip_methods:
        _info(f"Skipping: {', '.join(sorted(skip_methods))}")

    ALL_METHODS = [
        "npepfn_misspec",
        "npepfn_calib",
        "npepfn_mixed",
        "npe_sbi",
        "npepfn_y_fmpe",
        "npepfn_y_npepfn",
        "npepfn_y_fmpe_concat",
        "npepfn_y_npepfn_concat",
        "npepfn_ythetaonly_npepfn",
    ]
    active_methods = [m for m in ALL_METHODS if m not in skip_methods]
    total_methods = len(active_methods)
    _step = 0

    def _next_step(method):
        nonlocal _step
        _step += 1
        _header(_step, total_methods, method)

    torch.manual_seed(seed)

    task = sbibm.get_task(task_name)
    prior = task.get_prior_dist()
    _raw_true_simulator = task.get_simulator()
    misspec_simulator = get_misspecified_simulator(task_name, misspec_type, **misspec_kwargs)

    # Wrap true simulator to suppress Julia ODE solver warnings (SIR, Lotka-Volterra)
    _uses_julia = task_name in ("sir", "lotka_volterra")
    if _uses_julia:

        def true_simulator(theta):
            with suppress_julia_output():
                return _raw_true_simulator(theta)
    else:
        true_simulator = _raw_true_simulator

    if use_prior_transform:
        forward, inverse, est_prior, log_abs_det_jac = get_parameter_transform(prior)
    else:
        forward, inverse, est_prior, log_abs_det_jac = (
            lambda t: t,
            lambda z: z,
            prior,
            lambda z: torch.zeros(z.shape[0]),
        )

    # Number of simulation pairs: enough for both npepfn_mixed and npepfn_misspec
    n_sim_generate = max(num_sim_mixed, num_context)

    # --- Shared data generation ---
    if cache_data and artifacts_dir is not None:
        sim_cache = artifacts_dir.parent / f"simdata_seed{seed}.pt"
        calib_cache = artifacts_dir / f"calibdata_seed{seed}.pt"

        # Simulation data (shared across calib sizes)
        if sim_cache.exists():
            cached = torch.load(sim_cache, weights_only=True)
            theta_sim, x_sim = cached["theta_sim"], cached["x_sim"]
            _detail(f"cached sim data: {sim_cache}")
        else:
            theta_sim = prior.sample((n_sim_generate,))
            x_sim = misspec_simulator(theta_sim)
            sim_cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"theta_sim": theta_sim, "x_sim": x_sim}, sim_cache)

        # Calibration data (per calib size)
        if calib_cache.exists():
            cached = torch.load(calib_cache, weights_only=True)
            theta_calib, y_calib, x_calib = (
                cached["theta_calib"],
                cached["y_calib"],
                cached["x_calib"],
            )
            _detail(f"cached calib data: {calib_cache}")
        else:
            theta_calib = prior.sample((num_calibration,))
            y_calib = true_simulator(theta_calib)
            x_calib = misspec_simulator(theta_calib)
            artifacts_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"theta_calib": theta_calib, "y_calib": y_calib, "x_calib": x_calib}, calib_cache
            )
    else:
        theta_sim = prior.sample((n_sim_generate,))
        x_sim = misspec_simulator(theta_sim)
        theta_calib = prior.sample((num_calibration,))
        y_calib = true_simulator(theta_calib)
        x_calib = misspec_simulator(theta_calib)

    # Augment calibration data by calling misspec simulator M times per theta
    if augment_M > 1:
        theta_calib = theta_calib.repeat(augment_M, 1)
        y_calib = y_calib.repeat(augment_M, 1)
        x_calib_parts = [x_calib]
        for _ in range(augment_M - 1):
            x_calib_parts.append(misspec_simulator(theta_calib[:num_calibration]))
        x_calib = torch.cat(x_calib_parts, dim=0)

    # Transform theta to unbounded space for TabPFN-based methods
    theta_sim_t = forward(theta_sim)
    theta_calib_t = forward(theta_calib)

    # --- Shared synthetic y generation (used by all y-corrector methods) ---
    any_y_method = {
        "npepfn_y_fmpe",
        "npepfn_y_npepfn",
        "npepfn_y_fmpe_concat",
        "npepfn_y_npepfn_concat",
    }
    if any_y_method - skip_methods:
        y_cache_path = artifacts_dir / f"synthetic_y_seed{seed}.pt" if artifacts_dir else None
        if use_cache and y_cache_path is not None and y_cache_path.exists():
            cached_y = torch.load(y_cache_path, weights_only=True)
            theta_syn, y_tilde = cached_y["theta_syn"], cached_y["y_tilde"]
            _info(f"Loaded cached synthetic y (theta={theta_syn.shape[0]}, y={y_tilde.shape[0]})")
        else:
            y_predictor = build_y_predictor(theta_calib, x_calib, y_calib)
            theta_syn = prior.sample((num_context,))
            x_syn = misspec_simulator(theta_syn)
            t0 = time.perf_counter()
            y_tilde = generate_synthetic_y(
                y_predictor, theta_syn, x_syn, batched_mode=True, batch_size=batch_size
            )
            t_gen = time.perf_counter() - t0
            _info(f"Generated {num_context} synthetic y samples ({t_gen:.1f}s)")
            if artifacts_dir is not None:
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                torch.save({"theta_syn": theta_syn, "y_tilde": y_tilde}, y_cache_path)
                _save_y_distributional_diag(
                    y_predictor,
                    prior,
                    misspec_simulator,
                    true_simulator,
                    artifacts_dir,
                    seed,
                    batched_mode=True,
                    batch_size=batch_size,
                )
    else:
        theta_syn = y_tilde = None

    # --- Theta-only synthetic y generation ---
    if "npepfn_ythetaonly_npepfn" not in skip_methods:
        y_cache_thetaonly_path = (
            artifacts_dir / f"synthetic_y_thetaonly_seed{seed}.pt" if artifacts_dir else None
        )
        if use_cache and y_cache_thetaonly_path is not None and y_cache_thetaonly_path.exists():
            cached_y = torch.load(y_cache_thetaonly_path, weights_only=True)
            theta_syn_to, y_tilde_to = cached_y["theta_syn"], cached_y["y_tilde"]
            _info(f"Loaded cached theta-only synthetic y (theta={theta_syn_to.shape[0]}, y={y_tilde_to.shape[0]})")
        else:
            y_predictor_to = build_y_predictor_theta_only(theta_calib, y_calib)
            theta_syn_to = prior.sample((num_context,))
            t0 = time.perf_counter()
            y_tilde_to = generate_synthetic_y_theta_only(
                y_predictor_to, theta_syn_to, batched_mode=True, batch_size=batch_size
            )
            t_gen = time.perf_counter() - t0
            _info(f"Generated {num_context} theta-only synthetic y samples ({t_gen:.1f}s)")
            if artifacts_dir is not None:
                artifacts_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"theta_syn": theta_syn_to, "y_tilde": y_tilde_to},
                    y_cache_thetaonly_path,
                )
    else:
        theta_syn_to = y_tilde_to = None

    dim_x = x_sim.shape[1]
    all_results = []

    def _try_cached_or_skip(method):
        """Try loading cached results; skip or fall through depending on mode.

        - If use_cache is True and cache exists, return cached results.
        - If the method is in skip_methods and no cache, return [] (skip).
        - Otherwise return None (run normally).
        """
        if (use_cache or method in skip_methods) and artifacts_dir is not None:
            cached = _load_cached_results(
                artifacts_dir,
                method,
                seed,
                task,
                num_observations,
                task_name,
                misspec_type,
                misspec_kwargs,
                num_sim_mixed,
                metrics_to_compute=metrics_to_compute,
            )
            if cached is not None:
                return cached
        if method in skip_methods:
            _info(f"Skipping {method} (no cached results)")
            return []
        return None  # not cached, run normally

    def _nan_results(method, reason):
        """Generate NaN results for a failed method."""
        warnings.warn(f"[{method}] failed: {reason}")
        return [
            EvalResult(
                task_name=task_name,
                misspec_type=misspec_type,
                misspec_kwargs=misspec_kwargs,
                num_observation=obs_idx,
                num_context_size=num_sim_mixed,
                c2st=float("nan"),
                mmd=float("nan"),
                method=method,
            )
            for obs_idx in range(1, num_observations + 1)
        ]

    # --- Method 1: NPE-PFN (misspecified) --- uses num_context sim pairs
    cached = _try_cached_or_skip("npepfn_misspec")
    if cached is not None:
        all_results.extend(cached)
    else:
        _next_step("npepfn_misspec")
        estimator_misspec = TabPFN_Based_NPE_PFN(prior=est_prior)
        estimator_misspec.append_simulations(theta_sim_t[:num_context], x_sim[:num_context])
        for obs_idx in range(1, num_observations + 1):
            y_obs = task.get_observation(obs_idx)
            ref_samples = task.get_reference_posterior_samples(obs_idx)
            t0 = time.perf_counter()
            posterior_samples = inverse(
                estimator_misspec.sample(
                    sample_shape=torch.Size([num_posterior_samples]),
                    x=y_obs,
                )
            )
            t_inference = time.perf_counter() - t0
            if artifacts_dir is not None:
                _save_posterior(
                    artifacts_dir,
                    "npepfn_misspec",
                    obs_idx,
                    seed,
                    posterior_samples,
                )
                _save_posterior(artifacts_dir, "reference", obs_idx, seed, ref_samples)
            sample_metrics = _compute_sample_metrics(
                posterior_samples, ref_samples, metrics_to_compute
            )
            log_prob_val = float("nan")
            if compute_log_prob:
                try:
                    ref_t = forward(ref_samples)
                    lp = estimator_misspec.log_prob(ref_t, x=y_obs) + log_abs_det_jac(ref_t)
                    log_prob_val = float(lp.mean())
                except Exception:
                    pass
            sample_metrics["log_prob"] = log_prob_val
            if artifacts_dir is not None:
                _save_metrics(artifacts_dir, "npepfn_misspec", obs_idx, seed, sample_metrics)
            result = EvalResult(
                task_name=task_name,
                misspec_type=misspec_type,
                misspec_kwargs=misspec_kwargs,
                num_observation=obs_idx,
                num_context_size=num_context,
                c2st=sample_metrics.get("c2st", float("nan")),
                mmd=sample_metrics.get("mmd", float("nan")),
                log_prob=log_prob_val,
                method="npepfn_misspec",
            )
            all_results.append(result)
            _obs_line(
                "npepfn_misspec",
                obs_idx,
                {"c2st": result.c2st, "mmd": result.mmd, "log_prob": result.log_prob},
                f"({t_inference:.1f}s)",
            )

    # --- Method 2: NPE-PFN (calibration only) ---
    cached = _try_cached_or_skip("npepfn_calib")
    if cached is not None:
        all_results.extend(cached)
    else:
        _next_step("npepfn_calib")
        try:
            all_results.extend(
                evaluate_npepfn_calib_only(
                    task,
                    theta_calib_t,
                    y_calib,
                    inverse,
                    est_prior,
                    num_posterior_samples,
                    num_observations,
                    task_name,
                    misspec_type,
                    misspec_kwargs,
                    seed=seed,
                    artifacts_dir=artifacts_dir,
                    forward=forward,
                    log_abs_det_jac=log_abs_det_jac,
                    metrics_to_compute=metrics_to_compute,
                )
            )
        except (ValueError, RuntimeError) as e:
            all_results.extend(_nan_results("npepfn_calib", e))

    # --- Method 3: NPE-PFN (mixed, ours) ---
    cached = _try_cached_or_skip("npepfn_mixed")
    if cached is not None:
        all_results.extend(cached)
    else:
        _next_step("npepfn_mixed")
        try:
            estimator_mixed = build_calibrated_estimator(
                theta_sim_t,
                x_sim,
                theta_calib_t,
                x_calib,
                y_calib,
                est_prior,
            )
            for obs_idx in range(1, num_observations + 1):
                y_obs = task.get_observation(obs_idx)
                ref_samples = task.get_reference_posterior_samples(obs_idx)
                t0 = time.perf_counter()
                posterior_samples = inverse(
                    sample_calibrated(
                        estimator_mixed,
                        y_obs,
                        dim_x,
                        num_posterior_samples,
                    )
                )
                t_inference = time.perf_counter() - t0
                if artifacts_dir is not None:
                    _save_posterior(
                        artifacts_dir,
                        "npepfn_mixed",
                        obs_idx,
                        seed,
                        posterior_samples,
                    )
                sample_metrics = _compute_sample_metrics(
                    posterior_samples, ref_samples, metrics_to_compute
                )
                log_prob_val = float("nan")
                if compute_log_prob:
                    try:
                        ref_t = forward(ref_samples)
                        query = torch.cat([torch.full((1, dim_x), float("nan")), y_obs], dim=1)
                        lp = estimator_mixed.log_prob(ref_t, x=query) + log_abs_det_jac(ref_t)
                        log_prob_val = float(lp.mean())
                    except Exception:
                        pass
                sample_metrics["log_prob"] = log_prob_val
                if artifacts_dir is not None:
                    _save_metrics(artifacts_dir, "npepfn_mixed", obs_idx, seed, sample_metrics)
                result = EvalResult(
                    task_name=task_name,
                    misspec_type=misspec_type,
                    misspec_kwargs=misspec_kwargs,
                    num_observation=obs_idx,
                    num_context_size=num_sim_mixed,
                    c2st=sample_metrics.get("c2st", float("nan")),
                    mmd=sample_metrics.get("mmd", float("nan")),
                    log_prob=log_prob_val,
                    method="npepfn_mixed",
                )
                all_results.append(result)
                _obs_line(
                    "npepfn_mixed",
                    obs_idx,
                    {"c2st": result.c2st, "mmd": result.mmd, "log_prob": result.log_prob},
                    f"({t_inference:.1f}s)",
                )
        except (ValueError, RuntimeError) as e:
            all_results.extend(_nan_results("npepfn_mixed", e))

    # --- Method 4: NPE (sbi) ---
    cached = _try_cached_or_skip("npe_sbi")
    if cached is not None:
        all_results.extend(cached)
    else:
        _next_step("npe_sbi")
        try:
            all_results.extend(
                evaluate_npe_sbi(
                    task,
                    prior,
                    theta_calib,
                    y_calib,
                    num_posterior_samples,
                    num_observations,
                    task_name,
                    misspec_type,
                    misspec_kwargs,
                    seed=seed,
                    artifacts_dir=artifacts_dir,
                    metrics_to_compute=metrics_to_compute,
                )
            )
        except (ValueError, RuntimeError) as e:
            all_results.extend(_nan_results("npe_sbi", e))

    # --- Method 5: TabPFN y-corrector + FMPE --- syn only, num_context pairs
    cached = _try_cached_or_skip("npepfn_y_fmpe")
    if cached is not None:
        all_results.extend(cached)
    else:
        _next_step("npepfn_y_fmpe")
        try:
            all_results.extend(
                evaluate_npepfn_y_fmpe(
                    task,
                    prior,
                    theta_syn[:num_context],
                    y_tilde[:num_context],
                    theta_calib,
                    y_calib,
                    num_posterior_samples,
                    num_observations,
                    task_name,
                    misspec_type,
                    misspec_kwargs,
                    seed=seed,
                    artifacts_dir=artifacts_dir,
                    metrics_to_compute=metrics_to_compute,
                )
            )
        except (ValueError, RuntimeError) as e:
            all_results.extend(_nan_results("npepfn_y_fmpe", e))

    # --- Method 6: NPE-PFN on synthetic y --- syn only, num_context pairs
    cached = _try_cached_or_skip("npepfn_y_npepfn")
    if cached is not None:
        all_results.extend(cached)
    else:
        _next_step("npepfn_y_npepfn")
        try:
            all_results.extend(
                evaluate_npepfn_y_npepfn(
                    task,
                    theta_syn[:num_context],
                    y_tilde[:num_context],
                    theta_calib_t,
                    y_calib,
                    inverse,
                    est_prior,
                    forward,
                    num_posterior_samples,
                    num_observations,
                    task_name,
                    misspec_type,
                    misspec_kwargs,
                    seed=seed,
                    artifacts_dir=artifacts_dir,
                    log_abs_det_jac=log_abs_det_jac,
                    metrics_to_compute=metrics_to_compute,
                )
            )
        except (ValueError, RuntimeError) as e:
            all_results.extend(_nan_results("npepfn_y_npepfn", e))

    # --- Method 7: FMPE on synthetic y + calibration concat --- total = num_context
    n_syn_concat = max(0, num_context - len(theta_calib))
    cached = _try_cached_or_skip("npepfn_y_fmpe_concat")
    if cached is not None:
        all_results.extend(cached)
    else:
        _next_step("npepfn_y_fmpe_concat")
        try:
            all_results.extend(
                evaluate_npepfn_y_fmpe(
                    task,
                    prior,
                    theta_syn[:n_syn_concat],
                    y_tilde[:n_syn_concat],
                    theta_calib,
                    y_calib,
                    num_posterior_samples,
                    num_observations,
                    task_name,
                    misspec_type,
                    misspec_kwargs,
                    seed=seed,
                    artifacts_dir=artifacts_dir,
                    concat_calib=True,
                    metrics_to_compute=metrics_to_compute,
                )
            )
        except (ValueError, RuntimeError) as e:
            all_results.extend(_nan_results("npepfn_y_fmpe_concat", e))

    # --- Method 8: NPE-PFN on synthetic y + calibration concat --- total = num_context
    cached = _try_cached_or_skip("npepfn_y_npepfn_concat")
    if cached is not None:
        all_results.extend(cached)
    else:
        _next_step("npepfn_y_npepfn_concat")
        try:
            all_results.extend(
                evaluate_npepfn_y_npepfn(
                    task,
                    theta_syn[:n_syn_concat],
                    y_tilde[:n_syn_concat],
                    theta_calib_t,
                    y_calib,
                    inverse,
                    est_prior,
                    forward,
                    num_posterior_samples,
                    num_observations,
                    task_name,
                    misspec_type,
                    misspec_kwargs,
                    seed=seed,
                    artifacts_dir=artifacts_dir,
                    concat_calib=True,
                    log_abs_det_jac=log_abs_det_jac,
                    metrics_to_compute=metrics_to_compute,
                )
            )
        except (ValueError, RuntimeError) as e:
            all_results.extend(_nan_results("npepfn_y_npepfn_concat", e))

    # --- Method 9: NPE-PFN on theta-only synthetic y --- syn only, num_context
    cached = _try_cached_or_skip("npepfn_ythetaonly_npepfn")
    if cached is not None:
        all_results.extend(cached)
    else:
        _next_step("npepfn_ythetaonly_npepfn")
        try:
            all_results.extend(
                evaluate_npepfn_y_npepfn(
                    task,
                    theta_syn_to[:num_context],
                    y_tilde_to[:num_context],
                    theta_calib_t,
                    y_calib,
                    inverse,
                    est_prior,
                    forward,
                    num_posterior_samples,
                    num_observations,
                    task_name,
                    misspec_type,
                    misspec_kwargs,
                    seed=seed,
                    artifacts_dir=artifacts_dir,
                    log_abs_det_jac=log_abs_det_jac,
                    metrics_to_compute=metrics_to_compute,
                    method_name="npepfn_ythetaonly_npepfn",
                )
            )
        except (ValueError, RuntimeError) as e:
            all_results.extend(_nan_results("npepfn_ythetaonly_npepfn", e))

    return all_results
