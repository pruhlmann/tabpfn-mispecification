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
    training_batch_size=1024,
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
    effective_bs = min(training_batch_size, len(theta_calib))
    density_estimator = inference.train(training_batch_size=effective_bs)
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


def evaluate_mf_npe(
    task,
    prior,
    theta_sim,
    x_sim,
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
    pretrain_lr=5e-4,
    finetune_lr=1e-4,
    training_batch_size=1024,
    num_finetune_epochs=200,
    finetune_batch_size=64,
    finetune_val_fraction=0.2,
    finetune_patience=20,
):
    """MF-NPE: NPE pretrained on misspecified sims, fine-tuned on calibration.

    Recipe (Krouglova et al. 2025): pretrain a flow on (theta_sim, x_sim) from
    the misspecified simulator, then fine-tune the same network on calibration
    pairs only. The fine-tune is a manual torch loop because sbi 0.26's
    `train(resume_training=True)` (1) trains on the union of accumulated rounds
    rather than calib-only and (2) reverts to the pretrain "best validation"
    weights at the end, silently making fine-tune a no-op.

    Fine-tune uses 20% of calib as validation (if n_calib >= 5) with
    `finetune_patience`-epoch early stopping; best-val weights are restored.
    For n_calib < 5 we fall back to fixed-epoch training without validation.
    """
    from copy import deepcopy
    from sbi.inference import NPE

    compute_log_prob = "log_prob" in metrics_to_compute
    inference = NPE(prior=prior)
    inference.append_simulations(theta_sim.float(), x_sim.float())
    t0 = time.perf_counter()
    inference.train(
        learning_rate=pretrain_lr,
        training_batch_size=training_batch_size,
    )
    t_pretrain = time.perf_counter() - t0
    _info(f"MF-NPE pretrain completed ({t_pretrain:.1f}s, n_sim={len(theta_sim)})")

    density_estimator = inference._neural_net
    device = next(density_estimator.parameters()).device
    theta_c = theta_calib.float().to(device)
    y_c = y_calib.float().to(device)
    n_calib = theta_c.shape[0]
    optimizer = torch.optim.Adam(density_estimator.parameters(), lr=finetune_lr)

    g = torch.Generator(device="cpu").manual_seed(seed)
    split_perm = torch.randperm(n_calib, generator=g)
    n_val = int(round(finetune_val_fraction * n_calib))
    use_val = n_val >= 1 and (n_calib - n_val) >= 1 and n_calib >= 5
    if use_val:
        val_idx = split_perm[:n_val]
        train_idx = split_perm[n_val:]
    else:
        val_idx = None
        train_idx = split_perm
    n_train = len(train_idx)
    bs = min(finetune_batch_size, n_train)

    t0 = time.perf_counter()
    best_val = float("inf")
    best_state = deepcopy(density_estimator.state_dict())
    epochs_since_improve = 0
    last_epoch = 0
    for epoch in range(num_finetune_epochs):
        last_epoch = epoch + 1
        density_estimator.train()
        perm = train_idx[torch.randperm(n_train, generator=g)]
        for i in range(0, n_train, bs):
            idx = perm[i : i + bs]
            optimizer.zero_grad()
            loss = density_estimator.loss(theta_c[idx], condition=y_c[idx]).mean()
            loss.backward()
            optimizer.step()
        if use_val:
            density_estimator.eval()
            with torch.no_grad():
                val_loss = density_estimator.loss(
                    theta_c[val_idx], condition=y_c[val_idx]
                ).mean().item()
            if val_loss < best_val:
                best_val = val_loss
                best_state = deepcopy(density_estimator.state_dict())
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= finetune_patience:
                    break
    if use_val:
        density_estimator.load_state_dict(best_state)
    density_estimator.eval()
    t_finetune = time.perf_counter() - t0
    posterior = inference.build_posterior(density_estimator)
    _info(
        f"MF-NPE fine-tune completed ({t_finetune:.1f}s, "
        f"n_calib={n_calib}, n_train={n_train}, n_val={n_val if use_val else 0}, "
        f"epochs={last_epoch}/{num_finetune_epochs}, bs={bs})"
    )

    results = []
    for obs_idx in range(1, num_observations + 1):
        y_obs = task.get_observation(obs_idx)
        ref_samples = task.get_reference_posterior_samples(obs_idx)

        t0 = time.perf_counter()
        posterior_samples = posterior.sample((num_posterior_samples,), x=y_obs)
        t_inference = time.perf_counter() - t0

        if artifacts_dir is not None:
            _save_posterior(artifacts_dir, "mf_npe", obs_idx, seed, posterior_samples)

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
            _save_metrics(artifacts_dir, "mf_npe", obs_idx, seed, sample_metrics)
        result = EvalResult(
            task_name=task_name,
            misspec_type=misspec_type,
            misspec_kwargs=misspec_kwargs,
            num_observation=obs_idx,
            num_context_size=len(theta_calib),
            c2st=sample_metrics.get("c2st", float("nan")),
            mmd=sample_metrics.get("mmd", float("nan")),
            log_prob=log_prob_val,
            method="mf_npe",
        )
        results.append(result)
        _obs_line(
            "mf_npe",
            obs_idx,
            {"c2st": result.c2st, "mmd": result.mmd, "log_prob": result.log_prob},
            f"({t_inference:.1f}s)",
        )

    return results


def evaluate_fmcpe(
    task,
    prior,
    theta_sim,
    x_sim,
    theta_calib,
    x_calib,
    y_calib,
    num_posterior_samples,
    num_observations,
    task_name,
    misspec_type,
    misspec_kwargs,
    seed=42,
    artifacts_dir=None,
    metrics_to_compute=("c2st", "mmd", "log_prob"),
    hidden=(64, 64),
    num_steps=50,
    npe_epochs=1000,
    dual_epochs=1000,
    lr=1e-4,
    batch_size=128,
    patience=20,
):
    """FMCPE: flow-matching corrected posterior (port of ropefm `fm_post_transform`).

    Three-stage pipeline:
      1. NPE proposal — conditional flow matching on (theta_sim, x_sim).
      2. flow_x — CFM on (x_calib, y_calib) learning p(x | y).
      3. flow_theta — CFM on (theta_calib, y_calib) with the NPE proposal as
         source distribution; refines proposal samples conditional on y.

    `prior` is accepted for API parity but is not used (FMCPE is a continuous
    density model and does not constrain samples to prior support).

    `log_prob` is not implemented in this v1 port (the dual-flow log_prob in
    ropefm uses a reverse-ODE Hutchinson trace estimator). Returns NaN.
    """
    from tabpfn_misspec.fmcpe import sample_fmcpe, train_fmcpe

    del prior  # unused; FMCPE operates entirely on tensors

    device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.perf_counter()
    models = train_fmcpe(
        theta_sim,
        x_sim,
        theta_calib,
        x_calib,
        y_calib,
        device=device,
        hidden=hidden,
        num_steps=num_steps,
        npe_epochs=npe_epochs,
        dual_epochs=dual_epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        seed=seed,
    )
    t_train = time.perf_counter() - t0
    _info(
        f"FMCPE training completed ({t_train:.1f}s, n_sim={len(theta_sim)}, "
        f"n_calib={len(theta_calib)})"
    )

    results = []
    for obs_idx in range(1, num_observations + 1):
        y_obs = task.get_observation(obs_idx)
        ref_samples = task.get_reference_posterior_samples(obs_idx)

        t0 = time.perf_counter()
        posterior_samples = sample_fmcpe(models, y_obs, num_posterior_samples, device)
        t_inference = time.perf_counter() - t0

        if artifacts_dir is not None:
            _save_posterior(artifacts_dir, "fmcpe", obs_idx, seed, posterior_samples)

        sample_metrics = _compute_sample_metrics(
            posterior_samples, ref_samples, metrics_to_compute
        )
        log_prob_val = float("nan")
        sample_metrics["log_prob"] = log_prob_val
        if artifacts_dir is not None:
            _save_metrics(artifacts_dir, "fmcpe", obs_idx, seed, sample_metrics)
        result = EvalResult(
            task_name=task_name,
            misspec_type=misspec_type,
            misspec_kwargs=misspec_kwargs,
            num_observation=obs_idx,
            num_context_size=len(theta_calib),
            c2st=sample_metrics.get("c2st", float("nan")),
            mmd=sample_metrics.get("mmd", float("nan")),
            log_prob=log_prob_val,
            method="fmcpe",
        )
        results.append(result)
        _obs_line(
            "fmcpe",
            obs_idx,
            {"c2st": result.c2st, "mmd": result.mmd, "log_prob": result.log_prob},
            f"({t_inference:.1f}s)",
        )

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
    training_batch_size=1024,
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
    density_estimator = fmpe.train(training_batch_size=training_batch_size)
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
