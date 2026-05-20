"""Slice-MCMC + NSF + rejection-sampling reference-posterior pipeline.

Mirrors sbibm Appendix B.1 (Slice-MCMC -> NSF density estimator -> rejection
sampling) but uses sbi's vectorized slice sampler from
``sbi.samplers.mcmc.slice_numpy.SliceSamplerVectorized``. The original sbibm
``Slice`` kernel lived under ``sbi.mcmc.slice``, which was removed in sbi>=0.20.
"""

import time
from typing import Any, Optional

import numpy as np
import torch
from sbi.samplers.mcmc.slice_numpy import SliceSamplerVectorized
from sbibm.algorithms.pytorch.baseline_rejection import run as run_rejection
from sbibm.algorithms.pytorch.utils.proposal import get_proposal
from sbibm.tasks.task import Task

# sbibm.utils.torch.get_log_abs_det_jacobian has a typo: `vals.numel == batch_size`
# compares a method object to an int -> always False -> assertion always fires.
# Blocks every code path through get_proposal(bounded=True). Hot-patch rather
# than fork. Patch the consuming module's binding because nflows.py does a
# `from sbibm.utils.torch import get_log_abs_det_jacobian` (already-imported
# reference is not affected by patching the source module).
def _patched_get_log_abs_det_jacobian(
    transform, parameters_constrained, parameters_unconstrained,
):
    batch_size, dim_parameters = parameters_constrained.shape
    vals = transform.log_abs_det_jacobian(
        parameters_constrained, parameters_unconstrained,
    )
    if vals.ndim > 1 and vals.shape[1] == dim_parameters:
        vals = vals.sum(-1)
    assert vals.numel() == batch_size, (
        f"log_abs_det_jacobian shape mismatch: {vals.shape} vs batch {batch_size}"
    )
    return vals


import sbibm.utils.nflows as _sbibm_nflows  # noqa: E402
import sbibm.utils.torch as _sbibm_torch_utils  # noqa: E402

_sbibm_nflows.get_log_abs_det_jacobian = _patched_get_log_abs_det_jacobian
_sbibm_torch_utils.get_log_abs_det_jacobian = _patched_get_log_abs_det_jacobian


def run_slice_nsf_rejection_pipeline(
    task: Task,
    num_samples: int,
    num_observation: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    num_warmup: int = 2_000,
    num_chains: int = 1,
    slice_samples_per_chain: Optional[int] = None,
    tuning: int = 100,
    init_width: float = 0.01,
    prior_weight: float = 0.1,
    flow_model: str = "nsf",
    batch_size: int = 10_000,
    num_batches_without_new_max: int = 1_000,
    multiplier_M: float = 1.2,
    **_: Any,
) -> torch.Tensor:
    """Draw reference-posterior samples for ``task``.

    Stage 1: Slice-MCMC on the unnormalized posterior log-density returned by
    ``task._get_log_prob_fn(implementation="experimental")``.
    Stage 2: fit an NSF defensive proposal on the slice samples.
    Stage 3: rejection-sample from that proposal against the same log-density.
    """
    assert (num_observation is None) ^ (observation is None)

    # Slice budget per chain: keep total slice samples (= num_chains *
    # slice_samples_per_chain) roughly fixed when bumping num_chains, otherwise
    # K chains do K times more slice work for no gain.
    if slice_samples_per_chain is None:
        slice_samples_per_chain = num_samples

    print(
        f"[gt-pipeline] task={task.name} num_obs={num_observation} "
        f"num_samples={num_samples} num_warmup={num_warmup} "
        f"num_chains={num_chains} "
        f"slice_samples_per_chain={slice_samples_per_chain} "
        f"batch_size={batch_size}",
        flush=True,
    )

    torch_log_prob_fn = task._get_log_prob_fn(
        num_observation=num_observation,
        observation=observation,
        implementation="experimental",
        posterior=True,
    )

    n_calls = [0]
    n_failures = [0]
    heartbeat_every = 1000
    next_heartbeat = [heartbeat_every]

    def np_log_prob_fn(params_np: np.ndarray) -> np.ndarray:
        arr = np.atleast_2d(params_np)
        n_calls[0] += arr.shape[0]
        params = torch.as_tensor(arr, dtype=torch.float32)
        with torch.no_grad():
            log_p = torch_log_prob_fn(params)
        out = log_p.detach().cpu().numpy().astype(np.float64).reshape(-1)
        bad = ~np.isfinite(out)
        n_failures[0] += int(bad.sum())
        out[bad] = -np.inf
        if n_calls[0] >= next_heartbeat[0]:
            elapsed = time.time() - t0
            ms_per = 1000.0 * elapsed / n_calls[0]
            fail_pct = 100.0 * n_failures[0] / n_calls[0]
            print(
                f"[gt-pipeline] log_prob heartbeat: calls={n_calls[0]} "
                f"elapsed={elapsed:.1f}s ms/call={ms_per:.1f} "
                f"failures={n_failures[0]} ({fail_pct:.1f}%)",
                flush=True,
            )
            next_heartbeat[0] += heartbeat_every
        return out

    if num_observation is not None:
        init_theta = task.get_true_parameters(num_observation=num_observation)
        init_theta = init_theta.reshape(1, -1).repeat(num_chains, 1)
    else:
        init_theta = task.get_prior()(num_samples=num_chains)
    init_np = init_theta.detach().cpu().numpy().astype(np.float64).reshape(
        num_chains, task.dim_parameters,
    )

    print("[gt-pipeline] stage 1: slice MCMC starting...", flush=True)
    t0 = time.time()
    sampler = SliceSamplerVectorized(
        log_prob_fn=np_log_prob_fn,
        init_params=init_np,
        num_chains=num_chains,
        thin=1,
        tuning=tuning,
        verbose=False,
        init_width=init_width,
        num_workers=1,
    )
    chain = sampler.run(num_warmup + slice_samples_per_chain)  # (num_chains, n, dim)
    chain = chain[:, num_warmup:, :]
    t1 = time.time()
    print(
        f"[gt-pipeline] stage 1 done in {t1 - t0:.1f}s | "
        f"log_prob calls={n_calls[0]} | -inf returns={n_failures[0]} "
        f"({100 * n_failures[0] / max(n_calls[0], 1):.1f}%)",
        flush=True,
    )

    proposal_samples = torch.as_tensor(
        chain.reshape(-1, task.dim_parameters), dtype=torch.float32,
    )

    print(
        f"[gt-pipeline] stage 2: NSF fit on {proposal_samples.shape[0]} "
        "samples starting...",
        flush=True,
    )
    proposal_dist = get_proposal(
        task=task,
        samples=proposal_samples,
        prior_weight=prior_weight,
        bounded=True,
        density_estimator="flow",
        flow_model=flow_model,
    )
    t2 = time.time()
    print(f"[gt-pipeline] stage 2 done in {t2 - t1:.1f}s", flush=True)

    print(
        f"[gt-pipeline] stage 3: rejection sampling starting "
        f"(batch_size={batch_size}, "
        f"num_batches_without_new_max={num_batches_without_new_max})...",
        flush=True,
    )
    samples = run_rejection(
        task=task,
        num_observation=num_observation,
        observation=observation,
        num_samples=num_samples,
        batch_size=batch_size,
        num_batches_without_new_max=num_batches_without_new_max,
        multiplier_M=multiplier_M,
        proposal_dist=proposal_dist,
    )
    t3 = time.time()
    print(
        f"[gt-pipeline] stage 3 done in {t3 - t2:.1f}s | "
        f"total {t3 - t0:.1f}s | output shape={tuple(samples.shape)}",
        flush=True,
    )
    return samples
