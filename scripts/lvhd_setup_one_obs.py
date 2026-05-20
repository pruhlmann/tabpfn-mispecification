"""Generate observation + reference posterior for ONE observation of
lotka_volterra_hd. Mirrors the per-obs ``run()`` inside ``task._setup``
(sbibm/tasks/task.py:442) so we can run observations as a SLURM array.

Each array task is independent — no shared Julia state, no joblib worker pool.
"""

import argparse
import time

import numpy as np
import torch

from tabpfn_misspec._gt_pipeline import run_slice_nsf_rejection_pipeline
from tabpfn_misspec.tasks import get_task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-observation", type=int, required=True,
        help="1-indexed observation number",
    )
    parser.add_argument("--num-chains", type=int, default=4)
    parser.add_argument(
        "--num-warmup", type=int, default=500,
        help="Slice burn-in per chain (default 500; slice samplers adapt fast).",
    )
    args = parser.parse_args()

    task = get_task("lotka_volterra_hd")
    num_obs = args.num_observation
    observation_seed = task.observation_seeds[num_obs - 1]

    np.random.seed(observation_seed)
    torch.manual_seed(observation_seed)
    task._save_observation_seed(num_obs, observation_seed)

    true_parameters = task.get_prior()(num_samples=1)
    task._save_true_parameters(num_obs, true_parameters)

    simulator = task.get_simulator()
    observation = simulator(true_parameters)
    task._save_observation(num_obs, observation)

    print(
        f"[lvhd-setup] num_observation={num_obs} seed={observation_seed} "
        f"true_parameters={tuple(true_parameters.shape)} "
        f"observation={tuple(observation.shape)}",
        flush=True,
    )

    # Keep total slice samples ~= num_reference_posterior_samples so K chains
    # don't multiply slice work; rejection still draws the full target.
    total_slice = task.num_reference_posterior_samples
    slice_per_chain = max(total_slice // args.num_chains, 500)

    t0 = time.time()
    samples = run_slice_nsf_rejection_pipeline(
        task=task,
        num_samples=task.num_reference_posterior_samples,
        num_observation=num_obs,
        num_warmup=args.num_warmup,
        num_chains=args.num_chains,
        slice_samples_per_chain=slice_per_chain,
    )
    print(
        f"[lvhd-setup] pipeline done in {time.time() - t0:.1f}s, "
        f"samples={tuple(samples.shape)}",
        flush=True,
    )

    num_unique = torch.unique(samples, dim=0).shape[0]
    assert num_unique == task.num_reference_posterior_samples, (
        f"got {num_unique} unique samples, expected "
        f"{task.num_reference_posterior_samples}"
    )
    task._save_reference_posterior_samples(num_obs, samples)


if __name__ == "__main__":
    main()
