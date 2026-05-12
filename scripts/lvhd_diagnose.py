"""LV-HD ground-truth pipeline diagnostic.

Two parts:
  1. Micro-benchmark of raw Julia ODE solves (task.de(...)) on the true
     parameters of observation 1. Isolates per-solve cost from pyro/sbibm
     overhead. First call is warm-up (JIT) and reported separately.
  2. Tiny end-to-end run of Slice + NSF + rejection on observation 1.

Does NOT write any task files (calls the helper directly instead of
task._setup()).
"""

import time

from tabpfn_misspec._gt_pipeline import run_slice_nsf_rejection_pipeline
from tabpfn_misspec.lotka_volterra_hd import _theta_to_p
from tabpfn_misspec.tasks import get_task


def benchmark_raw_de(task, n_warmup: int = 2, n_iter: int = 100) -> None:
    """Time raw task.de(u0, tspan, p) calls on the true theta of obs 1."""
    theta = task.get_true_parameters(num_observation=1)  # (1, 25)
    p = _theta_to_p(theta)[0]  # (30,)

    print(
        f"[diagnose] raw-ODE benchmark: n_warmup={n_warmup} n_iter={n_iter}",
        flush=True,
    )
    for k in range(n_warmup):
        t0 = time.time()
        task.de(task.u0, task.tspan, p)
        print(
            f"[diagnose] warmup call {k}: {1000 * (time.time() - t0):.1f} ms",
            flush=True,
        )

    t0 = time.time()
    for _ in range(n_iter):
        task.de(task.u0, task.tspan, p)
    elapsed = time.time() - t0
    print(
        f"[diagnose] raw-ODE: {n_iter} calls in {elapsed:.2f}s "
        f"-> {1000 * elapsed / n_iter:.2f} ms/call",
        flush=True,
    )


def main() -> None:
    task = get_task("lotka_volterra_hd")

    benchmark_raw_de(task, n_warmup=2, n_iter=100)

    t0 = time.time()
    samples = run_slice_nsf_rejection_pipeline(
        task=task,
        num_samples=10,
        num_observation=1,
        num_warmup=10,
        num_chains=1,
        tuning=5,
        batch_size=200,
        num_batches_without_new_max=5,
    )
    t1 = time.time()
    print(
        f"[diagnose] full pipeline: {t1 - t0:.1f}s, "
        f"samples shape={tuple(samples.shape)}, "
        f"finite={samples.isfinite().all().item()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
