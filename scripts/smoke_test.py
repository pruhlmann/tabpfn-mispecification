"""Minimal end-to-end smoke test.

Runs `evaluate_calibrated_misspecification` once on `gaussian_linear` with tiny
sample counts and only a few methods enabled, to verify the pipeline imports,
trains, samples, and scores without crashing. Intended for local CPU runs.
"""

from tabpfn_misspec import evaluate_calibrated_misspecification


SKIP_METHODS = [
    "npepfn_mixed",
    "npepfn_y_fmpe",
    "npepfn_y_npepfn",
    "npepfn_y_fmpe_concat",
    "npepfn_y_npepfn_concat",
    "npepfn_ythetaonly_npepfn",
]


def main():
    results = evaluate_calibrated_misspecification(
        task_name="gaussian_linear",
        misspec_type="additive_noise",
        misspec_kwargs={},
        num_sim_mixed=30,
        num_calibration=10,
        num_posterior_samples=5,
        num_observations=1,
        num_context=100,
        seed=0,
        use_prior_transform=True,
        skip_methods=SKIP_METHODS,
        use_cache=False,
        metrics_to_compute=("c2st",),
    )
    print(f"\nsmoke test ok — {len(results)} result(s)")
    for r in results:
        print(r.to_dict())


if __name__ == "__main__":
    main()
