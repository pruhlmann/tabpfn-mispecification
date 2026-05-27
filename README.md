# tabpfn-misspecification

Evaluate NPE-PFN (TabPFN-based Neural Posterior Estimation) under simulator
misspecification on the [sbibm](https://github.com/sbi-benchmark/sbibm) tasks
and on a few custom high-dimensional tasks. Compares calibration-set–based
correction strategies (synthetic-y / mixed-context / calibration-only) against
NPE and FMPE baselines.

## What's here

```
tabpfn_misspec/      # Library: tasks, misspecified simulators, evaluation, plotting
configs/             # ml_collections configs, one per task (+ debug.py and experiment.py)
scripts/             # CLI entry points (sweep, plot, smoke test, LV-HD setup)
slurm/               # SLURM batch scripts for Jean Zay (sweeps + reference-posterior setup)
tests/               # pytest unit tests
npe_pfn/             # NPE-PFN implementation used as a dependency
```

## Install

The repo uses [pixi](https://pixi.sh). Two environments:

- `default` — CPU torch (small smoke runs, tests, reference-posterior setup).
- `gpu` — CUDA 12 torch (sweeps that train density estimators on GPU).

```bash
pixi install              # default (CPU)
pixi install -e gpu       # GPU env
```

## Smoke test (local)

Runs `evaluate_calibrated_misspecification` end-to-end on `gaussian_linear`
with tiny sample counts; takes a couple of minutes on CPU:

```bash
pixi run smoke
```

## Run a sweep

A sweep evaluates every method on one task / misspecification across several
calibration-set sizes and seeds, and writes results to
`results/<task>_<misspec>_sweep.json` plus a per-run artifacts directory:

```bash
pixi run -e gpu sweep --config configs/gaussian_linear.py
pixi run -e gpu sweep --config configs/gaussian_linear.py --calib_sizes 10,50,200,1000
pixi run -e gpu sweep --config configs/gaussian_linear.py --output_dir results-custom
```

The `--calib_sizes` flag overrides the default `[10, 50, 200, 1000]` sweep.
Seeds and all other hyperparameters live in the config file.

## Plot

After one or more sweeps have finished, regenerate every plot from the saved
JSON + per-run artifacts (posterior pairplots, y-diagnostics, per-metric
calibration curves):

```bash
pixi run plot                                     # input/output = results/
pixi run plot -- --input_dir results-custom       # custom dir
```

## Available tasks and misspecifications

Configs and registered misspecifications:

| Config | Task | Misspecification |
| --- | --- | --- |
| `gaussian_linear.py` | `gaussian_linear` (sbibm) | `additive_noise` |
| `two_moons.py` | `two_moons` (sbibm) | `heavy_tail_radius` |
| `gaussian_mixture.py` | `gaussian_mixture` (sbibm) | `one_gaussian` |
| `slcp.py` | `slcp` (sbibm) | `diagonal_covariance` |
| `sir.py` | `sir` (sbibm) | `weekend_delay` |
| `lotka_volterra.py` | `lotka_volterra` (sbibm) | `carrying_capacity` |
| `gaussian_linear_hd.py` | `gaussian_linear_hd` (custom, dim θ = 25) | `linear_misspec` |
| `gaussian_linear_hd_nonlinear.py` | `gaussian_linear_hd` | `nonlinear_theta` |
| `gaussian_mixture_hd.py` | `gaussian_mixture_hd` (custom) | `ellipse_modes` |
| `gaussian_mixture_2d.py` | `gaussian_mixture_2d` (custom) | `ellipse_modes` |
| `lotka_volterra_hd.py` | `lotka_volterra_hd` (custom, 5-species CLV, dim θ = 25) | `wrong_noise_scale` |
| `debug.py` | `two_moons` | `additive_noise` — tiny budgets for smoke tests |

To add a misspecification, register a factory in
`tabpfn_misspec/simulators.py::_REGISTRY`. To add a custom task, register it
in `tabpfn_misspec/tasks.py::_CUSTOM_TASKS` (must implement the duck-typed
sbibm task interface).

## Tests

```bash
pixi run pytest -v                                  # all
pixi run pytest tests/test_custom_task.py -v        # custom-task interface
pixi run pytest -m "not slow"                       # skip integration tests
```

## Running on a SLURM cluster (Jean Zay)

The `slurm/` directory contains batch scripts targeting Jean Zay:

- `slurm/sweep.slurm <task>` — A100 sweep for one task (defaults to
  `gaussian_linear`).
- `slurm/sweep_largecal.slurm <task>` — single large-`n_calib` point, separate
  output dir.
- `slurm/lvhd_setup_array.slurm` — 3-task array to generate
  `lotka_volterra_hd` reference posteriors (Slice → NSF → rejection pipeline,
  one observation per array task).
- `slurm/lvhd_diagnose.slurm` — short diagnostic for the LV-HD pipeline.

`slurm/JEAN_ZAY_SETUP.md` documents the one-time setup (Julia 1.10 +
`diffeqtorch` patching + Hugging Face weight cache push). Diff-eq–based tasks
(SIR, Lotka-Volterra, LV-HD) need that setup; pure-Python tasks do not.

## Pixi tasks (shorthand)

| Task | Command |
| --- | --- |
| `pixi run smoke` | `python scripts/smoke_test.py` |
| `pixi run sweep` | `python scripts/run_calibration_sweep.py` |
| `pixi run sweep-debug` | sweep with `configs/debug.py` |
| `pixi run plot` | `python scripts/plot_sweep.py` |
| `pixi run sync` | rsync repo to Jean Zay (edit the path in `pixi.toml`) |
| `pixi run sync-tabpfn-weights` | rsync local TabPFN weight cache to Jean Zay |
| `pixi run fetch-results` | rsync `results/` back from Jean Zay |
| `pixi run fetch-and-plot` | fetch then plot |

The `sync*` tasks have a hardcoded Jean Zay path — edit `pixi.toml` if you
target a different host.
