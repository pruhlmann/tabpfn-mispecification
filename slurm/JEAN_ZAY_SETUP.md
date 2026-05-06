# Jean Zay setup (Julia + diffeqtorch + Lotka-Volterra HD)

Run all of this on a **login node** (`jean-zay.idris.fr`). Compute nodes have no
internet, so package install must happen here.

## 1. Get Julia 1.10 (skip system Julia 1.12)

System Julia 1.12 is incompatible with `OrdinaryDiffEq`'s precompile workload
(`Base.StaticData` / `OrdinaryDiffEqTag` errors). Use the official 1.10 tarball:

```bash
cd $WORK
wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.11-linux-x86_64.tar.gz
tar -xzf julia-1.10.11-linux-x86_64.tar.gz
export PATH="$WORK/julia-1.10.11/bin:$PATH"
julia --version   # → julia version 1.10.11
```

## 2. Sync the repo + install both pixi envs

The sweep runs in the `gpu` env (`pixi run -e gpu sweep`); `default` is CPU and
used for setup / tests. Install both:

```bash
cd $WORK/projects/tabpfn-mispecification
pixi install                  # default (CPU)
pixi install -e gpu           # gpu (CUDA torch)
```

## 3. Patch `diffeqtorch`'s Julia Project.toml in both envs

Drop the legacy `DiffEqSensitivity` + `Zygote` (they pin `OrdinaryDiffEq` to
v6.25 which won't precompile on Julia 1.10) and add the modern
`SciMLSensitivity` — needed for `ODEForwardSensitivityProblem` (NUTS gradient
path through the ODE solve). Each env has its own site-packages, so patch
both:

```bash
for ENV in default gpu; do
  DIFFEQ_JL=$PWD/.pixi/envs/$ENV/lib/python3.11/site-packages/diffeqtorch/julia
  rm -f $DIFFEQ_JL/Manifest.toml
  cat > $DIFFEQ_JL/Project.toml <<'EOF'
[deps]
ArgParse = "c7e460c6-2fb9-53a9-8c5b-16f535851c63"
DifferentialEquations = "0c46a032-eb83-5123-abaf-570d42b7fbaa"
InteractiveUtils = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
OrdinaryDiffEq = "1dea7af3-3e70-54e6-95c3-0bf5283fa5ed"
PackageCompiler = "9b87118b-4619-50d2-8e1e-99f35a4d4d9d"
PyCall = "438e738f-606a-5dbb-bf0a-cddfbfd45ab0"
SciMLSensitivity = "1ed8b502-d754-442c-8d5d-10ac956f44a1"
EOF
done
```

## 4. Instantiate + precompile Julia deps (~3 min per env)

The Julia depot is shared via `JULIA_DEPOT_PATH`, but each env has its own
Manifest.toml — instantiate once per env so each gets a resolved Manifest:

```bash
export JULIA_DEPOT_PATH=$WORK/.julia      # shared depot, off $HOME quota
for ENV in default gpu; do
  export JULIA_PROJECT=$PWD/.pixi/envs/$ENV/lib/python3.11/site-packages/diffeqtorch/julia
  julia -e 'using Pkg; Pkg.instantiate(); Pkg.precompile()'
done
julia -e 'using DifferentialEquations; println("OK")'   # smoke test under last env
```

If precompile fails with stale-cache errors:

```bash
rm -rf $JULIA_DEPOT_PATH/compiled/v1.12 $JULIA_DEPOT_PATH/compiled/v1.10
```

and rerun step 4.

## 5. Add env vars to SLURM scripts

Add before the `srun pixi run` line in `slurm/sweep.slurm` (and any other batch
script that touches the simulator):

```bash
export PATH="$WORK/julia-1.10.11/bin:$PATH"
export JULIA_PROJECT=$WORK/projects/tabpfn-mispecification/.pixi/envs/gpu/lib/python3.11/site-packages/diffeqtorch/julia
export JULIA_DEPOT_PATH=$WORK/.julia
export LD_LIBRARY_PATH=$WORK/projects/tabpfn-mispecification/.pixi/envs/gpu/lib:$LD_LIBRARY_PATH
```

(Use `envs/gpu/...` because `slurm/sweep.slurm` runs `pixi run -e gpu`. For
batch jobs that use the CPU env — e.g. the reference-posterior `_setup` job —
swap `gpu` for `default` in both paths.)

`LD_LIBRARY_PATH` is needed so matplotlib (transitive sbi dep) picks up the
pixi env's libstdc++ instead of the older system one.

## 6. Smoke test on a compute node

```bash
srun -A vxk@cpu --time=00:15:00 --pty bash
# inside the allocation:
export PATH="$WORK/julia-1.10.11/bin:$PATH"
export JULIA_DEPOT_PATH=$WORK/.julia
export LD_LIBRARY_PATH=$WORK/projects/tabpfn-mispecification/.pixi/envs/default/lib:$LD_LIBRARY_PATH
cd $WORK/projects/tabpfn-mispecification
pixi run python -m pytest tests/test_custom_task.py -k lotka_volterra_hd -v
```

Expect 4 passed in ~75s (first call ~60s of Julia compile; cached afterwards).

## 7. Reference-posterior generation (long, batch it)

Once tests pass, submit `task._setup(n_jobs=1)` as a batch job (CPU,
`--time=12:00:00`) — generation runs hours per observation. Then:

```bash
sbatch slurm/sweep.slurm lotka_volterra_hd
```
