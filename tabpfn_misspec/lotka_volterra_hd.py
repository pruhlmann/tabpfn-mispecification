"""High-dim 5-species Competitive Lotka-Volterra task.

ODE:    dx_i/dt = r_i * x_i * (1 - sum_j alpha_ij * x_j),   i = 1..5

Inferred parameters (dim = 25): 5 growth rates r_i + 20 off-diagonal alpha_ij
(self-competition alpha_ii = 1 fixed). Julia ODE via diffeqtorch; reference
posterior follows sbibm Appendix B.1 (Slice-MCMC + NSF + rejection sampling).
"""

import gc
from pathlib import Path

import torch
import pyro
from pyro import distributions as pdist
from sbibm.tasks.task import Task
from sbibm.tasks.simulator import Simulator
from sbibm.utils.decorators import lazy_property
from diffeqtorch import DiffEq


_JULIA_F = """
function f(du, u, p, t)
    A = reshape(p[6:30], 5, 5)
    for i in 1:5
        s = 0.0
        for j in 1:5
            s += A[i, j] * u[j]
        end
        du[i] = p[i] * u[i] * (1.0 - s)
    end
end
"""


def _theta_to_p(theta: torch.Tensor) -> torch.Tensor:
    """(N, 25) -> (N, 30): [r (5) | alpha flat 5x5 (25)] with alpha_ii = 1.

    theta[:, :5]  -> r
    theta[:, 5:]  -> 20 off-diagonal alpha entries in row-major skip-diagonal order.
    """
    N = theta.shape[0]
    r = theta[:, :5]
    A = torch.eye(5).unsqueeze(0).repeat(N, 1, 1)
    off = [(i, j) for i in range(5) for j in range(5) if i != j]
    for k, (i, j) in enumerate(off):
        A[:, i, j] = theta[:, 5 + k]
    return torch.cat([r, A.reshape(N, 25)], dim=1)


class LotkaVolterraHD(Task):
    def __init__(self, days: float = 20.0, saveat: float = 0.1):
        self.dim_data_raw = int(5 * (days / saveat + 1))  # 5 * 201 = 1005
        dim_data = 5 * 10  # subsample every 21 steps -> 10 timepoints/species

        observation_seeds = [2000001, 2000002, 2000003]

        super().__init__(
            dim_parameters=25,
            dim_data=dim_data,
            name="lotka_volterra_hd",
            name_display="Lotka-Volterra (5-species competitive)",
            num_observations=len(observation_seeds),
            num_posterior_samples=10000,
            num_reference_posterior_samples=10000,
            num_simulations=[1000, 10000, 100000],
            path=Path(__file__).parent / "lotka_volterra_hd",
            observation_seeds=observation_seeds,
        )

        # LogNormal prior:
        #   r_i:        median 1.0,  scale 0.3
        #   alpha_ij:   median 0.1,  scale 0.5  (weak competition; stable equilibrium)
        # With alpha_ij ~ 0.1 and r_i ~ 1, x_i* = 1/(1 + 4*0.1) = 0.71 (interior).
        loc = torch.cat([
            torch.zeros(5),
            torch.full((20,), float(torch.log(torch.tensor(0.1)))),
        ])
        scale = torch.cat([
            torch.full((5,), 0.3),
            torch.full((20,), 0.5),
        ])
        self.prior_params = {"loc": loc, "scale": scale}
        self.prior_dist = pdist.LogNormal(**self.prior_params).to_event(1)

        self.u0 = torch.ones(5)
        self.tspan = torch.tensor([0.0, days])
        self.days = days
        self.saveat = saveat

    @lazy_property
    def de(self):
        return DiffEq(
            f=_JULIA_F,
            saveat=self.saveat,
            using=["DifferentialEquations", "SciMLSensitivity"],
            debug=False,
        )

    def get_labels_parameters(self):
        labels = [rf"$r_{{{i+1}}}$" for i in range(5)]
        labels += [
            rf"$\alpha_{{{i+1},{j+1}}}$"
            for i in range(5) for j in range(5) if i != j
        ]
        return labels

    def get_prior(self):
        def prior(num_samples=1):
            return pyro.sample(
                "parameters", self.prior_dist.expand_by([num_samples])
            )
        return prior

    def get_simulator(self, max_calls=None):
        def simulator(parameters):
            num_samples = parameters.shape[0]
            p_full = _theta_to_p(parameters)

            us = []
            for n in range(num_samples):
                u, _t = self.de(self.u0, self.tspan, p_full[n, :])
                if u.shape != torch.Size([5, int(self.dim_data_raw / 5)]):
                    u = float("nan") * torch.ones(
                        (5, int(self.dim_data_raw / 5))
                    ).double()
                if n % 100 == 0:
                    gc.collect()
                    self.de.jl.eval("Base.GC.gc()")
                us.append(u.reshape(1, 5, -1))
            us = torch.cat(us).float()
            us = us[:, :, ::21].reshape(num_samples, -1)

            data = float("nan") * torch.ones((num_samples, self.dim_data))
            ok = ~torch.isnan(us).any(dim=1)
            if ok.any():
                data[ok] = pyro.sample(
                    "data",
                    pdist.LogNormal(
                        loc=torch.log(us[ok].clamp(1e-10, 1e4)),
                        scale=0.1,
                    ).to_event(1),
                )
            return data

        return Simulator(task=self, simulator=simulator, max_calls=max_calls)

    def _sample_reference_posterior(
        self, num_samples, num_observation=None, observation=None,
    ):
        # Deviation from sbibm Appendix B.1: B.1 uses sbi's Slice kernel via
        # sbibm.algorithms.pyro.mcmc, but that wrapper imports sbi.mcmc which
        # was removed in sbi>=0.20. Use Pyro's NUTS directly to draw the
        # proposal samples that feed the NSF + rejection-sampling stage.
        from pyro.infer.mcmc import MCMC, NUTS
        from sbibm.algorithms.pytorch.baseline_rejection import run as run_rejection
        from sbibm.algorithms.pytorch.utils.proposal import get_proposal

        initial_params = (
            self.get_true_parameters(num_observation=num_observation)
            if num_observation is not None else None
        )
        conditioned_model = self._get_pyro_model(
            num_observation=num_observation, observation=observation
        )
        transforms = self._get_transforms(
            num_observation=num_observation,
            observation=observation,
            automatic_transforms_enabled=True,
        )
        kernel = NUTS(model=conditioned_model, transforms=transforms, jit_compile=False)
        if initial_params is not None:
            initial_params = {"parameters": transforms["parameters"](initial_params)}
        mcmc = MCMC(
            kernel,
            num_samples=num_samples,
            warmup_steps=2_000,
            num_chains=1,
            initial_params=initial_params,
        )
        mcmc.run()
        proposal_samples = mcmc.get_samples()["parameters"].squeeze()
        proposal_dist = get_proposal(
            task=self,
            samples=proposal_samples,
            prior_weight=0.1,
            bounded=True,
            density_estimator="flow",
            flow_model="nsf",
        )
        return run_rejection(
            task=self,
            num_observation=num_observation,
            observation=observation,
            num_samples=num_samples,
            batch_size=10_000,
            num_batches_without_new_max=1_000,
            multiplier_M=1.2,
            proposal_dist=proposal_dist,
        )


if __name__ == "__main__":
    task = LotkaVolterraHD()
    task._setup(n_jobs=1)
