"""FMCPE (Flow-Matching Conditional Posterior Estimator).

Self-contained port of `fm_post_transform` from the `ropefm` repository.
Three coupled conditional flow-matching models:

  * `npe`        — proposal CFM trained on misspecified prior simulations
                   (theta_sim, x_sim); learns p(theta | x).
  * `flow_x`     — denoiser CFM trained on calibration pairs (x_calib, y_calib);
                   learns p(x | y) to map real observations back into
                   simulator-output space.
  * `flow_theta` — posterior-transform CFM with the NPE samples as source
                   distribution; trained on calibration (theta_calib, y_calib)
                   to refine misspecified proposal samples conditional on y.

Sampling at inference is a 3-stage pipeline:

    y_obs --flow_x--> x_tilde --npe--> theta_0 --flow_theta--> theta_final
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import Tensor, nn


# -----------------------------------------------------------------------------
# Rescaler
# -----------------------------------------------------------------------------


class _ZScoreRescaler(nn.Module):
    """Affine z-score normalisation with fit/transform/inverse_transform.

    Buffers move with `.to(device)` and serialise via `state_dict`.
    """

    def __init__(self, dim: int, eps: float = 1e-8):
        super().__init__()
        self.eps = float(eps)
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))

    @torch.no_grad()
    def fit(self, x: Tensor) -> None:
        m = x.mean(dim=0)
        s = x.std(dim=0).clamp_min(self.eps)
        self.mean.copy_(m.to(self.mean.device))
        self.std.copy_(s.to(self.std.device))

    def transform(self, x: Tensor) -> Tensor:
        return (x - self.mean) / self.std

    def inverse_transform(self, z: Tensor) -> Tensor:
        return z * self.std + self.mean


# -----------------------------------------------------------------------------
# Residual MLP drift (zero-init final layer for identity-map initialisation)
# -----------------------------------------------------------------------------


class _Residual(nn.Module):
    def __init__(self, body: nn.Module):
        super().__init__()
        self.body = body

    def forward(self, x: Tensor) -> Tensor:
        return x + self.body(x)


class _ResMLP(nn.Module):
    """Mirror of ropefm/utils/networks.py::ResMLP.

    Sequence of `Linear(before, after)` projections interleaved with
    `Residual(Linear -> ELU -> Linear)` blocks. Drops the trailing residual so
    the network ends in a linear projection to `out_dim`; this final linear is
    zero-initialised so the initial velocity field is the zero map and the ODE
    is the identity at init.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden: Iterable[int] = (64, 64),
    ):
        super().__init__()
        sizes = (in_dim, *hidden, out_dim)
        blocks: list[nn.Module] = []
        for before, after in zip(sizes[:-1], sizes[1:]):
            if after != before:
                blocks.append(nn.Linear(before, after))
            blocks.append(
                _Residual(
                    nn.Sequential(
                        nn.Linear(after, after),
                        nn.ELU(),
                        nn.Linear(after, after),
                    )
                )
            )
        blocks = blocks[:-1]
        self.blocks = nn.ModuleList(blocks)

        for block in reversed(self.blocks):
            if isinstance(block, nn.Linear):
                nn.init.zeros_(block.weight)
                nn.init.zeros_(block.bias)
                break

    def forward(self, inputs: tuple[Tensor, Tensor, Tensor]) -> Tensor:
        xt, t, cond = inputs
        x = torch.cat([xt, t, cond], dim=-1)
        for block in self.blocks:
            x = block(x)
        return x


# -----------------------------------------------------------------------------
# FlowMatching: conditional flow matching with OT2 path + z-score rescaling
# -----------------------------------------------------------------------------


class FlowMatching(nn.Module):
    """Conditional flow matching v_theta(x_t, t, cond).

    OT2 probability path: x_t = (1 - (1 - sigma_min) t) x_0 + t x_1, with
    target velocity x_1 - x_0. Time prior Beta(1.5, 1.0) ('power' in ropefm).
    Base distribution N(0, I) in rescaled target space.
    """

    def __init__(
        self,
        target_dim: int,
        cond_dim: int,
        hidden: Iterable[int] = (64, 64),
        num_steps: int = 50,
        sigma_min: float = 1e-4,
    ):
        super().__init__()
        self.target_dim = int(target_dim)
        self.cond_dim = int(cond_dim)
        self.num_steps = int(num_steps)
        self.sigma_min = float(sigma_min)
        self.drift = _ResMLP(
            self.target_dim + 1 + self.cond_dim, self.target_dim, hidden=hidden
        )
        self.target_rescaler = _ZScoreRescaler(self.target_dim)
        self.cond_rescaler = _ZScoreRescaler(self.cond_dim)

    def set_scales(self, data: Tensor, cond: Tensor) -> None:
        self.target_rescaler.fit(data)
        self.cond_rescaler.fit(cond)

    def _time_sample(self, n: int, device) -> Tensor:
        return torch.distributions.Beta(1.5, 1.0).sample((n,)).to(device)

    def _interp(self, t: Tensor, x0: Tensor, x1: Tensor) -> Tensor:
        t_ = t.view(-1, 1)
        alpha = 1.0 - (1.0 - self.sigma_min) * t_
        beta = t_
        return alpha * x0 + beta * x1

    def forward(self, xt: Tensor, cond: Tensor, t: Tensor) -> Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return self.drift((xt, t, cond))

    def compute_loss(self, target: Tensor, cond: Tensor) -> Tensor:
        target_s = self.target_rescaler.transform(target)
        cond_s = self.cond_rescaler.transform(cond)
        t = self._time_sample(target.shape[0], target.device)
        source_s = torch.randn_like(target_s)
        xt = self._interp(t, source_s, target_s)
        v_pred = self.forward(xt, cond_s, t)
        v_target = target_s - source_s
        return (v_pred - v_target).pow(2).mean()

    def compute_loss_with_source(
        self, source: Tensor, target: Tensor, cond: Tensor
    ) -> Tensor:
        # Source samples arrive in DATA space (caller's distribution, e.g. NPE
        # output); rescale with target_rescaler so the velocity field operates
        # in the same rescaled space it sees at inference.
        target_s = self.target_rescaler.transform(target)
        cond_s = self.cond_rescaler.transform(cond)
        source_s = self.target_rescaler.transform(source)
        t = self._time_sample(target.shape[0], target.device)
        xt = self._interp(t, source_s, target_s)
        v_pred = self.forward(xt, cond_s, t)
        v_target = target_s - source_s
        return (v_pred - v_target).pow(2).mean()

    @torch.no_grad()
    def sample(
        self, x0: Tensor, cond: Tensor, num_steps: int | None = None
    ) -> Tensor:
        # x0 is in target-rescaled space; cond is in data space.
        cond_s = self.cond_rescaler.transform(cond)
        steps = int(num_steps if num_steps is not None else self.num_steps)
        dt = 1.0 / steps
        xt = x0
        for k in range(steps):
            t = torch.full((xt.shape[0],), k * dt, device=xt.device)
            v = self.forward(xt, cond_s, t)
            xt = xt + v * dt
        return self.target_rescaler.inverse_transform(xt)

    @torch.no_grad()
    def sample_base(self, cond: Tensor) -> Tensor:
        """Gaussian sample in target-rescaled space; one row per cond row."""
        return torch.randn(cond.shape[0], self.target_dim, device=cond.device)

    def rescale_source(self, source: Tensor) -> Tensor:
        return self.target_rescaler.transform(source)


# -----------------------------------------------------------------------------
# Training orchestration
# -----------------------------------------------------------------------------


def _split_indices(n: int, frac: float, gen: torch.Generator) -> tuple[Tensor, Tensor]:
    perm = torch.randperm(n, generator=gen)
    n_train = max(1, int(round(frac * n)))
    if n_train >= n:
        return perm, perm[:0]
    return perm[:n_train], perm[n_train:]


def _clone_state(module: nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def _train_single_flow(
    *,
    flow: FlowMatching,
    target_train: Tensor,
    cond_train: Tensor,
    target_val: Tensor,
    cond_val: Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    device,
    gen: torch.Generator,
) -> None:
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_train = target_train.shape[0]
    bs = max(1, min(batch_size, n_train))
    val_available = target_val.shape[0] > 0

    best_val = float("inf")
    best_state: dict | None = None
    epochs_since_improve = 0

    for _epoch in range(epochs):
        flow.train()
        perm = torch.randperm(n_train, generator=gen)
        for i in range(0, n_train, bs):
            idx = perm[i : i + bs]
            tgt = target_train[idx].to(device)
            cnd = cond_train[idx].to(device)
            loss = flow.compute_loss(tgt, cnd)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

        if val_available:
            flow.eval()
            with torch.no_grad():
                vl = flow.compute_loss(
                    target_val.to(device), cond_val.to(device)
                ).item()
            if vl < best_val:
                best_val = vl
                best_state = _clone_state(flow)
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= patience:
                    break

    if best_state is not None:
        flow.load_state_dict(best_state)


def _train_dual_flows(
    *,
    flow_theta: FlowMatching,
    flow_x: FlowMatching,
    npe: FlowMatching,
    theta_train: Tensor,
    x_train: Tensor,
    y_train: Tensor,
    theta_val: Tensor,
    x_val: Tensor,
    y_val: Tensor,
    epochs: int,
    lr: float,
    batch_size: int,
    patience: int,
    device,
    gen: torch.Generator,
    lam: float = 0.5,
) -> None:
    """Coupled training of flow_x (CFM) and flow_theta (CFM with NPE source).

    Mirrors ropefm/baselines/trainers.py::_train_loop_flow_theta_x. For each
    batch (theta, x, y) we draw `x_pred = flow_x.sample(y)`, then
    `theta_0 = npe.sample(x_pred)`, and use theta_0 as the source distribution
    for flow_theta's CFM loss. flow_x is trained with the standard CFM loss
    against (x, y). Combined loss: lam * loss_theta + (1 - lam) * loss_x.
    """
    params = list(flow_theta.parameters()) + list(flow_x.parameters())
    optimizer = torch.optim.Adam(params, lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    n_train = theta_train.shape[0]
    bs = max(1, min(batch_size, n_train))
    val_available = theta_val.shape[0] > 0

    npe.eval()

    def _batch_loss(thb: Tensor, xb: Tensor, yb: Tensor) -> Tensor:
        with torch.no_grad():
            x_pred = flow_x.sample(flow_x.sample_base(yb), yb)
            theta_0 = npe.sample(npe.sample_base(x_pred), x_pred)
        loss_theta = flow_theta.compute_loss_with_source(theta_0, thb, yb)
        loss_x = flow_x.compute_loss(xb, yb)
        return lam * loss_theta + (1.0 - lam) * loss_x

    best_val = float("inf")
    best_state: tuple[dict, dict] | None = None
    epochs_since_improve = 0

    for _epoch in range(epochs):
        flow_theta.train()
        flow_x.train()
        perm = torch.randperm(n_train, generator=gen)
        for i in range(0, n_train, bs):
            idx = perm[i : i + bs]
            thb = theta_train[idx].to(device)
            xb = x_train[idx].to(device)
            yb = y_train[idx].to(device)
            loss = _batch_loss(thb, xb, yb)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(flow_theta.parameters(), 5.0)
            torch.nn.utils.clip_grad_norm_(flow_x.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

        if val_available:
            flow_theta.eval()
            flow_x.eval()
            with torch.no_grad():
                vl = _batch_loss(
                    theta_val.to(device),
                    x_val.to(device),
                    y_val.to(device),
                ).item()
            if vl < best_val:
                best_val = vl
                best_state = (_clone_state(flow_theta), _clone_state(flow_x))
                epochs_since_improve = 0
            else:
                epochs_since_improve += 1
                if epochs_since_improve >= patience:
                    break

    if best_state is not None:
        flow_theta.load_state_dict(best_state[0])
        flow_x.load_state_dict(best_state[1])


def train_fmcpe(
    theta_sim: Tensor,
    x_sim: Tensor,
    theta_calib: Tensor,
    x_calib: Tensor,
    y_calib: Tensor,
    *,
    device,
    hidden: Iterable[int] = (64, 64),
    num_steps: int = 50,
    npe_epochs: int = 1000,
    dual_epochs: int = 1000,
    lr: float = 1e-4,
    batch_size: int = 128,
    patience: int = 20,
    train_size: float = 0.8,
    seed: int = 42,
) -> dict:
    """Train the three coupled FMCPE flows and return them."""
    theta_sim = theta_sim.float()
    x_sim = x_sim.float()
    theta_calib = theta_calib.float()
    x_calib = x_calib.float()
    y_calib = y_calib.float()

    theta_dim = theta_sim.shape[1]
    x_dim = x_sim.shape[1]
    y_dim = y_calib.shape[1]

    gen = torch.Generator(device="cpu").manual_seed(int(seed))

    # ---- Phase A: NPE proposal on (theta_sim, x_sim) ----
    npe = FlowMatching(theta_dim, x_dim, hidden=hidden, num_steps=num_steps).to(device)
    npe.set_scales(theta_sim.to(device), x_sim.to(device))
    sim_tr, sim_va = _split_indices(theta_sim.shape[0], train_size, gen)
    _train_single_flow(
        flow=npe,
        target_train=theta_sim[sim_tr],
        cond_train=x_sim[sim_tr],
        target_val=theta_sim[sim_va],
        cond_val=x_sim[sim_va],
        epochs=npe_epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        device=device,
        gen=gen,
    )

    # ---- Phase B: flow_x + flow_theta on calibration ----
    flow_x = FlowMatching(x_dim, y_dim, hidden=hidden, num_steps=num_steps).to(device)
    flow_theta = FlowMatching(theta_dim, y_dim, hidden=hidden, num_steps=num_steps).to(
        device
    )
    flow_x.set_scales(x_calib.to(device), y_calib.to(device))
    flow_theta.set_scales(theta_calib.to(device), y_calib.to(device))

    cal_tr, cal_va = _split_indices(theta_calib.shape[0], train_size, gen)
    _train_dual_flows(
        flow_theta=flow_theta,
        flow_x=flow_x,
        npe=npe,
        theta_train=theta_calib[cal_tr],
        x_train=x_calib[cal_tr],
        y_train=y_calib[cal_tr],
        theta_val=theta_calib[cal_va],
        x_val=x_calib[cal_va],
        y_val=y_calib[cal_va],
        epochs=dual_epochs,
        lr=lr,
        batch_size=batch_size,
        patience=patience,
        device=device,
        gen=gen,
    )

    return {"npe": npe, "flow_theta": flow_theta, "flow_x": flow_x}


@torch.no_grad()
def sample_fmcpe(models: dict, y_obs: Tensor, num_samples: int, device) -> Tensor:
    """3-stage sampling: y -> flow_x -> NPE proposal -> flow_theta."""
    flow_x: FlowMatching = models["flow_x"]
    flow_theta: FlowMatching = models["flow_theta"]
    npe: FlowMatching = models["npe"]
    flow_x.eval()
    flow_theta.eval()
    npe.eval()

    y_obs = y_obs.float().to(device)
    if y_obs.dim() == 1:
        y_obs = y_obs.unsqueeze(0)
    y_exp = y_obs[:1].expand(num_samples, -1).contiguous()

    x_tilde = flow_x.sample(flow_x.sample_base(y_exp), y_exp)
    theta_0 = npe.sample(npe.sample_base(x_tilde), x_tilde)
    source_scaled = flow_theta.rescale_source(theta_0)
    theta = flow_theta.sample(source_scaled, y_exp)
    return theta.cpu()
