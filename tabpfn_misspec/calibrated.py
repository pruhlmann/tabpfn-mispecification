"""Mixed-context NPE-PFN with NaN masking for calibration data."""

import torch

from npe_pfn import TabPFN_Based_NPE_PFN


def build_calibrated_estimator(theta_sim_t, x_sim, theta_calib_t, x_calib, y_calib, unbounded_prior):
    """NPE-PFN with mixed sim + calibration context using NaN masking.

    Extends feature space from [x] to [x, y]:
    - Simulation rows: [x_sim, NaN] — y unknown
    - Calibration rows: [x_calib, y_calib] — both present

    Args:
        theta_sim_t: Parameters from simulation set in transformed space, shape (N_sim, dim_theta).
        x_sim: Misspecified simulator outputs, shape (N_sim, dim_x).
        theta_calib_t: Parameters from calibration set in transformed space, shape (N_calib, dim_theta).
        x_calib: Misspecified simulator outputs for calibration params, shape (N_calib, dim_x).
        y_calib: True simulator outputs for calibration params, shape (N_calib, dim_y).
        unbounded_prior: Prior in transformed space (for rejection sampling).

    Returns:
        TabPFN_Based_NPE_PFN estimator with mixed context appended.
    """
    dim_y = y_calib.shape[1]
    x_sim_ext = torch.cat([x_sim, torch.full((len(x_sim), dim_y), float("nan"))], dim=1)
    x_calib_ext = torch.cat([x_calib, y_calib], dim=1)

    estimator = TabPFN_Based_NPE_PFN(prior=unbounded_prior)
    estimator.append_simulations(
        torch.cat([theta_sim_t, theta_calib_t]),
        torch.cat([x_sim_ext, x_calib_ext]),
    )
    return estimator


def sample_calibrated(estimator, y_obs, dim_x, num_samples):
    """Query the mixed-context estimator with [NaN_x, y_obs].

    At inference we don't know x (we don't have theta), but we have the real
    observation y_obs. The query is [NaN, ..., NaN, y_obs].

    Args:
        estimator: TabPFN_Based_NPE_PFN with mixed context.
        y_obs: Real observation, shape (1, dim_y).
        dim_x: Dimensionality of the misspecified simulator output.
        num_samples: Number of posterior samples to draw.

    Returns:
        Posterior samples, shape (num_samples, dim_theta).
    """
    query = torch.cat([torch.full((1, dim_x), float("nan")), y_obs], dim=1)
    return estimator.sample(sample_shape=torch.Size([num_samples]), x=query)


def build_y_predictor(theta_calib, x_calib, y_calib):
    """Build a TabPFN that predicts y from (theta, x).

    Context is calibration data only. TabPFN target = y_calib,
    features = [theta_calib, x_calib]. Uses raw (untransformed) theta
    since it's a feature, not a parameter.

    Args:
        theta_calib: Calibration parameters in original space, shape (N_calib, dim_theta).
        x_calib: Misspecified simulator outputs, shape (N_calib, dim_x).
        y_calib: True simulator outputs, shape (N_calib, dim_y).

    Returns:
        TabPFN_Based_NPE_PFN estimator for y prediction.
    """
    dim_y = y_calib.shape[1]
    dummy_prior = torch.distributions.Independent(
        torch.distributions.Normal(torch.zeros(dim_y), 100.0 * torch.ones(dim_y)), 1
    )
    features = torch.cat([theta_calib, x_calib], dim=1)
    estimator = TabPFN_Based_NPE_PFN(prior=dummy_prior)
    estimator.append_simulations(y_calib, features)
    return estimator


def generate_synthetic_y(y_predictor, theta, x):
    """Generate synthetic y samples using the y-predictor.

    Uses batched sampling to get one ỹ per (theta_i, x_i) in a single call.

    Args:
        y_predictor: TabPFN estimator from build_y_predictor.
        theta: Parameter values, shape (N, dim_theta).
        x: Misspecified simulator outputs, shape (N, dim_x).

    Returns:
        Synthetic y values, shape (N, dim_y).
    """
    query = torch.cat([theta, x], dim=1)
    # sample_batched returns shape (N, 1, dim_y)
    y_tilde = y_predictor.sample_batched(x=query, sample_shape=torch.Size([1]))
    return y_tilde.squeeze(1)
