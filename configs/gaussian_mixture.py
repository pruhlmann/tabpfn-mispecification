"""Task-specific config for gaussian_mixture."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.task = "gaussian_mixture"
    config.misspec_type = "one_gaussian"
    config.misspec_kwargs = {}
    config.num_simulations = 5000
    config.num_posterior_samples = 10000
    config.num_observations = 1
    config.seed = 42
    config.num_calibration = 50
    config.num_synthetic = 10000
    config.use_prior_transform = False
    config.seeds = [42, 123, 456]
    return config
