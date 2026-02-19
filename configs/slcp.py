"""Task-specific config for slcp."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.task = "slcp"
    config.misspec_type = "additive_noise"
    config.misspec_kwargs = {}
    config.num_simulations = 1000
    config.num_posterior_samples = 1000
    config.num_observations = 3
    config.seed = 42
    config.num_calibration = 50
    config.num_synthetic = 1000
    config.use_prior_transform = True
    config.seeds = [42, 123, 456, 789, 1024]
    return config
