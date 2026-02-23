"""Task-specific config for two_moons."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.task = "two_moons"
    config.misspec_type = "heavy_tail_radius"
    config.misspec_kwargs = {"df": 2}
    config.num_simulations = 5000
    config.num_posterior_samples = 1000
    config.num_observations = 3
    config.seed = 42
    config.num_calibration = 50
    config.num_synthetic = 10000
    config.use_prior_transform = False
    config.seeds = [42, 123, 456, 789, 1024]
    return config
