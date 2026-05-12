"""Debug config: low sample counts for quick smoke-testing."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.task = "two_moons"
    config.misspec_type = "additive_noise"
    config.misspec_kwargs = {}
    config.num_sim_mixed = 50
    config.num_posterior_samples = 5
    config.num_observations = 2
    config.seed = 42
    config.num_calibration = 10
    config.num_context = 2000
    config.use_prior_transform = True
    config.seeds = [42, 123]
    config.augment_M = 1
    config.batch_size = None
    config.train_batch_size = 1024
    return config
