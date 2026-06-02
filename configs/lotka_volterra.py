"""Task-specific config for Lotka-Volterra with carrying capacity misspecification."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.task = "lotka_volterra"
    config.misspec_type = "carrying_capacity"
    config.misspec_kwargs = {"K": 100}
    config.num_sim_mixed = 5000
    config.num_posterior_samples = 5000
    config.num_observations = 1
    config.seed = 42
    config.num_calibration = 50
    config.num_context = 2000
    config.skip_methods = ["npepfn_mixed", "fmcpe"]
    config.use_prior_transform = False
    config.seeds = (42, 123, 512)
    config.batch_size = 2500
    config.cache_data = True
    config.augment_M = 1
    config.train_batch_size = 1024
    return config
