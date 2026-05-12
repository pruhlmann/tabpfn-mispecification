"""Task-specific config for gaussian_linear."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.task = "gaussian_linear"
    config.misspec_type = "additive_noise"
    config.misspec_kwargs = {}
    config.num_sim_mixed = 2000
    config.num_posterior_samples = 5000
    config.num_observations = 1
    config.seed = 42
    config.num_calibration = 50
    config.num_context = 2000
    config.use_prior_transform = False
    config.seeds = [42, 123, 45]
    config.skip_methods = ["npepfn_misspec", "npepfn_calib", "npepfn_mixed"]
    config.batch_size = 1000
    config.augment_M = 1
    config.train_batch_size = 1024
    return config
