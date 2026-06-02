"""Task-specific config for slcp."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.task = "slcp"
    config.misspec_type = "diagonal_covariance"
    config.misspec_kwargs = {}
    config.num_sim_mixed = 5000
    config.num_posterior_samples = 10000
    config.num_observations = 1
    config.seed = 42
    config.num_calibration = 50
    config.num_context = 2000
    config.use_prior_transform = False
    config.seeds = [42, 123, 456]
    config.skip_methods = ["npepfn_mixed", "fmcpe"]
    config.batch_size = 2500
    config.augment_M = 1
    config.use_cache = True
    config.train_batch_size = 1024
    return config
