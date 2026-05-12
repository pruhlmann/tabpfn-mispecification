"""Task-specific config for gaussian_mixture."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.task = "gaussian_mixture"
    config.misspec_type = "one_gaussian"
    config.misspec_kwargs = {}
    config.num_sim_mixed = 5000
    config.num_posterior_samples = 10000
    config.num_observations = 1
    config.seed = 42
    config.num_calibration = 50
    config.num_context = 2000
    config.use_prior_transform = False
    config.seeds = [42, 123, 456]
    config.skip_methods = ["npepfn_mixed"]  # expensive simulator for synthetic y
    config.augment_M = 1
    config.batch_size = 2000
    config.metrics_to_compute = ("c2st", "mmd")
    config.cache_data = True
    config.train_batch_size = 1024
    return config
