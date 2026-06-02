"""Task-specific config for two_moons."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.task = "two_moons"
    config.misspec_type = "heavy_tail_radius"
    config.misspec_kwargs = {"df": 2}
    config.num_sim_mixed = 5000
    config.num_posterior_samples = 10000
    config.num_observations = 1
    config.seed = 42
    config.num_calibration = 50
    config.num_context = 2000
    config.use_prior_transform = False
    config.seeds = [42, 123, 456]
    config.skip_methods = ["npepfn_mixed", "fmcpe"]  # no expensive simulator for synthetic y
    config.augment_M = 1
    config.batch_size = 1500
    config.cache_data = True
    config.train_batch_size = 1024
    return config
