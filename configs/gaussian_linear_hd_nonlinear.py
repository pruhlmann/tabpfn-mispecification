"""Gaussian linear HD with quadratic-in-theta misspec (alpha controls strength)."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.task = "gaussian_linear_hd"
    config.misspec_type = "nonlinear_theta"
    config.misspec_kwargs = {"sigma_x": 0.5, "alpha": 0.1}
    config.num_sim_mixed = 5000
    config.num_posterior_samples = 5000
    config.num_observations = 1
    config.seed = 42
    config.num_calibration = 50
    config.num_context = 2000
    config.use_prior_transform = False
    config.seeds = [42, 123, 512]
    config.skip_methods = ["npepfn_mixed"]
    config.batch_size = 5000
    config.cache_data = True
    config.augment_M = 1
    config.train_batch_size = 1024
    return config
