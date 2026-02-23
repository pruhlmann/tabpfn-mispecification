"""Task-specific config for SIR with weekend delay misspecification."""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.task = "sir"
    config.misspec_type = "weekend_delay"
    config.misspec_kwargs = {"delay_fraction": 0.05}
    config.num_simulations = 10000
    config.num_posterior_samples = 1000
    config.num_observations = 3
    config.seed = 42
    config.num_calibration = 50
    config.num_synthetic = 10000
    config.skip_methods = ["npepfn_y_fmpe"]  # expensive simulator for synthetic y
    config.use_prior_transform = False
    config.seeds = (42, 123)
    return config
