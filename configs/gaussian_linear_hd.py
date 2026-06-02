"""Task-specific config for the high-dim Gaussian linear custom task.

dim_theta = 25, dim_y = dim_x = 8.  Misspecification: x = A theta + b + eps_x
with A != C (different random matrices), so x is a linearly mismatched view of
theta. Reference posterior is closed-form (Bayesian linear regression).
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.task = "gaussian_linear_hd"
    config.misspec_type = "linear_misspec"
    config.misspec_kwargs = {"sigma_x": 0.5}
    config.num_sim_mixed = 5000
    config.num_posterior_samples = 5000
    config.num_observations = 1
    config.seed = 42
    config.num_calibration = 50
    config.num_context = 2000
    config.use_prior_transform = False
    config.seeds = [42, 123, 512]
    config.skip_methods = ["npepfn_mixed", "fmcpe"]
    config.batch_size = 5000
    config.cache_data = True
    config.augment_M = 1
    config.train_batch_size = 1024
    config.num_sbc = 0  # SBC/TARP test pairs from the true simulator (0 = skip)
    config.num_sbc_samples = 250
    return config
