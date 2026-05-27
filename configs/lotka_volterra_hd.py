"""Task-specific config for the high-dim 5-species Competitive Lotka-Volterra task.

dim_theta = 25 (5 r + 20 off-diagonal alpha; alpha_ii=1 fixed). Misspecification:
LogNormal observation noise with scale=0.5 instead of true scale=0.1.
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.task = "lotka_volterra_hd"
    config.misspec_type = "wrong_noise_scale"
    config.misspec_kwargs = {"scale": 0.5}
    config.num_sim_mixed = 5000
    config.num_posterior_samples = 5000
    config.num_observations = 1
    config.seed = 42
    config.num_calibration = 50
    config.num_context = 2000
    config.skip_methods = ["npepfn_mixed", "fmcpe"]
    config.use_prior_transform = False
    config.seeds = (42, 123, 512)
    config.batch_size = 1000
    config.cache_data = True
    config.augment_M = 1
    config.train_batch_size = 1024
    config.num_sbc = 500  # SBC/TARP test pairs from the true simulator (0 = skip)
    config.num_sbc_samples = 1000
    return config
