"""Task-specific config for the high-dim Gaussian-mixture custom task.

dim_theta = dim_x = 20. Well-specified posterior is the "eight gaussians on a
circle" (closed-form, conjugate Gaussian prior + 8-component mixture
likelihood). Misspecification: anchors placed on an ellipse (axis_ratio != 1).
"""

import ml_collections


def get_config():
    config = ml_collections.ConfigDict()
    config.task = "gaussian_mixture_hd"
    config.misspec_type = "ellipse_modes"
    config.misspec_kwargs = {"axis_ratio": 0.5}
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
