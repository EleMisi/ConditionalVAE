import tensorflow as tf


def reconstruction_loss(original, reconstruction, eps=1e-10):
    """
    Reconstruction loss 
    """
    _tmp = original * tf.math.log(eps + reconstruction) + (1 - original) * tf.math.log(eps + 1 - reconstruction)
    return -tf.compat.v1.reduce_sum(_tmp, 1)


def latent_loss(latent_mean, latent_log_sigma_sq):
    """
    Latent loss (KL divergence)
    """
    latent_log_sigma_sq = tf.compat.v1.clip_by_value(latent_log_sigma_sq, clip_value_min=-1e-10, clip_value_max=1e+2)
    return -0.5 * tf.compat.v1.reduce_sum(1 + latent_log_sigma_sq - tf.math.square(latent_mean) - tf.math.exp(latent_log_sigma_sq), 1)

