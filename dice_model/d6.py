import numpy as np
from scipy.special import gamma
import scipy.stats as stats
from sklearn.preprocessing import OneHotEncoder

# example of each possible die observation category (6, 1)
obs_cat = np.asarray([[i for i in range(1, 7)]]).T
# encoder to transform (N, 1) observations to one hot (N, K)
encoder = OneHotEncoder(sparse=False, dtype=np.int)
encoder.fit(obs_cat)

# priors
prior_alpha = np.asarray([20.0] * 6)
prior_dist = stats.dirichlet(prior_alpha)


def map_alpha(observations, prior_alpha):
    """MAP estimate of K-dimensional Dirichlet alpha

    Args:
        observations (numpy.ndarray): One-hot encoding of observations (N, K)
        prior_alpha (numpy.ndarray): Alpha for Dirichlet prior (K,)

    Returns:
        numpy.ndarray: MAP estimate of Dirichlet alpha (K,)
    """
    # number of observations
    n = observations.shape[0]
    # count of each category in observations
    obs_count = np.sum(observations, axis=0)
    # how prior_alpha is used as exponent in Dirichlet distribution
    prior_exponent = prior_alpha - 1
    # MAP calculation
    return (obs_count + prior_exponent) / (n + np.sum(prior_exponent))
