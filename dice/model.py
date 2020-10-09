import numpy as np
from scipy.special import gamma


class DirichletCat:
    def __init__(self, alpha):
        """K-way categorical distribution with Dirichlet prior

        Args:
            alpha (numpy.ndarray): Alpha for Dirichlet prior (1, K)
        """
        self.ALPHA = alpha
        self.ALPHA_SUM = np.sum(alpha)

    def likelihood(self, obs_count):
        """Likelihood of data given prior

        Args:
            obs_count (numpy.ndarray): Count of observations (1, K)

        Returns:
            float: Likelihood of data given prior
        """
        # number of observations
        n = np.sum(obs_count)
        # regularization term of likelihood
        reg = gamma(self.ALPHA_SUM) / gamma(n + self.ALPHA_SUM)
        # likelihood term for each category
        cat_likelihood = gamma(obs_count + self.ALPHA) / gamma(self.ALPHA)
        return reg * np.prod(cat_likelihood)

    def map(self, obs_count):
        """MAP estimate of the K-way categorical distribution

        Args:
            obs_count (numpy.ndarray): Count of observations (1, K)

        Returns:
            numpy.ndarray: MAP estimate of the K-way categorical distribution (1, K)
        """
        # number of observations
        n = np.sum(obs_count)
        # how alpha is used as exponent in Dirichlet distribution prior
        prior_exponent = self.ALPHA - 1
        # MAP calculation
        return (obs_count + prior_exponent) / (n + np.sum(prior_exponent))
