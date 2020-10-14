import numpy as np
from scipy.special import gamma, gammaln


class DirichletCat:
    def __init__(self, alpha):
        """K-way categorical distribution with Dirichlet prior

        Args:
            alpha (numpy.ndarray): Alpha for Dirichlet prior (1, K)
        """
        self.ALPHA = alpha
        self.ALPHA_SUM = np.sum(alpha)

    def marginal(self, obs_count):
        """Marginal likelihood of data given prior

        Args:
            obs_count (numpy.ndarray): Count of observations (1, K)

        Returns:
            float: Marginal likelihood of data given prior
        """
        # number of observations
        n = np.sum(obs_count)
        # regularization term of marginal
        reg = gamma(self.ALPHA_SUM) / gamma(n + self.ALPHA_SUM)
        # marginal term for each category
        cat_marginal = gamma(obs_count + self.ALPHA) / gamma(self.ALPHA)
        return reg * np.prod(cat_marginal)

    def log_marginal(self, obs_count):
        """Log-marginal likelihood of data given prior

        Args:
            obs_count (numpy.ndarray): Count of observations (1, K)

        Returns:
            float: Log-marginal likelihood of data given prior
        """
        # number of observations
        n = np.sum(obs_count)
        # regularization term of log-marginal
        reg = gammaln(self.ALPHA_SUM) - gammaln(n + self.ALPHA_SUM)
        # log-marginal term for each category
        cat_log_marginal = gammaln(obs_count + self.ALPHA) - gammaln(self.ALPHA)
        return reg + np.sum(cat_log_marginal)

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
