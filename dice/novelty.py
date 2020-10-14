import numpy as np
from sklearn.preprocessing import OneHotEncoder


class DiceNoveltyDetector:
    """Interface for determining whether die distribution is novel
    """
    def __init__(self, fair_model, cheat_model, obs_cat):
        """Initialize detector.

        Args:
            fair_model (DirichletCat): Model used for a fair die
            cheat_model (DirichletCat): Model used for an unfair die
            obs_cat (numpy.ndarray): Example of all observation category labels (K, 1)
        """
        self.fair_model = fair_model
        self.cheat_model = cheat_model
        self.obs_cat = obs_cat
        # encoder to transform (N, 1) observations to one hot (N, K)
        self.encoder = OneHotEncoder(sparse=False, dtype=np.int)
        self.encoder.fit(self.obs_cat)

    def count_obs(self, obs):
        """Count observation labels

        Args:
            obs (numpy.ndarray): Observation labels (N, 1)

        Returns:
            numpy.ndarray: Count of observation labels (1, K)
        """
        return np.sum(self.one_hot_obs(obs), axis=0, keepdims=True)

    def is_novelty(self, obs, thresh=10):
        """Determine if die distribution is novel from observations

        Args:
            obs (numpy.ndarray): Observation labels (N, 1)
            thresh (float, optional): Threshold for lowest Bayes factor considered novelty.
                                      Defaults to 10.

        Returns:
            bool: Whether die distribution is novel from observations
        """
        return self.log_bayes_factor(obs) > np.log(thresh)

    def log_bayes_factor(self, obs):
        """Log Bayes factor with favoring cheat model as positive Bayes factors

        Args:
            obs (numpy.ndarray): Observation labels (N, 1)

        Returns:
            float: Log Bayes factor of cheat model versus fair model
        """
        obs_count = self.count_obs(obs)
        log_bayes_factor = self.cheat_model.log_marginal(obs_count) - \
            self.fair_model.log_marginal(obs_count)
        return log_bayes_factor

    def map_distribution(self, obs):
        """MAP estimate of the die distribution with fair prior

        Args:
            obs (numpy.ndarray): Observation labels (N, 1)

        Returns:
            tuple: (observation categories (K, 1), MAP estimate (K, 1))
        """
        return (self.obs_cat, self.fair_model.map(self.count_obs(obs)))

    def one_hot_obs(self, obs):
        """Transform observation labels into one-hot encoding

        Args:
            obs (numpy.ndarray): Observation labels (N, 1)

        Returns:
            numpy.ndarray: One-hot encoding of observation labels (N, K)
        """
        return self.encoder.transform(obs)
