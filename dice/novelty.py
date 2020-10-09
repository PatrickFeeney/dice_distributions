import numpy as np
from sklearn.preprocessing import OneHotEncoder


class DiceNoveltyDetector:
    """Interface for determining whether die distribution is novel
    """
    def __init__(self, model, obs_cat):
        """Initialize detector.

        Args:
            model (DirichletCat): Model used for probability calculations
            obs_cat (numpy.ndarray): Example of all observation category labels (K, 1)
        """
        self.model = model
        # encoder to transform (N, 1) observations to one hot (N, K)
        self.encoder = OneHotEncoder(sparse=False, dtype=np.int)
        self.encoder.fit(obs_cat)

    def is_novelty(self, obs, thresh=.1):
        """Determine if die distribution is novel from observations

        Args:
            obs (numpy.ndarray): Observation labels (N, 1)
            thresh (float, optional): Threshold for lowest likelihood not considered novelty.
                                      Defaults to .1.

        Returns:
            bool: Whether die distribution is novel from observations
        """
        return self.model.likelihood(self.count_obs(obs)) < thresh

    def count_obs(self, obs):
        """Count observation labels

        Args:
            obs (numpy.ndarray): Observation labels (N, 1)

        Returns:
            numpy.ndarray: Count of observation labels (1, K)
        """
        return np.sum(self.one_hot_obs(obs), axis=0, keepdims=True)

    def one_hot_obs(self, obs):
        """Transform observation labels into one-hot encoding

        Args:
            obs (numpy.ndarray): Observation labels (N, 1)

        Returns:
            numpy.ndarray: One-hot encoding of observation labels (N, K)
        """
        return self.encoder.transform(obs)
