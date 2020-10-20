import numpy as np
from sklearn.preprocessing import OneHotEncoder

from dice.model import DirichletCat


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
        # constants for expanding model for new values
        self.NEW_ALPHA_FAIR = 0.
        self.NEW_ALPHA_CHEAT = 1.
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

    def is_valid_obs(self, obs):
        """Whether observations are valid for current model

        Args:
            obs (numpy.ndarray): Observation labels (N, 1)

        Returns:
            bool: Whether observations are valid for current model
        """
        try:
            self.one_hot_obs(obs)
            return True
        except ValueError:
            return False
        except Exception as err:
            raise err

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

    def map_distribution(self, obs, fair=True):
        """MAP estimate of the die distribution with fair prior

        Args:
            obs (numpy.ndarray): Observation labels (N, 1)

        Returns:
            tuple: (observation categories (K, 1), MAP estimate (K, 1))
        """
        if fair:
            map_dist = self.fair_model.map(self.count_obs(obs))
        else:
            map_dist = self.cheat_model.map(self.count_obs(obs))
        return (self.obs_cat, map_dist)

    def one_hot_obs(self, obs):
        """Transform observation labels into one-hot encoding

        Args:
            obs (numpy.ndarray): Observation labels (N, 1)

        Returns:
            numpy.ndarray: One-hot encoding of observation labels (N, K)
        """
        return self.encoder.transform(obs)

    def update_model(self, obs):
        """Update the model to handle invalid observations

        Args:
            obs (numpy.ndarray): Observation labels (N, 1)

        Returns:
            DiceNoveltyDetector: self if valid observations, otherwise new DiceNoveltyDetector that
                                 handles new categories in obs
        """
        if self.is_valid_obs(obs):
            return self
        else:
            # update the known categories (dice values)
            new_cat = self.obs_cat
            unique_cat = np.unique(obs)
            for cat in unique_cat:
                if cat not in new_cat:
                    new_cat = np.append(new_cat, [[cat]], axis=0)
            # update the models
            fill_num = len(new_cat) - len(self.obs_cat)
            new_fair = DirichletCat(np.append(self.fair_model.ALPHA,
                                              [[self.NEW_ALPHA_FAIR] * fill_num],
                                              axis=1))
            new_cheat = DirichletCat(np.append(self.cheat_model.ALPHA,
                                               [[self.NEW_ALPHA_CHEAT] * fill_num],
                                               axis=1))
            return DiceNoveltyDetector(new_fair, new_cheat, new_cat)
