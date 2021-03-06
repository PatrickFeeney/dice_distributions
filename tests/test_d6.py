import numpy as np
import pytest

import dice.d6 as d6


# encoder tests
def test_encoder_expected():
    """Test encoder with all expected inputs
    """
    assert np.all(d6.nov_detect.one_hot_obs(d6.obs_cat) == np.identity(6))


def test_encoder_unexpected():
    """Test encoder with an unexpected value
    """
    with pytest.raises(ValueError):
        d6.nov_detect.one_hot_obs(np.asarray([[7]]))


def test_is_valid_obs():
    assert not d6.nov_detect.is_valid_obs(np.asarray([[7]]))


def test_update_model():
    new_nov = d6.nov_detect.update_model(np.asarray([[7]]))
    assert np.all(new_nov.obs_cat == np.append(d6.nov_detect.obs_cat, [[7]], axis=0))
    assert np.all(new_nov.fair_model.ALPHA ==
                  np.append(d6.nov_detect.fair_model.ALPHA,
                            [[d6.nov_detect.NEW_ALPHA_FAIR]],
                            axis=1))
    assert np.all(new_nov.cheat_model.ALPHA ==
                  np.append(d6.nov_detect.cheat_model.ALPHA,
                            [[d6.nov_detect.NEW_ALPHA_CHEAT]],
                            axis=1))


# modeling tests
def test_map_prob():
    """Test map_prob with one observation of each category
    """
    assert np.all(np.isclose(d6.fair_model.map(d6.nov_detect.count_obs(d6.obs_cat)),
                             [[.166666666] * 6]))
    obs_cat, map_dist = d6.nov_detect.map_distribution(d6.obs_cat)
    assert np.all(obs_cat == d6.obs_cat)
    assert np.all(np.isclose(map_dist, [[.166666666] * 6]))


def test_marginal():
    """Test marginal likelihood with one observation of each category
    """
    assert np.isclose(d6.fair_model.marginal(d6.nov_detect.count_obs(d6.obs_cat)),
                      1.8950327390001377e-05)


def test_log_marginal():
    """Test log-marginal likelihood with one observation of each category
    """
    assert np.isclose(d6.fair_model.log_marginal(d6.nov_detect.count_obs(d6.obs_cat)),
                      -10.873689350067835)


def test_log_bayes_factor():
    """Test log Bayes factor with one observation of each category
    """
    assert np.isclose(d6.nov_detect.log_bayes_factor(d6.obs_cat), -1.8411267530240067)
