import numpy as np
import pytest

import dice_model.d6 as d6


def test_encoder_expected():
    """Test encoder with all expected inputs
    """
    assert np.all(d6.encoder.transform(d6.obs_cat) == np.identity(6))


def test_encoder_unexpected():
    """Test encoder with an unexpected value
    """
    with pytest.raises(ValueError):
        d6.encoder.transform(np.asarray([[7]]))
