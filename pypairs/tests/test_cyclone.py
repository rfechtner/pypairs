from pypairs import pairs
from pypairs import datasets
from pypairs import utils

import numpy as np


ref_prediction = list(np.repeat("G2M", 76)) + list(np.repeat("S", 80)) + list(np.repeat("G1", 91))


def test_cyclone():
    training_data = datasets.leng15(mode='sorted')

    scores = pairs.cyclone(training_data, datasets.default_cc_marker(), iterations=1000, min_iter=10, min_pairs=1)

    test_quality = utils.evaluate_prediction(prediction=scores['max_class'], reference=ref_prediction)

    assert np.allclose(np.array(test_quality.values, dtype=float), np.ones(shape=(4, 4)), atol=0.1)
