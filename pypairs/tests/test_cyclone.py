from pypairs import pairs, datasets, utils, settings
import numpy as np

settings.verbosity = 4

ref_prediction = list(np.repeat("G2M", 76)) + list(np.repeat("S", 80)) + list(np.repeat("G1", 91))


def test_cyclone_train_on_train():
    training_data = datasets.leng15(mode='sorted')

    scores = pairs.cyclone(training_data, datasets.default_cc_marker(), iterations=1000, min_iter=10, min_pairs=1)

    test_quality = utils.evaluate_prediction(prediction=scores['max_class'], reference=ref_prediction)

    if not np.allclose(np.array(test_quality.values, dtype=float), np.ones(shape=(4, 4)), atol=0.1):
        raise AssertionError()

# TBFL
"""
def test_cyclone_train_on_test():
    testing_data = datasets.leng15(mode='unsorted')

    scores = pairs.cyclone(testing_data, datasets.default_cc_marker(), iterations=500, min_iter=10, min_pairs=1)
    test_quality = utils.evaluate_prediction(prediction=scores['max_class'], reference=ref_prediction)

    if not np.allclose(np.array(test_quality.values, dtype=float), np.ones(shape=(4, 4)), atol=0.2):
        raise AssertionError()
"""
