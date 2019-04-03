from pypairs import pairs, datasets, utils, settings
import numpy as np

settings.verbosity = 4

ref_prediction = list(np.repeat("G2M", 76)) + list(np.repeat("S", 80)) + list(np.repeat("G1", 91))


def test_cyclone_train_on_train():
    print("")
    print("")

    print("## Testing correctness of cyclone()")

    print("")
    print("# Testing algorithm on training data")
    print("")

    settings.enable_fastmath = True
    settings.verbosity = 4

    training_data = datasets.leng15(mode='sorted')

    scores = pairs.cyclone(training_data, datasets.default_cc_marker(), iterations=1000, min_iter=10, min_pairs=1)

    print(scores)

    test_quality = utils.evaluate_prediction(prediction=scores['max_class'], reference=ref_prediction)

    print(test_quality)

    assert np.allclose(np.array(test_quality.values, dtype=float), np.ones(shape=(4, 4)), atol=0.1)


def test_cyclone_speed():
    print("")
    print("")

    print("## Testing speed of cyclone()")

    print("")
    print("# Testing algorithm on minimal data, unjitted and single core")
    print("")

    marker = datasets.default_cc_marker()
    for p, m in marker.items():
        marker[p] = m[:500]

    settings.enable_jit = False
    settings.n_jobs = 1

    utils.benchmark_test(pairs.cyclone, {'data': datasets.leng15(mode='sorted'), 'marker_pairs': marker})

    print("")
    print("# Testing algorithm on minimal data, jitted and single core")
    print("")

    settings.enable_jit = True
    settings.n_jobs = 1

    utils.benchmark_test(pairs.cyclone, {'data': datasets.leng15(mode='sorted'), 'marker_pairs': marker})

    print("")
    print("# Testing algorithm on minimal data, jitted and multi core")
    print("")

    settings.enable_jit = True
    settings.n_jobs = 4

    utils.benchmark_test(pairs.cyclone, {'data': datasets.leng15(mode='sorted'), 'marker_pairs': marker})
