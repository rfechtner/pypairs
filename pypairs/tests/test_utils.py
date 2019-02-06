from pypairs import settings
from pypairs import utils, datasets
import numpy as np
import os
import pytest

def test_evaluate_prediction():
    pred = ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D']
    ref = ['A', 'A', 'B', 'B',  'C', 'C', 'D', 'D']

    e = utils.evaluate_prediction(prediction=pred, reference=ref)

    assert np.array_equal(np.array(e.values, dtype=float), np.ones(shape=(5, 4)))

    ref = ['D', 'D', 'B', 'B', 'C', 'C', 'A', 'A']

    e = utils.evaluate_prediction(prediction=pred, reference=ref)

    qual = np.array([
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [0.5, 0.5, 0.5, 0.5]], dtype=float
    )

    assert np.array_equal(e.values, qual)

def test_import_export_marker():
    m = datasets.default_cc_marker()

    fname = "marker_exp.json"
    settings.writedir = "./"

    fname_false =  "/not/existing/path/marker_exp.json"

    utils.export_marker(m, fname_false, defaultpath=False)
    assert not os.path.isfile(fname_false)

    utils.export_marker(m, fname)
    assert os.path.isfile(settings.writedir + fname)

    m2 = utils.load_marker(fname_false, defaultpath=False)
    assert m2 is None

    settings.verbosity = 4

    m2 = utils.load_marker(fname)
    assert utils.same_marker(m, m2)
