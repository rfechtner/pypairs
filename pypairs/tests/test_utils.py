from pypairs import settings, utils, datasets
import numpy as np
import os

settings.verbosity = 4

def test_evaluate_prediction():
    pred = ['A', 'A', 'B', 'B', 'C', 'C', 'D', 'D']
    ref = ['A', 'A', 'B', 'B',  'C', 'C', 'D', 'D']

    e = utils.evaluate_prediction(prediction=pred, reference=ref)

    if not np.array_equal(np.array(e.values, dtype=float), np.ones(shape=(5, 4))):
        raise AssertionError()

    ref = ['D', 'D', 'B', 'B', 'C', 'C', 'A', 'A']

    e = utils.evaluate_prediction(prediction=pred, reference=ref)

    qual = np.array([
        [0, 0, 0, 0],
        [1, 1, 1, 1],
        [1, 1, 1, 1],
        [0, 0, 0, 0],
        [0.5, 0.5, 0.5, 0.5]], dtype=float
    )

    if not np.array_equal(e.values, qual):
        raise AssertionError()


def test_import_export_marker():
    m = datasets.default_cc_marker()

    fname = "marker_exp.json"

    fname_false = "/not/existing/path/marker_exp.json"

    utils.export_marker(m, fname_false, defaultpath=False)
    if os.path.isfile(fname_false):
        raise AssertionError()

    utils.export_marker(m, fname)
    if not os.path.isfile(settings.writedir + fname):
        raise AssertionError()

    m2 = utils.load_marker(fname_false, defaultpath=False)
    if m2 is not None:
        raise AssertionError()

    settings.verbosity = 4

    m2 = utils.load_marker(fname)
    if not utils.same_marker(m, m2):
        raise AssertionError()


def test_to_boolean_mask():
    arr = ['A','B','C','D','E']

    mask = utils.to_boolean_mask([1,2], arr)
    if not np.array_equal([False, True, True, False, False], mask):
        raise AssertionError()

    mask = utils.to_boolean_mask(['A', 'B'], arr)
    if not np.array_equal([True, True, False, False, False], mask):
        raise AssertionError()

    mask = utils.to_boolean_mask(None, arr)
    if not np.array_equal([True, True, True, True, True], mask):
        raise AssertionError()

    mask = utils.to_boolean_mask([], arr)
    if not np.array_equal([True, True, True, True, True], mask):
        raise AssertionError()

    mask = utils.to_boolean_mask([False, True, True, False, False], arr)
    if not np.array_equal([False, True, True, False, False], mask):
        raise AssertionError()

    mask = utils.to_boolean_mask(1, arr)
    if not np.array_equal([False, True, False, False, False], mask):
        raise AssertionError()


def test_filter_unexpressed_genes():
    data = datasets.leng15(mode='sorted', gene_sub=list(range(0, 10)), sample_sub=list(range(0, 10)))
    filtered, gene_names = utils.filter_unexpressed_genes(data.X, list(data.var_names))

    if not filtered.shape == (10, 8):
        raise AssertionError()
    if not len(gene_names) == 8:
        raise AssertionError()

def test_same_marker():
    a = {
        'a': [(1,2),(3,4),(5,6)],
        'b': [(7,8),(9,10)]
    }

    b = {
        'a': [(1,2),(3,4),(5,6)],
        'b': [(7,8),(9,10)]
    }

    if not utils.same_marker(a, b):
        raise AssertionError()

    c = {
        'a': [(1,2),(3,4)],
        'b': [(7,8),(9,10)]
    }

    if utils.same_marker(a, c):
        raise AssertionError()

    d = {
        'a': [(1,2),(0,0),(5,6)],
        'b': [(7,8),(9,10)]
    }

    if utils.same_marker(a, d):
        raise AssertionError()

    e = {
        'a': [(1, 2), (0, 0), (5, 6)],
        'c': [(7, 8), (9, 10)]
    }

    if utils.same_marker(a, e):
        raise AssertionError()
