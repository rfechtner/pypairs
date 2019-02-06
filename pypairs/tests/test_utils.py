from pypairs import settings, utils, datasets
import numpy as np
import os

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

    fname_false = "/not/existing/path/marker_exp.json"

    utils.export_marker(m, fname_false, defaultpath=False)
    assert not os.path.isfile(fname_false)

    utils.export_marker(m, fname)
    assert os.path.isfile(settings.writedir + fname)

    m2 = utils.load_marker(fname_false, defaultpath=False)
    assert m2 is None

    settings.verbosity = 4

    m2 = utils.load_marker(fname)
    assert utils.same_marker(m, m2)


def test_to_boolean_mask():
    arr = ['A','B','C','D','E']

    mask = utils.to_boolean_mask([1,2], arr)
    assert np.array_equal([False, True, True, False, False], mask)

    mask = utils.to_boolean_mask(['A', 'B'], arr)
    assert np.array_equal([True, True, False, False, False], mask)

    mask = utils.to_boolean_mask(None, arr)
    assert np.array_equal([True, True, True, True, True], mask)

    mask = utils.to_boolean_mask([], arr)
    assert np.array_equal([True, True, True, True, True], mask)

    mask = utils.to_boolean_mask([False, True, True, False, False], arr)
    assert np.array_equal([False, True, True, False, False], mask)

    mask = utils.to_boolean_mask(1, arr)
    assert np.array_equal([False, True, False, False, False], mask)


def test_filter_unexpressed_genes():
    data = datasets.leng15(mode='sorted', gene_sub=list(range(0, 10)), sample_sub=list(range(0, 10)))
    filtered, gene_names = utils.filter_unexpressed_genes(data.X, list(data.var_names))

    assert filtered.shape == (10, 8)
    assert len(gene_names) == 8

def test_same_marker():
    a = {
        'a': [(1,2),(3,4),(5,6)],
        'b': [(7,8),(9,10)]
    }

    b = {
        'a': [(1,2),(3,4),(5,6)],
        'b': [(7,8),(9,10)]
    }

    assert utils.same_marker(a, b)

    c = {
        'a': [(1,2),(3,4)],
        'b': [(7,8),(9,10)]
    }

    assert not utils.same_marker(a, c)

    d = {
        'a': [(1,2),(0,0),(5,6)],
        'b': [(7,8),(9,10)]
    }

    assert not utils.same_marker(a, d)

    e = {
        'a': [(1, 2), (0, 0), (5, 6)],
        'c': [(7, 8), (9, 10)]
    }

    assert not utils.same_marker(a, e)
