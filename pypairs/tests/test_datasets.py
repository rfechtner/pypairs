from pypairs import datasets


def test_leng15():
    training_data = datasets.leng15(mode='all')

    if not training_data.shape == (460, 19084):
        raise AssertionError()

    training_data = datasets.leng15(mode='sorted')
    if not training_data.shape == (247, 19084):
        raise AssertionError()

    training_data = datasets.leng15(mode='unsorted')
    if not training_data.shape == (213, 19084):
        raise AssertionError()
