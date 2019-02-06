import pytest


def test_version():
    import pypairs
    print(pypairs.__version__)

    with pytest.raises(NotImplementedError):
        from pypairs import preprocessing

    with pytest.raises(NotImplementedError):
        from pypairs import plotting
