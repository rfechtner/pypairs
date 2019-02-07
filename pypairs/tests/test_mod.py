def test_version():
    import pypairs
    print(pypairs.__version__)

    try:
        from pypairs import preprocessing
        raise AssertionError()
    except NotImplementedError:
        pass

    try:
        from pypairs import plotting
        raise AssertionError()
    except NotImplementedError:
        pass
