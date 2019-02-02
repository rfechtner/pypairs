Installation
------------

This package is hosted at `PyPi <https://pypi.org>`_ ( https://pypi.org/project/pypairs/ ) and can be installed on any system running
Python3 via pip with::

    pip install pypairs

Alternatively, pypairs can be installed using `Conda <https://conda.io/docs/>`_ (most easily obtained via the `Miniconda Python distribution <https://conda.io/miniconda.html>`_::

    conda install -c bioconda pypairs

Minimal Example
---------------

:ref:`data` provide a example scRNA dataset and default marker pairs for cell cycle prediction::

    from pypairs import pairs, datasets

    # Load samples from the oscope scRNA-Seq dataset with known cell cycle
    training_data = datasets.leng15(mode='sorted')

    # Run sandbag() to identify marker pairs
    marker_pairs = pairs.sandbag(training_data, fraction=0.6)

    # Load samples from the oscope scRNA-Seq dataset without known cell cycle
    testing_data = datasets.leng15(mode='unsorted')

    # Run cyclone() score and predict cell cycle classes
    result = pairs.cyclone(testing_data, marker_pairs)

    # Further downstream analysis
    print(result)

