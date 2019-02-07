| |PyPI| |bioconda| |Build Status| |Docs|  
| |coverage| |codacy|

.. |PyPI| image:: https://img.shields.io/pypi/v/pypairs.svg
    :target: https://pypi.org/project/pypairs
.. |Docs| image:: https://readthedocs.org/projects/pypairs/badge/?version=latest
   :target: https://pypairs.readthedocs.io
.. |Build Status| image:: https://travis-ci.org/rfechtner/pypairs.svg?branch=master
   :target: https://travis-ci.org/rfechtner/pypairs
.. |bioconda| image:: https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat-square
   :target: http://bioconda.github.io/recipes/pypairs/README.html
.. |coverage| image:: https://codecov.io/gh/rfechtner/pypairs/branch/master/graph/badge.svg
  :target: https://codecov.io/gh/rfechtner/pypairs
.. |codacy| image:: https://api.codacy.com/project/badge/Grade/5af00781e82d46e5a3f55a36f55728d7
  :target: https://www.codacy.com/app/rfechtner/pypairs

PyPairs - A python scRNA-Seq classifier
=======================================

This is a python-reimplementation of the Pairs algorithm as described by A. Scialdone et. al. (2015).
Original Paper available under: https://doi.org/10.1016/j.ymeth.2015.06.021

A supervided maschine learning algorithm aiming to classify single cells based on their transcriptomic signal.
Initially created to predict cell cycle phase from scRNA-Seq data, this algorithm can be used for various applications.

Build to be fully compatible with `Scanpy <https://scanpy.readthedocs.io/en/stable/>`_. For more details see the
`full documentation <https://pypairs.readthedocs.io/en/latest/>`_.

Getting Started
---------------

Note: Version 3 still under development.
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


Installation
~~~~~~~~~~~~

This package is hosted at PyPi ( https://pypi.org/project/pypairs/ ) and can be installed on any system running
Python3 via pip with::

    pip install pypairs

Alternatively, pypairs can be installed using `Conda <https://conda.io/docs/>`_ (most easily obtained via the
`Miniconda Python distribution <https://conda.io/miniconda.html>`_::

    conda install -c bioconda pypairs

Minimal Example
~~~~~~~~~~~~~~~

Assuming you have two scRNA count files (csv, columns = samples, rows = genes) and one annotation file (csv, no header,
two rows: "gene, class") a minimal example would look like this

.. code:: python

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

Core Dependencies
-----------------

- `Numpy <http://www.numpy.org/>`_
- `Numba <https://numba.pydata.org/>`_
- `Pandas <https://pandas.pydata.org/>`_
- `AnnData <https://github.com/theislab/anndata>`_

Authors
-------

* **Antonio Scialdone** - *original algorithm*
* **Ron Fechtner** - *implementation and extension in Python*

License
-------

This project is licensed under the BSD 3-Clause License - see the `LICENSE <LICENSE>`_ file for details
