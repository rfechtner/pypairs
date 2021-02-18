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

A supervised maschine learning algorithm aiming to classify single cells based on their transcriptomic signal.
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

Provided a scRNA count file (csv, columns = samples, rows = genes) one can use `pairs.cyclone()` in order to predict the i.e. cell cycle phase based on a pre-trained marker set, described here: `Default Cell Cycle Marker <https://pypairs.readthedocs.io/en/latest/pypairs.datasets.default_cc_marker.html#pypairs.datasets.default_cc_marker>`_  

.. code:: python

    from pypairs import pairs
    import pandas as pd
    
    # Load the CSV, i.e. using pandas's read_csv() function.
    scRNA_data = pd.read_csv('scRNA_data.csv', sep=',', index_col=0)
    
    # Run cyclone(), that uses datasets.default_cc_marker() as default if no `marker_pairs` are passed 
    prediction = pairs.cyclone(scRNA_data)
    
    # Shows table with scores for each category for each sample. 
    # Per default cc_prediction is only available when using cell cycle classes
    print(prediction)
    
Alternatively one can use any other set of categories and marker_pairs by running `pairs.sandbag()` in order to generate a custom set of marker pairs. All thats needed is a scRNA training dataset along with an annotation to learn from. 

Assuming you have two scRNA count files (training and testing, csv, columns = samples, rows = genes) and one annotation file (for training, csv, no header, two rows: "gene, class") a minimal example would look like this:

.. code:: python

    from pypairs import pairs
    import pandas as pd

    # Load training data and annotation
    training_data = pd.read_csv('training_scRNA_data.csv', sep=',', index_col=0)
    annotation = pd.read_csv('annotation.csv', sep=',', index_col=0)

    # Run sandbag() to identify marker pairs
    marker_pairs = pairs.sandbag(training_data, annotation)

    # Load samples from the oscope scRNA-Seq dataset without known cell cycle
    testing_data = pd.read_csv('scRNA_data.csv', sep=',', index_col=0)

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
