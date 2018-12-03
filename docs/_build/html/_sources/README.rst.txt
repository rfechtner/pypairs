|PyPI| |Docs| |Build Status| |bioconda|

.. |PyPI| image:: https://img.shields.io/pypi/v/pypairs.svg
    :target: https://pypi.org/project/pypairs
.. |Docs| image:: https://readthedocs.org/projects/pypairs/badge/?version=latest
   :target: https://pypairs.readthedocs.io
.. |Build Status| image:: https://travis-ci.org/rfechtner/pypairs.svg?branch=master
   :target: https://travis-ci.org/rfechtner/pypairs
.. |bioconda| image:: https://img.shields.io/badge/install%20with-bioconda-brightgreen.svg?style=flat-square
   :target: http://bioconda.github.io/recipes/pypairs/README.html


.. role:: smallcaps

PyPairs - A python scRNA-Seq classifier
=======================================

This is a python-reimplementation of the :smallcaps:`Pairs` algorithm as described by A. Scialdone et. al. (2015).
Original Paper available under: <https://doi.org/10.1016/j.ymeth.2015.06.021>

A supervided maschine learning algorithm aiming to classify single cells based on their transcriptomic signal.
Initially created to predict cell cycle phase from scRNA-Seq data, this algorithm can be used for various applications.

Build to be fully compatible with `Scanpy <https://scanpy.readthedocs.io/en/stable/>`_ [Wolf18]_.

Code available on `GitHub <https://github.com/rfechter/pypairs>`_.

Core Dependencies
-----------------

- `Numpy <http://www.numpy.org/>`_
- `Numba <https://numba.pydata.org/>`_
- `Pandas <https://pandas.pydata.org/>`_
- `AnnData <https://github.com/theislab/anndata>`_

Authors
-------

- **Antonio Scialdone** - *original algorithm*
- **Ron Fechtner** - *implementation and extension in Python*
