.. role:: small

.. role:: smaller

.. role:: smallcaps

Release notes
-------------

Announcement
""""""""""""
.. note::

    Please only use :mod:`pypairs` >= 3.1.0


Versions
""""""""
Version 3.1.0, :small:`Apr 4, 2019`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
- New feature:
    - Multithreading now available for `pais.cyclone()`
- Minor changes and fixes:
    - `pais.sandbag()` now significally faster
    - `pais.sandbag()` more stable in terms of memory access

Version 3.0.1 - 3.0.13, :small:`Mar 13, 2019`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Various bug fixes, including:
    - Bioconda compability
    - Dataset loading
    - Cache file required
    - Cell Cycle specific scoring

Version 3.0.0, :small:`Jan 18, 2019 - Jan 31, 2019`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Complete restructuring of the package. Now fully compatiple with :mod:`scanpy` .
- Added:
    - This documentation
    - Default (oscope) dataset & marker pairs [Leng15]_
- Changed:
    - :func:`~pypairs.pairs.sandbag` and :func:`~pypairs.pairs.cyclone` now accept :class:`~anndata.AnnData`,
        :class:`~pandas.DataFrame` and :class:`~numpy.ndarray`
    - Multiprocessing now completely handled by :mod:`numba` .


Version 2.0.1 - 2.0.6, :small:`Nov 22, 2018`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Minor bug fixes and improvements.


Version 2.0.0, :small:`Aug 14, 2018`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Major restructuring of the package
- Improved parallel processing
- New features:
    - :func:`~pypairs.pairs.sandbag` and :func:`~pypairs.pairs.cyclone` can now deal with any number of classes to predict

Version 1.0.1 - 1.0.3, :small:`Jul 29, 2018`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Bug fixes and improvements. (Mostly bugs though)
- Added multi-core processing

Version 1.0.0, :small:`Mar 4, 2018`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Speed and performance improvements.

Version 0.1, :small:`Feb 22, 2018`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- Simple python reimplementation of the :smallcaps:`Pairs` algorithm.
- Included sandbag() and cyclone() algorithms
