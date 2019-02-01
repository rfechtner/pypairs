.. automodule:: pypairs

Documentation
-------------

To use PyPairs import the package as i.e. follows::

   import pypairs as pp


.. _Methods:

Sandbag
~~~~~~~

This function implements the classification step of the pair-based prediction method described by
Scialdone et al. (2015) [Scialdone15]_.

To illustrate, consider classification of cells into G1 phase.
Pairs of marker genes are identified with :func:`~pairs.sandbag`, where the expression of the first gene in the training
data is greater than the second in G1 phase but less than the second in all other phases.

.. autosummary::
   :toctree: .

   pairs.sandbag

Cyclone
~~~~~~~

For each cell, :func:`~pairs.cyclone` calculates the proportion of all marker pairs where the expression of the first gene is
greater than the second in the new data (pairs with the same expression are ignored). A high
proportion suggests that the cell is likely to belong to this category, as the expression ranking in the
new data is consistent with that in the training data. Proportions are not directly comparable between phases
due to the use of different sets of gene pairs for each phase. Instead, proportions are converted into scores
that account for the size and precision of the proportion estimate. The same process is repeated for
all phases, using the corresponding set of marker pairs in pairs.

.. autosummary::
   :toctree: .

   pairs.cyclone

While this method is described for cell cycle phase classification, any biological groupings can be
used here. However, for non-cell cycle phase groupings users should manually apply their own score thresholds
for assigning cells into specific groups.

.. _data:

Datasets
~~~~~~~~

.. autosummary::
   :toctree: .

   datasets.leng15
   datasets.default_cc_marker

Quality Assesment
~~~~~~~~~~~~~~~~~

.. autosummary::
   :toctree: .

   utils.evaluate_prediction

Utils
~~~~~

.. autosummary::
   :toctree: .

   utils.export_marker
   utils.load_marker


.. _settings:

Settings
~~~~~~~~

The default directories for saving figures and caching files.

==============================================  ===================================
:class:`settings.figdir`                        Directory for saving figures (default: ``'./figures/'``).
:class:`settings.cachedir`                      Directory for cache files (default: ``'./cache/'``).
==============================================  ===================================

The verbosity of logging output, where verbosity levels have the following
meaning: 0='error', 1='warning', 2='info', 3='hint'

==============================================  ===================================
:class:`settings.verbosity`                     Verbosity level (default: 1).
==============================================  ===================================

Print versions of packages that might influence numerical results.

.. autosummary::
   :toctree: .

   log.print_versions