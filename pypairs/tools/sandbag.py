from collections import defaultdict
from math import ceil
from typing import Union, Optional, Tuple, Mapping, Collection

import numpy as np
from anndata import AnnData
from numba import prange
from pandas import DataFrame

from pypairs import log as logg
from pypairs import settings
from pypairs import utils


def sandbag(
        data: Union[AnnData, DataFrame, np.ndarray, Collection[Collection[float]]],
        annotation: Optional[Mapping[str, Collection[Union[str, int, bool]]]] = None,
        gene_names: Optional[Collection[str]] = None,
        sample_names: Optional[Collection[str]] = None,
        fraction: float = 0.65,
        filter_genes: Optional[Collection[Union[str, int, bool]]] = None,
        filter_samples: Optional[Collection[Union[str, int, bool]]] = None
) -> Mapping[str, Collection[Tuple[str, str]]]:
    """
    Calculate 'marker pairs' from a genecount matrix. Cells x Genes.

    A Pair of genes `(g1, g2)` is considered a marker for a category if its expression changes from `g1 > g2`
    in one category to `g1 < g2` in all other categories, for at least a ``fraction`` of cells in this category.

    ``data`` can be of type :class:`~anndata.AnnData`, :class:`~pandas.DataFrame` or :class:`~numpy.ndarray` and should
    contain the raw or normalized gene counts of shape ``n_obs`` * ``n_vars``. Rows correspond to cells and columns to
    genes.

        *
            If data is :class:`~anndata.AnnData` object, the category for each sample should be in in
            ``data.vars['category']``, gene names in ``data.var_names`` and sample names in ``data.obs_names``.

        *
            If data is :class:`~pandas.DataFrame` object, gene names can be in ``df.columns`` or passed via
            ``gene_names`` and sample names in ``df.index`` or passed via ``sample_names``. The category for each
            sample must be passed via ``annotation``.

            *
                ``annotation`` must be in form of `{'category1': ['sample_1','sample_2',...], ...}`. List of samples
                for indexing can be integer, str or a boolean mask of ``len(sample_names)``.

        *
            If data :class:`~numpy.ndarray`, all information must be passed via ``annotation``, ``gene_names`` and
            ``sample_names`` parameters.

    Marker pairs are returned as a mapping from category to list of 2-tuple Genes: `{'category': [(Gene_1,Gene_2), ...],
    ...}`

    Parameters
    ----------

    data
        The (annotated) data matrix of shape ``n_obs`` * ``n_vars``.
        Rows correspond to cells and columns to genes.
    annotation
        Mapping from category to genes. If ``data`` is not :class:`~anndata.AnnData`, this is required.
        List of genes can be index, names or logical mask.
    gene_names
        Names for genes, must be same length as ``n_vars``. If ``data`` is not :class:`~anndata.AnnData`, this is
        required.
    sample_names
        Names for samples, must be same length as ``n_obs``. If ``data`` is not :class:`~anndata.AnnData`, this is
        required.
    fraction
        Fraction of cells per category where marker criteria must be satisfied. Default: 0.65
    filter_genes
        A list of genes to keep. If not ``None`` all genes not in this list will be removed.
        List can be index, names or logical mask.
    filter_samples
         A list of samples to keep. If not ``None`` all samples not in this list will be removed.
         List can be index, names or logical mask.

    Returns
    -------

    marker_pairs_dict
        A dict mapping from str to a list of 2-tuple, where the key is the category and the list contains the marker
        pairs: `{'Category_1': [(Gene_1, Gene_2), ...], ...}`.

    Examples
    --------
        To generate marker pairs for a different fraction
        than the default (0.65) based on the bundled ``oscope``-dataset [Leng15]_ run::

            from pypairs import pairs, datasets

            adata = datasets.leng15()
            marker_pairs = pairs.sandbag(adata, fraction=0.5)

    """
    logg.info('identifying marker pairs with sandbag', r=True)
    logg.hint('sandbag running with fraction of {}'.format(fraction))

    # AnnData or DataFrame or ndarray -> ndarray + meta information
    data, gene_names, sample_names, category_names, categories = utils.parse_data_and_annotation(
        data, annotation, gene_names, sample_names
    )

    # Get filter mask based on filter selection, and filter out unexpressed genes
    gene_mask, sample_mask = utils.get_filter_masks(
        data, gene_names, sample_names, categories, filter_genes, filter_samples
    )

    # Apply mask to gene names and categories, samples are not needed
    gene_names = np.array(gene_names)[gene_mask]
    categories = categories[:, sample_mask]

    # Remove empty categories
    categories, category_names = remove_empty_categories(categories, category_names)

    # Cells in category * fraction
    thresholds = calc_thresholds(categories, fraction)

    # Turn array of boolean into array of index
    cats = np.where(categories.T == True)[1]

    # Decorate check_pairs according to settings and platform
    check_pairs_decorated = utils.parallel_njit(check_pairs)
    pairs = check_pairs_decorated(data[sample_mask][:, gene_mask], cats, thresholds)

    # Convert to easier to read dict and return
    marker_pos = np.where(pairs != -1)

    marker_pairs_dict = defaultdict(list)

    for i in range(0, len(marker_pos[0])):
        g1 = marker_pos[0][i]
        g2 = marker_pos[1][i]
        cat = pairs[g1, g2]

        marker_pairs_dict[
            category_names[cat]
        ].append((gene_names[g1], gene_names[g2]))

    logg.info('finished', time=True)

    # Print count of marker pairs per category
    if settings.verbosity > 2:
        count_total = 0
        count_str = []
        for m, p in marker_pairs_dict.items():
            c = len(p)
            count_total += c
            count_str.append("\t{}: {}".format(m, c))

        logg.hint("found {} marker pairs".format(count_total))
        for s in count_str:
            logg.hint(s)

    return dict(marker_pairs_dict)


def check_pairs(
        raw_data_in: np.ndarray,
        cats: np.ndarray,
        thresholds: np.array
) -> Collection[int]:
    raw_data = np.ascontiguousarray(raw_data_in.T)
    result = np.full((raw_data.shape[0], raw_data.shape[0]), - 1)

    for g1 in prange(0, raw_data.shape[0]):
        for g2 in range(g1 + 1, raw_data.shape[0]):
            valid_phase_up = 0
            valid_phase_up_idx = -1
            valid_phase_down = 0
            valid_phase_down_idx = -1

            num_pos = np.zeros(thresholds.shape[0])
            num_neg = np.zeros(thresholds.shape[0])

            for i in range(0, raw_data.shape[1]):
                if raw_data[g1, i] > raw_data[g2, i]:
                    num_pos[cats[i]] += 1
                elif raw_data[g1, i] < raw_data[g2, i]:
                    num_neg[cats[i]] += 1

            for i in range(0, thresholds.shape[0]):
                if num_pos[i] >= thresholds[i]:
                    valid_phase_up += 1
                    valid_phase_up_idx = i
                if num_neg[i] >= thresholds[i]:
                    valid_phase_down += 1
                    valid_phase_down_idx = i

            if valid_phase_up == 1:
                if valid_phase_down == thresholds.shape[0] - 1:
                    result[g1, g2] = valid_phase_up_idx
            elif valid_phase_down == 1:
                if valid_phase_up == thresholds.shape[0] - 1:
                    result[g2, g1] = valid_phase_down_idx

    return result


def remove_empty_categories(categories, category_names):
    valid_cat_names = []
    valid_cats = []
    for i, cat in enumerate(categories):
        if sum(cat) > 0:
            valid_cat_names.append(category_names[i])
            valid_cats.append(cat)
    return np.array(valid_cats), np.array(valid_cat_names)


def calc_thresholds(categories, fraction):
    thresholds = np.apply_along_axis(sum, 1, categories)
    for i, t in enumerate(thresholds):
        thresholds[i] = ceil(t * fraction)

    return thresholds
