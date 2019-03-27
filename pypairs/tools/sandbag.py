from typing import Union, Optional, Tuple, Mapping, Iterable, Any

from anndata import AnnData
from pandas import DataFrame
import numpy as np

from math import ceil
from collections import defaultdict
from numba import njit, prange

from pypairs import utils
from pypairs import settings
from pypairs import log as logg


def sandbag(
        data: Union[AnnData, DataFrame, np.ndarray, Iterable[Iterable[float]]],
        annotation: Optional[Mapping[str, Iterable[Union[str, int, bool]]]] = None,
        gene_names: Optional[Iterable[str]] = None,
        sample_names: Optional[Iterable[str]] = None,
        fraction: float = 0.65,
        filter_genes: Optional[Iterable[Union[str, int, bool]]] = None,
        filter_samples: Optional[Iterable[Union[str, int, bool]]] = None
) -> Mapping[str, Iterable[Tuple[str, str]]]:
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

    raw_data, gene_names, sample_names, category_names, categories = utils.parse_data_and_annotation(
        data, annotation, gene_names, sample_names
    )

    raw_data, gene_names, sample_names, categories = utils.filter_matrix(
        raw_data, gene_names, sample_names, categories, filter_genes, filter_samples
    )

    valid_cat_names = []
    valid_cats = []
    for i, cat in enumerate(categories):
        if sum(cat) > 0:
            valid_cat_names.append(category_names[i])
            valid_cats.append(cat)
    category_names = np.array(valid_cat_names)
    categories = np.array(valid_cats)

    logg.hint('sandbag running with fraction of {}'.format(fraction))

    # Define thresholds for each category based on fraction
    thresholds = np.apply_along_axis(sum, 1, categories)
    for i, t in enumerate(thresholds):
        thresholds[i] = ceil(t * fraction)

    # Dynamic jitting of function check_pairs based on platform compatibility for multiprocessing via numba
    check_pairs_decorated = utils.parallel_njit(check_pairs)

    # BUG(?): I have to pass a copy to get rid of all references to the pre_filtered raw_data object, otherwise numba
    # will fail with a lowering error
    raw_data = raw_data.copy()

    pairs = check_pairs_decorated(raw_data, categories, thresholds, len(gene_names))

    # Convert to easier to read dict and return
    marker_pairs_dict = defaultdict(list)

    for i in range(0, len(pairs) - 2, 3):
        cat = pairs[i]
        g1 = pairs[i+1]
        g2 = pairs[i+2]
        marker_pairs_dict[
            category_names[cat]
        ].append((gene_names[g1], gene_names[g2]))

    logg.info('finished', time=True)

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
        raw_data: np.ndarray,
        categories: np.ndarray,
        thresholds: np.ndarray,
        n_genes: int
) -> Iterable[int]:
    """Loops over all 2-tuple combinations of genes and checks if they fullfil the 'marker pair' criteria

    We return marker pairs as dict mapping category to list of 2-tuple: {'C': [(A,B), ...], ...}

    This function will be compiled via numba's jit decorator.
    """
    # Number of categories
    n_cats = len(categories)

    # Will hold the tuples with the pairs
    pairs = []

    # Iterate over all possible gene combinations
    for g1 in range(0, n_genes):
        # Parallelized if jitted
        for g2 in prange(g1+1, n_genes):
            # Gene counts
            x1 = raw_data[:, g1]
            x2 = raw_data[:, g2]

            # Subtract all gene counts of gene 2 from gene counts of gene 1
            diff = np.subtract(x1, x2)

            no_up, last_idx_up = valid_phases_up(diff, thresholds, categories)
            no_down, last_idx_down = valid_phases_down(diff, thresholds, categories)

            if no_up == 1:
                if no_down == n_cats - 1:
                    pairs.extend([last_idx_up, g1, g2])
            elif no_down == 1:
                if no_up == n_cats - 1:
                    pairs.extend([last_idx_down, g2, g1])

    return pairs

@njit()
def valid_phases_up(diff, thresholds, categories, min_diff=0):
    last_idx = -1
    count = 0

    for i in range(0, len(categories)):
        if count_up(diff[categories[i]], min_diff) >= thresholds[i]:
            count += 1
            last_idx = i

    return count, last_idx


@njit()
def valid_phases_down(diff, thresholds, categories, min_diff=0):
    last_idx = -1
    count = 0

    for i in range(0, len(categories)):
        if count_down(diff[categories[i]], min_diff) >= thresholds[i]:
            count += 1
            last_idx = i

    return count, last_idx


@njit()
def count_up(diff, min_diff=0):
    return len(np.where(diff > min_diff)[0])


@njit()
def count_down(diff, min_diff=0):
    return len(np.where(diff < min_diff)[0])
