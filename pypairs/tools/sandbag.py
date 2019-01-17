from typing import Union, Optional, Tuple, Mapping, Iterable

from anndata import AnnData
from pandas import DataFrame
import numpy as np

from math import ceil
from collections import defaultdict
from numba import njit, prange

from pypairs import utils
from pypairs import settings
from pypairs import log as logg

# TODO: Add optional filter not highly variable genes

def sandbag(
        data: Union[AnnData, DataFrame, np.ndarray, Iterable[Iterable[float]]],
        annotation: Optional[Mapping[str, Iterable[Union[str, int, bool]]]] = None,
        gene_names: Optional[Iterable[str]] = None,
        sample_names: Optional[Iterable[str]] = None,
        filter_genes: Optional[Iterable[Union[str, int, bool]]] = None,
        filter_samples: Optional[Iterable[Union[str, int, bool]]] = None,
        fraction: float = 0.65
) -> Mapping[str, Iterable[Tuple[str, str]]]:
    """
    Calculate 'marker pairs' from a trainings set.

    ``data`` can be of type :class:`~anndata.AnnData`, :class:`~pandas.DataFrame` or :class:`~numpy.ndarray` and should
    contain the raw or normalized gene counts of shape ``n_obs`` * ``n_vars``. Rows correspond to cells and columns to
    genes.

    If a :class:`~anndata.AnnData` object is passed, the category for each sample should be in in
    ``data.vars['category']``, gene names in ``data.var_names`` and sample names in ``data.obs_names``.
    If not this information must be passed via ``annotation``, ``gene_names`` and ``sample_names``.

    ``annotation`` must be in form of `{'category1': ['sample_1','sample_2',...], ...}`.
    List of samples for indexing can be integer, str or a boolean mask of ``len(sample_names)``.

    A Pair of genes `(g1, g2)` is considered a marker for a category if its expression changes from `g1 > g2`
    in one category to `g1 < g2` in all other categories, for at least a ``fraction`` of cells in this category.

    Marker pairs are returned as a mapping from category to list of 2-tuple Genes: `{'category': [(Gene_1,Gene_2), ...],
    ...}`

    Parameters
    ----------

    data
        The (annotated) data matrix of shape ``n_obs`` * ``n_vars``.
        Rows correspond to cells and columns to genes.
    annotation
        Mapping from category to genes if ``data`` is not :class:`~anndata.AnnData`.
        List of genes can be index, names or logical mask.
    gene_names
        Names for genes, must be same length as ``n_vars``.
    sample_names
        Names for samples, must be same length as ``n_obs``.
    fraction
        Fraction of cells per category where marker criteria must be satisfied.

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
            print(marker_pairs)

    """
    logg.info('identifying marker pairs with sandbag', r=True)

    raw_data, gene_names, sample_names, category_names, categories = utils.parse_data_and_annotation(
        data, annotation, gene_names, sample_names
    )

    # Check that categories dont overlapp. TBD if needed
    """
    for i in range(len(categories)):
        for j in range(i+1, len(categories)):
            overlapp = np.logical_and(categories[i], categories[j])
            if overlapp.sum() > 0:
                raise ValueError("A observation can not be in multiple categories")
    """

    raw_data, gene_names, sample_names, categories = utils.filter_matrix(
        raw_data, gene_names, sample_names, categories, filter_genes, filter_samples
    )

    logg.hint('sandbag running with fraction of {}'.format(fraction))

    # Define thresholds for each category based on fraction
    thresholds = np.apply_along_axis(sum, 1, categories)
    for i, t in enumerate(thresholds):
        thresholds[i] = ceil(t * fraction)

    # Dynamic jitting of function check_pairs based on platform compability for multiprocessing via numba
    check_pairs_decorated = utils.parallel_njit(check_pairs)

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

    """
    for i, pair in enumerate(pairs):
        try:
            marker_pairs_dict[category_names[mapping[i]]].append(
                (gene_names[pair[0]], gene_names[pair[1]])
            )
        except:

            try:
                m = mapping[i]
            except:
                m = "N/A"
            print("failed for i.e.: i = {} for mapping = {} pair ({},{}). With {} gene_names and {} category_names and {} mappings".format(i, m, pair[0], pair[1], len(gene_names), len(category_names), len(mapping)))
    """
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
        raw_data: Iterable[Iterable[int]],
        categories: Iterable[Iterable[bool]],
        thresholds: Iterable[int],
        n_genes: int
) -> Iterable[int]:
    #Tuple[Iterable[Tuple[int, int]], Iterable[int]]:
    """Loops over all 2-tuple combinations of genes and checks if they fullfil the 'marker pair' criteria

    We return marker pairs as dict mapping category to list of 2-tuple: {'C': [(A,B), ...], ...}

    This function will be compiled via numba's jit decorator.

    Parameters
    ----------

    raw_data
        The data matrix of shape ``n_obs`` * ``n_vars`` of class :class:`~numpy.ndarray`.
        Axis 0 correspond to cells and Axis 1 to genes.
    categories
        List of logical mask for each category.
    thresholds
        List of threshold for each category.
    n_genes
        Number of genes / ``n_vars``.

    Returns
    -------

    A tuple: list of valid marker pairs and mapping for each index to its respective class:
    [(a,b),(c,f),...] and [0,1,...] where (a,b) is a marker for category 0
    """
    # Number of categories
    n_cats = len(categories)

    # Will hold the tuples with the pairs
    pairs = []
    # Will hold the categories for the respective tuple. pairs[i] is in category mapping[i]
    #mapping = []

    # Iterate over all possible gene combinations
    for g1 in range(0, n_genes):
        # Parallized
        for g2 in prange(g1+1, n_genes):
            # Gene counts
            x1 = raw_data[:, g1]
            x2 = raw_data[:, g2]

            # Subtract all gene counts of gene 2 from gene counts of gene 1
            diff = np.subtract(x1, x2)

            # Counter for phases in which gene 1 > gene 2
            up = 0

            # Stores last phase in which gene 1 < gene 2
            down = -1

            # Check each category
            for i in range(n_cats):

                # Check if g1 − g2 > 0 in at least a fraction f of G1 cells
                frac = count_up(diff[categories[i]])
                if frac >= thresholds[i]:

                    # If up > 1 (g1,g2) can't be a marker pair
                    # but if up == (number cats - 1) (g2,g1) might be
                    up += 1
                    passed_other = True

                    # Check other phases
                    for j in range(n_cats):
                        if i != j:

                            # g1 − g2 < 0 in at least a fraction f of all other categories
                            sub_frac = count_down(diff[categories[j]])
                            if not sub_frac >= thresholds[j]:
                                # (g1,g2) is not a marker pair
                                passed_other = False

                                # But this (g2, g1) might be pair for current category
                                sub_frac = count_up(diff[categories[j]])
                                if sub_frac >= thresholds[j]:
                                    up += 1

                                # Stop checking other categories
                                break
                            else:
                                # Store last passed
                                down = j

                    # If up in cat[i] and down in all other cat
                    if passed_other:
                        # Store marker pair for cat[i] = pair
                        pairs.extend([i, g1, g2])
                    else:
                        break

            # If all but one are up (g2,g1) might be marker pair
            if up == n_cats - 1:
                # If any down was found, (g2,g1) is marker pair for down
                if down != -1:
                    pairs.extend([down, g2, g1])
                # None found, check remaining category
                else:
                    left_over = n_cats-1
                    # If its down, (g2,g1) is marker pair for that
                    sub_frac = count_down(diff[categories[left_over]])
                    if sub_frac >= thresholds[left_over]:
                        pairs.extend([left_over, g2, g1])

    return pairs


@njit()
def count_up(diff, min_diff=0):
    return len(np.where(diff > min_diff)[0])


@njit()
def count_down(diff, min_diff=0):
    return len(np.where(diff < min_diff)[0])

# New approach, less code, imo better readable but unfortunately slower
"""
possible_combinations = itertools.combinations(range(0, len(gene_names)), 2)

marker_pairs = defaultdict(list)

for pair in possible_combinations:
    g1 = raw_data[:, pair[0]]
    g2 = raw_data[:, pair[1]]

    diff = np.subtract(g1, g2)

    up = [
        count_up(diff[categories[i]]) >= thresholds[i]
        for i in range(len(categories))
    ]
    down = [
        count_down(diff[categories[i]]) >= thresholds[i]
        for i in range(len(categories))
    ]

    if sum(up) == 1:
        group = np.argmax(up)

        if sum(down) == (len(down) - 1):
            marker_pairs[category_names[group]].append(
                pair
            )
    elif sum(down) == 1:
        group = np.argmax(down)

        if sum(up) == (len(up) - 1):
            marker_pairs[category_names[group]].append(
                (gene_names[pair[1]],
                 gene_names[pair[0]])
            )
"""