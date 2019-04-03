from typing import Union, Optional, Tuple, Mapping, Collection

from anndata import AnnData
from pandas import DataFrame
import numpy as np
from numba import njit, guvectorize, prange

from pypairs import utils
from pypairs import datasets
from pypairs import settings
from pypairs import log as logg


def cyclone(
    data: Union[AnnData, DataFrame, np.ndarray, Collection[Collection[float]]],
    marker_pairs: Optional[Mapping[str, Collection[Tuple[str, str]]]] = None,
    gene_names: Optional[Collection[str]] = None,
    sample_names: Optional[Collection[str]] = None,
    iterations: Optional[int] = 1000,
    min_iter: Optional[int] = 100,
    min_pairs: Optional[int] = 50
) -> DataFrame:
    """Score samples for each category based on marker pairs.

    ``data`` can be of type :class:`~anndata.AnnData`, :class:`~pandas.DataFrame` or :class:`~numpy.ndarray` and should
    contain the raw or normalized gene counts of shape ``n_obs`` * ``n_vars``. Rows correspond to cells and columns to
    genes.

        *
            If a :class:`~anndata.AnnData` object is passed, the category scores and the final prediction will be added
            to ``data.obs`` with key ``pypairs_{category}_score`` and ``pypairs_max_class``.

            *
                If marker pairs contain only the cell cycle categories G1, S and G2M an additional column
                ``pypairs_cc_prediction`` will be added. Where category S is assigned to samples where G1 and G2M score
                are below 0.5, as described in [Scialdone15]_.


    ``marker_pairs``, i.e. output from :func:`~pypairs.tools.sandbag()`, must be a mapping from category to list of
    2-tuple Genes: `{'category': [(Gene_1,Gene_2), ...], ...}`.

        *
            If no ``marker_pairs`` are passed the default are used from :func:`~pypairs.datasets.default_marker()`
            based on [Leng15]_ (marker pairs for cell cycle prediction).

    Parameters
    ----------

    data
        The (annotated) data matrix of shape ``n_obs`` * ``n_vars``.
        Rows correspond to cells and columns to genes.
    marker_pairs
        A dict mapping from str to a list of 2-tuple, where the key is the category and the list contains the marker
        pairs: {'Category_1': [(Gene_1, Gene_2), ...], ...}. If not provided default marker pairs are used
    gene_names
        Names for genes, must be same length as ``n_vars``.
    sample_names
        Names for samples, must be same length as ``n_obs``.
    iterations
        An integer specifying the number of iterations for random sampling to obtain a cycle score. Default: 1000
    min_iter
        An integer specifying the minimum number of iterations for score estimation. Default: 100
    min_pairs
        An integer specifying the minimum number of pairs for cycle estimation. Default: 50

    Returns
    -------

    A :class:`~pandas.DataFrame` with samples as index and categories as columns with scores for each category for each
    sample and a additional column with the name of the max scoring category for each sample.

        *
            If marker pairs contain only the cell cycle categories G1, S and G2M an additional column
            ``pypairs_cc_prediction`` will be added. Where category S is assigned to samples where G1 and G2M score are
            below 0.5, as described in [Scialdone15]_.


    Examples
    --------
        To predict the cell cycle phase of the unsorted cell from the [Leng15]_  dataset run::

            import pypairs import pairs, datasets

            adata = datasets.leng15('unsorted')
            marker_pairs = datasets.default_cc_marker()
            scores = pairs.cyclone(adata, marker_pairs)
            print(scores)

    """
    logg.info('predicting category scores with cyclone', r=True)

    if marker_pairs is None:
        logg.hint('no marker pairs passed, using default cell cycle prediction marker')
        marker_pairs = datasets.default_cc_marker()

    raw_data, gene_names, sample_names = utils.parse_data(data, gene_names, sample_names)

    # Filter marker pairs to those where both genes are present in `data`
    marker_pairs, used = filter_marker_pairs(marker_pairs, gene_names)

    logg.hint('staring processing with {} thread'.format(settings.n_jobs))

    get_phase_scores_decorated = utils.parallel_njit(get_phase_scores)

    scores = {
        cat: get_phase_scores_decorated(raw_data, iterations, min_iter, min_pairs, pairs, used[cat]) for
        cat, pairs in marker_pairs.items()
    }

    scores_df = DataFrame(scores, columns=marker_pairs.keys())
    scores_df.index = sample_names
    scores_df['max_class'] = scores_df.idxmax(axis=1)

    if len(marker_pairs.items()) == 3 and all(elem in marker_pairs.keys() for elem in ["G1", "S", "G2M"]):
        scores_cc = scores_df.loc[:, ["G1", "G2M"]].idxmax(axis=1)
        scores_df['cc_prediction'] = [
            "S" if x < 0.5 else scores_cc[i] for i, x in
            enumerate(scores_df.loc[:, ["G1", "G2M"]].max(axis=1).values)
        ]

    logg.info("finished", time=True)

    if isinstance(data, AnnData):
        logg.hint('adding scores with key "pypairs_{category}" to `data.obs`"')
        logg.hint('adding max_class with key "pypairs_max_class" to `data.obs`"')
        if len(marker_pairs.items()) == 3 and all(elem in marker_pairs.keys() for elem in ["G1", "S", "G2M"]):
            logg.hint('adding cc_prediction with key "pypairs_cc_prediction" to `data.obs`"')

        for name, values in scores_df.iteritems():
            key_name = 'pypairs_{}'.format(name)
            data.obs[key_name] = values

    logg.info('finished', time=True)
    return scores_df


def marker_pairs_to_nd(pairs):
    lis = list([p.tolist() for p in pairs.values()])

    n = len(lis)
    lengths = np.array([len(x) for x in lis])
    max_len = max(lengths)
    arr = np.zeros((n, max_len, 2), dtype='int32')

    for i in range(n):
        arr[i, :lengths[i]] = lis[i]
    return arr, lengths


@njit()
def get_proportion(sample, min_pairs, pairs):
    hits = 0
    total = 0

    for i in range(len(pairs)):
        pair = pairs[i]

        a = sample[pair[0]]
        b = sample[pair[1]]

        if a != b:
            if a > b:
                hits += 1
            total += 1

    if hits < min_pairs:
        return None

    if total == 0:
        return 0
    return hits / total


@njit()
def get_sample_score(sample, iterations, min_iter, min_pairs, pairs):
    cur_score = get_proportion(sample, min_pairs, pairs)

    if cur_score is None:
        return 0

    below = 0
    total = 0
    idx = sample
    for _ in range(0, iterations):
        np.random.shuffle(idx)
        new_score = get_proportion(idx, min_pairs, pairs)
        if new_score is not None:
            if new_score < cur_score:
                below += 1
            total += 1

    if total == 0:
        return 0
    if total >= min_iter:
        return below / total


def filter_marker_pairs(marker_pairs, gene_names):
    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
    removed = 0

    marker_pairs_idx = {}

    used_masks = {}
    for cat, pairs in marker_pairs.items():
        used_mask = np.zeros(len(gene_names), dtype='bool')
        for pair in pairs:
            try:
                g1_idx = gene_name_to_idx[pair[0]]
                g2_idx = gene_name_to_idx[pair[1]]

                used_mask[g1_idx] = True
                used_mask[g2_idx] = True
            except KeyError:
                removed += 1
        used_masks[cat] = used_mask
        used_idx = np.where(used_mask)[0].tolist()

        new_idx = {u: i for i, u in enumerate(used_idx)}

        new_pairs_idx = []
        for pair in pairs:
            try:
                g1_idx = gene_name_to_idx[pair[0]]
                g2_idx = gene_name_to_idx[pair[1]]

                new_pairs_idx.append([new_idx[g1_idx], new_idx[g2_idx]])
            except KeyError:
                logg.hint("genepair ({}, {}) not present in dataset".format(pair[0], pair[1]))

        marker_pairs_idx[cat] = np.array(new_pairs_idx)

    logg.hint("translated marker pairs, {} removed".format(removed))
    return marker_pairs_idx, used_masks


def get_phase_scores(matrix, iterations, min_iter, min_pairs, pairs, used):

    phase_scores = np.zeros(len(matrix))
    for s in prange(0, matrix.shape[0]):
        phase_scores[s] = get_sample_score(
            matrix[s][used], iterations, min_iter, min_pairs, pairs
        )

    return phase_scores
