"""Utility functions
"""

from typing import Union, Mapping, Iterable, Tuple, Any, Callable, Optional
from anndata import AnnData
from pandas import DataFrame
from sklearn.metrics import (precision_score, recall_score, f1_score)
import numpy as np
from numba import njit
import sys
import json
import os

from pypairs import settings
from pypairs import log as logg


def parallel_njit(
        func: Callable[[], Any],
        jitted: Optional[bool] = True
) -> Callable[[], Any]:
    """Dynamic decorator for jit-compiled functions.
    Adds parallel=True if settings.n_jobs > 1

    Parameters
    ----------

    func
        jit-able function
    jitted
        If False do not compile function. For debugging purposed.

    Returns
    -------

    Decorated function ``func``
    """
    if jitted is False:
        logg.warn('staring uncompiled processing. Should only be used for debug and testing!'.format(settings.n_jobs))
        return func

    if settings.n_jobs > 1:
        if is_win32() is False:
            logg.hint('staring parallel processing with {} threads'.format(settings.n_jobs))
            return njit(func, parallel=True)
        else:
            logg.error(
                'n_jobs is set to {} but multiprocessing is not supported for your platform! '
                'falling back to single core... '.format(settings.n_jobs))
            return njit(func)
    else:
        logg.hint('staring processing with 1 thread')
        return njit(func)


def is_win32():
    if os.name == 'nt' and is_64bit_arch() is False:
        return True
    else:
        return False


def is_64bit_arch():
    return sys.maxsize > 2**32


def parse_annotation(
        annotation: Mapping[str, Iterable[Union[str, int, bool]]],
        sample_names: Iterable[str]
) -> Tuple[Iterable[str], Iterable[Iterable[bool]]]:
    """ Translates a dictionary annotation {'category': [sample1, sample2, ...], ..} into a list of boolean masks.
    Accepts index, names and boolean.

    Parameters
    ----------

    annotation
        Mapping from "category" to list of "samples", e.g. {'category': [sample1, sample2, ...], ..}
    sample_names
        List of sample names, order must match data

    Returns
    -------

    A :class:`~np.array` with the category names and a :class:`~np.ndarray` with the boolean mask for each category
    """
    category_names = np.array(list(annotation.keys()))
    categories = np.ndarray(shape=(len(category_names), len(sample_names)), dtype=bool)

    logg.hint("passed {} categories: {}".format(len(category_names), str(category_names)))
    for i, k in enumerate(annotation.keys()):
        logg.hint("\t{}: {}".format(k, len(annotation[k])))
        categories[i] = to_boolean_mask(np.array(annotation[k]), sample_names)

    return category_names, categories


def parse_data(
        data: Union[AnnData, DataFrame, np.ndarray],
        gene_names: Optional[Iterable[str]] = None,
        sample_names: Optional[Iterable[str]] = None
) -> Tuple[np.ndarray, Iterable[str], Iterable[str]]:
    """Reduces :class:`~anndata.AnnData` and :class:`~pandas.DataFrame` to a :class:`~numpy.dnarray` and extracts
    `gene_names` and `sample_names` from index and column names.

    Parameters
    ----------
    data
        The (annotated) data matrix of shape ``n_obs`` * ``n_vars``.
        Rows correspond to cells and columns to genes.
    gene_names
        Names for genes, must be same length as ``n_vars``.
    sample_names
        Names for samples, must be same length as ``n_obs``.
    """
    if isinstance(data, AnnData):
        if sample_names is None:
            sample_names = list(data.obs_names)

        if gene_names is None:
            gene_names = list(data.var_names)

        raw_data = data.X
    else:
        if isinstance(data, DataFrame):
            if gene_names is None:
                gene_names = list(data.columns)
            if sample_names is None:
                sample_names = list(data.index)

            raw_data = data.values
        elif isinstance(data, np.ndarray):
            if gene_names is None or sample_names is None:
                raise ValueError("Provide gene names and sample names in ``gene_names`` and ``sample_names``")

            raw_data = data
        else:
            raise ValueError("data can only be of type AnnData, DataFrame or ndarray")

    logg.hint("passed data of shape {} x {} (genes x samples)".format(*raw_data.shape))
    return raw_data, gene_names, sample_names

def to_boolean_mask(
        selected: Iterable[Any],
        l: Iterable[str]
) -> Iterable[bool]:
    """Converts a index, str or boolean array to an boolean array based on a list

    Parameters
    ----------

    selected
        List of index, str or bool
    l
        List of str containing the names

    Returns
    -------
        List of bool of size len(names), where a position matchin index or str is True
    """
    if selected is None:
        all_mask = np.ones(len(l), dtype=bool)
        return all_mask

    selected = np.array(selected)

    if selected.size == 0:
        all_mask = np.ones(len(l), dtype=bool)
        return all_mask

    mask = np.zeros(len(l), dtype=bool)

    if selected.dtype.type is np.int_:
        mask[selected] = True
    elif selected.dtype.type is np.str_:
        for i, l in enumerate(l):
            if l in selected:
                mask[i] = True
    elif selected.dtype == 'bool':
        return selected
    else:
        raise ValueError("Categories must be array-like of type bool, int or str")

    return mask


def write_dict_to_json(d, fout):
    with open(fout, 'w') as f:
        json.dump(d, f)


def read_dict_from_json(fin):
    return json.load(open(fin, 'r'))


def evaluate_prediction(prediction, reference):
    ref = np.array(reference)
    pred = np.array(prediction['prediction'])

    labels = np.unique(list(ref) + list(pred))

    f1 = f1_score(ref, pred, average=None, labels=labels)
    recall = recall_score(ref, pred, average=None, labels=labels)
    precision = precision_score(ref, pred, average=None, labels=labels)

    df = DataFrame(columns=labels, index=["f1", "recall", "precision"])

    df.loc["f1"] = f1
    df.loc["recall"] = recall
    df.loc["precision"] = precision

    return df.T