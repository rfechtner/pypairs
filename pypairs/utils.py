"""Utility functions
"""

from typing import Union, Mapping, Iterable, Tuple, Any, Callable, Optional
from anndata import AnnData
from pandas import DataFrame
from sklearn.metrics import (precision_score, recall_score, f1_score)
import numpy as np
from numba import njit
import sys, json, os
from pathlib import Path
import pandas as pd

from pypairs import settings
from pypairs import log as logg

# ===== DOCUMENTED =====


def evaluate_prediction(
    prediction: Iterable[str],
    reference: Iterable[str],
) -> DataFrame:
    """Calculates F1 Score, Recall and Precision of a :func:`~pypairs.cyclone` prediction.

    Parameters
    ----------

    prediction
        List of predicted classes.
    reference
        List of actual classes

    Returns
    -------

        A :class:`~pandas.DataFrame` with columns "f1", "recall", "precision" and "average"
        for all categories and a overall average containing the respective score.

    Example
    -------

        To get the prediction quality for the example usecase of :func:`~pypairs.cyclone` run::

            from pypairs import pairs, datasets, utils, plotting
            import numpy as np

            adata = datasets.leng15('sorted')
            marker_pairs = datasets.default_cc_marker()
            scores = pairs.cyclone(adata, marker_pairs)

            ref_labels = list(np.repeat("G2M", 76)) + list(np.repeat("S", 80)) + list(np.repeat("G1", 91))

            prediction_quality = utils.evaluate_prediction(scores['max_class'], ref_labels)

            print(prediction_quality)

    """
    ref = np.array(reference)
    pred = np.array(prediction)

    labels_cats = np.unique(list(ref) + list(pred))

    f1 = np.append(
        f1_score(ref, pred, average=None, labels=labels_cats),
        f1_score(ref, pred, average='macro', labels=labels_cats)
    )
    recall = np.append(
        recall_score(ref, pred, average=None, labels=labels_cats),
        recall_score(ref, pred, average='macro', labels=labels_cats)
    )
    precision = np.append(
        precision_score(ref, pred, average=None, labels=labels_cats),
        precision_score(ref, pred, average='macro', labels=labels_cats)
    )

    labels = np.append(labels_cats, "average")

    df = DataFrame(columns=labels, index=["f1", "recall", "precision"])

    df.loc["f1"] = f1
    df.loc["recall"] = recall
    df.loc["precision"] = precision

    average = np.average(df.values, axis=0)

    df.loc["average"] = average

    df = df.apply(pd.to_numeric, errors='coerce')

    return df.T


def export_marker(
    marker: Mapping[str, Iterable[Tuple[str,str]]],
    fname: str,
    defaultpath: Optional[bool] = True
):
    """Export marker pairs to json-File.

    Parameters
    ----------
    marker
        Marker pairs as from :func:`~pypairs.pairs.sandbag`
    fname
        Name of the json-File in the writedir (see settings)
    defaultpath
        Use settings.writedir as root. Default: True
    """

    if defaultpath:
        fpath = settings.writedir + fname
    else:
        fpath = fname

    try:
        write_dict_to_json(marker, fpath)
        logg.hint("marker pairs written to: " + str(fpath))
    except IOError as e:
        msg = "could not write to {}.".format(fpath) + \
              "Please verify that the path exists and is writable. Or change `writedir` via `pypairs.settings.writedir`"
        logg.error(msg)
        logg.error(str(e))


def load_marker(
    fname: str,
    defaultpath: Optional[bool] = True
):
    """Export marker pairs to json-File.

    Parameters
    ----------
    fname
        Name of the json-File to write to
    defaultpath
        Use settings.writedir as root. Default: True
    """

    if defaultpath:
        fpath = settings.writedir + fname
    else:
        fpath = fname

    try:
        marker = read_dict_from_json(fpath)
    except IOError:
        logg.error("could not read from {}.\n Please verify that the path exists and is writable.".format(fpath))
        return None

    if settings.verbosity > 2:
        count_total = 0
        count_str = []
        for m, p in marker.items():
            c = len(p)
            count_total += c
            count_str.append("\t{}: {}".format(m, c))

        logg.hint("loaded {} marker pairs".format(count_total))
        for s in count_str:
            logg.hint(s)

    return marker

# ===== UNDOCUMENTED =====

def parallel_njit(
    func: Callable[[], Any],
    jitted: Optional[bool] = True
) -> Callable[[], Any]:
    """Dynamic decorator for jit-compiled functions.
    Adds parallel=True if settings.n_jobs > 1
    """
    if jitted is False:
        logg.warn('staring uncompiled processing. Should only be used for debug and testing!')
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


def parse_data_and_annotation(
    data: Union[AnnData, DataFrame, np.ndarray, Iterable[Iterable[float]]],
    annotation: Optional[Mapping[str, Iterable[Union[str, int, bool]]]] = None,
    gene_names: Optional[Iterable[str]] = None,
    sample_names: Optional[Iterable[str]] = None
) -> Tuple[np.ndarray, list, list, np.ndarray, np.ndarray]:
    raw_data, gene_names, sample_names = parse_data(data, gene_names, sample_names)

    if isinstance(data, AnnData):
        if annotation:
            category_names, categories = parse_annotation(annotation, sample_names)
        else:
            if 'category' in data.obs_keys():
                category_names = np.unique(data.obs['category'])

                categories = np.ndarray(shape=(len(category_names), len(sample_names)), dtype=bool)

                logg.hint("passed {} categories: {}".format(len(category_names), str(category_names)))
                for i, name in enumerate(category_names):
                    categories[i] = np.isin(data.obs['category'], name)
                    logg.hint("\t{}: {}".format(name, len(categories[i])))
            else:
                raise ValueError("Provide categories as data.var['category'] or in ``annotation``")
    else:
        if annotation:
            category_names, categories = parse_annotation(annotation, sample_names)
        else:
            raise ValueError("Provide categories in ``annotation``")

    return raw_data, gene_names, sample_names, category_names, categories


def parse_annotation(
    annotation: Mapping[str, Iterable[Union[str, int, bool]]],
    sample_names: Iterable[str]
) -> Tuple[np.ndarray, np.ndarray]:
    """ Translates a dictionary annotation {'category': [sample1, sample2, ...], ..} into a list of boolean masks.
    Accepts index, names and boolean."""
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
) -> Tuple[np.ndarray, list, list]:
    """Reduces :class:`~anndata.AnnData` and :class:`~pandas.DataFrame` to a :class:`~numpy.dnarray` and extracts
    `gene_names` and `sample_names` from index and column names."""
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

    logg.hint("passed data of shape {} x {} (samples x genes)".format(*raw_data.shape))
    return raw_data, gene_names, sample_names


def to_boolean_mask(
    selected: Iterable[Any],
    l: Iterable[str]
) -> Iterable[bool]:
    """Converts a index, str or boolean array to an boolean array based on a list"""
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


def filter_unexpressed_genes(data, gene_names):
    mask = np.invert(np.all(data == 0, axis=0))
    x = data[:, mask]
    gene_names = np.array(gene_names)[mask]

    return x, list(gene_names)


def filter_matrix(data, gene_names, sample_names, categories, filter_genes, filter_samples):
    dim_befor_filter = data.shape
    filtered = False

    if filter_genes is not None:
        gene_mask = to_boolean_mask(filter_genes, gene_names)
        gene_names = list(np.array(gene_names)[gene_mask])
        data = np.copy(data[:, gene_mask])
        filtered = True

    if filter_samples is not None:
        sample_mask = to_boolean_mask(filter_samples, sample_names)
        sample_names = list(np.array(sample_names)[sample_mask])
        data = np.copy(data[sample_mask, :])
        categories = categories[:, sample_mask]
        filtered = True

    if filtered:
        logg.hint("filtered out {} samples and {} genes based on passed subsets".format(
            dim_befor_filter[0] - data.shape[0],
            dim_befor_filter[1] - data.shape[1]
        ))
        logg.hint("new data is of shape {} x {}".format(*data.shape))

    return np.copy(data), gene_names, sample_names, categories


def save_pandas(fname, data):
    """Save DataFrame or Series"""
    if isinstance(data, DataFrame):
        try:
            data.to_csv(fname, header=True)
        except IOError as e:
            logg.warn("could not store cache to {}".format(fname))
            logg.warn(str(e))
    else:
        logg.error("could not save object, `data` must be DataFrame")


def load_pandas(fname):
    """Load DataFrame or Series"""
    try:
        return pd.read_csv(fname, index_col=0)
    except OSError as e:
        logg.error("could not load cached files {}".format(fname))
        logg.error(str(e))


def same_marker(a, b):
    if len(a) != len(b):
        return False

    if sorted(a.keys()) != sorted(b.keys()):
        return False

    for cat, values in a.items():
        set_a = set([tuple(v) for v in values])
        set_b = set([tuple(v) for v in b[cat]])

        if set_a - set_b or set_b - set_a:
            return False

    return True


def is_cached(fname):
    if settings.cachedir is None:
        return False

    cached_fname = os.path.join(settings.cachedir, fname)

    if os.path.isdir(settings.cachedir):
        if os.path.isfile(cached_fname):
            return True
        else:
            return False
    else:
        try:
            os.mkdir(settings.cachedir)
            dir_abs = Path(settings.cachedir).absolute()
            logg.info("created specified cache dir: {}".format(dir_abs))
        except OSError:
            logg.warn("could not create specified cache directory: {}.\n No caching will be used for this session."
                      " You can change `cachedir` via `pypairs.settings.cachedir`".format(settings.cachedir))
            settings.cachedir = None
            return False
