import itertools
from os import cpu_count
from multiprocessing import Pool
from math import ceil
import numpy as np
from numba import njit
from collections import defaultdict
from pandas import DataFrame
import operator
from functools import partial

def __set_matrix(adata, phases=None, subset_genes=None, subset_samples=None, rm_zeros=True, fraction=None, verboose=False):
    """
    Sets the parameter for the algorithms and trims the 'x'-matrix to contain only necessary elements

    :param x: Pandas-Matrix with gene counts, index must contain gene names, columns must contain sample names
    :param phases: Dictionary of Lists, i.e. {phase: [sample, ...]}, containing annotation of samples to their phase
    :param subset_genes: List of Indices, Names or Boolean of genes to look at excluding all other
    :param subset_samples: List of Indices, Names or Boolean of samples to look at excluding all other
    :return: nothing
    """

    current_shape = adata.shape

    if verboose:
        print('[__set_matrix] Original Matrix \'x\' has shape {} x {}'.format(
            current_shape[0], current_shape[1]
        ))

    # Copy x to ensure original matrix is not altered
    x_copy = adata

    # Eliminate rows where genes not in 'subset_genes', if provided
    if subset_genes is not None:
        # Make sure 'subset_genes' is index array if boolean or named array is supplied
        # And remove genes provided in 'subset_genes' but not contained in 'x'
        genes_mask = to_index(subset_genes, x_copy.obs_names)
        x_copy = x_copy[genes_mask, :]

        if verboose:
            print('[__set_matrix] Removed {} genes that were not in \'subset_genes\'. {} genes remaining.'.format(
                (current_shape[0] - x_copy.shape[0]), x_copy.shape[0])
            )

    current_shape = x_copy.shape

    if rm_zeros:
        # Eliminate not expressed genes
        rows = x_copy.X
        not_zero = [i for i in range(0, x_copy.shape[0]) if np.sum(rows[i]) != 0]
        x_copy = x_copy[not_zero , :]
        #x_copy = x_copy[(x_copy.T != 0).any()]

        if verboose:
            print('[__set_matrix] Removed {} genes that were not expressed in any samples. {} genes remaining.'.format(
                (current_shape[0] - x_copy.shape[0]), x_copy.shape[0])
            )

    # Store remaining gene names for later use, rename for readability
    gene_names = list(x_copy.obs_names)

    # Store all sample names for re calculation of indices
    all_samples = x_copy.var_names

    # Eliminate columns where samples not in 'subset_samples', if provided
    if subset_samples is not None:
        # Make sure 'subset_samples' is index array if boolean or named array is supplied
        # And remove samples provided in 'subset_genes' but not contained in 'x'
        sample_mask = to_index(subset_samples, all_samples)
        x_copy = x_copy[:, sample_mask]

        if verboose:
            print('[__set_matrix] Removed {} samples that were not in \'subset_samples\'. {} samples remaining.'.format(
                (current_shape[1] - x_copy.shape[1]), x_copy.shape[1])
            )

    current_shape = x_copy.shape

    thresholds = None
    phases_copy = None

    # Eliminate samples not annotated in 'phases'
    if phases is not None:
        # Get 1D index based mask from samples per phase
        # And remove all samples not contained in this list
        phase_mask = [
            idx for _, samples in phases.items()
            for idx in to_index(
                to_named(samples, all_samples),
                x_copy.var_names
            )
        ]

        x_copy = x_copy[:, phase_mask]

        if verboose:
            print(
                '[__set_matrix] Removed {} samples that were not annotated in \'phases\'. {} samples remaining.'.format(
                    (current_shape[1] - x_copy.shape[1]), x_copy.shape[1])
            )

        # Re-calculate phases indices based on truncated sample list
        phases_copy = {
            phase: to_index(
                to_named(samples, all_samples),
                x_copy.var_names
            ) for phase, samples in phases.items()
        }

        # Pre Calculate thresholds for phases
        thresholds = {phase: ceil(len(samples) * fraction) for phase, samples in phases_copy.items()}

    # Store remaining sample names for later use, rename for readability
    sample_names = list(x_copy.var_names)

    if verboose:
        print('[__set_matrix] Matrix truncation done. Working with {} genes for {} samples.'.format(
            x_copy.shape[0], x_copy.shape[1])
        )

    # Transform to ndarray for faster calculations
    x_copy = x_copy.X

    return {
        "x": x_copy,
        "phases": phases_copy,
        "sample_names": sample_names,
        "gene_names": gene_names,
        "thresholds": thresholds
    }


def sandbag(adata, phases, fraction=0.5, processes=1, subset_genes=None, subset_samples=None, verboose=False):
    """ Calculates the pairs of genes serving as marker pairs for each phase, based on a matrix of gene counts and
    an annotation of known phases.

    :param verboose: Debug info
    :param processes: Number of processes to use for multiprocess.pool
    :param fraction: Fraction
    :param x: Pandas-Matrix with gene counts, index must contain gene names, columns must contain sample names
    :param phases: Dictionary of Lists, i.e. {phase: [sample, ...]}, containing annotation of samples to their phase
    :param subset_genes: List of Indices, Names or Boolean of genes to look at excluding all other
    :param subset_samples: List of Indices, Names or Boolean of samples to look at excluding all other
    :return: Dictionary of List of Tuples, i.e. {phase: [(Gene1, Gene2), ...]}, containing marker pairs per phase
    """

    # Set the parameter to the class instance and remove unnecessary elements in 'x'
    params = __set_matrix(adata, fraction=fraction, phases=phases, subset_genes=subset_genes, subset_samples=subset_samples,
                          verboose=verboose)

    if verboose:
        print('[sandbag] Identifying marker pairs...', end='')

    possible_combinations = itertools.combinations(range(0, len(params["gene_names"])), 2)

    if processes == 0:
        processes = cpu_count() - 1

    masks = (params["phases"]["G1"], params["phases"]["S"], params["phases"]["G2M"])
    thresholds = [params["thresholds"]["G1"], params["thresholds"]["S"], params["thresholds"]["G2M"]]

    check_phase_for_pair_wrapper_par = partial(check_phase_for_pair_wrapper, x=params["x"], masks=masks, thresholds=thresholds)

    # Multi cored calculation if requested
    if processes != 1:
        # Worker pool of processes

        with Pool(processes=processes) as pool:
            if verboose:
                print("Processing in parallel with {} processes...".format(processes))
            annotations = pool.map(
                check_phase_for_pair_wrapper_par, possible_combinations)

        annotations = list(annotations)
    else:
        annotations = (check_phase_for_pair_wrapper_par(pair) for pair in possible_combinations)

    # Create container for marker pairs
    marker_pairs = {phase: [] for phase in phases.keys()}

    # Puts marker pairs into the 'marker_pairs' dictionary and removes 'None' phase annotation
    for annotation in annotations:
        if annotation[0]:
            marker_pairs[annotation[0]].append(
                (params["gene_names"][annotation[1][0]], params["gene_names"][annotation[1][1]]))

    if verboose:
        count_pairs = 0
        for _, pairs in marker_pairs.items():
            count_pairs = count_pairs + len(pairs)
        print(" Done!")
        print("[sandbag] Identified {} marker pairs (phase: count):".format(count_pairs), end=' ')
        print({phase: len(pairs) for phase, pairs in marker_pairs.items()})

    # Return 'marker_pairs' dictionary: {phase: [(Gene1, Gene2), ...]}
    return marker_pairs


def cyclone(adata, marker_pairs, subset_genes=None, iterations=1000, min_iter=100, min_pairs=50,
            subset_samples=None, verboose=False, rm_zeros=False, processes=1):

    params = __set_matrix(adata, subset_genes=subset_genes, subset_samples=subset_samples, rm_zeros=rm_zeros,
                          verboose=verboose)

    if verboose:
        print('[cyclone] Preparing marker pairs, where at least one gene was not present in \'x\'...', end='')

    # Eliminate all gene pairs where at least one gene is not present in gene_names and convert to index
    marker_pairs_idx = defaultdict(list)

    removed = 0

    used = defaultdict(list)
    used_idx = defaultdict(list)

    gene_name_idx = {g: i for i, g in enumerate(params["gene_names"])}

    # Generate used list
    for phase, pairs in marker_pairs.items():
        u = []

        for pair in pairs:
            try:
                idx_pair = (gene_name_idx[pair[0]], gene_name_idx[pair[1]])
                marker_pairs_idx[phase].append(idx_pair)
                u.extend([idx_pair[0], idx_pair[1]])
            except KeyError:
                removed = removed + 1

        used[phase] = list(np.unique(u))

    for phase, pairs in marker_pairs.items():
        u_idx = np.empty(len(params["gene_names"]), dtype=int)

        for i, u in enumerate(used[phase]):
            u_idx[u] = i

        used_idx[phase] = u_idx

    if verboose:
        count_pairs = 0
        for _, pairs in marker_pairs.items():
            count_pairs = count_pairs + len(pairs)
        print(' Done!')
        print('[cyclone] Removed {} marker pairs. {} marker pairs remaining.'.format(removed, count_pairs))
        print('[cyclone] Calculating scores and predicting cell cycle phase...', end='')

    if processes == 0:
        processes = cpu_count() - 1

    # Iterate over phases
    scores = {phase: __get_phase_scores(params["x"], iterations, min_iter, min_pairs,
                                        pairs, used[phase], used_idx[phase], processes
                                        ) for phase, pairs
              in marker_pairs_idx.items()}

    scores_df = DataFrame(scores)
    normalized_df = scores_df.div(scores_df.sum(axis=1), axis=0)

    prediction = {}

    for index, score in scores_df.iterrows():
        ph = ["G1", "S", "G2M"]
        for p in ph:
            if p not in score:
                score[p] = 0

        sc = [score["G1"], score["S"], score["G2M"]]
        max_idx, max_val = max(enumerate(sc), key=operator.itemgetter(1))

        if max_val == 0:
            prediction[params["sample_names"][index]] = None
        elif score["G1"] >= 0.5 or score["G2M"] >= 0.5:
            if score["G1"] >= score["G2M"]:
                prediction[params["sample_names"][index]] = "G1"
            else:
                prediction[params["sample_names"][index]] = "G2M"
        else:
            if score["S"] != 0:
                prediction[params["sample_names"][index]] = "S"
            else:
                prediction[params["sample_names"][index]] = ph[max_idx]

    output = {
        "prediction": prediction,
        "scores": scores_df,
        "normalized": normalized_df
    }

    if verboose:
        print(' Done!')
        print("[cyclone] Calculated scores and prediction (phase: count): ", end='')
        counts = defaultdict(int)
        for _, pred in prediction.items():
            counts[pred] = counts[pred] + 1
        print(', '.join('{}: {}'.format(phase, count) for phase, count in counts.items()))

    return output


def __get_phase_scores(x, iterations, min_iter, min_pairs, pairs, used, used_idx, processes):
    # Multi cored calculation if requested
    if processes != 1:
        get_sample_score_par = partial(__get_sample_score, iterations=iterations, min_iter=min_iter,
                                       min_pairs=min_pairs, pairs=pairs, used_idx=used_idx)

        # Worker pool of processes
        with Pool(processes=processes) as pool:
            phase_scores = pool.map(get_sample_score_par, x.T)

        return list(phase_scores)
    else:
        return [__get_sample_score(
            sample[used], iterations, min_iter, min_pairs, pairs, used_idx
        ) for sample in x.T]


@njit()
def __get_sample_score(sample, iterations, min_iter, min_pairs, pairs, used_idx):
    cur_score = get_proportion(sample, min_pairs, pairs, used_idx)
    if cur_score is None:
        return 0

    below = 0
    total = 0
    idx = sample
    for i in range(0, iterations):
        np.random.shuffle(idx)
        new_score = get_proportion(idx, min_pairs, pairs, used_idx)
        if new_score is not None:
            if new_score < cur_score:
                below += 1
            total += 1

    if total >= min_iter:
        return below / total


@njit()
def get_proportion(sample, min_pairs, pairs, used_idx):
    hits = 0
    total = 0

    for pair in pairs:
        a = sample[used_idx[pair[0]]]
        b = sample[used_idx[pair[1]]]
        if a > b:
            hits += 1
        if a != b:
            total += 1

    if hits < min_pairs:
        return None

    return hits / total


def check_phase_for_pair_wrapper(pair, x, masks, thresholds):
    phases = [None, "G1", "S", "G2M"]

    phase = __check_phase_for_pair(pair, x, masks, thresholds)

    if phase > 0:
        return phases[abs(phase)], pair
    else:
        return phases[abs(phase)], (pair[1], pair[0])


@njit()
def __check_phase_for_pair(pair, x, masks, thresholds):
    """ Calculates the phase for which a pair of genes is a valid marker pair.
    Returns the phase in which gene 1 is higher expressed (in more than fraction * number of cells in phase)
    as gene 2 while being lower expressed (in more than fraction * number of cells in phases) in all other phases
    Return None if pair is not a marker pair for any phase

    :param pair: Tuple (Gene 1 index, Gene 2 index) of genes to be checked
    :param x: 'x' with gene counts
    :param masks: Masked of genes annotated
    :param thresholds: Pre calculated dict of thresholds, i.e. {phase: 'fraction' * 'number of cells in phases'}
    :return: Phase for which this pair is a marker, or None if not a marker pair, along with the pair Tuple
    """
    x1 = x[pair[0], :]
    x2 = x[pair[1], :]

    # Subtract all gene counts of gene 2 from gene counts of gene 1
    diff = __expression_diff(x1, x2)

    # Counter for phases in which gene 1 > gene 2
    up = 0

    # Stores last phase in which gene 1 < gene 2
    down = 0

    # Test each phase
    for i in range(0, 3):

        # Check if gene 1 > gene 2 in more than set fraction of samples in current phase
        if __check_if_up(mask(diff, masks[i]), thresholds[i]):
            up += 1
            passed_other = True

            # Check if gene 2 > gene 1 in all other phases
            for j in range(0, 3):

                # Skip same phase
                if i != j:

                    # Check if gene 1 < gene 2 in more than set fraction of samples in current 'sub_phase'
                    if not __check_if_down(mask(diff, masks[j]), thresholds[j]):
                        passed_other = False

                        # If not, check if gene 1 > gene 2 in current 'sub_phase'
                        if __check_if_up(mask(diff, masks[j]), thresholds[j]):
                            up += 1

                        # Don't check other phases
                        break
                    else:
                        # Store down phase as it could be up for the reversed pair
                        down = j+1

            # Return phase and pair if found
            if passed_other:
                return i+1
            else:
                break

    # When gene 1 > gene 2 in all but one phase, consider that (Gene2, Gene1) is marker pair in the remaining phase
    if up == 2:
        # When the loop above already revealed the remaining phase and checked that Gene 2 > Gene 1 not only '>='
        if down != 0:
            # Return reversed pair with phase
            return 0-down
        # Else look at the remaining phase and check if Gene 2 > Gene 1 not only '>='
        else:
            if __check_if_down(mask(diff, masks[2]), thresholds[2]):
                # Return reversed pair with checked phase
                return 3

    # Return 'None' if no phase if current pair, and reversed, is not a marker pair for any phase
    return 0


@njit()
def mask(x, arr):
    y = np.zeros(len(arr))
    for i in range(0, len(arr)):
        y[i] = x[arr[i]]
    return y


@njit()
def __expression_diff(x1, x2):
    """ Fast matrix subtraction

    :param x1: Row 1
    :param x2: Row 2
    :return: Difference
    """
    return np.subtract(x1, x2)


@njit()
def __check_if_up(diff, threshold):
    """ Checks if Gene 1 is higher expressed than Gene 2

    :param diff: Difference expression gene 1 - gene 2
    :param threshold: Number of required samples
    :return: True if more than threshold samples are above 1
    """
    return (diff > 0).sum() >= threshold


@njit()
def __check_if_down(diff, threshold):
    """ Checks if Gene 1 is lower expressed than Gene 2

    :param diff: Difference expression gene 1 - gene 2
    :param threshold: Number of required samples
    :return: True if more than threshold samples are below 1
    """
    return (diff < 0).sum() >= threshold


def to_index(m, names=None):
    names = list(names)
    # Check idx
    if all(isinstance(i, int) for i in m):
        return m

    # Named mask
    if all(isinstance(i, str) for i in m):
        return list([names.index(i) for i in m if i in names])

    # Boolean mask
    if all(isinstance(i, bool) for i in m):
        return list([i for i, j in enumerate(m) if j])

    raise ValueError("Only homogeneous Index, Name or Boolean arrays valid")


def to_named(m, names):
    names = list(names)
    # Named mask
    if all(isinstance(i, str) for i in m):
        return list(set(names).intersection(m))

    # Check idx
    if all(isinstance(i, int) for i in m):
        return list(names[i] for i in m)

    # Boolean mask
    if all(isinstance(i, bool) for i in m):
        return list([names[i] for i, j in enumerate(m) if j])

    raise ValueError("Only homogeneous Index, Name or Boolean arrays valid")
