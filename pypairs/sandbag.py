import functools
import itertools
import multiprocessing
from collections import defaultdict
from math import ceil

import numpy as np
from numba import jit
from scanpy import api

from pypairs import helper


def filter_matrix(matrix, categories, gene_names, sample_names, subset_genes, subset_samples, rm_zeros=True,
                  filter_genes_dispersion=False, always_keep_genes=None):
    subset_genes_mask = helper.to_boolean(subset_genes, gene_names)

    if rm_zeros:
        zeros_mask = np.invert(np.all(matrix == 0, axis=1))
        genes_mask = np.logical_and(subset_genes_mask, zeros_mask)
    else:
        genes_mask = subset_genes_mask

    notde = 0
    if filter_genes_dispersion:
        x = api.pp.filter_genes_dispersion(matrix.T)
        dispersion_mask = x.gene_subset
        if always_keep_genes is not None:
            dispersion_mask = np.logical_or(helper.to_boolean(always_keep_genes, gene_names), dispersion_mask)

        genes_mask = np.logical_and(dispersion_mask, genes_mask)
        notde = len(dispersion_mask) - sum(dispersion_mask)

    helper.print_info("O", "Filtered out {num} genes: {subset} subsetted, {notde} based on dispersion".format(
        num=len(genes_mask) - sum(genes_mask),
        subset=len(subset_genes_mask) - sum(subset_genes_mask),
        notde=notde
    ), "sandbag")

    categories_np = np.ndarray(shape=(len(categories.keys()), len(sample_names)), dtype='bool')

    subset_samples_mask = helper.to_boolean(subset_samples, sample_names)
    last = None
    last_cat = ""
    for i, k in enumerate(categories.keys()):
        selected_np = np.array(categories[k])
        categories_np[i] = np.logical_and(helper.to_boolean(selected_np, sample_names), subset_samples_mask)

    samples_mask = np.logical_or.reduce(categories_np)

    helper.print_info("O", "Filtered out {num} samples: {subset} subsetted, {notan} not annotated".format(
        num=len(samples_mask) - sum(samples_mask),
        subset=len(subset_samples_mask) - sum(subset_samples_mask),
        notan=((len(samples_mask) - sum(samples_mask)) - (len(subset_samples_mask) - sum(subset_samples_mask)))),
                      "sandbag")

    matrix = matrix[genes_mask, :]
    matrix = matrix[:, samples_mask]

    categories_np = categories_np[:, samples_mask]

    gene_names_np = np.array(gene_names)[genes_mask]
    sample_names_np = np.array(sample_names)[samples_mask]

    return matrix, categories_np, gene_names_np, sample_names_np


@jit(nopython=True)
def expression_diff(x1, x2):
    return np.subtract(x1, x2)


@jit(nopython=True)
def count_up(diff, min_diff=0):
    return (diff > min_diff).sum()


@jit(nopython=True)
def count_down(diff, min_diff=0):
    return (diff < min_diff).sum()


@jit(nopython=True)
def check_phase_for_pair(pair, matrix, categories, thresholds, min_diff):
    x1 = matrix[pair[0], :]
    x2 = matrix[pair[1], :]

    # Subtract all gene counts of gene 2 from gene counts of gene 1
    diff = expression_diff(x1, x2)

    # Counter for phases in which gene 1 > gene 2
    up = 0

    min_diff_down = min_diff * -1

    # Stores last phase in which gene 1 < gene 2
    down = 0

    frac = 0
    cats = len(categories)

    for i in range(0, cats):
        frac = count_up(diff[categories[i]], min_diff=min_diff)

        if frac >= thresholds[i]:
            up += 1
            passed_other = True

            for j in range(0, cats):

                if i != j:
                    sub_frac = count_down(diff[categories[j]], min_diff=min_diff_down)

                    if not sub_frac >= thresholds[j]:
                        passed_other = False

                        sub_frac = count_up(diff[categories[j]], min_diff=min_diff)

                        if sub_frac >= thresholds[j]:
                            frac += sub_frac
                            up += 1

                        break
                    else:
                        frac += sub_frac
                        down = j + 1

            if passed_other:
                return i + 1, frac
            else:
                break

    if up == cats - 1:
        if down != 0:
            return 0 - down, frac
        else:
            sub_frac = count_down(diff[categories[cats - 1]], min_diff=min_diff_down)
            if sub_frac >= thresholds[cats - 1]:
                frac += sub_frac
                return cats * -1, frac

    return 0, 0


def check_phase_for_pair_wrapper(pair, matrix, category_names, categories, thresholds, min_diff):
    cats = np.insert(category_names, 0, [None], axis=0)

    cat = check_phase_for_pair(pair, matrix, categories, thresholds, min_diff)

    if cat[0] > 0:
        return cats[abs(cat[0])], pair, cat[1]
    else:
        return cats[abs(cat[0])], (pair[1], pair[0]), cat[1]


def sandbag(matrix, categories, gene_names, sample_names, fraction=0.5, subset_genes=None, subset_samples=None,
            rm_zeros=True, filter_genes_dispersion=False, always_keep_genes=None, min_diff=0, processes=1):
    helper.print_info("I", "Started and filtering matrix...", "sandbag")
    category_names = np.array(list(categories.keys()))
    matrix, categories_filtered, gene_names_filtered, sample_names_filtered = filter_matrix(matrix, categories,
                                                                                            gene_names, sample_names,
                                                                                            subset_genes,
                                                                                            subset_samples,
                                                                                            rm_zeros=rm_zeros,
                                                                                            filter_genes_dispersion=filter_genes_dispersion,
                                                                                            always_keep_genes=always_keep_genes)

    helper.print_init_sandbag(matrix, categories, fraction, processes)

    thresholds = np.apply_along_axis(sum, 1, categories_filtered)
    for i, t in enumerate(thresholds):
        thresholds[i] = ceil(t * fraction)

    possible_combinations = itertools.combinations(range(0, len(gene_names_filtered)), 2)

    if processes == 0:
        processes = multiprocessing.cpu_count() - 1

    check_phase_for_pair_wrapper_par = functools.partial(check_phase_for_pair_wrapper, matrix=matrix,
                                                         category_names=category_names, categories=categories_filtered,
                                                         thresholds=thresholds, min_diff=min_diff)

    helper.print_info("I", "Identifying marker pairs...", "sandbag")
    if processes != 1:
        with multiprocessing.Pool(processes=processes) as pool:
            annotations = pool.map(check_phase_for_pair_wrapper_par, possible_combinations)

        annotations = list(annotations)
    else:
        annotations = (check_phase_for_pair_wrapper_par(pair) for pair in possible_combinations)

    marker_pairs = defaultdict(list)

    for annotation in annotations:
        if annotation[0] not in ['N', 'None', 'Non', None]:
            marker_pairs[annotation[0]].append(
                (gene_names_filtered[annotation[1][0]], gene_names_filtered[annotation[1][1]]))

    count_total = 0
    count_str = []
    for m, p in marker_pairs.items():
        c = len(p)
        count_total += c
        count_str.append("\t{}: {}".format(m, c))

    helper.print_info("I", "Found: ", "sandbag")
    for s in count_str:
        helper.print_info("\tI", s, "sandbag")
    helper.print_info("I", "\tTotal: {}".format(count_total), "sandbag")

    helper.print_info("O", "is done!", "sandbag")

    return marker_pairs
