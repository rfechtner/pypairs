import functools
import multiprocessing

import numpy as np
from numba import jit
from pandas import DataFrame

from pypairs import helper


def filter_matrix(matrix, gene_names, sample_names, subset_genes, subset_samples, rm_zeros=True):
    subset_genes_mask = helper.to_boolean(subset_genes, gene_names)

    zero = 0
    if rm_zeros:
        zeros_mask = np.invert(np.all(matrix == 0, axis=1))
        genes_mask = np.logical_and(subset_genes_mask, zeros_mask)
        zero = len(zeros_mask) - sum(zeros_mask)
    else:
        genes_mask = subset_genes_mask

    helper.print_info("O", "Filtered out {num} genes: {subset} subsetted, {notde} not expressed".format(
        num=len(genes_mask) - sum(genes_mask),
        subset=len(subset_genes_mask) - sum(subset_genes_mask),
        notde=zero
    ), "cyclone")

    subset_samples_mask = helper.to_boolean(subset_samples, sample_names)

    helper.print_info("O", "Filtered out {subset} (subsetted) samples".format(
        subset=len(subset_samples_mask) - sum(subset_samples_mask)), "cyclone")

    matrix = matrix[genes_mask, :]
    matrix = matrix[:, subset_samples_mask]

    gene_names_np = np.array(gene_names)[genes_mask]
    sample_names_np = np.array(sample_names)[subset_samples_mask]

    return matrix, gene_names_np, sample_names_np


@jit(nopython=True)
def get_proportion(sample, min_pairs, pairs):
    hits = 0
    total = 0

    for i in range(len(pairs)):
        pair = pairs[i]

        a = sample[pair[0]]
        b = sample[pair[1]]

        if a > b:
            hits += 1
        if a != b:
            total += 1

    if hits < min_pairs:
        return None

    if total == 0:
        return 0
    return hits / total


@jit(nopython=True)
def get_sample_score(sample, iterations, min_iter, min_pairs, pairs):
    cur_score = get_proportion(sample, min_pairs, pairs)

    if cur_score is None:
        return 0

    below = 0
    total = 0
    idx = sample
    for i in range(0, iterations):
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


def get_phase_scores(matrix, cat, iterations, min_iter, min_pairs, pairs, used, processes):
    if pairs.size == 0:
        helper.print_info("E", "No marker pairs for category {}".format(cat), "cyclone")
        return [0.0 for _ in matrix.T]

    if processes != 1:
        get_sample_score_par = functools.partial(get_sample_score, iterations=iterations, min_iter=min_iter,
                                                 min_pairs=min_pairs, pairs=pairs)

        samples = [sample[used] for sample in matrix.T]

        with multiprocessing.Pool(processes=processes) as pool:
            phase_scores = pool.map(get_sample_score_par, samples)

        return list(phase_scores)
    else:
        phase_scores = [get_sample_score(sample[used], iterations, min_iter, min_pairs, pairs) for sample in matrix.T]

        return phase_scores


def filter_marker_pairs(marker_pairs, gene_names):
    gene_name_to_idx = {g: i for i, g in enumerate(gene_names)}
    removed = 0
    removed2 = 0

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
                print("Genepair ({}, {}) not present in dataset".format(pair[0], pair[1]))
                removed2 += 1

        marker_pairs_idx[cat] = np.array(new_pairs_idx)

    helper.print_info("O", "Translated marker pairs, {} removed".format(removed), "cyclone")
    return marker_pairs_idx, used_masks


def cyclone(matrix, marker_pairs, gene_names, sample_names, subset_genes=None, subset_samples=None, iterations=1000,
            min_iter=100, min_pairs=50, rm_zeros=True, processes=1):
    helper.print_info("I", "Started and preprocessing matrix and marker pairs...", "cyclone")
    matrix, gene_names_filtered, sample_names_filtered = filter_matrix(matrix, gene_names=gene_names,
                                                                       sample_names=sample_names,
                                                                       subset_genes=subset_genes,
                                                                       subset_samples=subset_samples, rm_zeros=rm_zeros)
    marker_pairs, used = filter_marker_pairs(marker_pairs, gene_names_filtered)

    helper.print_init_cyclone(matrix, marker_pairs, iterations, min_iter, min_pairs, processes)

    if processes == 0:
        processes = multiprocessing.cpu_count() - 1

    scores = {cat: get_phase_scores(matrix, cat, iterations, min_iter, min_pairs, pairs, used[cat], processes) for
              cat, pairs in marker_pairs.items()}

    scores_df = DataFrame(scores, columns=marker_pairs.keys())
    scores_df.index = sample_names_filtered
    scores_df['Prediction'] = scores_df.idxmax(axis=1)

    helper.print_info("O", "is done!", "cyclone")

    return scores_df
