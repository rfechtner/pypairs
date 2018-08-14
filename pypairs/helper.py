import operator
import random
import time
from collections import defaultdict

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from colorama import Fore, Style


def print_info(type, msg, func="PyPairs"):
    if type == "W":
        type = f'{Fore.YELLOW}WARN {Style.RESET_ALL}'
    elif type == "E":
        type = f'{Fore.RED}ERROR{Style.RESET_ALL}'
    elif type == "I":
        type = f'{Fore.BLUE}INFO {Style.RESET_ALL}'
    elif type == "O":
        type = f'{Fore.GREEN}OK   {Style.RESET_ALL}'

    print("{time} [{type}] '{func}' {msg}".format(time=time.strftime("%H:%M:%S"), type=type, msg=msg, func=func))


def print_init_sandbag(matrix, categories, fraction, processes):
    print("############################################################")
    print("#", end=" ")
    print_info("I", "Running sangbag with:", "sangbag")
    print("#", end=" ")
    print_info("I", "\tMatrix: {} gene for {} samples".format(*matrix.shape), "sandbag")
    print("#", end=" ")
    print_info("I", "\tNo of Categories: {}".format(len(categories.keys())), "sandbag")
    print("#", end=" ")
    print_info("I", "\tSize of Categories:", "sandbag")
    for c, s in categories.items():
        print("#", end=" ")
        print_info("I", "\t\t{cat}: {size}".format(cat=c, size=len(s)), "sandbag")
    print("#", end=" ")
    print_info("I", "\tFraction: {}".format(fraction), "sandbag")
    print("#", end=" ")
    print_info("I", "\tProcesses: {}".format(processes), "sandbag")
    print("############################################################")


def print_init_cyclone(matrix, marker_pairs, iterations, min_iter, min_pairs, processes):
    print("##############################")
    print("#", end=" ")
    print_info("I", "Running cyclone with:", "cyclone")
    print("#", end=" ")
    print_info("I", "\tMatrix: {} gene for {} samples".format(*matrix.shape), "cyclone")
    print("#", end=" ")
    print_info("I", "\tIterations: {}".format(iterations), "cyclone")
    print("#", end=" ")
    print_info("I", "\tMin Iterations: {}".format(min_iter), "cyclone")
    print("#", end=" ")
    print_info("I", "\tMin Pairs: {}".format(min_pairs), "cyclone")
    print("#", end=" ")
    print_info("I", "\tProcesses: {}".format(processes), "cyclone")
    print("#", end=" ")
    print_info("I", "\tNo of Categories: {}".format(len(marker_pairs.keys())), "cyclone")
    print("#", end=" ")
    print_info("I", "\tMarkers (avaible in this set):", "cyclone")
    total = 0
    for m, pairs in marker_pairs.items():
        print("#", end=" ")
        print_info("I", "\t\t{cat}: {size}".format(cat=m, size=len(pairs)), "cyclone")
        total += len(pairs)
    print("#", end=" ")
    print_info("I", "\t\tTotal: {}".format(total), "cyclone")
    print("##############################")


def to_boolean(selected, l):
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
        print_info("E", "Categories must be array-like of type bool, int or str")
        exit(1)

    return mask


def print_marker_stats(marker_pairs):
    for c, pairs in marker_pairs.items():
        counter_up = defaultdict(int)
        counter_down = defaultdict(int)
        for p in pairs:
            counter_up[p[0]] = counter_up[p[0]] + 1
            counter_down[p[1]] = counter_down[p[1]] + 1

        s_up = sorted(counter_up.items(), key=operator.itemgetter(1), reverse=True)
        s_down = sorted(counter_down.items(), key=operator.itemgetter(1), reverse=True)

        print("Top first pair of markers for category {}".format(c))
        print(s_up[:10])
        print("Top second pair of markers for category {}".format(c))
        print(s_down[:10])


def show_curve_for_pair(matrix, gene_names, sample_names, categories, pair):
    g1_points = []
    g2_points = []

    g1_gene_mask = to_boolean(pair[0], gene_names)
    g2_gene_mask = to_boolean(pair[1], gene_names)

    for cat, samples in categories.items():
        cat_mask = to_boolean(samples, sample_names)
        g1_points.append(np.average(matrix[g1_gene_mask, cat_mask]))
        g2_points.append(np.average(matrix[g2_gene_mask, cat_mask]))

    plt.plot(categories.keys(), g1_points, 'r-', categories.keys(), g2_points, 'b-')
    g1_patch = mpatches.Patch(color='red', label=pair[0])
    g2_patch = mpatches.Patch(color='blue', label=pair[1])
    plt.legend(handles=[g1_patch, g2_patch])
    plt.show()


def get_mean_expression(matrix, gene_names, sample_names, categories, gene):
    g1_gene_mask = to_boolean(gene, gene_names)

    for cat, samples in categories.items():
        cat_mask = to_boolean(samples, sample_names)
        print(cat + ":")
        print(matrix[g1_gene_mask, cat_mask])
        print(np.average(matrix[g1_gene_mask, cat_mask]))


def random_subset(iterator, k):
    result = []
    n = 0

    for item in iterator:
        n += 1
        if len(result) < k:
            result.append(item)
        else:
            s = int(random.random() * n)
            if s < k:
                result[s] = item

    return result
