from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from pypairs import sandbag, helper


def sandbag_from_file(genecounts_file, annotation_file, sep_genecounts=",", sep_annotation=",",
                      annotation_fields=(0, 1), dtype=np.float32, index_col="Unnamed: 0", random_subset=None):
    try:
        gencounts_training = pd.read_csv(Path(genecounts_file), sep=sep_genecounts)
    except:
        helper.print_info("E", "File not found: {}".format(genecounts_file), "sandbag")
        exit(1)

    try:
        gencounts_training.set_index(index_col, inplace=True)
    except:
        helper.print_info("E", "Index column not found {}".format(index_col), "sandbag")
        exit(1)

    matrix = np.array(gencounts_training.values, dtype=dtype)

    annotation = defaultdict(list)

    try:
        for line in open(annotation_file, "r"):
            line = line.replace("\n", "").replace("\r", "")
            infos = line.split(sep_annotation)
            annotation[infos[annotation_fields[1]]] = infos[annotation_fields[0]]
    except FileNotFoundError:
        helper.print_info("E", "File not found: {}".format(genecounts_file), "sandbag")
        exit(1)

    if random_subset is not None:
        for cat, samples in annotation.items():
            annotation[cat] = helper.random_subset(samples, random_subset)

    return sandbag.sandbag(
        matrix, categories=annotation,
        gene_names=list(gencounts_training.index), sample_names=list(gencounts_training.columns),
        rm_zeros=True, fraction=0.65, processes=10, filter_genes_dispersion=True
    )
