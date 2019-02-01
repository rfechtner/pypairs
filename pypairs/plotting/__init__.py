from typing import Union, Optional, Tuple, Mapping, Iterable

from pypairs import utils, settings
from pypairs import log as logg

import numpy as np

from anndata import AnnData
from pandas import DataFrame

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import plotly.plotly as py

import os

def plot_evaluation(
        evaluation: DataFrame,
        label: Optional[str] = "",
        show: Optional[bool] = True,
        save: Optional[bool] = True,
        overwrite: Optional[bool] = False
):
    category_names = list(evaluation.index)
    plt.plot(
        category_names, evaluation.loc[:,"f1"], "ro",
        category_names, evaluation.loc[:, "recall"], "bs",
        category_names, evaluation.loc[:, "precision"], "g^",
        category_names, evaluation.loc[:, "average"], "b_",  markersize=12
    )
    f1_patch = mpatches.Patch(color='red', label="F1 Score")
    recall_patch = mpatches.Patch(color='blue', label="Recall")
    precision_patch = mpatches.Patch(color='green', label="Precision")
    average_patch = mpatches.Patch(color='black', label="Average")
    plt.legend(handles=[f1_patch, recall_patch, precision_patch, average_patch])
    axes = plt.gca()
    axes.set_ylim([0, 1.1])
    plt.title("{} prediction quality".format(label))
    plt.xlabel("Category")
    plt.ylabel("Score")

    if show:
        if settings._is_run_from_ipython():
            logg.hint("running from ipython.. wrapping plot into interactive plotly")
            py.plot_mpl(plt)
        else:
            plt.show()

    if save:
        savepath = settings.figdir + '{}_prediction_quality.png'.format(label)
        logg.hint("saving plot to: {}".format(savepath))
        if os.path.isfile(savepath):
            if overwrite:
                plt.savefig(savepath)
            else:
                logg.warn("file already exists. set `overwrite` to True if you want to replace the file")
        else:
            plt.savefig(savepath)


def show_curve_for_pair(
        data: Union[AnnData, DataFrame, np.ndarray, Iterable[Iterable[float]]],
        pair: Tuple[str, str],
        annotation: Optional[Mapping[str, Iterable[Union[str, int, bool]]]] = None,
        gene_names: Optional[Iterable[str]] = None,
        sample_names: Optional[Iterable[str]] = None,
        show: Optional[bool] = True,
        save: Optional[bool] = True,
        overwrite: Optional[bool] = True
):
    raw_data, gene_names, sample_names, category_names, categories = utils.parse_data_and_annotation(
        data, annotation, gene_names, sample_names
    )

    g1_points = []
    g2_points = []

    g1_gene_mask = utils.to_boolean_mask(pair[0], gene_names)
    g2_gene_mask = utils.to_boolean_mask(pair[1], gene_names)

    for cat in categories:
        g1_points.append(np.average(raw_data[cat, g1_gene_mask]))
        g2_points.append(np.average(raw_data[cat, g2_gene_mask]))

    plt.plot(category_names, g1_points, 'r-', category_names, g2_points, 'b-')
    g1_patch = mpatches.Patch(color='red', label=pair[0])
    g2_patch = mpatches.Patch(color='blue', label=pair[1])
    plt.legend(handles=[g1_patch, g2_patch])
    plt.title("Change in average expression over categories")
    plt.xlabel("Category")
    plt.ylabel("Average gene expression per category")

    if show:
        if settings._is_run_from_ipython():
            logg.hint("running from ipython.. wrapping plot into interactive plotly")
            py.plot_mpl(plt)
        else:
            plt.show()

    if save:
        savepath = settings.figdir + 'pair_curve_{}_vs_{}.png'.format(pair[0], pair[1])
        logg.hint("saving plot to: {}".format(savepath))
        if os.path.isfile(savepath):
            if overwrite:
                plt.savefig(savepath)
            else:
                logg.warn("file already exists. set `overwrite` to True if you want to replace the file")
        else:
            plt.savefig(savepath)
