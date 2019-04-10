import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib.ticker import NullFormatter
from matplotlib import transforms
import numpy as np

def cc_scatter(
        prediction, actual_labels,
        title = "Cellcycle assignment of single cells with PyPairs",
        desc = "",
        figsize = (10,8),
        dpi = 80,
        save_to = None,
        format = "pdf"
    ):
    sample_ann = {
        i: [x for x, y in enumerate(actual_labels) if i in y] for i in ["G1", "G2M", "S"]
    }

    xs = np.linspace(0, 1, 200)

    # the random data
    x = prediction['G1']
    y = prediction['G2M']

    nullfmt = NullFormatter()         # no labels

    # definitions for the axes
    left, width = 0.1, 0.65
    bottom, height = 0.1, 0.65
    bottom_h = left_h = left + width + 0.02

    rect_scatter = [left, bottom, width, height]
    rect_histx = [left, bottom_h, width, 0.2]
    rect_histy = [left_h, bottom, 0.2, height]

    plt.figure(1, figsize=figsize, dpi=dpi)

    axScatter = plt.axes(rect_scatter)
    axHistx = plt.axes(rect_histx)
    axHisty = plt.axes(rect_histy)

    # no labels
    axHistx.xaxis.set_major_formatter(nullfmt)
    axHisty.yaxis.set_major_formatter(nullfmt)

    # the scatter plot:
    axScatter.plot(prediction.iloc[sample_ann["G1"],0].values, prediction.iloc[sample_ann["G1"],2].values, 'x', color="blue")
    axScatter.plot(prediction.iloc[sample_ann["G2M"],0].values, prediction.iloc[sample_ann["G2M"],2].values, '^', color="green")
    axScatter.plot(prediction.iloc[sample_ann["S"],0].values, prediction.iloc[sample_ann["S"],2].values, '.', color="red")

    axScatter.grid()
    axScatter.margins(0) # remove default margins (matplotlib verision 2+)

    g1_patch = plt.Polygon(
        [[0, 0.5], [0.5, 0.5], [1, 1], [0, 1]],
        closed=None, facecolor='lightgreen', edgecolor='black', alpha = 0.5
    )
    g2m_patch = plt.Polygon(
        [[0.5, 0], [0.5, 0.5], [1, 1], [1, 0]],
        closed=None, facecolor='lightblue', edgecolor='black', alpha = 0.5
    )
    s_patch = plt.Polygon(
        [[0, 0], [0, 0.5], [0.5, 0.5], [0.5, 0]],
        closed=None, facecolor='lightcoral', edgecolor='black', alpha = 0.5
    )
    axScatter.add_patch(g1_patch)
    axScatter.add_patch(g2m_patch)
    axScatter.add_patch(s_patch)

    # now determine nice limits by hand:
    binwidth = 0.05
    xymax = np.max([np.max(np.fabs(x)), np.max(np.fabs(y))])
    lim = (int(xymax/binwidth)) * binwidth

    axScatter.set_xlim((0, lim))
    axScatter.set_ylim((0, lim))

    density_g1 = gaussian_kde(prediction.iloc[sample_ann["G1"], 0].values)
    density_s = gaussian_kde(prediction.iloc[sample_ann["S"], 0].values)
    density_g2m = gaussian_kde(prediction.iloc[sample_ann["G2M"], 0].values)

    density_g1.covariance_factor = lambda : .25
    density_g1._compute_covariance()
    density_s.covariance_factor = lambda : .25
    density_s._compute_covariance()
    density_g2m.covariance_factor = lambda : .25
    density_g2m._compute_covariance()

    axHistx.plot(xs,density_g1(xs), color="Blue")
    axHistx.plot(xs,density_s(xs), color="Red")
    axHistx.plot(xs,density_g2m(xs), color="Green")

    axHistx.xaxis.grid(False)
    axHistx.axvline(x=0.5, ymin=0, ymax=1, color='black', alpha=0.7, linestyle="--")

    base = axHisty.transData
    rot = transforms.Affine2D().rotate_deg(90).scale(-1,1)

    density_g1 = gaussian_kde(prediction.iloc[sample_ann["G1"], 2].values)
    density_s = gaussian_kde(prediction.iloc[sample_ann["S"], 2].values)
    density_g2m = gaussian_kde(prediction.iloc[sample_ann["G2M"], 2].values)

    density_g1.covariance_factor = lambda : .25
    density_g1._compute_covariance()
    density_s.covariance_factor = lambda : .25
    density_s._compute_covariance()
    density_g2m.covariance_factor = lambda : .25
    density_g2m._compute_covariance()

    axHisty.plot(xs,density_g1(xs), color="Blue", transform= rot + base)
    axHisty.plot(xs,density_s(xs), color="Red", transform= rot + base)
    axHisty.plot(xs,density_g2m(xs), color="Green", transform= rot + base)

    axHisty.transData

    axHisty.xaxis.grid(False)
    axHisty.axhline(y=0.5, color='black', alpha=0.7, linestyle="--")

    axHisty.grid(False)
    axHistx.grid(False)

    axScatter.legend(('G1', 'G2M', 'S'),
               loc=1, bbox_to_anchor=(1, 1))

    axScatter.set_title(title, y=1.335)
    axScatter.set_xlabel('G1 score')
    axScatter.set_ylabel('G2M score')
    plt.figtext(0, 0, desc)
    if save_to is not None:
        plt.savefig(save_to, format=format)
    plt.show()
