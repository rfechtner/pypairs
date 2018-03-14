import random
import uuid
import numpy
from sklearn.metrics import (precision_score, recall_score, f1_score)
from plotly.offline import iplot
import plotly.graph_objs as go
from collections import defaultdict
from pandas import concat as df_concat
import math
import networkx as nx
import pandas
from plotly import tools
from pathlib import Path
import pypairs as pairs


# Returns a plotly Plot
def get_prediction_plot(g1, s, g2m, t="line", title="", xaxis=None, xaxislbl="", samples=None, width=950, height=600):
    xrange = xaxis
    if xrange is None:
        xrange = list(range(0, len(g1)))

    if t == "line":
        # Create traces
        g1_trace = go.Scatter(
            x=xrange,
            y=g1,
            mode='lines+markers',
            marker=dict(
                symbol='circle',
                size=10,
                color='red',
            ),
            name='G1'
        )

        s_trace = go.Scatter(
            x=xrange,
            y=s,
            mode='lines+markers',
            marker=dict(
                symbol='triangle-up',
                size=10,
                color='green',
            ),
            name='S'
        )

        g2m_trace = go.Scatter(
            x=xrange,
            y=g2m,
            mode='lines+markers',
            marker=dict(
                symbol='square',
                size=10,
                color='blue',
            ),
            name='G2M'
        )

        layout = go.Layout(
            title=title,
            width=width,
            height=height,
            xaxis=dict(
                title=xaxislbl,
            ),
            yaxis=dict(
                title='Score [0-1]',
            )
        )
        data = go.Figure(data=[g1_trace, s_trace, g2m_trace], layout=layout)

        return data
    elif t == "pie":
        x = []
        y = []
        for i in range(0, len(g1)):
            x.append(
                (g1[i] - 0.5) /
                math.sqrt(
                    math.pow((g1[i] - 0.5), 2) + math.pow((g2m[i] - 0.5), 2)
                )
            )
            y.append(
                (g2m[i] - 0.5) /
                math.sqrt(
                    math.pow((g1[i] - 0.5), 2) + math.pow((g2m[i] - 0.5), 2)
                )
            )

        txt = ["{} {}".format(xaxislbl, i) for i in xrange]

        datapoints = go.Scatter(
            x=x,
            y=y,
            mode='markers+text',
            marker=dict(
                symbol='circle',
                size=10,
                color='black',
            ),
            name='Sample0',
            text=txt,
            textposition='top left',
            textfont=dict(
                size=12
            )
        )

        lbls = go.Scatter(
            x=[0.625, -0.25, -0.375],
            y=[-0.125, 0.5, -0.375],
            text=['G1',
                  'G2M',
                  'S'],
            mode='text',
            showlegend=False,
            hoverinfo='none'
        )

        data = [datapoints, lbls]

        layout = {
            'title': title,
            'xaxis': {
                'showgrid': False,
                'showticklabels': False,
                'showline': False,
                'range': [-1.5, 1.5],
                'zeroline': False,
            },
            'yaxis': {
                'showgrid': False,
                'showticklabels': False,
                'showline': False,
                'range': [-1.5, 1.5]
            },
            'width': width,
            'height': height,
            'shapes': [
                # G1
                {
                    'type': 'path',
                    'path': ' M 0,0 L0,-1 C 0.552,-1 1,-0.552 1,0 C 1,0.15 0.972,0.442 0.707,0.707 Z',
                    'fillcolor': 'rgba(255,0,0,0.2)'
                },
                # S
                {
                    'type': 'path',
                    'path': ' M 0,0 L-1,0 C -1,-0.552 -0.552,-1 0,-1 Z',
                    'fillcolor': 'rgba(255,255,0,0.2)'
                },
                # G2M
                {
                    'type': 'path',
                    'path': ' M 0,0 L0.707,0.707 C 0.442,0.952 0.205,1 0,1 C -0.552,1 -1,0.552 -1,0 Z',
                    'fillcolor': 'rgba(0,0,255,0.2)'
                }
            ]
        }
        fig = {
            'data': data,
            'layout': layout,
        }

        return fig
    elif t == "scatter":
        if samples is None:
            txt = ["{} {}".format(xaxislbl, i) for i in xrange]
        else:
            txt = samples

        datapoints = go.Scatter(
            x=g1,
            y=g2m,
            mode='markers+text',
            marker=dict(
                symbol='circle',
                size=10,
                color='black',
            ),
            name='Sample0',
            text=txt,
            textposition='top left',
            textfont=dict(
                size=12
            )
        )

        lbls = go.Scatter(
            x=[0.8, 0.4, 0.25],
            y=[0.4, 0.8, 0.25],
            text=['G1',
                  'G2M',
                  'S'],
            mode='text',
            showlegend=False,
            hoverinfo='none'
        )

        data = [datapoints, lbls]

        layout = {
            'title': title,
            'xaxis': {
                'range': [-0.1, 1.1],
            },
            'yaxis': {
                'range': [-0.1, 1.1]
            },
            'width': width,
            'height': height,
            'shapes': [
                # G1
                {
                    'type': 'path',
                    'path': ' M 0.5,0 L1,0 L1,1 L0.5,0.5 Z',
                    'fillcolor': 'rgba(255,0,0,0.2)'
                },
                # S
                {
                    'type': 'path',
                    'path': ' M 0,0 L0.5,0 L0.5,0.5 L0,0.5 Z',
                    'fillcolor': 'rgba(255,255,0,0.2)'
                },
                # G2M
                {
                    'type': 'path',
                    'path': ' M 0,0.5 L0,1 L1,1 L0.5,0.5 Z',
                    'fillcolor': 'rgba(0,0,255,0.2)'
                }
            ]
        }
        fig = {
            'data': data,
            'layout': layout,
        }

        return fig


# Randomly selects k elements from a iterator
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


# Prints a pandas.DataFrame using jQuery DataTable plugin
def DataTable(df):
    from IPython.display import HTML
    output = """<div id="datatable-%(uuid)s">%(html)s
            <script type="text/javascript">
                $(document).ready(function() {
                    require(['dataTables'], function() {
                        $('#datatable-%(uuid)s').find('table.datatable').dataTable({
                        columnDefs: [{ targets: %(sci_cols)s, type: 'scientific' }]});
                    });
                });
            </script>
        </div>
    """ % {'uuid': uuid.uuid1(), 'html': df.to_html(classes="datatable display"),
          'sci_cols': '[%s]' % ",".join([str(i) for i, _ in enumerate(df.dtypes == numpy.float64)])}
    return HTML(output)


# Reformats cyclone's output as one table
def get_prediction_table(prediction):
    samples = list(prediction['prediction'].keys())
    pred = list(prediction['prediction'].values())
    result = df_concat([prediction['scores'], prediction['normalized']], axis=1)
    result = result.assign(prediction=pred)
    result = result.assign(sample=samples)
    result.set_index('sample', inplace=True)
    return result


# Return F1, Recall and Precision Score based on a get_prediction_table output
def evaluate_prediction(predictiction_table, label):
    classes = ["G1", "S", "G2M", None]

    y_label = [classes.index(i) for i in label]
    y_pred = [classes.index(row['prediction']) for index, row in predictiction_table.iterrows()]

    f1 = f1_score(y_label, y_pred, average=None, labels=[0, 1, 2])
    recall = recall_score(y_label, y_pred, average=None, labels=[0, 1, 2])
    precision = precision_score(y_label, y_pred, average=None, labels=[0, 1, 2])

    print("F1 Score: G1: {}, S: {}, G2M: {}".format(*f1))
    print("Reacall: G1: {}, S: {}, G2M: {} ".format(*recall))
    print("Precision: G1: {}, S: {}, G2M: {} ".format(*precision))

    return f1, recall, precision


# Plots the output of a prediction evaluation
def plot_evaluation(f1, recall, precision, average=False, xaxis=None, xaxislbl="", title="", lines=False):
    xrange = xaxis
    if xrange is None:
        xrange = list(range(0, len(f1)))

    if lines:
        mode = 'lines+markers'
    else:
        mode = 'markers'

    # Create traces
    f1_trace = go.Scatter(
        x=xrange,
        y=f1,
        mode=mode,
        marker=dict(
            symbol='circle',
            size=10,
            color='red',
        ),
        name='F1-Score'
    )

    recall_trace = go.Scatter(
        x=xrange,
        y=recall,
        mode=mode,
        marker=dict(
            symbol='square',
            size=10,
            color='blue',
        ),
        name='Recall-Score'
    )

    precision_trace = go.Scatter(
        x=xrange,
        y=precision,
        mode=mode,
        marker=dict(
            symbol='triangle-up',
            size=10,
            color='green',
        ),
        name='Precision-Score'
    )



    layout = go.Layout(
        title=title,
        xaxis=dict(
            title=xaxislbl,
        ),
        yaxis=dict(
             title='F1, Recall, Precision Score',
        )
    )

    if average:
        average_f1_trace = go.Scatter(
            x=xrange,
            y=[numpy.average(f1),numpy.average(f1), numpy.average(f1)],
            mode="lines",
            marker=dict(
                size=10,
                color='red',
            ),
            name='Average F1'
        )

        average_recall_trace = go.Scatter(
            x=xrange,
            y=[numpy.average(recall), numpy.average(recall),numpy.average(recall)],
            mode="lines",
            marker=dict(
                size=10,
                color='blue',
            ),
            name='Average Recall'
        )
        average_precision_trace = go.Scatter(
            x=xrange,
            y=[numpy.average(precision), numpy.average(precision),numpy.average(precision)],
            mode="lines",
            marker=dict(
                size=10,
                color='green',
            ),
            name='Average Precision'
        )

        data = go.Figure(data=[
            f1_trace, recall_trace, precision_trace,
            average_f1_trace, average_recall_trace, average_precision_trace
        ], layout=layout)
    else:
        data = go.Figure(data=[f1_trace, recall_trace, precision_trace], layout=layout)

    return data


# Returns marker pairs as plotly network
def get_pairs_as_network_plot(marker_pairs, genes, triplets=False):
    marker_edges = defaultdict(list)

    for phase, pairs in marker_pairs.items():
        for pair in pairs:
            if triplets:
                marker_edges[phase].append((genes.index(pair[0]), genes.index(pair[1])))
                marker_edges[phase].append((genes.index(pair[1]), genes.index(pair[2])))
            else:
                marker_edges[phase].append((genes.index(pair[0]), genes.index(pair[1])))

    fig = tools.make_subplots(rows=3, cols=1, subplot_titles=('G1', 'S', 'G2M'))

    i = 0
    for phase, pairs in marker_pairs.items():

        net = nx.DiGraph()

        net.add_edges_from(marker_edges[phase])

        pos = nx.spring_layout(net)

        dmin = 1
        for n in pos:
            x, y = pos[n]
            d = (x - 0.5) ** 2 + (y - 0.5) ** 2
            if d < dmin:
                dmin = d

        edge_trace = go.Scatter(
            x=[],
            y=[],
            line=go.Line(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines')

        for edge in net.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace['x'] += [x0, x1, None]
            edge_trace['y'] += [y0, y1, None]

        node_trace = go.Scatter(
            x=[],
            y=[],
            text=[],
            mode='markers',
            hoverinfo='text',
            marker=go.Marker(
                showscale=True,
                # colorscale options
                # 'Greys' | 'Greens' | 'Bluered' | 'Hot' | 'Picnic' | 'Portland' |
                # Jet' | 'RdBu' | 'Blackbody' | 'Earth' | 'Electric' | 'YIOrRd' | 'YIGnBu'
                colorscale='YIGnBu',
                reversescale=True,
                color=[],
                size=10,
                colorbar=dict(
                    thickness=15,
                    title='Node Connections',
                    xanchor='left',
                    titleside='right'
                ),
                line=dict(width=2)))

        for node in net.nodes():
            x, y = pos[node]
            node_trace['x'].append(x)
            node_trace['y'].append(y)

        for node in net.nodes:
            node_trace['marker']['color'].append(net.degree(node))
            node_info = '# of connections: ' + str(net.degree(node))
            node_trace['text'].append(node_info)

        fig.append_trace(edge_trace, i+1, 1)
        fig.append_trace(node_trace, i+1, 1)

        i += 1
    fig['layout'].update(

        title='Network of Marker pairs',
        height=1200,
        showlegend=False,
        hovermodel='closest',
        xaxis1=go.XAxis(showgrid=False, zeroline=False, showticklabels=False),
        xaxis2=go.XAxis(showgrid=False, zeroline=False, showticklabels=False),
        xaxis3=go.XAxis(showgrid=False, zeroline=False, showticklabels=False),
        yaxis1=go.YAxis(showgrid=False, zeroline=False, showticklabels=False),
        yaxis2=go.YAxis(showgrid=False, zeroline=False, showticklabels=False),
        yaxis3=go.YAxis(showgrid=False, zeroline=False, showticklabels=False)
    )

    return fig


# Loads marker pairs from the oscope dataset
def load_ocope_marker(relPath, fraction=0.6, cc_only=True, weighted=False, triplets=False):
    # Load matrix
    gencounts_oscope = pandas.read_csv(Path(relPath + "GSE64016_H1andFUCCI_normalized_EC_human.csv"))

    # Set index right
    gencounts_oscope.set_index("Unnamed: 0", inplace=True)

    # Subset sorted
    gencounts_oscope_sorted = gencounts_oscope.iloc[:,
                              [gencounts_oscope.columns.get_loc(c)
                               for c in gencounts_oscope.columns
                               if "G1_" in c or "G2_" in c or "S_" in c]]

    # Define annotation
    is_G1 = [gencounts_oscope_sorted.columns.get_loc(c) for c in gencounts_oscope_sorted.columns if "G1_" in c]
    is_S = [gencounts_oscope_sorted.columns.get_loc(c) for c in gencounts_oscope_sorted.columns if "S_" in c]
    is_G2M = [gencounts_oscope_sorted.columns.get_loc(c) for c in gencounts_oscope_sorted.columns if "G2_" in c]

    annotation = {
        "G1": list(is_G1),
        "S": list(is_S),
        "G2M": list(is_G2M)
    }

    go_0007049 = [line.replace("\n", "").replace("\r", "") for line in
                  open(relPath + "go_0007049_homoSapiens.csv", "r")]
    cycle_base = [line.split("\t")[0] for i, line in enumerate(open(relPath + "cyclebase_top1000_genes.tsv", "r")) if
                  0 < i]
    cycle_genes = numpy.unique(numpy.concatenate((go_0007049, cycle_base), 0))

    if cc_only:
        return pairs.sandbag(gencounts_oscope_sorted, phases=annotation, subset_genes=list(cycle_genes), fraction=fraction,
                              processes=10, verbose=True, weighted=weighted, triplets=triplets)
    else:
        return pairs.sandbag(gencounts_oscope_sorted, phases=annotation, fraction=fraction,
                             processes=10, verbose=True, weighted=weighted, triplets=triplets)
