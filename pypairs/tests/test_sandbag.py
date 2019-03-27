from pypairs import datasets, settings, utils, pairs
from pandas import DataFrame
import numpy as np

settings.verbosity = 4

ref_markers = {
    "G1": [
        ("OGFOD1","ZMYND8"),("TAT","HIST1H3B"), ("APOL4","ANKRD36B"), ("PRR5","HIST1H3B"), ("MFSD4","HIST1H3B"),
        ("PARP3","HIST1H3B"), ("IL10RB","HIST1H3B")
    ],
    "S": [
        ("SNX5","CKS2"),("SLC29A2","GTSE1"),("ZMYND8","CKS2"),("PLEKHJ1","NDUFAF2"),("NSUN5","NDUFAF2"),
        ("CLSPN","CKS2"),("LITAF","CKS2"),("ANKRD36B","METTL11A"),("ANKRD36B","HNRNPUL1"),("ANKRD36B","RPL39L"),
        ("PRELP","CKS2"),("GABRB3","CKS2"),("RBPJ","RPL27"),("TXLNA","RPL27"),("PLEKHJ1","CHMP4A"),("FLNA","TSTD1"),
        ("FLOT1","NEK3"),("BSG","COX6C"),("JKAMP","CKS2"),("TUBB2B","CKS2"),("INSR","CKS2"),("NANOG","CKS2"),
        ("ARIH2","CKS2"),("SEPW1","CKS2"),("SLC7A3","CKS2"),("CHD4","CKS2"),("FTH1","CKS2"),("ZWINT","CKS2"),
        ("ZNF195","CKS2"),("DHX36","CKS2"),("EEF1D","CKS2"),("MCM5","CKS2"),("SLC3A2","CKS2"),("NDUFB8","CKS2"),
        ("FLNA","NDUFA7"),("SLC25A3","SNRPD1"),("RBPJ","RPL31"),("TXLNA","RPL31"),("SLC29A2","CENPL"),("WDR27","BAX"),
        ("SMARCA4","UBLCP1"),("KLRD1","COX17"),("ACBD7","COX17"),("PDDC1","COX17"),("BSG","SF3B14"),("FLNA","COX17")
    ],
    "G2M": [
        ("KPNA2","H2AFZ"),("RPS6","H2AFZ")
    ]}

min_ref = {
    'G1': [
        ('0', '2'), ('0', '7'), ('10', '3'), ('5', '4'), ('8', '4'),
        ('5', '7'), ('8', '7'), ('10', '7'), ('8', '9')],
    'S': [
        ('3', '0'), ('5', '0'), ('3', '2'), ('4', '2'), ('7', '2'),
        ('3', '8'), ('3', '9'), ('5', '8'), ('6', '8')],
    'G2M': [
        ('0', '10'), ('2', '10'), ('4', '10'), ('9', '5'), ('6', '10'),
        ('8', '10'), ('9', '10')]
}

def test_sandbag_min():
    import string

    data = np.array([
        [48, 10, 63, 30, 1, 8, 4, 15, 20, 50, 20, 18],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [12, 42, 26, 8, 12, 1, 8, 38, 20, 15, 61, 52],
        [22, 8, 8, 4, 43, 33, 21, 3, 1, 6, 4, 6],
        [1, 41, 6, 5, 15, 12, 16, 8, 9, 3, 33, 29],
        [30, 4, 41, 38, 3, 3, 8, 1, 0, 51, 6, 2],
        [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3],
        [2, 31, 6, 5, 31, 26, 29, 19, 28, 13, 29, 31],
        [38, 12, 12, 43, 0, 18, 1, 4, 6, 51, 8, 9],
        [10, 18, 18, 11, 6, 1, 5, 21, 5, 14, 18, 16],
        [99, 1, 87, 66, 18, 19, 11, 2, 19, 75, 2, 4]
    ])

    data = data.T

    cats = {
        "G1": [0, 2, 3, 8],
        "S": [4, 5, 6],
        "G2M": [1, 7, 9, 10]
    }

    sample_names = list(string.ascii_uppercase[:12])
    gene_names = list([str(x) for x in range(11)])

    marker_pairs = pairs.sandbag(
        data=data,
        annotation=cats,
        gene_names=gene_names,
        sample_names=sample_names
    )

    print(marker_pairs)

    assert utils.same_marker(marker_pairs, min_ref)


def test_sandbag_1j():
    settings.n_jobs = 1

    training_data = datasets.leng15(mode='sorted', gene_sub=list(range(0, 1000)))
    marker_pairs = pairs.sandbag(training_data)

    if not utils.same_marker(marker_pairs, ref_markers):
        raise AssertionError()


def test_sandbag_2j_ann():
    settings.n_jobs = 2

    training_data = datasets.leng15(mode='sorted', gene_sub=list(range(0, 1000)))
    annotation = {
        cat: [i for i, x in enumerate(training_data.obs['category']) if x == cat]
        for cat in ["G1", "S", "G2M"]
    }

    marker_pairs = pairs.sandbag(training_data, annotation=annotation)

    if not utils.same_marker(marker_pairs, ref_markers):
        raise AssertionError()


def test_sandbag_unjitted():
    training_data = datasets.leng15(mode='sorted', gene_sub=list(range(0, 1000)))

    marker_pairs_unjitted = pairs.sandbag(training_data)

    if not utils.same_marker(marker_pairs_unjitted, ref_markers):
        raise AssertionError()


def test_sandbag_df():
    training_data = datasets.leng15(mode='sorted', gene_sub=list(range(0, 1000)))
    training_data_df = DataFrame(training_data.X)

    sample_names = list(training_data.obs_names)
    gene_names = list(training_data.var_names)

    training_data_df.Index = sample_names
    training_data_df.columns = gene_names

    annotation = {
        cat: [i for i, x in enumerate(training_data.obs['category']) if x == cat]
        for cat in ["G1", "S", "G2M"]
    }

    marker_pairs_df = pairs.sandbag(training_data_df, annotation)

    if not utils.same_marker(marker_pairs_df, ref_markers):
        raise AssertionError()

    marker_pairs_df = pairs.sandbag(training_data_df, annotation, gene_names, sample_names)

    if not utils.same_marker(marker_pairs_df, ref_markers):
        raise AssertionError()


def test_sandbag_np():
    training_data = datasets.leng15(mode='sorted', gene_sub=list(range(0, 1000)))
    training_data_df = DataFrame(training_data.X)
    training_data_np = training_data_df.values

    sample_names = list(training_data.obs_names)
    gene_names = list(training_data.var_names)
    annotation = {
        cat: [i for i, x in enumerate(training_data.obs['category']) if x == cat]
        for cat in ["G1", "S", "G2M"]
    }

    marker_pairs_np = pairs.sandbag(training_data_np, annotation, gene_names, sample_names)

    if not utils.same_marker(marker_pairs_np, ref_markers):
        raise AssertionError()


def test_sandbag_filtered():
    training_data = datasets.leng15(mode='sorted', gene_sub=list(range(0, 1000)))
    sample_names = list(training_data.obs_names)
    sample_names.pop(3)

    marker_pairs_filtered = pairs.sandbag(
        training_data, filter_genes=list(range(0, 999)), filter_samples=sample_names
    )

    ref2 = ref_markers
    ref2['G2M'].append(('CENPL', 'APOL4'))

    if not utils.same_marker(marker_pairs_filtered, ref2):
        raise AssertionError()
