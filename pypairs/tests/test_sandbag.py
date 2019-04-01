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

min_ref_mat = np.array([
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

min_ref_cats = {
    "G1": [0, 2, 3, 8],
    "S": [4, 5, 6],
    "G2M": [1, 7, 9, 10]
}

min_ref_sample_names = ["A_G1", "B_G2M", "C_G1", "D_G1", "E_S", "F_S", "G_S", "H_G2M", "I_G1", "J_G2M", "K_G2M", "L_X"]
min_ref_gene_names = list(["X{}".format(i) for i in range(0,11)])

min_ref = {
    'G1': [
        ('X0', 'X2'), ('X0', 'X7'), ('X10', 'X3'), ('X5', 'X4'), ('X8', 'X4'),
        ('X5', 'X7'), ('X8', 'X7'), ('X10', 'X7'), ('X8', 'X9')],
    'S': [
        ('X3', 'X0'), ('X5', 'X0'), ('X3', 'X2'), ('X4', 'X2'), ('X7', 'X2'),
        ('X3', 'X8'), ('X3', 'X9'), ('X5', 'X8'), ('X6', 'X8')],
    'G2M': [
        ('X0', 'X10'), ('X2', 'X10'), ('X4', 'X10'), ('X9', 'X5'), ('X6', 'X10'),
        ('X8', 'X10'), ('X9', 'X10')]
}


def test_sandbag_min():
    print("")
    print("")

    print("## Testing correctness of sandbag()")

    print("")
    print("# Testing algorithm on minimal data unjitted")
    print("")

    settings.enable_jit = False

    params = {
        'data': min_ref_mat.T,
        'annotation': min_ref_cats,
        'gene_names': min_ref_gene_names,
        'sample_names': min_ref_sample_names
    }

    marker_pairs = pairs.sandbag(**params)

    assert utils.same_marker(marker_pairs, min_ref)

    utils.benchmark_test(pairs.sandbag, params)

    print("")
    print("# Testing algorithm on minimal data jitted, single core")
    print("")

    settings.verbosity = 4
    settings.n_jobs = 1
    settings.enable_jit = True

    marker_pairs = pairs.sandbag(**params)

    assert utils.same_marker(marker_pairs, min_ref)

    utils.benchmark_test(pairs.sandbag, params)

    print("")
    print("# Testing algorithm on minimal data jitted, multi core")
    print("")

    settings.verbosity = 4
    settings.n_jobs = 4
    settings.enable_jit = True

    marker_pairs = pairs.sandbag(**params)

    assert utils.same_marker(marker_pairs, min_ref)

    utils.benchmark_test(pairs.sandbag, params)


def test_benchmark_sandbag():
    print("")
    print("")

    print("## Testing speed of sandbag()")

    print("")
    print("# Testing on (50, 500) with single core, unjitted")
    print("")

    settings.n_jobs = 1
    settings.enable_jit = False
    utils.benchmark_test(pairs.sandbag, utils.generate_random_testdata(50, 500))

    print("")
    print("# Testing on (50, 500) with single core, jitted")
    print("")

    settings.n_jobs = 1
    settings.enable_jit = True
    utils.benchmark_test(pairs.sandbag, utils.generate_random_testdata(50, 500))

    print("")
    print("# Testing on (150, 1000) with single core, jitted")
    print("")

    settings.n_jobs = 1
    utils.benchmark_test(pairs.sandbag, utils.generate_random_testdata(150, 1000))

    print("")
    print("# Testing on (150, 1000) with multi core, jitted")
    print("")

    settings.n_jobs = 4
    utils.benchmark_test(pairs.sandbag, utils.generate_random_testdata(150, 1000))

    print("")
    print("# Testing on (250, 2500) with single core")
    print("")

    settings.n_jobs = 1
    utils.benchmark_test(pairs.sandbag, utils.generate_random_testdata(250, 2500))

    print("")
    print("# Testing on (250, 2500) with multi core")
    print("")

    settings.n_jobs = 4
    utils.benchmark_test(pairs.sandbag, utils.generate_random_testdata(250, 2500))

def test_sandbag_inputs():
    print("")
    print("")

    print("## Testing different input types for sandbag()")

    settings.n_jobs = 4
    settings.verbosity = 4

    print("")
    print("# Testing AnnData obj, including annotation")
    print("")

    training_data = datasets.leng15(mode='sorted', gene_sub=list(range(0, 1000)))

    marker_pairs = pairs.sandbag(training_data)

    assert utils.same_marker(marker_pairs, ref_markers)

    print("")
    print("# Testing AnnData obj, with separate annotation")
    print("")

    annotation = {
        cat: [i for i, x in enumerate(training_data.obs['category']) if x == cat]
        for cat in ["G1", "S", "G2M"]
    }

    marker_pairs = pairs.sandbag(training_data, annotation=annotation)

    assert utils.same_marker(marker_pairs, ref_markers)

    print("")
    print("# Testing DataFrame obj, with separate annotation")
    print("")

    training_data_df = DataFrame(training_data.X)

    sample_names = list(training_data.obs_names)
    gene_names = list(training_data.var_names)

    training_data_df.Index = sample_names
    training_data_df.columns = gene_names

    marker_pairs = pairs.sandbag(training_data_df, annotation=annotation)

    assert utils.same_marker(marker_pairs, ref_markers)

    print("")
    print("# Testing DataFrame obj, with separate annotation and separate gene-/sample_names")
    print("")

    marker_pairs = pairs.sandbag(training_data_df, annotation, gene_names, sample_names)

    assert utils.same_marker(marker_pairs, ref_markers)

    print("")
    print("# Testing ndarray obj, with separate annotation and separate gene-/sample_names")
    print("")

    training_data_np = training_data_df.values

    marker_pairs = pairs.sandbag(training_data_np, annotation, gene_names, sample_names)

    assert utils.same_marker(marker_pairs, ref_markers)
